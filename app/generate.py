"""Generation dispatcher: a UI request -> a rendered .png path.

Boogu exposes one custom diffusers pipeline per variant. The HF card documents only `pipe(prompt)`,
so rather than hard-coding kwarg names we introspect the pipeline's `__call__` signature and pass
only the arguments it accepts — mapping guidance/image to whichever names the variant uses. This
keeps the app correct even if the exact kwarg spelling differs from the docs.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

from app import config
from app.pipelines import MANAGER

log = logging.getLogger(__name__)

# Boogu's pipeline __call__ kwargs, confirmed by reading boogu/pipelines/boogu/pipeline_boogu.py:
# instruction / negative_instruction / input_images / text_guidance_scale / image_guidance_scale /
# num_inference_steps / generator / height / width. The Turbo subclass takes the same via *args/
# **kwargs (so signature introspection would see only **kwargs — don't filter, pass names directly).


@dataclass
class GenRequest:
    variant: str                       # base | turbo | edit
    prompt: str
    negative_prompt: str = config.DEFAULTS["negative_prompt"]
    width: int = config.DEFAULTS["width"]
    height: int = config.DEFAULTS["height"]
    seed: int = config.DEFAULTS["seed"]
    randomize_seed: bool = False
    num_inference_steps: int = config.DEFAULTS["num_inference_steps"]
    text_guidance_scale: float = config.DEFAULTS["text_guidance_scale"]
    image_guidance_scale: float = config.DEFAULTS["image_guidance_scale"]
    image_paths: list[str] = field(default_factory=list)   # edit: 1+ input images
    acceleration: str = config.DEFAULT_ACCELERATION         # none | taylorseer | teacache


class _StepBar:
    """Drop-in for diffusers' `progress_bar` that logs each denoising step. The pipeline's own
    tqdm bar writes to stderr and never reaches the UI log panel; logging each `.update()` via the
    `app` logger surfaces a live step count there."""

    def __init__(self, total, iterable=None):
        self.total = total
        self.iterable = iterable
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for x in self.iterable or []:
            yield x
            self.update()

    def update(self, k=1):
        self.n += k
        log.info("diffusion step %d/%s", self.n, self.total)

    # no-op tqdm API the pipeline may touch
    def set_description(self, *a, **k):
        pass

    def set_postfix(self, *a, **k):
        pass

    def close(self):
        pass


def _apply_acceleration(pipe, mode: str) -> None:
    """Toggle Boogu's step-caching on the resident pipeline (idempotent). Always resets first so
    switching modes between runs takes effect. Flags set on both pipeline and transformer because
    TaylorSeer's master flag lives on the pipeline while the transformer reads its own attrs."""
    t = pipe.transformer
    pipe.enable_taylorseer = False
    t.enable_taylorseer = False
    t.enable_taylorseer_for_all_layers = False
    t.enable_teacache = False
    t.enable_teacache_for_all_layers = False
    if mode == "taylorseer":
        pipe.enable_taylorseer = True
        t.enable_taylorseer = True
        t.enable_taylorseer_for_all_layers = True
    elif mode == "teacache":
        t.enable_teacache = True
        t.enable_teacache_for_all_layers = True
        t.teacache_rel_l1_thresh = config.TEACACHE_REL_L1_THRESH
    log.info("Acceleration: %s", mode)


def _install_step_logger(pipe) -> None:
    """Replace the resident pipeline's progress bar with one that logs steps (idempotent)."""
    def progress_bar(iterable=None, total=None):
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        return _StepBar(total, iterable)

    pipe.progress_bar = progress_bar


def _load_image(path: str):
    """Load an Edit input image as RGB (exif-transposed), as Boogu's inference.py does. The
    pipeline resizes inputs internally (max_input_image_pixels), so we don't pre-crop."""
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(path).convert("RGB"))


def _output_path(req: GenRequest) -> str:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return str(config.OUTPUT_DIR / f"{req.variant}_seed{req.seed}_{stamp}.png")


def run(req: GenRequest) -> str:
    """Validate, build/load the pipeline, generate one image, save it, return the png path."""
    variant = config.VARIANT_REGISTRY[req.variant]

    # Snap dims to a valid size so any typed number works.
    sw, sh = config.snap_dim(req.width), config.snap_dim(req.height)
    if (sw, sh) != (req.width, req.height):
        log.info("Snapped resolution %dx%d -> %dx%d (multiple of 64)", req.width, req.height, sw, sh)
        req.width, req.height = sw, sh
    config.validate_dims(req.width, req.height)

    if req.randomize_seed:
        req.seed = random.randint(0, 2**31 - 1)
        log.info("Randomized seed: %d", req.seed)

    # Turbo is distilled: fixed few-step / CFG 0 — ignore the UI's steps/cfg.
    steps = variant.default_steps if variant.fast else int(req.num_inference_steps)
    text_cfg = variant.default_cfg if variant.fast else float(req.text_guidance_scale)

    if variant.task == "edit" and not req.image_paths:
        raise ValueError("Edit mode requires at least one input image.")

    import torch

    pipe = MANAGER.get(req.variant)
    _install_step_logger(pipe)
    # Turbo's 4-step DMD loop doesn't benefit from step-caching, so force it off there.
    _apply_acceleration(pipe, "none" if variant.fast else req.acceleration)
    gen = torch.Generator(device="cuda").manual_seed(int(req.seed))

    kwargs: dict = {
        "instruction": req.prompt,
        "num_inference_steps": steps,
        "height": int(req.height),
        "width": int(req.width),
        "text_guidance_scale": text_cfg,
        "generator": gen,
    }
    if req.negative_prompt:
        kwargs["negative_instruction"] = req.negative_prompt
    if variant.task == "edit":
        kwargs["input_images"] = [_load_image(p) for p in req.image_paths]
        kwargs["image_guidance_scale"] = float(req.image_guidance_scale)

    log.info("Generating (%s): %d steps, %dx%d, seed %d, text_cfg=%s, keys=%s",
             req.variant, steps, req.width, req.height, req.seed, text_cfg, sorted(kwargs))

    with torch.inference_mode():
        result = pipe(**kwargs)

    image = result.images[0]
    out_path = _output_path(req)
    image.save(out_path)
    log.info("Wrote %s", out_path)
    return out_path
