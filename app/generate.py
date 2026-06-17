"""Generation dispatcher: a UI request -> a rendered .png path.

Boogu exposes one custom diffusers pipeline per variant. The HF card documents only `pipe(prompt)`,
so rather than hard-coding kwarg names we introspect the pipeline's `__call__` signature and pass
only the arguments it accepts — mapping guidance/image to whichever names the variant uses. This
keeps the app correct even if the exact kwarg spelling differs from the docs.
"""

from __future__ import annotations

import inspect
import logging
import random
import time
from dataclasses import dataclass, field

from app import config
from app.pipelines import MANAGER

log = logging.getLogger(__name__)

# Boogu's pipeline __call__ uses `instruction`/`negative_instruction`/`input_images` and the
# *_guidance_scale names (confirmed by reading boogu/pipelines/boogu/pipeline_boogu.py). We still
# resolve against the live signature so a future rename doesn't silently drop an argument.
_TEXT_GUIDANCE_NAMES = ("text_guidance_scale", "true_cfg_scale", "guidance_scale")
_IMAGE_GUIDANCE_NAMES = ("image_guidance_scale", "image_cfg_scale")
_IMAGE_INPUT_NAMES = ("input_images", "image", "images")
_PROMPT_NAMES = ("instruction", "prompt")
_NEGATIVE_NAMES = ("negative_instruction", "negative_prompt")


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


def _supported(pipe) -> set[str]:
    """Parameter names accepted by the pipeline's __call__ (its forward signature)."""
    return set(inspect.signature(pipe.__call__).parameters)


def _first_supported(params: set[str], names: tuple[str, ...]) -> str | None:
    """First candidate kwarg name the pipeline actually accepts (or None)."""
    return next((n for n in names if n in params), None)


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
    params = _supported(pipe)
    gen = torch.Generator(device="cuda").manual_seed(int(req.seed))

    kwargs: dict = {"num_inference_steps": steps, "height": int(req.height),
                    "width": int(req.width), "generator": gen}
    kwargs[_first_supported(params, _PROMPT_NAMES) or "instruction"] = req.prompt
    if req.negative_prompt and (ng := _first_supported(params, _NEGATIVE_NAMES)):
        kwargs[ng] = req.negative_prompt
    if (tg := _first_supported(params, _TEXT_GUIDANCE_NAMES)):
        kwargs[tg] = text_cfg

    if variant.task == "edit":
        images = [_load_image(p) for p in req.image_paths]
        if (ik := _first_supported(params, _IMAGE_INPUT_NAMES)):
            # singular `image` takes one PIL; plural names take the list of inputs.
            kwargs[ik] = images[0] if ik == "image" and len(images) == 1 else images
        if (ig := _first_supported(params, _IMAGE_GUIDANCE_NAMES)):
            kwargs[ig] = float(req.image_guidance_scale)

    # Drop anything the pipeline doesn't accept (defensive against signature differences).
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    log.info("Generating (%s): %d steps, %dx%d, seed %d, kwargs=%s",
             req.variant, steps, req.width, req.height, req.seed, sorted(kwargs))

    with torch.inference_mode():
        result = pipe(**kwargs)

    image = result.images[0]
    out_path = _output_path(req)
    image.save(out_path)
    log.info("Wrote %s", out_path)
    return out_path
