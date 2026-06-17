"""Static configuration: paths, variant registry, defaults, validators.

Pure Python — no torch / diffusers imports — so it is importable (and testable) without a GPU
or the heavy packages installed. Everything heavy lives in pipelines.py / generate.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- paths
WORKSPACE = Path(os.environ.get("BOOGU_WORKSPACE", "/workspace"))
# Downloaded models live on LOCAL disk (default /models) — the /workspace network FS is slow to
# load a 10B (~20GB) checkpoint from. Local disk is ephemeral (re-downloaded after a pod restart)
# but loads fast; set BOOGU_MODELS_DIR=/dev/shm/models for RAM-speed loads if you have the RAM.
MODELS_DIR = Path(os.environ.get("BOOGU_MODELS_DIR", "/models"))
# User content stays on the persistent network volume so it survives restarts.
OUTPUT_DIR = WORKSPACE / "outputs"


# --------------------------------------------------------------------------- variant registry
@dataclass(frozen=True)
class Variant:
    """A selectable Boogu-Image variant = one HF repo + how its pipeline is called."""

    id: str
    label: str
    repo: str
    task: str            # "t2i" (text→image) | "edit" (text+image→image)
    default_steps: int
    default_cfg: float   # text-guidance scale
    fast: bool = False   # distilled (Turbo): fixed few-step / CFG 0 — UI hides steps/cfg

    @property
    def local_dir(self) -> Path:
        return MODELS_DIR / self.repo.split("/")[-1]


# All three are 10B, base = Qwen3-VL-8B, Apache-2.0, ungated. bf16, no fp8 (H200 141GB).
VARIANT_REGISTRY: dict[str, Variant] = {
    "base": Variant(
        id="base", label="Base — quality text→image (25–50 steps)",
        repo="Boogu/Boogu-Image-0.1-Base", task="t2i",
        default_steps=50, default_cfg=4.0,
    ),
    "turbo": Variant(
        id="turbo", label="Turbo — fast text→image (4 steps, distilled)",
        repo="Boogu/Boogu-Image-0.1-Turbo", task="t2i",
        default_steps=4, default_cfg=0.0, fast=True,
    ),
    "edit": Variant(
        id="edit", label="Edit — text + image → image",
        repo="Boogu/Boogu-Image-0.1-Edit", task="edit",
        default_steps=50, default_cfg=5.0,
    ),
}
DEFAULT_VARIANT = "turbo"


# --------------------------------------------------------------------------- generation defaults
DEFAULTS = {
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "num_inference_steps": 50,
    "text_guidance_scale": 4.0,   # CFG for Base/Edit; Turbo overrides to 0.0
    "image_guidance_scale": 1.5,  # Edit only
    "negative_prompt": "",
}


# --------------------------------------------------------------------------- validators
def snap_dim(value: int, factor: int = 64) -> int:
    """Snap a width/height to the nearest positive multiple of `factor` (min one factor).

    Boogu uses the FLUX.1 VAE (/8 spatial) plus transformer patchify; the released model is tested
    at 1024 and 2048 (both %64). Snapping to 64 lets the user type any number and get a valid size
    instead of an error.
    """
    if value <= 0:
        raise ValueError(f"dimension={value} must be positive.")
    return max(factor, round(value / factor) * factor)


def validate_dims(width: int, height: int, factor: int = 64) -> None:
    for name, val in (("width", width), ("height", height)):
        if val <= 0 or val % factor != 0:
            raise ValueError(f"{name}={val} must be a positive multiple of {factor}.")


def ensure_dirs() -> None:
    for d in (MODELS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
