"""On-demand model downloader. Pulls a whole variant repo (checkpoint + mllm + vae + the custom
pipeline `.py` that `trust_remote_code` loads) from HuggingFace to local /models, idempotently.

`huggingface_hub` is a light dependency (no torch), so tests mock its one function.
"""

from __future__ import annotations

import logging

from app import config

log = logging.getLogger(__name__)


def ensure_variant(variant_id: str) -> str:
    """Download a variant's full repo snapshot to its local dir. Returns the local path.

    Boogu ships its custom pipeline/transformer code inside the repo (loaded via
    trust_remote_code), so we need the whole snapshot, not single files.
    """
    from huggingface_hub import snapshot_download  # lazy: keeps host imports light

    variant = config.VARIANT_REGISTRY[variant_id]
    variant.local_dir.mkdir(parents=True, exist_ok=True)
    log.info("Ensuring %s -> %s", variant.repo, variant.local_dir)
    return snapshot_download(repo_id=variant.repo, local_dir=str(variant.local_dir))


def variant_is_present(variant_id: str) -> bool:
    """True if a variant looks downloaded (has its model_index.json). No network."""
    variant = config.VARIANT_REGISTRY[variant_id]
    return (variant.local_dir / "model_index.json").exists()


def local_status() -> dict[str, bool]:
    """Per-variant presence, for the UI status line. No network."""
    return {vid: variant_is_present(vid) for vid in config.VARIANT_REGISTRY}
