"""PipelineManager — owns the single resident diffusers pipeline.

Loads a Boogu variant via the official diffusers route:
`DiffusionPipeline.from_pretrained(local_path, dtype=bf16, trust_remote_code=True)` → returns the
repo's custom BooguImagePipeline / BooguImageTurboPipeline. Rebuilt only when the selected variant
changes (one model resident on the single GPU).

All torch / diffusers imports are lazy so this module imports cleanly under test.
"""

from __future__ import annotations

import gc
import logging
import os

from app import config, models

log = logging.getLogger(__name__)


class PipelineManager:
    def __init__(self) -> None:
        self._pipeline = None
        self._variant: str | None = None

    def get(self, variant_id: str):
        """Return a ready pipeline, rebuilding only if the selected variant changed."""
        if variant_id == self._variant and self._pipeline is not None:
            log.info("Reusing resident pipeline for '%s'", variant_id)
            return self._pipeline
        self._free()
        local_path = models.ensure_variant(variant_id)
        log.info("Building DiffusionPipeline for '%s' from %s", variant_id, local_path)
        self._pipeline = self._build(local_path)
        self._variant = variant_id
        return self._pipeline

    def _build(self, local_path: str):
        import torch
        from diffusers import DiffusionPipeline

        # Some Boogu modules read os.getenv("device") at construction to decide whether to use
        # CUDA/Flash-Attention operators (per the inference guide). Set it before loading.
        os.environ.setdefault("device", "cuda:0")
        pipe = DiffusionPipeline.from_pretrained(
            local_path, dtype=torch.bfloat16, trust_remote_code=True
        )
        pipe.to("cuda")
        return pipe

    def _free(self) -> None:
        if self._pipeline is None:
            return
        self._pipeline = None
        self._variant = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cache clear is best-effort
            pass


# Process-wide singleton (one resident model on the single GPU).
MANAGER = PipelineManager()
