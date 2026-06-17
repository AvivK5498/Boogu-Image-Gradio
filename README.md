# Boogu-Image-0.1 Gradio Studio (bf16 · no fp8)

A Gradio app to run **Boogu-Image-0.1 at full extent** — full bf16 weights, **no fp8/quantization**.
Text→image (**Base**, quality · **Turbo**, 4-step fast) and text+image→image (**Edit**), with a
variant selector that auto-downloads on demand, multi-image editing, full settings, and a live
inference log. Built on the official `boogu` package pipelines (`BooguImagePipeline` /
`BooguImageTurboPipeline`) — not ComfyUI.

## Quick start

1. **Launch a GPU pod.** Boogu supports **CUDA ≤ 12.8 / torch ≤ 2.11** — do **not** use a CUDA-13 image
   (NGC 25.x); flash-attn + the custom ops won't match. Verified on **`runpod/pytorch:1.0.6-cu1281-torch271-ubuntu2204`**
   (CUDA 12.8.1 / torch 2.7.1 — Boogu's tested env, and it runs sshd). Sized for an **H200 SXM** (Hopper,
   FA via prebuilt wheel); the 10B model is ~20GB in bf16 so any ≥24GB GPU also works (smaller GPUs can
   offload — not wired into the UI). Optional: mount a network volume at `/workspace` (persists your outputs).
2. **Clone + run:**
   ```bash
   git clone https://github.com/AvivK5498/Boogu-Image-Gradio.git
   cd Boogu-Image-Gradio
   export HF_TOKEN=hf_xxxxx          # OPTIONAL — Boogu is ungated; a token just speeds downloads
   ./run.sh                          # installs everything (incl. flash-attn) then starts the app
   ```
   `run.sh` installs once (marker at `/opt/.boogu_setup_done`; `./run.sh --reinstall` to redo), then
   serves Gradio on `0.0.0.0:7860`. On RunPod open `https://<YOUR_POD_ID>-7860.proxy.runpod.net`.
3. **Manage:** `./serve.sh` (restart) · `./stop.sh` (stop). Logs: `/root/boogu_app.log`.

First use of a variant downloads it to local `/models` (fast NVMe; ephemeral — re-downloaded after a
pod restart). Set `BOOGU_MODELS_DIR=/dev/shm/models` for RAM-speed loads.

### Why these scripts are shaped this way (inherited from the LTX pod, hard-won)
- **`torch.inference_mode()`** around generation — without it autograd retains activations (VRAM bloat).
- **Gradio 6 needs `GRADIO_SSR_MODE=False`** — SSR-on uses a Node sidecar that breaks reverse proxies.
- **Outputs need `allowed_paths`** — Gradio only serves files under cwd/tmp unless told otherwise.
- **Persistence via tmux**, not systemd — these containers have no systemd (PID 1 is `docker-init`).
- **Never `pkill -f app.ui`** — that substring also matches the SSH command running it. Use `pkill -f "[m] app.ui"`.
- **`os.environ["device"]` is set before load** — some Boogu modules read it at construction to pick CUDA/FA ops.

## Variants

| Variant | Task | Steps | CFG | Repo |
|---------|------|-------|-----|------|
| **Base** | text→image (quality, fine-tune base) | 25–50 | ~4.0 | `Boogu/Boogu-Image-0.1-Base` |
| **Turbo** | text→image (fast, distilled) | **4** (fixed) | **0.0** | `Boogu/Boogu-Image-0.1-Turbo` |
| **Edit** | text + image→image (1+ input images) | 25–50 | ~5.0 | `Boogu/Boogu-Image-0.1-Edit` |

All variants are 10B (base = Qwen3-VL-8B), Apache-2.0, ungated. Switching the variant in the UI rebuilds
the resident pipeline. Turbo hides the steps/CFG fields (they're fixed by the distillation).

## How calls are dispatched

The pipeline `__call__` uses Boogu's own argument names (verified against
`boogu/pipelines/boogu/pipeline_boogu.py`): **`instruction`**, `negative_instruction`, `input_images`
(a list of PIL images for Edit), `text_guidance_scale`, `image_guidance_scale`, `num_inference_steps`,
`generator`, `height`, `width`; output is `result.images`. Turbo is the `BooguImageTurboPipeline`
subclass (DMD few-step) and needs **text/image guidance = 1.0** (1.0 = no CFG — the model card's "0.0"
is misleading); the app sets that automatically.

## Constraints

- Width/height are snapped to multiples of **64** in the backend (FLUX VAE /8 + patchify; tested 1024/2048).

## Tests

Host-side logic tests (no GPU/torch needed): `python -m pytest tests/ -q`.

See [`docs/specs/2026-06-17-boogu-image-gradio/`](docs/specs/2026-06-17-boogu-image-gradio/) for the design spec.
