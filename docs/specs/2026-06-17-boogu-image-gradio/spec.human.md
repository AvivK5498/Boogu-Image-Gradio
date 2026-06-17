<!-- spec.human.md — Aviv's review surface. ~30-second read. Optimized for VETO, not detail. -->

# Boogu-Image-0.1 Gradio Studio (bf16 · H200) — t2i + fast t2i + image editing

**Type:** `feature/app` · **Full spec:** [`spec.claude.md`](./spec.claude.md)

## ✅ What you'll see when this is done
A Gradio app on the H200 pod (`https://<pod>-7860.proxy.runpod.net`) that runs Boogu-Image-0.1 at full bf16:
pick **Base** (quality t2i) or **Turbo** (4-step fast t2i), or switch to **Edit** mode to upload image(s)
and transform them with a text instruction. Full settings exposed (steps, text/image guidance, size, seed,
randomize), live inference log, outputs saved to `/workspace/outputs`. Same shape as the LTX pod.

## ⚠️ Decisions you're approving
- **Full bf16 only, no fp8** — chose this over *exposing an fp8 toggle* because you confirmed it; H200 (141GB) runs 10B unquantized with no offload. ← change if wrong
- **No LLM prompt-rewriter** — chose this over *the Qwen3-VL-32B enhancer* because you confirmed skip; avoids a ~60GB extra model. ← change if wrong
- **Pure diffusers route (`trust_remote_code`)**, not cloning the github `inference.py` CLI — chose this over *subprocess-ing their CLI* because it mirrors the LTX `PipelineManager` (resident pipeline object, clean Gradio calls). ← change if wrong
- **No multi-LoRA in v1** — LTX had it; Boogu's inference model doesn't document LoRA inference and diffusers LoRA on its custom transformer is unproven. Left out (see cut-list).

## 🎲 Riding on these assumptions
- **`DiffusionPipeline.from_pretrained(repo, trust_remote_code=True)` is fully self-contained** — i.e. the custom `transformer_boogu` / time-shifting scheduler `.py` ship in the HF repo so we don't need to `pip install -e .` the github package. If wrong, setup also clones `boogu-project/Boogu-Image`. (couldn't confirm: didn't read the repo's `.py` files, only `model_index.json`.)
- **The diffusers `__call__` kwargs** are `prompt`, `negative_prompt`, `num_inference_steps`, `true_cfg_scale`/`text_guidance_scale`, `height`, `width`, `generator`, and (Edit) `image=[PIL,...]` + `image_guidance_scale`. Exact names come from the pipeline `.py` — verified on first load on the pod. If different, only `generate.py`'s call mapping changes.
- **Width/height want multiples of 64** (FLUX VAE /8 + patchify); tested sizes are 1024/2048. Backend snaps to 64. If the real constraint is 16/32, snapping is just slightly coarse — harmless.
- **flash-attn installs on Hopper** via the repo's `utils/get_flash_attn.py` (prebuilt wheel). If it fails, falls back to SDPA — slower, still works.

## 🪤 Gotchas
- `trust_remote_code=True` runs the repo's Python on download — expected for this model, but it's why we pin the exact repo.
- Edit is **TI2I** and accepts **multiple** input images (`input_image_paths` plural) — UI supports 1+.
- Turbo is hard-wired to ~4 steps / CFG 0.0; the steps/guidance fields are ignored for it (like LTX distilled).
- 10B bf16 ≈ 20GB; the "80GB" on the card is the no-offload VRAM tier, not the file size.

## Done when
- [ ] On the pod, Base and Turbo generate a 1024² image from a prompt; Edit transforms an uploaded image from an instruction.
- [ ] Model selector auto-downloads the chosen variant to `/models` on first use; outputs land in `/workspace/outputs`.
- [ ] Full settings exposed + randomize-seed; live log; preview capped to page (not full-page sprawl).
- [ ] `./run.sh` installs everything (incl. flash-attn) and serves on 7860; `serve.sh`/`stop.sh` manage it.
- [ ] Host-side logic tests pass (`pytest tests/ -q`, no GPU needed).

## The plan
1. **Scaffold** the new repo by adapting the proven ltx-pod skeleton (`config/models/pipelines/generate/ui` + `run/serve/stop/setup.sh`).
2. **Build-first:** `config.py` (variant registry, dim-snap, defaults) → `models.py` (HF snapshot download) → `pipelines.py` (resident `DiffusionPipeline`).
3. **Parallel slices:** `generate.py` (t2i + edit dispatch, image fitting), `ui.py` (Gradio), `setup.sh`+scripts, `README` + tests.
4. **Verify on pod:** first real Base/Turbo/Edit generations; confirm the `__call__` kwargs against the live pipeline.

## ✂️ Not asked for — cut?
- **Multi-LoRA UI** — you said "mimic" LTX, but LoRA isn't a documented Boogu inference ability. Default: drop for v1, add later if it works.
- **2048² + tiling, batch generation, prompt-rewriter** — default: drop for v1 (2K still selectable as a size, just untuned).
