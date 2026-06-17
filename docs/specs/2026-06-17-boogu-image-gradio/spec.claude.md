<!-- spec.claude.md — execution contract for the fan-out. Mirrors spec.human.md; they must not disagree. -->

# Boogu-Image-0.1 Gradio Studio (bf16 · H200)

- **Work type:** `feature/app`
- **Status:** `draft` → awaiting Aviv approval (do NOT dispatch until approved)
- **Review surface:** [`spec.human.md`](./spec.human.md)

## 1. Problem / Context
Stand up a reusable RunPod studio for **Boogu-Image-0.1** (10B unified image gen/edit model, base = Qwen3-VL-8B,
Apache-2.0, no gating) on a single **H200 SXM** — mirroring the structure/quality bar of the existing LTX-2.3 pod
(`~/src/ltx-pod`). Expose all three release variants through one Gradio app: **Base** (quality t2i), **Turbo**
(4-step distilled t2i), **Edit** (text+image → image). New repo, GitHub remote
`git@github-personal:AvivK5498/Boogu-Image-Gradio.git` (already set), no push without explicit approval.

## 2. Approach & why
Run via the **official diffusers route**: `DiffusionPipeline.from_pretrained(local_path, dtype=torch.bfloat16,
trust_remote_code=True)` → returns `BooguImagePipeline` (Base/Edit) / `BooguImageTurboPipeline` (Turbo).
Confirmed by the HF card snippet (`DiffusionPipeline.from_pretrained("Boogu/Boogu-Image-0.1-Turbo", dtype=bf16, device_map="cuda"); pipe(prompt)`)
and `model_index.json` (`_class_name: BooguImagePipeline`; components `mllm`=Qwen3VLForConditionalGeneration,
`processor`=Qwen3VLProcessor, `scheduler`=FlowMatchEulerDiscreteScheduler, `transformer`=BooguImageTransformer2DModel,
`vae`=AutoencoderKL). This mirrors the LTX `PipelineManager` (resident pipeline, rebuilt only on signature change) —
`app/pipelines.py:75-98` (LTX). Full bf16, **no fp8, no offload** (H200 141GB; card's "80GB" row = no-offload tier).

## 3. Acceptance Criteria
- [ ] On the pod, **Base** and **Turbo** produce a 1024² image from a text prompt; **Edit** transforms an uploaded image from an instruction → (ask: "all its capabilities ... Base Turbo Edit")
- [ ] Model selector **auto-downloads** the chosen variant to `/models` on first use → (ask: "model selector that auto-downloads", mimic LTX)
- [ ] Gradio exposes prompt, negative, steps, text_guidance, image_guidance (Edit), width, height, seed, **randomize-seed**; live inference log → (ask: "supports all of its abilities", mimic LTX)
- [ ] Edit mode reveals image input(s) supporting **1+ images** (`input_image_paths` is plural) → (ask: "Edit")
- [ ] Output preview capped to page height; outputs saved to `/workspace/outputs` → (ask: "mimic the same")
- [ ] `./run.sh` installs (incl. flash-attn) + serves 7860; `serve.sh`/`stop.sh` manage; `pytest tests/ -q` green host-side → (ask: "exact same approach")

## 4. Scope & Non-Goals
**In scope:** new repo `~/src/boogu-image-gradio` — `app/{config,models,pipelines,generate,ui}.py`,
`{run,serve,stop,setup}.sh`, `tests/`, `README.md`, `.gitignore`.
**Non-goals (v1):** fp8 path; LLM prompt-rewriter (Qwen3-VL-32B); multi-LoRA UI; 2K tiling/torch-compile tuning;
batch generation; pushing to GitHub or deploying (approval-gated, separate step).

## 5. Key Decisions & Constraints
- **Decided:** full bf16, no fp8 (Aviv). No prompt-rewriter (Aviv). No multi-LoRA v1 (undocumented for Boogu inference).
- **Decided:** diffusers `trust_remote_code` route, not subprocess of github `inference.py`.
- **Constraint:** set `os.environ["device"]="cuda:0"` BEFORE `from_pretrained` — repo modules read `os.getenv("device")` at construction (INFERENCE_GUIDE).
- **Constraint:** `trust_remote_code=True` executes repo Python — pin exact repo ids; download to local `/models`.
- **Mirror existing:** `~/src/ltx-pod/app/pipelines.py` (PipelineManager), `app/models.py` (ensure_bundle/_fetch),
  `app/ui.py` (Blocks + mode-change visibility + 70vh preview CSS), `app/generate.py` (GenRequest + ffmpeg/PIL fitting),
  `setup.sh`/`run.sh`/`serve.sh`/`stop.sh`, `tests/test_config.py`.
- **Scale:** personal/single-operator pod, one GPU, one resident pipeline. Not multi-tenant.

## 6. Code Surface Map (to create; pattern source in ltx-pod)
- `app/config.py` — `Variant` dataclass + `MODEL_REGISTRY` {base, turbo, edit} (repo id, task t2i/edit, default steps/cfg);
  `DEFAULTS`; `snap_dim(v, 64)`; `OUTPUTS_DIR=/workspace/outputs`, `MODELS_DIR=/models`. ← mirror `ltx-pod/app/config.py`.
- `app/models.py` — `ensure_variant(id)` = HF `snapshot_download(repo, local_dir=/models/<name>)`; `variant_is_present`. ← mirror `ltx-pod/app/models.py`.
- `app/pipelines.py` — `PipelineManager.get(variant)`: build `DiffusionPipeline.from_pretrained(path, dtype=bf16, trust_remote_code=True)`, `.to("cuda")`, resident, rebuild on variant change. ← mirror `ltx-pod/app/pipelines.py:75-98`.
- `app/generate.py` — `GenRequest{variant, prompt, negative, steps, text_guidance, image_guidance, width, height, seed, randomize_seed, image_paths}`; `run()` dispatches t2i vs edit; PIL cover-crop/resize inputs to snapped size; save PNG to OUTPUTS_DIR.
- `app/ui.py` — Gradio Blocks: variant dropdown (base/turbo/edit), Edit reveals `gr.File`/image input(s), settings row, randomize checkbox, output gallery (70vh cap), live log textbox.
- `setup.sh` — NGC PyTorch base; `pip install -U diffusers transformers accelerate huggingface_hub`; flash-attn via repo `utils/get_flash_attn.py` (fallback: SDPA); marker file. `run/serve/stop.sh` — copy LTX verbatim (rename app module).

## 7. Ultracode Dispatch Notes
**Build first (sequential — freezes interfaces):**
- `app/config.py` (registry + dim-snap + defaults) — every other module imports it.
- `app/models.py` (`ensure_variant`) and `app/pipelines.py` (`PipelineManager.get`) — generate/ui depend on these signatures.

**Parallel slices (one agent each):**
- **Slice A** — `app/generate.py` (t2i + edit dispatch, PIL fitting, save). Writes: `app/generate.py`, `tests/test_generate.py`.
- **Slice B** — `app/ui.py` (Gradio Blocks, mode visibility, preview cap). Writes: `app/ui.py`.
- **Slice C** — `setup.sh`/`run.sh`/`serve.sh`/`stop.sh`/`.gitignore`. Writes: those files.
- **Slice D** — `README.md` + `tests/test_config.py` + `tests/test_models.py`. Writes: `README.md`, `tests/test_config.py`, `tests/test_models.py`.

**⛓ Collision audit:** A writes `tests/test_generate.py`, D writes `tests/test_config.py`/`tests/test_models.py` — disjoint. No two slices write the same file. config/models/pipelines are frozen build-first artifacts (in `frozen`).

**Each agent must:** implement its slice + write/green its own host-side tests (no GPU) + self-verify against §3.

```yaml
dispatch:
  frozen: [app/config.py, app/models.py, app/pipelines.py]
  slices:
    - {key: sliceA, writes: [app/generate.py, tests/test_generate.py]}
    - {key: sliceB, writes: [app/ui.py]}
    - {key: sliceC, writes: [setup.sh, run.sh, serve.sh, stop.sh, .gitignore]}
    - {key: sliceD, writes: [README.md, tests/test_config.py, tests/test_models.py]}
  testRunner: "cd ~/src/boogu-image-gradio && python -m pytest tests/ -q"
```

## 8. Assumptions & Open Questions
- **ASSUMPTION:** `from_pretrained(..., trust_remote_code=True)` is self-contained (custom `transformer_boogu`/time-shifting scheduler `.py` ship in the HF repo). Couldn't verify — didn't read repo `.py`, only `model_index.json`. Impact if wrong: setup must also `git clone boogu-project/Boogu-Image && pip install -e .`.
- **ASSUMPTION:** `__call__` kwargs = `prompt`, `negative_prompt`, `num_inference_steps`, `text_guidance_scale` (Turbo ignores/uses 0.0), `height`, `width`, `generator`, + Edit `image=[PIL,...]`, `image_guidance_scale`. HF card confirms only `pipe(prompt)`. Verify on pod first load. Impact: only `generate.py` call mapping changes.
- **ASSUMPTION:** width/height snap to 64 is valid (FLUX VAE /8; tested 1024/2048). Impact if real constraint is 16/32: harmlessly coarser.
- **ASSUMPTION:** `utils/get_flash_attn.py` installs a Hopper wheel on the NGC image. Impact if not: fall back to SDPA (slower, works).
- **OPEN:** does Boogu expose an attention-backend selector like LTX? Not evident; v1 leaves attention to the install default (no UI knob).
