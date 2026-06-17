"""Gradio interface for the Boogu-Image-0.1 studio."""

from __future__ import annotations

import collections
import logging
import threading
import time
import traceback

import gradio as gr

from app import config, models
from app.generate import GenRequest, run

log = logging.getLogger(__name__)

# Loggers shown in the panel, with per-source levels. `app` carries our status + live step count;
# `transformers` is pinned to ERROR to drop the repetitive "Kwargs passed to processor.__call__"
# warning that otherwise floods the panel.
_LOG_LEVELS = {"app": logging.INFO, "diffusers": logging.INFO, "transformers": logging.ERROR}
_LOG_SOURCES = tuple(_LOG_LEVELS)


class _BufferHandler(logging.Handler):
    def __init__(self, buf: collections.deque) -> None:
        super().__init__()
        self.buf = buf

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buf.append(self.format(record))
        except Exception:  # noqa: BLE001 - logging must never raise
            pass


def _attach_log_capture():
    buf: collections.deque = collections.deque(maxlen=500)
    handler = _BufferHandler(buf)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    for name, level in _LOG_LEVELS.items():
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        lg.setLevel(level)
    return buf, handler


def _detach_log_capture(handler) -> None:
    for name in _LOG_SOURCES:
        logging.getLogger(name).removeHandler(handler)


_VARIANT_CHOICES = [(v.label, vid) for vid, v in config.VARIANT_REGISTRY.items()]


def _status_text() -> str:
    s = models.local_status()
    return "  ·  ".join(
        f"{'✅' if s.get(vid) else '⬇️'} {v.label}" for vid, v in config.VARIANT_REGISTRY.items()
    )


def _download(variant_id: str) -> str:
    try:
        models.ensure_variant(variant_id)
        return "Download complete.\n" + _status_text()
    except Exception as e:  # noqa: BLE001 - surface the error to the UI
        log.exception("download failed")
        return f"❌ Download failed: {e}"


def _on_variant_change(variant_id: str):
    is_edit = config.VARIANT_REGISTRY[variant_id].task == "edit"
    is_fast = config.VARIANT_REGISTRY[variant_id].fast
    return (
        gr.update(visible=is_edit),                 # input image(s) — Edit only
        gr.update(visible=is_edit),                 # image-guidance slider — Edit only
        gr.update(visible=not is_fast),             # steps — hidden for Turbo (fixed)
        gr.update(visible=not is_fast),             # text-guidance — hidden for Turbo (CFG 0)
    )


def build_app() -> gr.Blocks:
    config.ensure_dirs()

    # Cap the preview so a tall/portrait output doesn't expand down the whole page.
    css = """
    #out_image img { max-height: 70vh !important; width: auto !important;
        margin: 0 auto !important; display: block !important; }
    #out_image { display: flex !important; justify-content: center !important; }
    """
    default = config.DEFAULT_VARIANT
    default_is_edit = config.VARIANT_REGISTRY[default].task == "edit"
    default_is_fast = config.VARIANT_REGISTRY[default].fast

    with gr.Blocks(title="Boogu-Image-0.1 Studio (H200 · bf16 · no fp8)", css=css) as app:
        gr.Markdown("# Boogu-Image-0.1 Studio\nFull bf16 weights on a single H200 — no fp8. "
                    "Base · Turbo · Edit.")

        with gr.Row():
            variant = gr.Dropdown(_VARIANT_CHOICES, value=default, label="Variant")
            dl_btn = gr.Button("⬇️ Download selected variant", variant="secondary")
        status = gr.Markdown(_status_text())

        prompt = gr.Textbox(label="Prompt", lines=3, placeholder="A cinematic photograph of...")
        negative = gr.Textbox(label="Negative prompt", value=config.DEFAULTS["negative_prompt"])

        image_in = gr.File(label="Input image(s) — Edit (1 or more)", file_count="multiple",
                           file_types=["image"], visible=default_is_edit)

        with gr.Accordion("Settings", open=True):
            with gr.Row():
                width = gr.Number(value=config.DEFAULTS["width"], label="Width", precision=0, step=64,
                                  info="Snapped to a multiple of 64 in the backend")
                height = gr.Number(value=config.DEFAULTS["height"], label="Height", precision=0, step=64)
            with gr.Row():
                seed = gr.Number(value=config.DEFAULTS["seed"], label="Seed", precision=0)
                randomize_seed = gr.Checkbox(value=False, label="🎲 Randomize seed")
            with gr.Row():
                steps = gr.Number(value=config.DEFAULTS["num_inference_steps"], label="Steps",
                                  precision=0, visible=not default_is_fast,
                                  info="Turbo ignores this (fixed 4-step distilled)")
                text_guidance = gr.Slider(0, 10, config.DEFAULTS["text_guidance_scale"], step=0.5,
                                          label="Text guidance (CFG)", visible=not default_is_fast)
                image_guidance = gr.Slider(0, 5, config.DEFAULTS["image_guidance_scale"], step=0.1,
                                           label="Image guidance (Edit)", visible=default_is_edit)

        gen_btn = gr.Button("🎨 Generate", variant="primary")
        out_image = gr.Image(label="Output", elem_id="out_image", type="filepath", height="70vh")
        logs_box = gr.Textbox(label="Inference log", lines=12, max_lines=12, interactive=False,
                              autoscroll=True, value="")

        # ---- wiring
        dl_btn.click(_download, variant, status)
        variant.change(_on_variant_change, variant, [image_in, image_guidance, steps, text_guidance])

        scalar_inputs = [variant, prompt, negative, image_in, width, height, seed, randomize_seed,
                         steps, text_guidance, image_guidance]

        def _generate(variant_v, prompt_v, negative_v, files_v, width_v, height_v, seed_v,
                      randomize_v, steps_v, text_g_v, image_g_v):
            # gr.File(multiple) yields a list of file objects/paths (or None).
            paths = [getattr(f, "name", f) for f in (files_v or [])]
            req = GenRequest(
                variant=variant_v, prompt=prompt_v, negative_prompt=negative_v,
                width=int(width_v), height=int(height_v), seed=int(seed_v),
                randomize_seed=bool(randomize_v), num_inference_steps=int(steps_v),
                text_guidance_scale=float(text_g_v), image_guidance_scale=float(image_g_v),
                image_paths=paths,
            )
            buf, handler = _attach_log_capture()
            result: dict = {}

            def _work():
                try:
                    result["path"] = run(req)
                except Exception:  # noqa: BLE001 - surface the traceback to the log panel
                    result["error"] = traceback.format_exc()

            t = threading.Thread(target=_work, daemon=True)
            t0 = time.time()
            t.start()
            try:
                while t.is_alive():
                    yield "\n".join(buf) + f"\n… running ({int(time.time() - t0)}s)", None
                    time.sleep(1.0)
                t.join()
            finally:
                _detach_log_capture(handler)

            if "error" in result:
                yield "\n".join(buf) + "\n\n❌ ERROR:\n" + result["error"], None
            else:
                yield "\n".join(buf) + f"\n✅ done in {int(time.time() - t0)}s", result["path"]

        gen_btn.click(_generate, scalar_inputs, [logs_box, out_image])

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Outputs live on /workspace (outside cwd/tmp); Gradio must be told it may serve them.
    build_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(config.OUTPUT_DIR), str(config.WORKSPACE)],
    )


if __name__ == "__main__":
    main()
