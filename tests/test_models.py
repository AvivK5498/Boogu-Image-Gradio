import sys
import types

from app import config, models


def _fake_hf(monkeypatch):
    """Install a fake huggingface_hub module exposing snapshot_download; return the call log."""
    calls = []
    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = lambda repo_id, local_dir: calls.append((repo_id, local_dir)) or local_dir
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return calls


def test_ensure_variant_downloads_repo_to_local_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    calls = _fake_hf(monkeypatch)

    path = models.ensure_variant("base")
    repo, local_dir = calls[-1]
    assert repo == config.VARIANT_REGISTRY["base"].repo
    assert local_dir == str(tmp_path / "Boogu-Image-0.1-Base")
    assert path == local_dir


def test_variant_is_present_checks_model_index(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    assert models.variant_is_present("turbo") is False
    d = config.VARIANT_REGISTRY["turbo"].local_dir
    d.mkdir(parents=True)
    (d / "model_index.json").write_text("{}")
    assert models.variant_is_present("turbo") is True


def test_local_status_covers_all_variants(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    status = models.local_status()
    assert set(status) == set(config.VARIANT_REGISTRY)
