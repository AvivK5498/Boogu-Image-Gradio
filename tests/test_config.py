import pytest

from app import config


def test_registry_has_base_turbo_edit():
    assert set(config.VARIANT_REGISTRY) == {"base", "turbo", "edit"}
    assert config.VARIANT_REGISTRY["base"].task == "t2i"
    assert config.VARIANT_REGISTRY["turbo"].task == "t2i"
    assert config.VARIANT_REGISTRY["edit"].task == "edit"
    assert config.DEFAULT_VARIANT in config.VARIANT_REGISTRY


def test_turbo_is_fast_few_step_cfg0():
    turbo = config.VARIANT_REGISTRY["turbo"]
    assert turbo.fast is True
    assert turbo.default_steps == 4
    assert turbo.default_cfg == 0.0
    # Base/Edit are full multi-step with CFG.
    assert config.VARIANT_REGISTRY["base"].fast is False
    assert config.VARIANT_REGISTRY["edit"].fast is False


def test_variant_local_dir_under_models_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    assert config.VARIANT_REGISTRY["edit"].local_dir == tmp_path / "Boogu-Image-0.1-Edit"


@pytest.mark.parametrize("value,expected", [
    (1080, 1088),   # classic 1080p height -> 1088
    (1024, 1024),   # already %64
    (2048, 2048),
    (1000, 1024),
    (10, 64),       # floors at one factor
])
def test_snap_dim_to_multiple_of_64(value, expected):
    n = config.snap_dim(value)
    assert n == expected
    config.validate_dims(n, n)  # snapped value is always valid


def test_snap_dim_rejects_nonpositive():
    with pytest.raises(ValueError):
        config.snap_dim(0)


@pytest.mark.parametrize("w,h", [(1024, 1024), (2048, 1152), (1024, 576)])
def test_validate_dims_ok(w, h):
    config.validate_dims(w, h)


@pytest.mark.parametrize("w,h", [(1000, 1024), (1024, 1010), (0, 64)])
def test_validate_dims_rejects_non_multiple_of_64(w, h):
    with pytest.raises(ValueError):
        config.validate_dims(w, h)
