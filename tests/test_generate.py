import pytest

from app import config, generate
from app.generate import GenRequest


class _FakeT:
    pass


class _FakePipe:
    def __init__(self):
        self.transformer = _FakeT()


def test_apply_acceleration_taylorseer_single_stream_then_reset():
    pipe = _FakePipe()
    generate._apply_acceleration(pipe, "taylorseer")
    assert pipe.enable_taylorseer is True
    assert pipe.transformer.enable_taylorseer is True
    # all-layers must stay OFF (it OOMs the 141GB H200 on this 10B model)
    assert pipe.transformer.enable_taylorseer_for_all_layers is False
    # switching modes must reset prior flags
    generate._apply_acceleration(pipe, "none")
    assert pipe.enable_taylorseer is False
    assert pipe.transformer.enable_teacache is False


def test_apply_acceleration_teacache_single_stream():
    pipe = _FakePipe()
    generate._apply_acceleration(pipe, "teacache")
    assert pipe.transformer.enable_teacache is True
    assert pipe.transformer.enable_teacache_for_all_layers is False
    assert pipe.transformer.teacache_rel_l1_thresh == config.TEACACHE_REL_L1_THRESH
    assert pipe.enable_taylorseer is False


def test_input_pixel_budget_splits_across_images():
    one = generate._input_pixel_budget(1)
    assert one == config.EDIT_MAX_INPUT_PIXELS                 # single image gets the full ~1MP
    assert generate._input_pixel_budget(2) == config.EDIT_MAX_INPUT_PIXELS // 2
    assert generate._input_pixel_budget(4) == config.EDIT_MAX_INPUT_PIXELS // 4
    # many images floor at the minimum so each stays usable
    assert generate._input_pixel_budget(1000) == config.EDIT_MIN_INPUT_PIXELS


def test_default_acceleration_is_taylorseer():
    # Single-stream TaylorSeer: measured ~1.5x faster at ~56GB on the H200 (OOM-safe, no offload).
    assert config.DEFAULT_ACCELERATION == "taylorseer"


def test_acceleration_registry_and_default():
    assert set(config.ACCELERATION) == {"none", "taylorseer", "teacache"}
    assert config.DEFAULT_ACCELERATION in config.ACCELERATION


def test_edit_without_image_raises_before_torch():
    # The edit-needs-image guard runs before any torch import, so this is host-testable.
    with pytest.raises(ValueError, match="input image"):
        generate.run(GenRequest(variant="edit", prompt="make it night", image_paths=[]))


def test_genrequest_defaults():
    req = GenRequest(variant="turbo", prompt="a cat")
    assert req.width == 1024 and req.height == 1024
    assert req.image_paths == []
    assert req.randomize_seed is False
