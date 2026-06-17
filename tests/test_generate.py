import pytest

from app import generate
from app.generate import GenRequest


def test_first_supported_picks_existing_name():
    params = {"prompt", "true_cfg_scale", "height"}
    assert generate._first_supported(params, generate._TEXT_GUIDANCE_NAMES) == "true_cfg_scale"
    # falls through to guidance_scale if that's the only one present
    assert generate._first_supported({"guidance_scale"}, generate._TEXT_GUIDANCE_NAMES) == "guidance_scale"
    # none present -> None
    assert generate._first_supported({"prompt"}, generate._IMAGE_GUIDANCE_NAMES) is None


def test_image_input_name_preference():
    assert generate._first_supported({"image"}, generate._IMAGE_INPUT_NAMES) == "image"
    assert generate._first_supported({"images", "image"}, generate._IMAGE_INPUT_NAMES) == "image"
    assert generate._first_supported({"input_images"}, generate._IMAGE_INPUT_NAMES) == "input_images"


def test_edit_without_image_raises_before_torch():
    # The edit-needs-image guard runs before any torch import, so this is host-testable.
    with pytest.raises(ValueError, match="input image"):
        generate.run(GenRequest(variant="edit", prompt="make it night", image_paths=[]))


def test_genrequest_defaults():
    req = GenRequest(variant="turbo", prompt="a cat")
    assert req.width == 1024 and req.height == 1024
    assert req.image_paths == []
    assert req.randomize_seed is False
