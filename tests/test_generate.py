import pytest

from app import generate
from app.generate import GenRequest


def test_edit_without_image_raises_before_torch():
    # The edit-needs-image guard runs before any torch import, so this is host-testable.
    with pytest.raises(ValueError, match="input image"):
        generate.run(GenRequest(variant="edit", prompt="make it night", image_paths=[]))


def test_genrequest_defaults():
    req = GenRequest(variant="turbo", prompt="a cat")
    assert req.width == 1024 and req.height == 1024
    assert req.image_paths == []
    assert req.randomize_seed is False
