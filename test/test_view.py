import pytest


def test_view(oh):
    oh.config.load_str(
        """
        [a]
        x = 42
        [c.d]
        z = "asdf"
        """
    )
    with oh.config.enter("c.d"):
        assert oh.config.z == "asdf"
        with pytest.raises(AttributeError):
            oh.config.a.x

    assert oh.config.a.x == 42
