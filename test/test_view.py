import pytest

from oh import ValidationError

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


def test_merge_schema(oh):
    oh.config.load_str(
        """
        [a]
        x = 42
        """
    )
    with pytest.raises(ValidationError):
        oh.config.load_str(
            """
            [c.d]
            z = "asdf"
            """
        )
    oh.config.load_str(
        """
        [c.d]
        z = "asdf"
        """,
        merge_schema=True,
    )
    with oh.config.enter("c.d"):
        assert oh.config.z == "asdf"
        with pytest.raises(AttributeError):
            oh.config.a.x

    assert oh.config.a.x == 42
