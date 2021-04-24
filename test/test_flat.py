import pytest
from oh import Config


def test_flat():
    c = Config.from_str(
        """
        [a]
        x = 42
        [b]
        y = "asdf"
        [c.d]
        z = {"hello": "world"}
        """
    )
    assert len(c.flat) == 3
    assert list(c.flat) == ["a.x", "b.y", "c.d.z.hello"]
    with pytest.raises(KeyError):
        c.flat["c.d.z.helloooo"]
    assert c.flat["c.d.z.hello"] == "world"
