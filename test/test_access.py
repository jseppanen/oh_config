import pytest

from oh import Config


def test_attribute_access():
    c = Config.from_str(
        """
        [a]
        x = 42
        [c.d]
        z = {"hello": "world"}
        """
    )
    assert c.a.x == 42
    assert c.c.d.z.hello == "world"

    c.a.y = 43
    c.c.c = 44
    assert c.a.x == 42
    assert c.a.y == 43
    assert c.c.c == 44
    assert c.c.d.z.hello == "world"


def test_dict_access():
    c = Config.from_str(
        """
        [a]
        x = 42
        [c.d]
        z = {"hello": "world"}
        """
    )
    assert c == {"a": {"x": 42}, "c": {"d": {"z": {"hello": "world"}}}}

    c["a"]["y"] = 43
    c["c"]["c"] = 44
    assert c["a"]["x"] == 42
    assert c["a"]["y"] == 43
    assert c["c"]["c"] == 44
    assert c["c"]["d"]["z"]["hello"] == "world"


def test_flat_access():
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

    with pytest.raises(KeyError):
        c.flat["x.y.z"] = 1

    with pytest.raises(KeyError):
        del c.flat["a.x"]
