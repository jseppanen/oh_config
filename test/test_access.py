import pytest

from oh import Config, ValidationError


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

    c.a.x = 43
    assert c.a.x == 43

    with pytest.raises(ValidationError):
        c.a.y = 43

    with pytest.raises(AttributeError):
        c.a.b.c.d.e = 99


def test_dict_access():
    c = Config.from_str(
        """
        [a]
        0 = -1
        x = 42
        [c.d]
        z = {"hello": "world"}
        """
    )
    assert c == {"a": {"0": -1, "x": 42}, "c": {"d": {"z": {"hello": "world"}}}}

    c["a"]["x"] = 43
    assert c["a"]["0"] == -1
    assert c["a"]["x"] == 43
    assert c["c"]["d"]["z"]["hello"] == "world"

    # test integer access
    assert c["a"][0] == -1
    c["a"][0] = 5
    assert c["a"][0] == 5

    with pytest.raises(ValueError):
        c["a"][-5] = 5

    with pytest.raises(TypeError):
        c["a"][5.0] = 5

    with pytest.raises(TypeError):
        c["a"][True] = 5

    with pytest.raises(TypeError):
        c["a"][None] = 5

    with pytest.raises(TypeError):
        c["a"][(1, 2)] = 5

    with pytest.raises(TypeError):
        c["a"][[1, 2]] = 5

    with pytest.raises(KeyError):
        c["a"]["b"]["c"]["d"]["e"] = 99


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

    with pytest.raises(TypeError):
        c.flat["x.y.z"] = 1

    with pytest.raises(TypeError):
        del c.flat["a.x"]
