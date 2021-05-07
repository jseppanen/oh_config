import json
from typing import Any

import numpy as np
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
        0 = -1
        x = 42
        [c.d]
        z = {"hello": "world"}
        """
    )
    assert c == {"a": {"0": -1, "x": 42}, "c": {"d": {"z": {"hello": "world"}}}}

    c["a"]["y"] = 43
    c["c"]["c"] = 44
    assert c["a"]["0"] == -1
    assert c["a"]["x"] == 42
    assert c["a"]["y"] == 43
    assert c["c"]["c"] == 44
    assert c["c"]["d"]["z"]["hello"] == "world"

    # test integer access
    assert c["a"][0] == -1
    c["a"][5] = 5
    assert c["a"][5] == 5

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


def test_update():
    c = Config.from_str(
        """
        [a]
        x = 42
        [b.c]
        y = "asdf"
        """
    )
    c.update({"b": {"d": 2}})
    assert c == {"a": {"x": 42}, "b": {"d": 2}}

    c.update({"b": {"d": np.int64(64)}})
    assert c == {"a": {"x": 42}, "b": {"d": 64}}
    assert_valid_json(c)


def assert_valid_json(data: Any) -> None:
    """Test that data is serializable as plain JSON."""
    json.dumps(data)
