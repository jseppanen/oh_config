import json
from typing import Any

import numpy as np
import pytest

from oh import Config, ValidationError


def test_load_merging():
    c = Config()
    c.load_str(
        """
        [a]
        b = 1
        [c]
        d = "foo"
        """
    )

    with pytest.raises(ValidationError):
        c.load_str(
            """
            [a]
            ab = "yeah"
            [aa]
            cd = "abba"
            """
        )

    assert c == {"a": {"b": 1}, "c": {"d": "foo"}}

    c.load_str(
        """
        [a]
        ab = "yeah"
        [aa]
        cd = "abba"
        """,
        merge_schema=True,
    )
    assert c == {"a": {"b": 1, "ab": "yeah"}, "c": {"d": "foo"}, "aa": {"cd": "abba"}}


def test_dict_merging():
    c = Config({"a": {"b": 1, "c": 42}})
    c.update({"a": {"b": 123}})
    assert c == {"a": {"b": 123, "c": 42}}

    c = Config.from_str(
        """
        [a]
        x = 42
        [b.c]
        y = "asdf"
        """
    )
    c.update({"a": {"z": 1}}, merge_schema=True)
    c.update({"b": {"d": 2}}, merge_schema=True)
    assert c == {"a": {"x": 42, "z": 1}, "b": {"c": {"y": "asdf"}, "d": 2}}


def test_update():
    c = Config.from_str(
        """
        [a]
        x = 42
        [b.c]
        y = "asdf"
        """
    )

    with pytest.raises(ValidationError):
        c.update({"b": {"d": 2}})

    c.update({"b": {"d": 2}}, merge_schema=True)
    assert c == {"a": {"x": 42}, "b": {"c": {"y": "asdf"}, "d": 2}}

    c.update({"b": {"d": np.int64(64)}})
    assert c == {"a": {"x": 42}, "b": {"c": {"y": "asdf"}, "d": 64}}
    assert_valid_json(c)


def assert_valid_json(data: Any) -> None:
    """Test that data is serializable as plain JSON."""
    json.dumps(data)
