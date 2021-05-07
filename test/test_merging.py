import pytest

from oh import Config


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
    c.load_str(
        """
        [a]
        ab = "yeah"
        [aa]
        cd = "abba"
        """
    )
    assert c == {"a": {"b": 1, "ab": "yeah"}, "c": {"d": "foo"}, "aa": {"cd": "abba"}}


def test_dict_merging():
    c = Config({"a": {"b": 1, "c": 42}})
    c.merge({"a": {"b": 123}})
    assert c == {"a": {"b": 123, "c": 42}}

    c = Config.from_str(
        """
        [a]
        x = 42
        [b.c]
        y = "asdf"
        """
    )
    c.merge({"a": {"z": 1}})
    c.merge({"b": {"d": 2}})
    assert c == {"a": {"x": 42, "z": 1}, "b": {"c": {"y": "asdf"}, "d": 2}}
