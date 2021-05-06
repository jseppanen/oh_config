import pytest

from oh import Config


def test_merging():
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


@pytest.mark.xfail(reason="WIP")
def test_dict_merging():
    c = Config({"a": {"b": 1, "c": 42}})
    c.update({"a": {"b": 123}})
    assert c == {"a": {"b": 123, "c": 42}}
