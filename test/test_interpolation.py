import pytest
from oh import Config, ParseError


def test_interpolation():
    with pytest.raises(ParseError):
        Config.from_str(
            """
            [a]
            foo = "hello"
            [b]
            bar = ${foo}
            """
        )

    c = Config.from_str(
        """
        [a]
        foo = "hello"
        [b]
        bar = ${a.foo}
        """
    )
    assert c["b"]["bar"] == "hello"

    c = Config.from_str(
        """
        [a]
        foo = "hello"
        [b]
        bar = "${a.foo}!"
        """
    )
    assert c["b"]["bar"] == "hello!"

    with pytest.raises(ParseError):
        Config.from_str(
            """
            [a]
            foo = "hello"
            [b]
            bar = ${a.foo}!
            """
        )

    with pytest.raises(ParseError):
        Config.from_str(
            """
            [a]
            foo = 15
            [b]
            bar = ${a.foo}!
            """
        )

    c = Config.from_str(
        """
        [a]
        foo = ["x", "y"]
        [b]
        bar = ${a.foo}
        """
    )
    assert c["b"]["bar"] == ["x", "y"]

    # Interpolation within the same section
    c = Config.from_str(
        """
        [a]
        foo = "x"
        bar = ${a.foo}
        baz = "${a.foo}y"
        """
    )
    assert c["a"]["bar"] == "x"
    assert c["a"]["baz"] == "xy"


def test_interpolation_lists():
    """Test that lists are preserved correctly"""
    c = Config.from_str(
        """
        [a]
        b = 1
        [c]
        d = ["hello ${a.b}", "world"]
        """,
        interpolate=False
    )
    assert c["c"]["d"] == ["hello ${a.b}", "world"]

    c = Config.from_str(
        """
        [a]
        b = 1
        [c]
        d = ["hello ${a.b}", "world"]
        """
    )
    assert c["c"]["d"] == ["hello 1", "world"]

    with pytest.raises(ParseError):
        Config.from_str(
            """
            [a]
            b = 1
            [c]
            d = [${a.b}, "hello ${a.b}", "world"]
            """,
            interpolate=False
        )

    c = Config.from_str(
        """
        [a]
        b = 1
        [c]
        d = [${a.b}, "hello ${a.b}", "world"]
        """
    )
    assert c["c"]["d"] == [1, "hello 1", "world"]

    c = Config.from_str(
        """
        [a]
        b = 1
        [c]
        d = ["hello", ${a}]
        """
    )
    assert c["c"]["d"] == ["hello", {"b": 1}]

    with pytest.raises(ParseError):
        Config.from_str(
            """
            [a]
            b = 1
            [c]
            d = ["hello", "hello ${a}"]
            """
        )

    c = Config.from_str(
        """
        [a]
        b = 1
        [c]
        d = ["hello", {"x": ["hello ${a.b}"], "y": 2}]
        """
    )
    assert c["c"]["d"] == ["hello", {"x": ["hello 1"], "y": 2}]

    c = Config.from_str(
        """
        [a]
        b = 1
        [c]
        d = ["hello", {"x": [${a.b}], "y": 2}]
        """
    )
    assert c["c"]["d"] == ["hello", {"x": [1], "y": 2}]

    c = Config.from_str(
        """
        [a]
        b = 1
        c = "fof"
        [d]
        e = ${a}
        """
    )
    assert c.d.e == {"b": 1, "c": "fof"}

    with pytest.raises(ParseError):
        Config.from_str(
            """
            [a]
            b = 1
            c = "fof"
            [d]
            e = "${a}"
            """
        )


def test_no_interpolation():
    """Test that interpolation is correctly preserved."""
    c = Config.from_str(
        """
        [a]
        b = 1
        [c]
        d = "${a.b}"
        e = "hello${a.b}"
        """,
        interpolate=False
    )
    assert c["c"]["d"] == "${a.b}"
    assert c["c"]["e"] == "hello${a.b}"

    d = Config.from_str(c.to_str(), interpolate=True)
    assert d["c"]["d"] == "1"
    assert d["c"]["e"] == "hello1"
