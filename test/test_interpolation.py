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

    # multiple string interpolations
    c = Config.from_str(
        """
        [a]
        x = "x"
        y = 1
        z = 3.14159
        zz = "foo ${a.x} ${a.y} ${a.z}"
        """
    )
    assert c.a.zz == "foo x 1 3.14159"

    # test all types
    c = Config.from_str(
        """
        [a]
        int = 42
        float = 3.14159
        str = "foobar"
        bool = true
        null = null
        x = "${a.int} ${a.float} ${a.str} ${a.bool} ${a.null}"
        y = {"a": ${a.int}, "b": ${a.float}, "c": ${a.str}, "d": ${a.bool}, "e": ${a.null}}
        z = [${a.int}, ${a.float}, ${a.str}, ${a.bool}, ${a.null}]
        """
    )
    assert c.a.x == "42 3.14159 foobar True None"
    assert c.a.y == {"a": 42, "b": 3.14159, "c": "foobar", "d": True, "e": None}
    assert c.a.z == [42, 3.14159, "foobar", True, None]

    # leading and trailing text
    c = Config.from_str(
        """
        [a]
        b = "ergo"
        c = "cogito ${a.b} sum"
        """
    )
    assert c.a.c == "cogito ergo sum"

    c = Config.from_str(
        """
        [a]
        b = "zip"
        c = "${a.b}${a.b}"
        """
    )
    assert c.a.c == "zipzip"

    with pytest.raises(ParseError):
        Config.from_str(
            """
            [a]
            b = "zip"
            c = ${a.b}${a.b}
            """
        )


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

    c = Config.from_str(
        """
        [a]
        b = 1
        [c]
        d = [${a.b}, "hello ${a.b}", "world"]
        """,
        interpolate=False
    )
    assert c["c"]["d"] == [{"@ref": "a.b"}, "hello ${a.b}", "world"]

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
        f = ${a}
        """,
        interpolate=False
    )

    assert c["c"]["d"] == "${a.b}"
    assert c["c"]["e"] == "hello${a.b}"
    assert c["c"]["f"] == {"@ref": "a"}

    d = Config.from_str(c.to_str(), interpolate=True)
    assert d["c"]["d"] == "1"
    assert d["c"]["e"] == "hello1"
    assert d["c"]["f"] == {"b": 1}

    c = Config.from_str(
        """
        [a]
        b = 1
        [c.d]
        @ref = a
        """
    )
    assert c.flat["a.b"] == 1
    assert c.flat["c.d"] == {"@ref": "a"}

    d = Config.from_str(c.to_str(), interpolate=True)
    assert d.flat["a.b"] == 1
    assert d.flat["c.d"] == {"b": 1}
