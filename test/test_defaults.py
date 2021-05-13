import pytest

from oh import Config


def test_defaults_aliasing(oh):
    @oh.register(with_defaults=True)
    def foo(x=0, y=1, z="asdf"):
        return x == 42

    oh.config.load_str(
        """
        [main]
        @call = foo
        0 = 42
        """
    )
    assert oh.config.main["0"] == 42
    assert "x" not in oh.config.main
    assert oh.config.main.y == 1
    assert oh.config.main.z == "asdf"
    res = oh.config.main()
    assert res == True


def test_fill_defaults_simple_config(oh):
    @oh.register(with_defaults=True)
    def foo(required: int, optional: str = "default value"):
        return 42

    # test valid config
    filled = Config.from_str(
        """
        [a]
        @call = foo
        required = 1
        """
    )
    assert filled.a["required"] == 1
    assert filled.a["optional"] == "default value"

    # test invalid config
    filled = Config.from_str(
        """
        [a]
        @call = foo
        optional = "some value"
        """
    )
    with pytest.raises(RuntimeError):
        filled.a()
