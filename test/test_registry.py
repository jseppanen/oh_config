import pathlib


def test_module_import(oh):
    oh.config.load_str(
        """
        [main]
        @call = pathlib/Path.cwd
        """
    )
    res = oh.config.main()
    assert isinstance(res, pathlib.Path)


def test_register_class(oh):
    @oh.register
    class Foo:
        def __init__(self):
            self.x = 42

    oh.config.load_str(
        """
        [main]
        @call = Foo
        """
    )
    res = oh.config.main()
    assert isinstance(res, Foo)
    assert res.x == 42


def test_register_function(oh):
    @oh.register
    def foo(x):
        return x + 1

    oh.config.load_str(
        """
        [main]
        @call = foo
        0 = 42
        """
    )
    res = oh.config.main()
    assert res == 43
