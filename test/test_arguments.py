
def test_positional_args(oh):
    @oh.register
    def fun(a, b):
        if a == 1 and b == 42:
            return "OK"

    oh.config.load_str(
        """
        [main]
        @call = fun
        1 = 42
        """
    )
    res = oh.config.main(1)
    assert res == "OK"


def test_keyword_args(oh):
    @oh.register
    def fun(a=3, b=4):
        if a == 1 and b == 42:
            return "OK"

    oh.config.load_str(
        """
        [main]
        @call = fun
        a = 1
        b = 42
        """
    )
    res = oh.config.main()
    assert res == "OK"


def test_no_positional_args(oh):
    @oh.register
    def fun(a, b):
        if a == 1 and b == 42:
            return "OK"

    oh.config.load_str(
        """
        [main]
        @call = fun
        a = 1
        b = 42
        """
    )
    res = oh.config.main()
    assert res == "OK"
