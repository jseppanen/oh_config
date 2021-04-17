import oh


def test_no_positional_args():
    @oh.register
    def fun(a, b):
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
