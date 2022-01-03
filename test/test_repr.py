import oh


def test_repr():
    conf = oh.Config({"a": 42, "b": {"c": 666}})
    assert repr(conf) == "Config({'a': 42, 'b': {'c': 666}})"
    assert repr(conf.b) == "ConfigDict({'c': 666})"
