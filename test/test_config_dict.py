import math
from oh import ConfigDict


def test_init():
    # init from dict
    conf = ConfigDict({
        "a": 1,
        "b": [1, 2, 3],
        "c": {
            "x": {
                "y": 42
            }
        }
    })

    assert conf.a == 1
    assert conf.b == [1, 2, 3]
    assert conf.c.x.y == 42

    # init from items
    conf = ConfigDict([("a", 1), ("b", {"c": 42})])
    assert conf.a == 1
    assert conf.b.c == 42


def test_call():
    conf = ConfigDict({
        "@call": "math:log",
        "0": 1,
    })
    assert conf() == 0.0
    assert conf(math.exp(1.0)) == 1.0


def test_update():
    conf = ConfigDict({
        "a": 1,
        "b": {"c": 42},
    })
    conf.update({"b": {"c": 43}})
    assert conf.b.c == 43
