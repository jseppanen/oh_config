
import pickle
import json
from oh import Config


TEST_CONFIG = \
"""
[a]
b = 1
c = "fof"
[d]
e = ${a}
[f]
@call = pathlib/Path.cwd
"""


def test_save_as_json():
    c = Config({"foo": "bar"})
    data = json.dumps(c)
    d = Config.from_json(data)
    assert d == {"foo": "bar"}


def test_save_as_pickle():
    c = Config({"foo": "bar"})
    data = pickle.dumps(c)
    d = pickle.loads(data)
    assert d == {"foo": "bar"}


def test_save_as_str():
    c = Config.from_str(TEST_CONFIG)
    d = Config.from_str(c.to_str())
    assert c == d

    e = Config.from_str(TEST_CONFIG, interpolate=False)
    f = Config.from_str(e.to_str(), interpolate=False)
    assert e == f

    h = Config.from_str(e.to_str())
    assert h == d
