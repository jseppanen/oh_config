import json
import pickle

from oh import Config, ConfigDict

TEST_CONFIG = """
[a]
b = 1
c = "fof"
[d]
e = ${a}
[f]
@call = pathlib:Path.cwd
[g]
x = ${a.c}
"""


def test_save_as_json():
    c = Config.from_str(TEST_CONFIG)
    data = json.dumps(c)
    d = Config.from_json(data)
    assert d == c

    c = Config.from_str(TEST_CONFIG, interpolate=False)
    data = json.dumps(c)
    d = Config.from_json(data)
    assert d == c


def test_save_as_pickle():
    c = Config.from_str(TEST_CONFIG)
    data = pickle.dumps(c)
    d = pickle.loads(data)
    assert isinstance(d, dict)
    assert not isinstance(d, Config)
    assert not isinstance(d, ConfigDict)
    assert not isinstance(d["a"], ConfigDict)
    assert d == c

    c = Config.from_str(TEST_CONFIG, interpolate=False)
    data = pickle.dumps(c)
    d = pickle.loads(data)
    assert isinstance(d, dict)
    assert not isinstance(d, Config)
    assert not isinstance(d, ConfigDict)
    assert d == c


def test_save_as_str():
    c = Config.from_str(TEST_CONFIG)
    d = Config.from_str(c.to_str())
    assert c == d

    e = Config.from_str(TEST_CONFIG, interpolate=False)
    f = Config.from_str(e.to_str(), interpolate=False)
    assert e == f

    h = Config.from_str(e.to_str())
    assert h == d


def test_save_as_dict():
    c = Config.from_str(TEST_CONFIG)
    d = Config(c.to_dict())
    assert c == d

    e = Config.from_str(TEST_CONFIG, interpolate=False)
    f = Config(e.to_dict())
    assert e == f
