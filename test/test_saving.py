
import pickle
import json
from oh import Config


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
    c = Config.from_str(
        """
        [a]
        b = 1
        c = "fof"
        [d]
        e = ${a}
        """
    )
    d = Config.from_str(c.to_str())
    assert c == d
