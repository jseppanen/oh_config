import datetime as dt
import json
from decimal import Decimal
from math import isnan
from string import printable
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst

from oh import Config

# ints are not strictly json but they're important & supported by Python's json module
jsons = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(printable),
    lambda children: st.lists(children) | st.dictionaries(st.text(printable), children),
)

numpy_scalar_dtypes = st.one_of(
    nst.boolean_dtypes(),
    nst.integer_dtypes(),
    nst.unsigned_integer_dtypes(),
    nst.floating_dtypes(),
)


@given(jsons)
def test_json_types(value):
    c = Config({"x": value})
    assert isinstance(c.x, type(value))
    assert c.x == value or (
        isinstance(c.x, float)
        and isnan(c.x)
        and isinstance(value, float)
        and isnan(value)
    )
    assert_valid_json(c)


@given(numpy_scalar_dtypes)
def test_numpy_scalars(dtype):
    value = np.ones(1, dtype=dtype)[0]
    c = Config({"x": value})
    assert isinstance(c.x, type(value.item())) and c.x == value.item()
    assert_valid_json(c)


@given(numpy_scalar_dtypes)
def test_1d_numpy_arrays(dtype):
    value = np.ones(5, dtype=dtype)
    c = Config({"x": value})
    assert isinstance(c.x, type(value.tolist())) and c.x == value.tolist()
    assert_valid_json(c)


@given(numpy_scalar_dtypes)
def test_2d_numpy_arrays(dtype):
    value = np.ones((5, 5), dtype=dtype)
    c = Config({"x": value})
    assert isinstance(c.x, type(value.tolist())) and c.x == value.tolist()
    assert_valid_json(c)


def test_illegal_types():
    with pytest.raises(TypeError):
        Config({"x": {None: 2}})

    with pytest.raises(TypeError):
        Config({"x": b"ugh"})

    with pytest.raises(TypeError):
        Config({"x": dt.datetime.utcnow()})

    with pytest.raises(TypeError):
        Config({"x": dt.timedelta(days=1)})

    with pytest.raises(TypeError):
        Config({"x": Decimal("3.3")})

    with pytest.raises(TypeError):
        Config({"x": complex(1, 2)})


def assert_valid_json(data: Any) -> None:
    """Test that data is serializable as plain JSON."""
    json.dumps(data)
