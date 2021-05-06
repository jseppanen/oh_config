import json
from math import isnan
from string import printable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst

from oh import Config

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
    c = Config()
    c.x = value
    assert isinstance(c.x, type(value))
    assert c.x == value or (isinstance(value, float) and isnan(value))
    json.dumps(c)  # this raises if values are not JSON compatible


@given(numpy_scalar_dtypes)
def test_numpy_scalars(dtype):
    value = np.ones(1, dtype=dtype)[0]
    c = Config()
    c.x = value
    assert isinstance(c.x, type(value.item())) and c.x == value.item()
    json.dumps(c)  # this raises if values are not JSON compatible


@given(numpy_scalar_dtypes)
def test_numpy_arrays(dtype):
    value = np.ones(5, dtype=dtype)
    c = Config()
    c.x = value
    assert isinstance(c.x, type(value.tolist())) and c.x == value.tolist()
    json.dumps(c)  # this raises if values are not JSON compatible


def test_illegal_types():
    c = Config()
    with pytest.raises(TypeError):
        c.x = {1: 2}
