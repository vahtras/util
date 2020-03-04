import numpy as np
import numpy.testing as npt

from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

from util.full import init, Matrix

@given(st.integers(min_value=1, max_value=100))
def test_sqrt_method(n):
    """
    Test for symmetric matrices
    """
    a = np.random.random((n, n))
    a = a + a.T
    a2 = a @ a
    _a2 = a2.view(Matrix)
    _a = _a2.sqrt()
    npt.assert_allclose(a2, _a@_a, rtol=1e-7, atol=1e-7)
