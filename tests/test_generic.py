import pytest
import numpy as np
from pandas import DataFrame
from numpy import random as npr


@pytest.fixture
def df():
    return DataFrame({"A": [1, 2, 3]})


def test_pd(df):

    assert df is not None

def test_np_randomness():
    np.random.seed(54)
    rand_ints_one = npr.randint(500, size=50)
    np.random.seed(54)
    rand_ints_two = npr.randint(500, size=50)
    assert rand_ints_one.all() == rand_ints_two.all()
