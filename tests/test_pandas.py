import pytest
import numpy as np 
from pandas import DataFrame
import xarray

@pytest.fixture
def df():
    return DataFrame({'A': [1, 2, 3]})


def test_pd(df):

    assert df is not None


def test_xarray(df):

    assert df.to_xarray() is not None