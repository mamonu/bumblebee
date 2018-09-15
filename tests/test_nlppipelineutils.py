import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises


def test_scikitlearn_classifier_exceptions():
    clf = DummyClassifier(strategy="The Ramones") ### whatever!
    assert_raises(ValueError, clf.fit, [], [])
    assert_raises(ValueError, clf.predict, [])
    assert_raises(ValueError, clf.predict_proba, [])


