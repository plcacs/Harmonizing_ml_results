from typing import Callable, Dict
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from fklearn.causal.cate_learning.meta_learners import TREATMENT_FEATURE, _append_treatment_feature, _create_treatment_flag, _filter_by_treatment, _fit_by_treatment, _get_learners, _get_model_fcn, _get_unique_treatments, _predict_by_treatment_flag, _simulate_t_learner_treatment_effect, _simulate_treatment_effect, causal_s_classification_learner, causal_t_classification_learner
from fklearn.exceptions.exceptions import MissingControlError, MissingTreatmentError, MultipleTreatmentsError
from fklearn.training.classification import logistic_classification_learner
from fklearn.types import LearnerFnType

@pytest.fixture
def base_input_df() -> DataFrame:
    return pd.DataFrame({'x1': [1.3, 1.0, 1.8, -0.1, 0.0, 1.0, 2.2, 0.4, -5.0], 'x2': [10, 4, 15, 6, 5, 12, 14, 5, 12], 'treatment': ['A', 'B', 'A', 'A', 'B', 'control', 'control', 'B', 'control'], 'target': [1, 1, 1, 0, 0, 1, 0, 0, 1]})

def test__append_treatment_feature() -> None:
    features = ['feat1', 'feat2', 'feat3']
    treatment_feature = 'treatment'
    assert _append_treatment_feature(features, treatment_feature) == features + [treatment_feature]
    assert len(features) > 0
    assert treatment_feature

def test__get_unique_treatments() -> None:
    df = pd.DataFrame({'feature': [1.0, 4.0, 1.0, 5.0, 3.0], 'treatment': ['treatment_A', 'treatment_C', 'treatment_B', 'treatment_A', 'control'], 'target': [1, 1, 0, 0, 1]})
    filtered = _get_unique_treatments(df, treatment_col='treatment', control_name='control')
    expected = ['treatment_A', 'treatment_B', 'treatment_C']
    assert sorted(filtered) == sorted(expected)

def test__filter_by_treatment() -> None:
    values = [[1.0, 'treatment_A', 1], [4.0, 'treatment_C', 1], [1.0, 'treatment_B', 0], [5.0, 'treatment_A', 0], [3.0, 'control', 1]]
    df = pd.DataFrame(data=values, columns=['feat1', 'treatment', 'target'])
    selected_treatment = 'treatment_A'
    expected_values = [[1.0, 'treatment_A', 1], [5.0, 'treatment_A', 0], [3.0, 'control', 1]]
    expected = pd.DataFrame(data=expected_values, columns=['feat1', 'treatment', 'target'])
    results = _filter_by_treatment(df, treatment_col='treatment', treatment_name=selected_treatment, control_name='control')
    assert_frame_equal(results.reset_index(drop=True), expected)

def test__create_treatment_flag_missing_control() -> None:
    with pytest.raises(Exception) as e:
        df = pd.DataFrame({'feature': [1.0, 4.0, 1.0, 5.0, 3.0], 'treatment': ['treatment_A', 'treatment_A', 'treatment_A', 'treatment_A', 'treatment_A'], 'target': [1, 1, 0, 0, 1]})
        _create_treatment_flag(df, treatment_col='treatment', treatment_name='treatment_A', control_name='control')
    assert e.type == MissingControlError
    assert e.value.args[0] == 'Data does not contain the specified control.'

# Add type annotations to the remaining test functions as well
