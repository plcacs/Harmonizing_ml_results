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

def test__append_treatment_feature() -> None:
    ...

def test__get_unique_treatments() -> None:
    ...

def test__filter_by_treatment() -> None:
    ...

def test__create_treatment_flag_missing_control() -> None:
    ...

def test__create_treatment_flag_missing_treatment() -> None:
    ...

def test__create_treatment_flag_multiple_treatments() -> None:
    ...

def test__create_treatment_flag() -> None:
    ...

def test__fit_by_treatment(base_input_df: DataFrame) -> None:
    ...

def ones_or_zeros_model(df: DataFrame) -> Callable:
    ...

def test__predict_by_treatment_flag_positive() -> None:
    ...

def test__predict_by_treatment_flag_negative() -> None:
    ...

def test__simulate_treatment_effect() -> None:
    ...

def test__simulate_t_learner_treatment_effect() -> None:
    ...

def test_get_model_fcn(base_input_df: DataFrame) -> None:
    ...

def test_get_model_fcn_exception(base_input_df: DataFrame) -> None:
    ...

def test_get_learners() -> None:
    ...

def test_causal_s_classification_learner(base_input_df: DataFrame) -> None:
    ...

def test_causal_t_classification_learner(base_input_df: DataFrame) -> None:
    ...
