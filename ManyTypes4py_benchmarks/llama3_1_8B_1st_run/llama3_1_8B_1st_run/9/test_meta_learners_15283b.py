from typing import Callable, Dict, List, Optional
from unittest.mock import MagicMock, call, create_autospec, patch
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from fklearn.causal.cate_learning.meta_learners import (
    TREATMENT_FEATURE,
    _append_treatment_feature,
    _create_treatment_flag,
    _filter_by_treatment,
    _fit_by_treatment,
    _get_learners,
    _get_model_fcn,
    _get_unique_treatments,
    _predict_by_treatment_flag,
    _simulate_t_learner_treatment_effect,
    _simulate_treatment_effect,
    causal_s_classification_learner,
    causal_t_classification_learner,
)
from fklearn.exceptions.exceptions import (
    MissingControlError,
    MissingTreatmentError,
    MultipleTreatmentsError,
)
from fklearn.training.classification import logistic_classification_learner
from fklearn.types import LearnerFnType

@pytest.fixture
def base_input_df() -> DataFrame:
    return pd.DataFrame(
        {
            "x1": [1.3, 1.0, 1.8, -0.1, 0.0, 1.0, 2.2, 0.4, -5.0],
            "x2": [10, 4, 15, 6, 5, 12, 14, 5, 12],
            "treatment": ["A", "B", "A", "A", "B", "control", "control", "B", "control"],
            "target": [1, 1, 1, 0, 0, 1, 0, 0, 1],
        }
    )

def test__append_treatment_feature(
    features: List[str], treatment_feature: str
) -> None:
    assert _append_treatment_feature(features, treatment_feature) == features + [treatment_feature]
    assert len(features) > 0
    assert treatment_feature

def test__get_unique_treatments(
    df: DataFrame,
) -> None:
    filtered = _get_unique_treatments(df, treatment_col="treatment", control_name="control")
    expected = ["treatment_A", "treatment_B", "treatment_C"]
    assert sorted(filtered) == sorted(expected)

def test__filter_by_treatment(
    values: List[List[float]],
    df: DataFrame,
    selected_treatment: str,
    expected_values: List[List[float]],
    expected: DataFrame,
) -> None:
    results = _filter_by_treatment(df, treatment_col="treatment", treatment_name=selected_treatment, control_name="control")
    assert_frame_equal(results.reset_index(drop=True), expected)

def test__create_treatment_flag_missing_control(
    df: DataFrame,
) -> None:
    with pytest.raises(Exception) as e:
        _create_treatment_flag(df, treatment_col="treatment", treatment_name="treatment_A", control_name="control")
    assert e.type == MissingControlError
    assert e.value.args[0] == "Data does not contain the specified control."

def test__create_treatment_flag_missing_treatment(
    df: DataFrame,
) -> None:
    with pytest.raises(Exception) as e:
        _create_treatment_flag(df, treatment_col="treatment", treatment_name="treatment_A", control_name="control")
    assert e.type == MissingTreatmentError
    assert e.value.args[0] == "Data does not contain the specified treatment."

def test__create_treatment_flag_multiple_treatments(
    df: DataFrame,
) -> None:
    with pytest.raises(Exception) as e:
        _create_treatment_flag(df, treatment_col="treatment", treatment_name="treatment_A", control_name="control")
    assert e.type == MultipleTreatmentsError
    assert e.value.args[0] == "Data contains multiple treatments."

def test__create_treatment_flag(
    df: DataFrame,
) -> None:
    expected = pd.DataFrame(
        {
            "feature": [1.3, 1.0, 1.8, -0.1],
            "group": ["treatment", "treatment", "control", "control"],
            "target": [1, 1, 1, 0],
            TREATMENT_FEATURE: [1.0, 1.0, 0.0, 0.0],
        }
    )
    results = _create_treatment_flag(df, treatment_col="group", control_name="control", treatment_name="treatment")
    assert_frame_equal(results, expected)

def test__fit_by_treatment(
    base_input_df: DataFrame,
    learner_binary: Callable[[DataFrame], LearnerFnType],
    treatments: List[str],
) -> None:
    learners, logs = _fit_by_treatment(
        base_input_df,
        learner=learner_binary,
        treatment_col="treatment",
        control_name="control",
        treatments=treatments,
    )
    assert len(learners) == len(treatments)
    assert len(logs) == len(treatments)
    assert type(logs) is dict
    assert [type(learner) is LearnerFnType for learner in learners]

def ones_or_zeros_model(
    df: DataFrame,
) -> DataFrame:
    pred = df[TREATMENT_FEATURE].values
    col_dict = {"prediction": pred[:]}
    return df.assign(**col_dict)

def test__predict_by_treatment_flag_positive(
    df: DataFrame,
) -> None:
    assert (_predict_by_treatment_flag(df, ones_or_zeros_model, True, "prediction") == np.ones(df.shape[0])).all()

def test__predict_by_treatment_flag_negative(
    df: DataFrame,
) -> None:
    assert (_predict_by_treatment_flag(df, ones_or_zeros_model, False, "prediction") == np.zeros(df.shape[0])).all()

@patch("fklearn.causal.cate_learning.meta_learners._predict_by_treatment_flag")
def test__simulate_treatment_effect(
    mock_predict_by_treatment_flag: MagicMock,
    df: DataFrame,
    treatments: List[str],
    control_name: str,
    learners: Dict[str, Callable[[DataFrame], LearnerFnType]],
    prediction_column: str,
) -> None:
    mock_predict_by_treatment_flag.side_effect = [
        [0.3, 0.3, 0.0, 1.0],
        [0.2, 0.5, 0.3, 0.0],
        [0.6, 0.7, 0.0, 1.0],
        [1.0, 0.5, 1.0, 1.0],
    ]
    results = _simulate_treatment_effect(
        df,
        treatments=treatments,
        control_name=control_name,
        learners=learners,
        prediction_column=prediction_column,
    )
    expected = pd.DataFrame(
        {
            "x1": [1.3, 1.0, 1.8, -0.1],
            "x2": [10, 4, 15, 6],
            "treatment": ["A", "B", "A", "control"],
            "target": [0, 0, 0, 1],
            "treatment_A__prediction_on_treatment": [0.3, 0.3, 0.0, 1.0],
            "treatment_A__prediction_on_control": [0.2, 0.5, 0.3, 0.0],
            "treatment_A__uplift": [0.1, -0.2, -0.3, 1.0],
            "treatment_B__prediction_on_treatment": [0.6, 0.7, 0.0, 1.0],
            "treatment_B__prediction_on_control": [1.0, 0.5, 1.0, 1.0],
            "treatment_B__uplift": [-0.4, 0.2, -1.0, 0.0],
            "uplift": [0.1, 0.2, -0.3, 1.0],
            "suggested_treatment": ["treatment_A", "treatment_B", "control", "treatment_A"],
        }
    )
    assert_frame_equal(results, expected)

@patch("fklearn.causal.cate_learning.meta_learners._simulate_treatment_effect")
@patch("fklearn.causal.cate_learning.meta_learners._fit_by_treatment")
@patch("fklearn.causal.cate_learning.meta_learners._get_unique_treatments")
@patch("fklearn.causal.cate_learning.meta_learners._append_treatment_feature")
@patch("fklearn.causal.cate_learning.meta_learners._get_learner_features")
def test_causal_s_classification_learner(
    mock_get_learner_features: MagicMock,
    mock_append_treatment_feature: MagicMock,
    mock_get_unique_treatments: MagicMock,
    mock_fit_by_treatment: MagicMock,
    mock_simulate_treatment_effect: MagicMock,
    base_input_df: DataFrame,
) -> None:
    mock_model = create_autospec(logistic_classification_learner)
    mock_fit_by_treatment.side_effect = [(ones_or_zeros_model, dict()), (ones_or_zeros_model, dict())]
    causal_s_classification_learner(
        base_input_df,
        treatment_col="treatment",
        control_name="control",
        prediction_column="prediction",
        learner=mock_model,
    )
    mock_get_learner_features.assert_called()
    mock_append_treatment_feature.assert_called()
    mock_get_unique_treatments.assert_called()
    mock_fit_by_treatment.assert_called()
    mock_simulate_treatment_effect.assert_called()

def test_simulate_t_learner_treatment_effect(
    df: DataFrame,
    treatments: List[str],
    control_name: str,
    prediction_column: str,
    control_learner: MagicMock,
    treatment_learner: MagicMock,
    learners: Dict[str, Callable[[DataFrame], LearnerFnType]],
) -> None:
    result = _simulate_t_learner_treatment_effect(
        df,
        learners=learners,
        treatments=treatments,
        control_name=control_name,
        prediction_column=prediction_column,
    )
    expected = pd.DataFrame(
        {
            "x1": [1.3, 1.0, 1.8, -0.1],
            "x2": [10, 4, 15, 6],
            "treatment": ["A", "B", "A", "control"],
            "target": [0, 0, 0, 1],
            "treatment_A__prediction_on_treatment": [3, 2, 4, 4],
            "treatment_A__uplift": [2, 0, 1, 0],
            "treatment_B__prediction_on_treatment": [3, 2, 4, 4],
            "treatment_B__uplift": [2, 0, 1, 0],
            "uplift": [2, 0, 1, 0],
            "suggested_treatment": ["treatment_A", "control", "treatment_A", "control"],
        }
    )
    assert isinstance(result, pd.DataFrame)
    assert_frame_equal(result, expected)

def test_get_model_fcn(
    base_input_df: DataFrame,
    fake_prediction_column: List[float],
    df_expected: DataFrame,
) -> None:
    def mock_learner(df: DataFrame) -> DataFrame:
        df["prediction"] = fake_prediction_column
        return (lambda x: x, df, dict())

    learner = MagicMock()
    learner.side_effect = mock_learner
    mock_fcn, mock_p_df, mock_logs = _get_model_fcn(
        base_input_df, "treatment", "A", learner
    )
    assert isinstance(mock_fcn, Callable)
    assert_frame_equal(mock_p_df, df_expected)
    assert isinstance(mock_logs, dict)

def test_get_model_fcn_exception(
    base_input_df: DataFrame,
    fake_prediction_column: List[float],
) -> None:
    def mock_learner(df: DataFrame) -> DataFrame:
        df["prediction"] = fake_prediction_column
        return (lambda x: x, df, dict())

    learner = MagicMock()
    learner.side_effect = mock_learner
    with pytest.raises(Exception) as e:
        _ = _get_model_fcn(base_input_df, "treatment", "C", learner)
    assert e.type == MissingTreatmentError

@patch("fklearn.causal.cate_learning.meta_learners._get_model_fcn")
def test_get_learners(
    mock_get_model_fcn: MagicMock,
    unique_treatments: List[str],
) -> None:
    mock_get_model_fcn.side_effect = [
        ("mocked_control_fcn", None, None),
        ("mocked_treatment_fcn_filtering_treatment_a", None, None),
        ("mocked_treatment_fcn_filtering_treatment_b", None, None),
        ("mocked_treatment_fcn_filtering_treatment_c", None, None),
    ]
    learners, logs = _get_learners(
        df="mocked_df",
        unique_treatments=unique_treatments,
        treatment_col="treatment",
        control_name="control",
        control_learner="mocked_control_fcn",
        treatment_learner="mocked_treatment_fcn",
    )
    assert learners["control"] == "mocked_control_fcn"
    assert learners["treatment_a"] == "mocked_treatment_fcn_filtering_treatment_a"
    assert learners["treatment_b"] == "mocked_treatment_fcn_filtering_treatment_b"
    assert learners["treatment_c"] == "mocked_treatment_fcn_filtering_treatment_c"
    assert isinstance(learners, dict)
    assert isinstance(logs, dict)
    calls = [
        call("mocked_df", "treatment", "control", "mocked_control_fcn"),
        call("mocked_df", "treatment", "treatment_a", "mocked_treatment_fcn"),
        call("mocked_df", "treatment", "treatment_b", "mocked_treatment_fcn"),
        call("mocked_df", "treatment", "treatment_c", "mocked_treatment_fcn"),
    ]
    mock_get_model_fcn.assert_has_calls(calls)

@patch("fklearn.causal.cate_learning.meta_learners._simulate_t_learner_treatment_effect")
@patch("fklearn.causal.cate_learning.meta_learners._get_learners")
@patch("fklearn.causal.cate_learning.meta_learners._get_unique_treatments")
def test_causal_t_classification_learner(
    mock_get_unique_treatments: MagicMock,
    mock_get_learners: MagicMock,
    mock_simulate_t_learner_treatment_effect: MagicMock,
    base_input_df: DataFrame,
) -> None:
    mock_get_learners.side_effect = [([], dict())]
    mock_model = create_autospec(logistic_classification_learner)
    causal_t_classification_learner(
        df=base_input_df,
        treatment_col="treatment",
        control_name="control",
        prediction_column="prediction",
        learner=mock_model,
    )
    mock_get_unique_treatments.assert_called()
    mock_get_learners.assert_called()
    mock_simulate_t_learner_treatment_effect.assert_called()
