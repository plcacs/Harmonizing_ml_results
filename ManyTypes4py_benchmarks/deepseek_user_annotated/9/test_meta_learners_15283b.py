from typing import Callable, Dict, List, Tuple, Union
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
    causal_t_classification_learner
)
from fklearn.exceptions.exceptions import (
    MissingControlError,
    MissingTreatmentError,
    MultipleTreatmentsError
)
from fklearn.training.classification import logistic_classification_learner
from fklearn.types import LearnerFnType


@pytest.fixture
def base_input_df() -> DataFrame:
    return pd.DataFrame(
        {
            "x1": [1.3, 1.0, 1.8, -0.1, 0.0, 1.0, 2.2, 0.4, -5.0],
            "x2": [10, 4, 15, 6, 5, 12, 14, 5, 12],
            "treatment": [
                "A",
                "B",
                "A",
                "A",
                "B",
                "control",
                "control",
                "B",
                "control",
            ],
            "target": [1, 1, 1, 0, 0, 1, 0, 0, 1],
        }
    )


def test__append_treatment_feature() -> None:
    features: List[str] = ["feat1", "feat2", "feat3"]
    treatment_feature: str = "treatment"

    assert _append_treatment_feature(features, treatment_feature) == features + [
        treatment_feature
    ]
    assert len(features) > 0
    assert treatment_feature


def test__get_unique_treatments() -> None:
    df: DataFrame = pd.DataFrame(
        {
            "feature": [1.0, 4.0, 1.0, 5.0, 3.0],
            "treatment": [
                "treatment_A",
                "treatment_C",
                "treatment_B",
                "treatment_A",
                "control",
            ],
            "target": [1, 1, 0, 0, 1],
        }
    )

    filtered: List[str] = _get_unique_treatments(
        df, treatment_col="treatment", control_name="control"
    )
    expected: List[str] = ["treatment_A", "treatment_B", "treatment_C"]

    assert sorted(filtered) == sorted(expected)


def test__filter_by_treatment() -> None:
    values: List[List[Union[float, str, int]]] = [
        [1.0, "treatment_A", 1],
        [4.0, "treatment_C", 1],
        [1.0, "treatment_B", 0],
        [5.0, "treatment_A", 0],
        [3.0, "control", 1],
    ]

    df: DataFrame = pd.DataFrame(data=values, columns=["feat1", "treatment", "target"])

    selected_treatment: str = "treatment_A"

    expected_values: List[List[Union[float, str, int]] = [
        [1.0, "treatment_A", 1],
        [5.0, "treatment_A", 0],
        [3.0, "control", 1],
    ]

    expected: DataFrame = pd.DataFrame(
        data=expected_values, columns=["feat1", "treatment", "target"]
    )

    results: DataFrame = _filter_by_treatment(
        df,
        treatment_col="treatment",
        treatment_name=selected_treatment,
        control_name="control",
    )

    assert_frame_equal(results.reset_index(drop=True), expected)


def test__create_treatment_flag_missing_control() -> None:
    with pytest.raises(Exception) as e:
        df: DataFrame = pd.DataFrame(
            {
                "feature": [1.0, 4.0, 1.0, 5.0, 3.0],
                "treatment": [
                    "treatment_A",
                    "treatment_A",
                    "treatment_A",
                    "treatment_A",
                    "treatment_A",
                ],
                "target": [1, 1, 0, 0, 1],
            }
        )

        _create_treatment_flag(
            df,
            treatment_col="treatment",
            treatment_name="treatment_A",
            control_name="control",
        )

    assert e.type == MissingControlError
    assert e.value.args[0] == "Data does not contain the specified control."


def test__create_treatment_flag_missing_treatment() -> None:
    with pytest.raises(Exception) as e:
        df: DataFrame = pd.DataFrame(
            {
                "feature": [1.0, 4.0, 1.0, 5.0, 3.0],
                "treatment": [
                    "control",
                    "control",
                    "control",
                    "control",
                    "control",
                ],
                "target": [1, 1, 0, 0, 1],
            }
        )

        _create_treatment_flag(
            df,
            treatment_col="treatment",
            treatment_name="treatment_A",
            control_name="control",
        )

    assert e.type == MissingTreatmentError
    assert e.value.args[0] == "Data does not contain the specified treatment."


def test__create_treatment_flag_multiple_treatments() -> None:
    with pytest.raises(Exception) as e:
        df: DataFrame = pd.DataFrame(
            {
                "feature": [1.0, 4.0, 1.0, 5.0, 3.0],
                "treatment": [
                    "treatment_A",
                    "treatment_C",
                    "treatment_B",
                    "treatment_A",
                    "control",
                ],
                "target": [1, 1, 0, 0, 1],
            }
        )

        _create_treatment_flag(
            df,
            treatment_col="treatment",
            treatment_name="treatment_A",
            control_name="control",
        )

    assert e.type == MultipleTreatmentsError
    assert e.value.args[0] == "Data contains multiple treatments."


def test__create_treatment_flag() -> None:
    df: DataFrame = pd.DataFrame(
        {
            "feature": [1.3, 1.0, 1.8, -0.1],
            "group": ["treatment", "treatment", "control", "control"],
            "target": [1, 1, 1, 0],
        }
    )

    expected: DataFrame = pd.DataFrame(
        {
            "feature": [1.3, 1.0, 1.8, -0.1],
            "group": ["treatment", "treatment", "control", "control"],
            "target": [1, 1, 1, 0],
            TREATMENT_FEATURE: [1.0, 1.0, 0.0, 0.0],
        }
    )

    results: DataFrame = _create_treatment_flag(
        df, treatment_col="group", control_name="control", treatment_name="treatment"
    )

    assert_frame_equal(results, expected)


def test__fit_by_treatment(base_input_df: DataFrame) -> None:
    learner_binary: LearnerFnType = logistic_classification_learner(
        features=["x1", "x2", TREATMENT_FEATURE],
        target="target",
        params={"max_iter": 10},
    )

    treatments: List[str] = ["A", "B"]

    learners: List[LearnerFnType]
    logs: Dict[str, Dict[str, float]]
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


def ones_or_zeros_model(df: DataFrame) -> Callable[[DataFrame], DataFrame]:
    def p(new_df: DataFrame) -> DataFrame:
        pred = new_df[TREATMENT_FEATURE].values

        col_dict = {"prediction": pred[:]}

        return new_df.assign(**col_dict)

    return p(df)


def test__predict_by_treatment_flag_positive() -> None:
    df: DataFrame = pd.DataFrame(
        {
            "x1": [1.3, 1.0, 1.8, -0.1],
            "x2": [10, 4, 15, 6],
            "target": [1, 1, 1, 0],
        }
    )

    assert (
        _predict_by_treatment_flag(df, ones_or_zeros_model, True, "prediction")
        == np.ones(df.shape[0])
    ).all()


def test__predict_by_treatment_flag_negative() -> None:
    df: DataFrame = pd.DataFrame(
        {
            "x1": [1.3, 1.0, 1.8, -0.1],
            "x2": [10, 4, 15, 6],
            "target": [1, 1, 1, 0],
        }
    )

    assert (
        _predict_by_treatment_flag(df, ones_or_zeros_model, False, "prediction")
        == np.zeros(df.shape[0])
    ).all()


@patch("fklearn.causal.cate_learning.meta_learners._predict_by_treatment_flag")
def test__simulate_treatment_effect(mock_predict_by_treatment_flag: MagicMock) -> None:
    df: DataFrame = pd.DataFrame(
        {
            "x1": [1.3, 1.0, 1.8, -0.1],
            "x2": [10, 4, 15, 6],
            "treatment": ["A", "B", "A", "control"],
            "target": [0, 0, 0, 1],
        }
    )

    expected: DataFrame = pd.DataFrame(
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
            "suggested_treatment": [
                "treatment_A",
                "treatment_B",
                "control",
                "treatment_A",
            ],
        }
    )

    treatments: List[str] = ["A", "B"]
    control_name: str = "control"

    mock_predict_by_treatment_flag.side_effect = [
        [0.3, 0.3, 0.0, 1.0],
        [0.2, 0.5, 0.3, 0.0],
        [0.6, 0.7, 0.0, 1.0],
        [1.0, 0.5, 1.0, 1.0]
    ]

    learners: Dict[str, LearnerFnType] = {
        "A": logistic_classification_learner,
        "B": logistic_classification_learner,
    }

    results: DataFrame = _simulate_treatment_effect(
        df,
        treatments=treatments,
        control_name=control_name,
        learners=learners,
        prediction_column="prediction",
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
    mock_model: MagicMock = create_autospec(logistic_classification_learner)
    mock_fit_by_treatment.side_effect = [
        (ones_or_zeros_model, dict()),
        (ones_or_zeros_model, dict()),
    ]

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


def test_simulate_t_learner_treatment_effect() -> None:
    df: DataFrame = pd.DataFrame(
        {
            "x1": [1.3, 1.0, 1.8, -0.1],
            "x2": [10, 4, 15, 6],
            "treatment": ["A", "B", "A", "control"],
            "target": [0, 0, 0, 1],
        }
    )

    treatments: List[str] = ["A", "B"]
    control_name: str = "control"
    prediction_column: str = "prediction"

    control_learner: MagicMock = MagicMock()
    control_learner.side_effect = lambda _: pd.DataFrame({"prediction": [1, 2, 3, 4]})

    treatment_learner: MagicMock = MagicMock()
    treatment_learner.side_effect = lambda _: pd.DataFrame({"prediction": [3, 2, 4, 4]})

    learners: Dict[str, MagicMock] = {
        "control": control_learner,
        "A": treatment_learner,
        "B": treatment_learner,
    }

    result: DataFrame = _simulate_t_learner_treatment_effect(
        df,
        learners,
        treatments,
        control_name,
        prediction_column,
    )

    expected: DataFrame = pd.DataFrame(
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

    assert isinstance(result, DataFrame)
    assert_frame_equal(result, expected)


def test_get_model_fcn(base_input_df: DataFrame) -> None:
    fake_prediction_column: List[float] = [0.1, 0.2, 0.3]
    df_expected: DataFrame = pd.DataFrame(
        {
            "x1": [1.3, 1.8, -0.1],
            "x2": [10, 15, 6],
            "treatment": [
                "A",
                "A",
                "A",
            ],
            "target": [1, 1, 0],
            "prediction": fake_prediction_column,
        }
    )

    def mock_learner(df: DataFrame) -> Tuple[Callable[[DataFrame], DataFrame], DataFrame, Dict[str, float]]:
        df["prediction"] = fake_prediction_column
        return (lambda x: x, df, dict())

    learner: MagicMock = MagicMock()
    learner.side_effect = mock_learner

    mock_fcn: Callable[[DataFrame], DataFrame]
    mock_p_df: DataFrame
    mock_logs: Dict[str, float]
    mock_fcn, mock_p_df, mock_logs = _get_model_fcn(
        base_input_df, "treatment", "A", learner
    )

    assert isinstance(mock_fcn, Callable)
    assert_frame_equal(mock_p_df, df_expected)
    assert isinstance(mock_logs, dict)


def test_get_model_fcn_exception(base_input_df: DataFrame) -> None:
    fake_prediction_column: List[float] = [0.1, 