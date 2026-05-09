from typing import Callable, Dict, List
import pandas as pd
import numpy as np

@pytest.fixture
def base_input_df() -> pd.DataFrame:
    return pd.DataFrame({'x1': [1.3, 1.0, 1.8, -0.1, 0.0, 1.0, 2.2, 0.4, -5.0], 
                         'x2': [10, 4, 15, 6, 5, 12, 14, 5, 12], 
                         'treatment': ['A', 'B', 'A', 'A', 'B', 'control', 'control', 'B', 'control'], 
                         'target': [1, 1, 1, 0, 0, 1, 0, 0, 1]})

def test__append_treatment_feature(features: List[str], treatment_feature: str) -> None:
    assert _append_treatment_feature(features, treatment_feature) == features + [treatment_feature]
    assert len(features) > 0
    assert treatment_feature

def test__get_unique_treatments(df: pd.DataFrame, treatment_col: str, control_name: str) -> List[str]:
    filtered = _get_unique_treatments(df, treatment_col, control_name)
    expected = ['treatment_A', 'treatment_B', 'control']
    assert sorted(filtered) == sorted(expected)

def test__filter_by_treatment(df: pd.DataFrame, selected_treatment: str) -> pd.DataFrame:
    expected_values = [[1.0, 'treatment_A', 1], [5.0, 'treatment_A', 0], [3.0, 'control', 1]]
    expected = pd.DataFrame(data=expected_values, columns=['feat1', 'treatment', 'target'])
    results = _filter_by_treatment(df, treatment_col='treatment', treatment_name=selected_treatment, control_name='control')
    assert_frame_equal(results.reset_index(drop=True), expected)

def test__create_treatment_flag(df: pd.DataFrame, treatment_col: str, control_name: str, treatment_name: str) -> pd.DataFrame:
    expected = pd.DataFrame({'feature': [1.3, 1.0, 1.8, -0.1], 'group': ['treatment', 'treatment', 'control', 'control'], 'target': [1, 1, 1, 0], TREATMENT_FEATURE: [1.0, 1.0, 0.0, 0.0]})
    results = _create_treatment_flag(df, treatment_col, control_name, treatment_name)
    assert_frame_equal(results, expected)

def test__fit_by_treatment(base_input_df: pd.DataFrame, learner: Callable) -> None:
    treatments = ['A', 'B']
    learners, logs = _fit_by_treatment(base_input_df, learner, treatment_col='treatment', control_name='control', treatments=treatments)
    assert len(learners) == len(treatments)
    assert len(logs) == len(treatments)
    assert type(logs) is dict
    assert [type(learner) is LearnerFnType for learner in learners]

def test__predict_by_treatment_flag_positive(df: pd.DataFrame, model: Callable, positive: bool) -> np.ndarray:
    assert (_predict_by_treatment_flag(df, model, positive, 'prediction') == np.ones(df.shape[0])).all()

def test__predict_by_treatment_flag_negative(df: pd.DataFrame, model: Callable, negative: bool) -> np.ndarray:
    assert (_predict_by_treatment_flag(df, model, negative, 'prediction') == np.zeros(df.shape[0])).all()

@patch('fklearn.causal.cate_learning.meta_learners._predict_by_treatment_flag')
def test__simulate_treatment_effect(mock_predict_by_treatment_flag) -> pd.DataFrame:
    df = pd.DataFrame({'x1': [1.3, 1.0, 1.8, -0.1], 'x2': [10, 4, 15, 6], 'treatment': ['A', 'B', 'A', 'control'], 'target': [0, 0, 0, 1]})
    treatments = ['A', 'B']
    control_name = 'control'
    mock_predict_by_treatment_flag.side_effect = [[0.3, 0.3, 0.0, 1.0], [0.2, 0.5, 0.3, 0.0], [0.6, 0.7, 0.0, 1.0], [1.0, 0.5, 1.0, 1.0]]
    learners = {'A': logistic_classification_learner, 'B': logistic_classification_learner}
    results = _simulate_treatment_effect(df, treatments, control_name, learners, prediction_column='prediction')
    expected = pd.DataFrame({'x1': [1.3, 1.0, 1.8, -0.1], 'x2': [10, 4, 15, 6], 'treatment': ['A', 'B', 'A', 'control'], 'target': [0, 0, 0, 1], 'treatment_A__prediction_on_treatment': [0.3, 0.3, 0.0, 1.0], 'treatment_A__prediction_on_control': [0.2, 0.5, 0.3, 0.0], 'treatment_A__uplift': [0.1, -0.2, -0.3, 1.0], 'treatment_B__prediction_on_treatment': [0.6, 0.7, 0.0, 1.0], 'treatment_B__prediction_on_control': [1.0, 0.5, 1.0, 1.0], 'treatment_B__uplift': [-0.4, 0.2, -1.0, 0.0], 'uplift': [0.1, 0.2, -0.3, 1.0], 'suggested_treatment': ['treatment_A', 'treatment_B', 'control', 'treatment_A']}
    assert isinstance(results, pd.DataFrame)
    assert_frame_equal(results, expected)

def test_get_model_fcn(df: pd.DataFrame, treatment_col: str, treatment_name: str, learner: Callable) -> Callable:
    fake_prediction_column = [0.1, 0.2, 0.3]
    df_expected = pd.DataFrame({'x1': [1.3, 1.8, -0.1], 'x2': [10, 15, 6], 'treatment': ['A', 'A', 'A'], 'target': [1, 1, 0], 'prediction': fake_prediction_column}

    def mock_learner(df):
        df['prediction'] = fake_prediction_column
        return (lambda x: x, df, dict())
    learner = MagicMock()
    learner.side_effect = mock_learner
    mock_fcn, mock_p_df, mock_logs = _get_model_fcn(df, treatment_col, treatment_name, learner)
    assert isinstance(mock_fcn, Callable)
    assert_frame_equal(mock_p_df, df_expected)
    assert isinstance(mock_logs, dict)

def test_get_model_fcn_exception(df: pd.DataFrame, treatment_col: str, treatment_name: str, learner: Callable) -> None:
    fake_prediction_column = [0.1, 0.2, 0.3]

    def mock_learner(df):
        df['prediction'] = fake_prediction_column
        return (lambda x: x, df, dict())
    learner = MagicMock()
    learner.side_effect = mock_learner
    with pytest.raises(Exception) as e:
        _ = _get_model_fcn(df, treatment_col, 'C', learner)
    assert e.type == MissingTreatmentError

@patch('fklearn.causal.cate_learning.meta_learners._get_model_fcn')
def test_get_learners(mock_get_model_fcn) -> Dict[str, Callable]:
    unique_treatments = ['treatment_a', 'treatment_b', 'treatment_c']
    mock_get_model_fcn.side_effect = [('mocked_control_fcn', None, None), ('mocked_treatment_fcn_filtering_treatment_a', None, None), ('mocked_treatment_fcn_filtering_treatment_b', None, None), ('mocked_treatment_fcn_filtering_treatment_c', None, None)]
    learners, logs = _get_learners(df='mocked_df', unique_treatments=unique_treatments, treatment_col='treatment', control_name='control', control_learner='mocked_control_fcn', treatment_learner='mocked_treatment_fcn')
    assert learners['control'] == 'mocked_control_fcn'
    assert learners['treatment_a'] == 'mocked_treatment_fcn_filtering_treatment_a'
    assert learners['treatment_b'] == 'mocked_treatment_fcn_filtering_treatment_b'
    assert learners['treatment_c'] == 'mocked_treatment_fcn_filtering_treatment_c'
    assert isinstance(learners, dict)
    assert isinstance(logs, dict)
    calls = [call('mocked_df', 'treatment', 'control', 'mocked_control_fcn'), call('mocked_df', 'treatment', 'treatment_a', 'mocked_treatment_fcn'), call('mocked_df', 'treatment', 'treatment_b', 'mocked_treatment_fcn'), call('mocked_df', 'treatment', 'treatment_c', 'mocked_treatment_fcn')]
    mock_get_model_fcn.assert_has_calls(calls)

@patch('fklearn.causal.cate_learning.meta_learners._simulate_t_learner_treatment_effect')
@patch('fklearn.causal.cate_learning.meta_learners._get_learners')
@patch('fklearn.causal.cate_learning.meta_learners._get_unique_treatments')
def test_causal_t_classification_learner(mock_get_unique_treatments, mock_get_learners, mock_simulate_t_learner_treatment_effect, base_input_df: pd.DataFrame) -> None:
    mock_get_learners.side_effect = [([], dict())]
    mock_model = create_autospec(logistic_classification_learner)
    causal_t_classification_learner(df=base_input_df, treatment_col='treatment', control_name='control', prediction_column='prediction', learner=mock_model)
    mock_get_unique_treatments.assert_called()
    mock_get_learners.assert_called()
    mock_simulate_t_learner_treatment_effect.assert_called()
