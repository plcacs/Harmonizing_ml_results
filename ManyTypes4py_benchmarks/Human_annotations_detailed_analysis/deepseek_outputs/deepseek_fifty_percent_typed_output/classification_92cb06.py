from typing import List, Any, Optional, Callable, Tuple, Union, TYPE_CHECKING, Literal, Dict

import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from toolz import curry, merge, assoc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import __version__ as sk_version

from fklearn.types import LearnerReturnType, LearnerLogType, LogType
from fklearn.common_docstrings import learner_return_docstring, learner_pred_fn_docstring
from fklearn.training.utils import log_learner_time, expand_features_encoded


if TYPE_CHECKING:
    from lightgbm import Booster


@curry
@log_learner_time(learner_name='logistic_classification_learner')
def logistic_classification_learner(df: pd.DataFrame,
                                    features: List[str],
                                    target: str,
                                    params: Optional[Dict[str, Any]] = None,
                                    prediction_column: str = "prediction",
                                    weight_column: Optional[str] = None,
                                    encode_extra_cols: bool = True) -> LearnerReturnType:
    """
    Fits an logistic regression classifier to the dataset. Return the predict function
    for the model and the predictions for the input dataset.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    features : list of str
        A list os column names that are used as features for the model. All this names
        should be in `df`.

    target : str
        The name of the column in `df` that should be used as target for the model.
        This column should be discrete, since this is a classification model.

    params : dict
        The LogisticRegression parameters in the format {"par_name": param}. See:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    prediction_column : str
        The name of the column with the predictions from the model.
        If a multiclass problem, additional prediction_column_i columns will be added for i in range(0,n_classes).

    weight_column : str, optional
        The name of the column with scores to weight the data.

    encode_extra_cols : bool (default: True)
        If True, treats all columns in `df` with name pattern fklearn_feat__col==val` as feature columns.
    """

    def_params: Dict[str, Any] = {"C": 0.1, "multi_class": "ovr", "solver": "liblinear"}
    merged_params: Dict[str, Any] = def_params if not params else merge(def_params, params)

    weights: Optional[npt.NDArray[np.float64]] = df[weight_column].values if weight_column else None

    features = features if not encode_extra_cols else expand_features_encoded(df, features)

    clf: LogisticRegression = LogisticRegression(**merged_params)
    clf.fit(df[features].values, df[target].values, sample_weight=weights)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        pred: npt.NDArray[np.float64] = clf.predict_proba(new_df[features].values)
        if merged_params["multi_class"] == "multinomial":
            col_dict: Dict[str, npt.NDArray[np.float64]] = {prediction_column + "_" + str(key): value for (key, value) in enumerate(pred.T)}
            col_dict.update({prediction_column: pred.argmax(axis=1)})
        else:
            col_dict = {prediction_column: pred[:, 1]}

        return new_df.assign(**col_dict)

    p.__doc__ = learner_pred_fn_docstring("logistic_classification_learner")

    log: LearnerLogType = {
        'logistic_classification_learner': {
            'features': features,
            'target': target,
            'parameters': merged_params,
            'prediction_column': prediction_column,
            'package': "sklearn",
            'package_version': sk_version,
            'feature_importance': dict(zip(features, clf.coef_.flatten())),
            'training_samples': len(df)
        },
        'object': clf
    }

    return p, p(df), log


logistic_classification_learner.__doc__ += learner_return_docstring("Logistic Regression")


@curry
@log_learner_time(learner_name='xgb_classification_learner')
def xgb_classification_learner(df: pd.DataFrame,
                               features: List[str],
                               target: str,
                               learning_rate: float = 0.1,
                               num_estimators: int = 100,
                               extra_params: Optional[LogType] = None,
                               prediction_column: str = "prediction",
                               weight_column: Optional[str] = None,
                               encode_extra_cols: bool = True) -> LearnerReturnType:
    """
    Fits an XGBoost classifier to the dataset. It first generates a DMatrix
    with the specified features and labels from `df`. Then, it fits a XGBoost
    model to this DMatrix. Return the predict function for the model and the
    predictions for the input dataset.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    features : list of str
        A list os column names that are used as features for the model. All this names
        should be in `df`.

    target : str
        The name of the column in `df` that should be used as target for the model.
        This column should be discrete, since this is a classification model.

    learning_rate : float
        Float in the range (0, 1]
        Step size shrinkage used in update to prevents overfitting. After each boosting step,
        we can directly get the weights of new features. and eta actually shrinks the
        feature weights to make the boosting process more conservative.
        See the eta hyper-parameter in:
        http://xgboost.readthedocs.io/en/latest/parameter.html

    num_estimators : int
        Int in the range (0, inf)
        Number of boosted trees to fit.
        See the n_estimators hyper-parameter in:
        http://xgboost.readthedocs.io/en/latest/python/python_api.html

    extra_params : dict, optional
        Dictionary in the format {"hyperparameter_name" : hyperparameter_value}.
        Other parameters for the XGBoost model. See the list in:
        http://xgboost.readthedocs.io/en/latest/parameter.html
        If not passed, the default will be used.

    prediction_column : str
        The name of the column with the predictions from the model.
        If a multiclass problem, additional prediction_column_i columns will be added for i in range(0,n_classes).

    weight_column : str, optional
        The name of the column with scores to weight the data.

    encode_extra_cols : bool (default: True)
        If True, treats all columns in `df` with name pattern fklearn_feat__col==val` as feature columns.
    """

    import xgboost as xgb

    params: Dict[str, Any] = extra_params if extra_params else {}
    params = assoc(params, "eta", learning_rate)
    params = params if "objective" in params else assoc(params, "objective", 'binary:logistic')

    weights: Optional[npt.NDArray[np.float64]] = df[weight_column].values if weight_column else None

    features = features if not encode_extra_cols else expand_features_encoded(df, features)

    dtrain: xgb.DMatrix = xgb.DMatrix(
        df[features].values,
        label=df[target].values,
        feature_names=list(map(str, features)),
        weight=weights
    )

    bst: xgb.Booster = xgb.train(params, dtrain, num_estimators)

    def p(new_df: pd.DataFrame, apply_shap: bool = False) -> pd.DataFrame:

        dtest: xgb.DMatrix = xgb.DMatrix(
            new_df[features].values,
            feature_names=list(map(str, features))
        )

        pred: npt.NDArray[np.float64] = bst.predict(dtest)
        if params["objective"] == "multi:softprob":
            col_dict: Dict[str, npt.NDArray[np.float64]] = {prediction_column + "_" + str(key): value
                        for (key, value) in enumerate(pred.T)}
            col_dict.update({prediction_column: pred.argmax(axis=1)})
        else:
            col_dict = {prediction_column: pred}

        if apply_shap:
            import shap
            explainer: shap.TreeExplainer = shap.TreeExplainer(bst)
            shap_values: Union[npt.NDArray[np.float64], List[npt.NDArray[np.float64]]] = explainer.shap_values(new_df[features])
            shap_expected_value: Union[float, npt.NDArray[np.float64]] = explainer.expected_value

            if params["objective"] == "multi:softprob":
                shap_values_multiclass: Dict[str, List[float]] = {f"shap_values_{class_index}": list(value)
                                          for (class_index, value) in enumerate(shap_values)}
                shap_expected_value_multiclass: Dict[str, npt.NDArray[np.float64]] = {
                    f"shap_expected_value_{class_index}":
                        np.repeat(expected_value, len(class_shap_values))
                    for (class_index, (expected_value, class_shap_values))
                    in enumerate(zip(shap_expected_value, shap_values))
                }
                shap_output: Dict[str, Any] = merge(shap_values_multiclass, shap_expected_value_multiclass)

            else:
                shap_values = list(shap_values)
                shap_output = {"shap_values": shap_values,
                               "shap_expected_value": np.repeat(shap_expected_value, len(shap_values))}

            col_dict = merge(col_dict, shap_output)

        return new_df.assign(**col_dict)

    p.__doc__ = learner_pred_fn_docstring("xgb_classification_learner", shap=True)

    log: LearnerLogType = {
        'xgb_classification_learner': {
            'features': features,
            'target': target,
            'prediction_column': prediction_column,
            'package': "xgboost",
            'package_version': xgb.__version__,
            'parameters': assoc(params, "num_estimators", num_estimators),
            'feature_importance': bst.get_score(),
            'training_samples': len(df)
        },
        'object': bst
    }

    return p, p(df), log


xgb_classification_learner.__doc__ += learner_return_docstring("XGboost Classifier")


@curry
def _get_catboost_shap_values(df: pd.DataFrame, cbr: Any,
                              features: List[str], target: str,
                              weights: Optional[List[float]], cat_features: Optional[List[str]]) -> npt.NDArray[np.float64]:
    """
    Auxiliar method to allow us to get shap values for Catboost multiclass models

    This method exists to allow us to serialize catboost models as pickle without any issues

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        Shap values will be calculated over this data.

    cbr: Any
        Catboost trained model

    features : List[str]
        A list of column names that are used as features for the model. All this names
        should be in `df`.

    target : str
        The name of the column in `df` that should be used as target for the model.

    weights : List
        Weight column values as a list

    cat_features: List[str]
        A list of column names that are used as categoriacal features for the model.
    """
    import catboost
    dtrain: catboost.Pool = catboost.Pool(df[features].values, df[target].values, weight=weights,
                           feature_names=list(map(str, features)),
                           cat_features=cat_features)
    return cbr.get_feature_importance(type=catboost.EFstrType.ShapValues,
                                      data=dtrain)


@curry
@log_learner_time(learner_name='catboost_classification_learner')
def catboost_classification_learner(df: pd.DataFrame,
                                    features: List[str],
                                    target: str,
                                    learning_rate: float = 0.1,
                                    num_estimators: int = 100,
                                    extra_params: Optional[LogType] = None,
                                    prediction_column: str = "prediction",
                                    weight_column: Optional[str] = None,
                                    encode_extra_cols: bool = True) -> LearnerReturnType:
    """
    Fits an CatBoost classifier to the dataset. It first generates a DMatrix
    with the specified features and labels from `df`. Then, it fits a CatBoost
    model to this DMatrix. Return the predict function for the model and the
    predictions for the input dataset.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    features : list of str
        A list os column names that are used as features for the model. All this names
        should be in `df`.

    target : str
        The name of the column in `df` that should be used as target for the model.
        This column should be discrete, since this is a classification model.

    learning_rate : float
        Float in the range (0, 1]
        Step size shrinkage used in update to prevents overfitting. After each boosting step,
        we can directly get the weights of new features. and eta actually shrinks the
        feature weights to make the boosting process more conservative.
        See the eta hyper-parameter in:
        https://catboost.ai/docs/concepts/python-reference_parameters-list.html

    num_estimators : int
        Int in the range (0, inf)
        Number of boosted trees to fit.
        See the n_estimators hyper-parameter in:
        https://catboost.ai/docs/concepts/python-reference_parameters-list.html

    extra_params : dict, optional
        Dictionary in the format {"hyperparameter_name" : hyperparameter_value}.
        Other parameters for the CatBoost model. See the list in:
        https://catboost.ai/docs/concepts/python-reference_catboostregressor.html
        If not passed, the default will be used.

    prediction_column : str
        The name of the column with the predictions from the model.
        If a multiclass problem, additional prediction_column_i columns will be added for i in range(0,n_classes).

    weight_column : str, optional
        The name of the column with scores to weight the data.

    encode_extra_cols : bool (default: True)
        If True, treats all columns in `df` with name pattern fklearn_feat__col==val` as feature columns.
    """
    from catboost import Pool, CatBoostClassifier
    import catboost

    weights: Optional[npt.NDArray[np.float64]] = df[weight_column].values if weight_column else None
    params: Dict[str, Any] = extra_params if extra_params else {}
    params = assoc(params, "eta", learning_rate)
    params = params if "objective" in params else assoc(params, "objective", 'Logloss')

    features = features if not encode_extra_cols else expand_features_encoded(df, features)

    cat_features: Optional[List[str]] = params["cat_features"] if "cat_features" in params else None

    dtrain: Pool = Pool(df[features].values, df[target].values, weight=weights,
                  feature_names=list(map(str, features)), cat_features=cat_features)

    cat_boost_classifier: CatBoostClassifier = CatBoostClassifier(iterations=num_estimators, **params)
    cbr: CatBoostClassifier = cat_boost_classifier.fit(dtrain, verbose=0)

    def p(new_df: pd.DataFrame, apply_shap: bool = False) -> pd.DataFrame:

        pred: npt.NDArray[np.float64] = cbr.predict_proba(new_df[features])
        if params["objective"] == "MultiClass":
            col_dict: Dict[str, npt.NDArray[np.float64]] = {prediction_column + "_" + str(key): value
                        for (key, value) in enumerate(pred.T)}
            col_dict.update({prediction_column: pred.argmax(axis=1)})
        else:
            col_dict = {prediction_column: pred[:, 1]}

        if apply_shap:
            import shap
            if params["objective"] == "MultiClass":
                shap_values: npt.NDArray[np.float64] = _get_catboost_shap_values(df, cbr, features, target, weights, cat_features)
                # catboost shap returns a list for each row, we reformat it to return
                # a list for each class
                shap_values = shap_values.transpose(1, 0, 2)
                shap_values_multiclass: Dict[str, List[float]] = {f"shap_values_{class_index}": list(value[:, :-1])
                                          for (class_index, value) in enumerate(shap_values)}
                shap_expected_value_multiclass: Dict[str, npt.NDArray[np.float64]] = {f"shap_expected_value_{class_index}": value[:, -1]
                                                  for (class_index, value) in enumerate(shap_values)}
                shap_output: Dict[str, Any] = merge(shap_values_multiclass, shap_expected_value_multiclass)

            else:
                explainer: shap.TreeExplainer = shap.TreeExplainer(cbr)
                shap_values: Union[npt.NDArray[np.float64], List[npt.NDArray[np.float64]]] = explainer.shap_values(new_df[features])
                shap_expected_value: Union[float, npt.NDArray[np.float64]] = explainer.expected_value
                shap_values = list(shap_values)
                shap_output = {"shap_values": shap_values,
                               "shap_expected_value": np.repeat(shap_expected_value, len(shap_values))}

            col_dict = merge(col_dict, shap_output)

       