```python
from typing import List, Any, Optional, Callable, Tuple, Union, TYPE_CHECKING, Literal, Dict, cast

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
    import xgboost as xgb
    import catboost


@curry
@log_learner_time(learner_name='logistic_classification_learner')
def logistic_classification_learner(df: pd.DataFrame,
                                    features: List[str],
                                    target: str,
                                    params: Optional[LogType] = None,
                                    prediction_column: str = "prediction",
                                    weight_column: Optional[str] = None,
                                    encode_extra_cols: bool = True) -> LearnerReturnType:
    def_params: LogType = {"C": 0.1, "multi_class": "ovr", "solver": "liblinear"}
    merged_params: LogType = def_params if not params else merge(def_params, params)

    weights: Optional[npt.NDArray[np.float_]] = df[weight_column].values if weight_column else None

    features = features if not encode_extra_cols else expand_features_encoded(df, features)

    clf: LogisticRegression = LogisticRegression(**merged_params)
    clf.fit(df[features].values, df[target].values, sample_weight=weights)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        pred: npt.NDArray[np.float_] = clf.predict_proba(new_df[features].values)
        col_dict: Dict[str, Union[npt.NDArray[np.int_], npt.NDArray[np.float_]]]
        if merged_params["multi_class"] == "multinomial":
            col_dict = {prediction_column + "_" + str(key): value for (key, value) in enumerate(pred.T)}
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
    import xgboost as xgb

    params: LogType = extra_params if extra_params else {}
    params = assoc(params, "eta", learning_rate)
    params = params if "objective" in params else assoc(params, "objective", 'binary:logistic')

    weights: Optional[npt.NDArray[np.float_]] = df[weight_column].values if weight_column else None

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

        pred: npt.NDArray[np.float_] = bst.predict(dtest)
        col_dict: Dict[str, Union[npt.NDArray[np.int_], npt.NDArray[np.float_]]]
        if params["objective"] == "multi:softprob":
            col_dict = {prediction_column + "_" + str(key): value
                        for (key, value) in enumerate(pred.T)}
            col_dict.update({prediction_column: pred.argmax(axis=1)})
        else:
            col_dict = {prediction_column: pred}

        if apply_shap:
            import shap
            explainer: shap.TreeExplainer = shap.TreeExplainer(bst)
            shap_values: List[npt.NDArray[np.float_]] = explainer.shap_values(new_df[features])
            shap_expected_value: npt.NDArray[np.float_] = explainer.expected_value

            if params["objective"] == "multi:softprob":
                shap_values_multiclass: Dict[str, List[npt.NDArray[np.float_]]] = {f"shap_values_{class_index}": list(value)
                                          for (class_index, value) in enumerate(shap_values)}
                shap_expected_value_multiclass: Dict[str, npt.NDArray[np.float_]] = {
                    f"shap_expected_value_{class_index}":
                        np.repeat(expected_value, len(class_shap_values))
                    for (class_index, (expected_value, class_shap_values))
                    in enumerate(zip(shap_expected_value, shap_values))
                }
                shap_output: Dict[str, Union[List[npt.NDArray[np.float_]], npt.NDArray[np.float_]]] = merge(shap_values_multiclass, shap_expected_value_multiclass)

            else:
                shap_values_list: List[float] = list(shap_values)
                shap_output = {"shap_values": shap_values_list,
                               "shap_expected_value": np.repeat(shap_expected_value, len(shap_values_list))}

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
                              weights: Optional[List[float]], cat_features: Optional[List[str]]) -> npt.NDArray[np.float_]:
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
    from catboost import Pool, CatBoostClassifier
    import catboost

    weights: Optional[npt.NDArray[np.float_]] = df[weight_column].values if weight_column else None
    params: LogType = extra_params if extra_params else {}
    params = assoc(params, "eta", learning_rate)
    params = params if "objective" in params else assoc(params, "objective", 'Logloss')

    features = features if not encode_extra_cols else expand_features_encoded(df, features)

    cat_features: Optional[List[str]] = params.get("cat_features", None)

    dtrain: Pool = Pool(df[features].values, df[target].values, weight=weights,
                  feature_names=list(map(str, features)), cat_features=cat_features)

    cat_boost_classifier: CatBoostClassifier = CatBoostClassifier(iterations=num_estimators, **params)
    cbr: CatBoostClassifier = cat_boost_classifier.fit(dtrain, verbose=0)

    def p(new_df: pd.DataFrame, apply_shap: bool = False) -> pd.DataFrame:
        pred: npt.NDArray[np.float_] = cbr.predict_proba(new_df[features])
        col_dict: Dict[str, Union[npt.NDArray[np.int_], npt.NDArray[np.float_]]]
        if params["objective"] == "MultiClass":
            col_dict = {prediction_column + "_" + str(key): value
                        for (key, value) in enumerate(pred.T)}
            col_dict.update({prediction_column: pred.argmax(axis=1)})
        else:
            col_dict = {prediction_column: pred[:, 1]}

        if apply_shap:
            import shap
            if params["objective"] == "MultiClass":
                shap_values: npt.NDArray[np.float_] = _get_catboost_shap_values(df, cbr, features, target, weights, cat_features)
                shap_values = shap_values.transpose(1, 0, 2)
                shap_values_multiclass: Dict[str, List[npt.NDArray[np.float_]]] = {f"shap_values_{class_index}": list(value[:, :-1])
                                          for (class_index, value) in enumerate(shap_values)}
                shap_expected_value_multiclass: Dict[str, npt.NDArray[np.float_]] = {f"shap_expected_value_{class_index}": value[:, -1]
                                                  for (class_index, value) in enumerate(shap_values)}
                shap_output: Dict[str, Union[List[npt.NDArray[np.float_]], npt.NDArray[np.float_]]] = merge(shap_values_multiclass, shap_expected_value_multiclass)

            else:
                explainer: shap.TreeExplainer = shap.TreeExplainer(cbr)
                shap_values: List[npt.NDArray[np.float_]] = explainer.shap_values(new_df[features])
                shap_expected_value: npt.NDArray[np.float_] = explainer.expected_value
                shap_values_list: List[float] = list(shap_values)
                shap_output = {"shap_values": shap_values_list,
                               "shap_expected_value": np.repeat(shap_expected_value, len(shap_values_list))}

            col_dict = merge(col_dict, shap_output)

        return new_df.assign(**col_dict)

    p.__doc__ = learner_pred_fn_docstring("catboost_classification_learner", shap=True)

    log: LearnerLogType = {
        'catboost_classification_learner': {
            'features': features,
            'target': target,
            'prediction_column': prediction_column,
            'package': "catboost",
            'package_version': catboost.__version__,
            'parameters': assoc(params, "num_estimators", num_estimators),
            'feature_importance': cbr.feature_importances_,
            'training_samples': len(df)
        },
        'object': cbr
    }

    return p, p(df), log


catboost_classification_learner.__doc__ += learner_return_docstring("catboost_classification_learner")


@curry
@log_learner_time(learner_name='nlp_logistic_classification_learner')
def nlp_logistic_classification_learner(df: pd.DataFrame,
                                        text_feature_cols: List[str],
                                        target: str,
                                        vectorizer_params: Optional[LogType] = None,
                                        logistic_params: Optional[LogType] = None,
                                        prediction_column: str = "prediction") -> LearnerReturnType:
    default_vect_params: LogType = {"strip_accents": "unicode", "min_df": 1}
    merged_vect_params: LogType = default_vect_params if not vectorizer_params else merge(default_vect_params, vectorizer_params)

    default_clf_params: LogType = {"C": 0.1, "multi_class": "ovr", "solver": "liblinear"}
    merged_logistic_params: LogType = default_clf_params if not logistic_params else merge(default_clf_params, logistic_params)

    vect: TfidfVectorizer = TfidfVectorizer(**merged_vect_params)
    clf: LogisticRegression = LogisticRegression(**merged_logistic_params)

    text_df: pd.Series = df[text_feature_cols].apply(lambda x: x + " ", axis=1).sum(axis=1)
    vect.fit(text_df.values)
    sparse_vect: npt.NDArray[np.float_] = vect.transform(text_df.values)
    clf.fit(sparse_vect, df[target].values)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        predict_text_df: pd.Series = new_df[text_feature_cols].apply(lambda x: x + " ", axis=1).sum(axis=1)
        predict_sparse_vect: npt.NDArray[np.float_] = vect.transform(predict_text_df)

        col_dict: Dict[str, npt.NDArray[np.float_]]
        if merged_logistic_params["multi_class"] == "multinomial":
            col_dict = {prediction_column + "_" + str(key): value
                        for (key, value) in enumerate(clf.predict_proba(predict_sparse_vect).T)}
        else:
            col_dict = {prediction_column: clf.predict_proba(predict_sparse_vect)[:, 1]}

        return new_df.assign(**col_dict)

    p.__doc__ = learner_pred_fn_docstring("nlp_logistic_classification_learner")

    params: Dict[str, LogType] = {"vectorizer_params": merged_vect_params,
              "logistic_params": merged_logistic_params}

    log: LearnerLogType = {'nlp_logistic_classification_learner': {
        'features': text_feature_cols,
        'target': target,
        'prediction_column': prediction_column,
        'parameters': assoc(params, "vocab_size", sparse_vect.shape[1]),
        'package': "sklearn",
        'package_version': sk_version,
        'training_samples': len(df)},
        'object': clf}

    return p, p(df), log


nlp_logistic_classification_learner.__doc__ += learner_return_docstring("NLP Logistic Regression")


@curry
@log_learner_time(learner_name='lgbm_classification_learner')
def lgbm_classification_learner(
        df: pd.DataFrame,
        features: List[str],
        target: str,
        learning_rate: float = 0.1,
        num_estimators: int = 100,
        extra_params: Optional[LogType] = None,
        prediction_column: str = "prediction",
        weight_column: Optional[str] = None,
        encode_extra_cols: bool = True,
        valid_sets: Optional[List[pd.DataFrame]] = None,
        valid_names: Optional[List[str]] = None,
        feval: Optional[Union[
            Union[Callable[[npt.NDArray[Any], 'lgbm.Dataset'], Tuple[str, float, bool]],
            List[Callable[[npt.NDArray[Any], 'lgbm.Dataset'], Tuple[str, float, bool]]]
        ]] = None,
        init_model: Optional[Union[str, Path, 'Booster']] = None,
        feature_name: Union[List[str], Literal['auto']] = 'auto',
        categorical_feature: Union[List[str], List[int], Literal['auto']] = 'auto',
        keep_training_booster: bool = False,
        callbacks: Optional[List[Callable[..., Any]]] = None,
        dataset_init_score: Optional[Union[List[float], List[List[float]], npt.NDArray[np.float_], pd.Series, pd.DataFrame]] = None
) -> LearnerReturnType:
    import lightgbm as lgbm

    LGBM_MULTICLASS_OBJECTIVES = {'multiclass', 'softmax', 'multiclassova', 'multiclass_ova', 'ova', 'ovr'}

    params: LogType = extra_params if extra_params else {}
    params = assoc(params, "eta", learning_rate)
    params = params if "objective" in params else assoc(params, "objective", 'binary')

    is_multiclass_classification: bool = params["objective"] in LGBM_MULTICLASS_OBJECTIVES

    weights: Optional[npt.NDArray[np.float_]] = df[weight_column].values if weight_column else None

    features = features if not encode_extra_cols else expand_features_encoded(df, features)

    dtrain: lgbm.Dataset = lgbm.Dataset(
        df[features].values,
       