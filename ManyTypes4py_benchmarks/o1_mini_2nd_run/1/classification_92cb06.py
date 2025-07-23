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
    import xgboost as xgb
    import catboost
    import lightgbm as lgbm

@curry
@log_learner_time(learner_name='logistic_classification_learner')
def logistic_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True
) -> LearnerReturnType:
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
    def_params: Dict[str, Any] = {'C': 0.1, 'multi_class': 'ovr', 'solver': 'liblinear'}
    merged_params: Dict[str, Any] = def_params if not params else merge(def_params, params)
    weights: Optional[np.ndarray] = df[weight_column].values if weight_column else None
    encoded_features: List[str] = features if not encode_extra_cols else expand_features_encoded(df, features)
    clf: LogisticRegression = LogisticRegression(**merged_params)
    clf.fit(df[encoded_features].values, df[target].values, sample_weight=weights)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        pred: np.ndarray = clf.predict_proba(new_df[encoded_features].values)
        if merged_params['multi_class'] == 'multinomial':
            col_dict: Dict[str, Union[np.ndarray, np.ndarray]] = {
                f"{prediction_column}_{key}": value for key, value in enumerate(pred.T)
            }
            col_dict[prediction_column] = pred.argmax(axis=1)
        else:
            col_dict = {prediction_column: pred[:, 1]}
        return new_df.assign(**col_dict)
    p.__doc__ = learner_pred_fn_docstring('logistic_classification_learner')
    log: LogType = {
        'logistic_classification_learner': {
            'features': encoded_features,
            'target': target,
            'parameters': merged_params,
            'prediction_column': prediction_column,
            'package': 'sklearn',
            'package_version': sk_version,
            'feature_importance': dict(zip(encoded_features, clf.coef_.flatten())),
            'training_samples': len(df)
        },
        'object': clf
    }
    return (p, p(df), log)
logistic_classification_learner.__doc__ += learner_return_docstring('Logistic Regression')


@curry
@log_learner_time(learner_name='xgb_classification_learner')
def xgb_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True
) -> LearnerReturnType:
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
    params = assoc(params, 'eta', learning_rate)
    params = params if 'objective' in params else assoc(params, 'objective', 'binary:logistic')
    weights: Optional[np.ndarray] = df[weight_column].values if weight_column else None
    encoded_features: List[str] = features if not encode_extra_cols else expand_features_encoded(df, features)
    dtrain: xgb.DMatrix = xgb.DMatrix(
        df[encoded_features].values,
        label=df[target].values,
        feature_names=list(map(str, encoded_features)),
        weight=weights
    )
    bst: xgb.Booster = xgb.train(params, dtrain, num_estimators)

    def p(new_df: pd.DataFrame, apply_shap: bool = False) -> pd.DataFrame:
        dtest: xgb.DMatrix = xgb.DMatrix(new_df[encoded_features].values, feature_names=list(map(str, encoded_features)))
        pred: np.ndarray = bst.predict(dtest)
        if params['objective'] == 'multi:softprob':
            col_dict: Dict[str, Any] = {
                f"{prediction_column}_{key}": value for key, value in enumerate(pred.T)
            }
            col_dict[prediction_column] = pred.argmax(axis=1)
        else:
            col_dict = {prediction_column: pred}
        if apply_shap:
            import shap
            explainer = shap.TreeExplainer(bst)
            shap_values: Union[np.ndarray, List[np.ndarray]] = explainer.shap_values(new_df[encoded_features].values)
            shap_expected_value = explainer.expected_value
            if params['objective'] == 'multi:softprob':
                shap_values_multiclass: Dict[str, List[float]] = {
                    f'shap_values_{class_index}': list(value) for class_index, value in enumerate(shap_values)
                }
                shap_expected_value_multiclass: Dict[str, np.ndarray] = {
                    f'shap_expected_value_{class_index}': np.repeat(expected_value, len(class_shap_values))
                    for class_index, (expected_value, class_shap_values) in enumerate(zip(shap_expected_value, shap_values))
                }
                shap_output: Dict[str, Any] = merge(shap_values_multiclass, shap_expected_value_multiclass)
            else:
                shap_values_list: List[float] = list(shap_values)
                shap_output: Dict[str, Any] = {
                    'shap_values': shap_values_list,
                    'shap_expected_value': np.repeat(shap_expected_value, len(shap_values_list))
                }
            col_dict = merge(col_dict, shap_output)
        return new_df.assign(**col_dict)
    p.__doc__ = learner_pred_fn_docstring('xgb_classification_learner', shap=True)
    log: LogType = {
        'xgb_classification_learner': {
            'features': encoded_features,
            'target': target,
            'prediction_column': prediction_column,
            'package': 'xgboost',
            'package_version': xgb.__version__,
            'parameters': assoc(params, 'num_estimators', num_estimators),
            'feature_importance': bst.get_score(),
            'training_samples': len(df)
        },
        'object': bst
    }
    return (p, p(df), log)
xgb_classification_learner.__doc__ += learner_return_docstring('XGboost Classifier')


@curry
def _get_catboost_shap_values(
    df: pd.DataFrame,
    cbr: Any,
    features: List[str],
    target: str,
    weights: List[float],
    cat_features: List[str]
) -> np.ndarray:
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
    dtrain: catboost.Pool = catboost.Pool(
        df[features].values,
        df[target].values,
        weight=weights,
        feature_names=list(map(str, features)),
        cat_features=cat_features
    )
    return cbr.get_feature_importance(type=catboost.EFstrType.ShapValues, data=dtrain)


@curry
@log_learner_time(learner_name='catboost_classification_learner')
def catboost_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True
) -> LearnerReturnType:
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
    weights: Optional[np.ndarray] = df[weight_column].values if weight_column else None
    params: Dict[str, Any] = extra_params if extra_params else {}
    params = assoc(params, 'eta', learning_rate)
    params = params if 'objective' in params else assoc(params, 'objective', 'Logloss')
    encoded_features: List[str] = features if not encode_extra_cols else expand_features_encoded(df, features)
    cat_features: Optional[List[str]] = params.get('cat_features')
    dtrain: Pool = Pool(
        df[encoded_features],
        df[target],
        weight=weights,
        feature_names=list(map(str, encoded_features)),
        cat_features=cat_features
    )
    cat_boost_classifier: CatBoostClassifier = CatBoostClassifier(iterations=num_estimators, **params)
    cbr: catboost.CatBoost = cat_boost_classifier.fit(dtrain, verbose=0)

    def p(new_df: pd.DataFrame, apply_shap: bool = False) -> pd.DataFrame:
        pred: np.ndarray = cbr.predict_proba(new_df[encoded_features])
        if params['objective'] == 'MultiClass':
            col_dict: Dict[str, Any] = {
                f"{prediction_column}_{key}": value for key, value in enumerate(pred.T)
            }
            col_dict[prediction_column] = pred.argmax(axis=1)
        else:
            col_dict = {prediction_column: pred[:, 1]}
        if apply_shap:
            import shap
            if params['objective'] == 'MultiClass':
                shap_values: np.ndarray = _get_catboost_shap_values(
                    df, cbr, encoded_features, target, weights.tolist() if weights is not None else [], cat_features
                )
                shap_values = shap_values.transpose(1, 0, 2)
                shap_values_multiclass: Dict[str, List[float]] = {
                    f'shap_values_{class_index}': list(value[:, :-1]) for class_index, value in enumerate(shap_values)
                }
                shap_expected_value_multiclass: Dict[str, List[float]] = {
                    f'shap_expected_value_{class_index}': value[:, -1] for class_index, value in enumerate(shap_values)
                }
                shap_output: Dict[str, Any] = merge(shap_values_multiclass, shap_expected_value_multiclass)
            else:
                explainer: shap.TreeExplainer = shap.TreeExplainer(cbr)
                shap_values: List[float] = list(explainer.shap_values(new_df[encoded_features]))
                shap_expected_value: float = explainer.expected_value
                shap_output: Dict[str, Any] = {
                    'shap_values': shap_values,
                    'shap_expected_value': np.repeat(shap_expected_value, len(shap_values))
                }
            col_dict = merge(col_dict, shap_output)
        return new_df.assign(**col_dict)
    p.__doc__ = learner_pred_fn_docstring('catboost_classification_learner', shap=True)
    log: LogType = {
        'catboost_classification_learner': {
            'features': encoded_features,
            'target': target,
            'prediction_column': prediction_column,
            'package': 'catboost',
            'package_version': catboost.__version__,
            'parameters': assoc(params, 'num_estimators', num_estimators),
            'feature_importance': cbr.feature_importances_,
            'training_samples': len(df)
        },
        'object': cbr
    }
    return (p, p(df), log)
catboost_classification_learner.__doc__ += learner_return_docstring('catboost_classification_learner')


@curry
@log_learner_time(learner_name='nlp_logistic_classification_learner')
def nlp_logistic_classification_learner(
    df: pd.DataFrame,
    text_feature_cols: List[str],
    target: str,
    vectorizer_params: Optional[Dict[str, Any]] = None,
    logistic_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction'
) -> LearnerReturnType:
    """
    Fits a text vectorizer (TfidfVectorizer) followed by
    a logistic regression (LogisticRegression).

    Parameters
    ----------
    
    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    text_feature_cols : list of str
        A list of column names of the text features used for the model. All these names
        should be in `df`.

    target : str
        The name of the column in `df` that should be used as target for the model.
        This column should be discrete, since this is a classification model.

    vectorizer_params : dict
        The TfidfVectorizer parameters in the format {"par_name": param}. See:
        http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

    logistic_params : dict
        The LogisticRegression parameters in the format {"par_name": param}. See:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    prediction_column : str
        The name of the column with the predictions from the model.
    """
    default_vect_params: Dict[str, Any] = {'strip_accents': 'unicode', 'min_df': 1}
    merged_vect_params: Dict[str, Any] = default_vect_params if not vectorizer_params else merge(default_vect_params, vectorizer_params)
    default_clf_params: Dict[str, Any] = {'C': 0.1, 'multi_class': 'ovr', 'solver': 'liblinear'}
    merged_logistic_params: Dict[str, Any] = default_clf_params if not logistic_params else merge(default_clf_params, logistic_params)
    vect: TfidfVectorizer = TfidfVectorizer(**merged_vect_params)
    clf: LogisticRegression = LogisticRegression(**merged_logistic_params)
    text_df: pd.Series = df[text_feature_cols].apply(lambda x: x + ' ', axis=1).sum(axis=1)
    vect.fit(text_df.values)
    sparse_vect: npt.NDArray = vect.transform(text_df.values)
    clf.fit(sparse_vect, df[target].values)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        predict_text_df: pd.Series = new_df[text_feature_cols].apply(lambda x: x + ' ', axis=1).sum(axis=1)
        predict_sparse_vect: npt.NDArray = vect.transform(predict_text_df)
        if merged_logistic_params['multi_class'] == 'multinomial':
            col_dict: Dict[str, Any] = {
                f"{prediction_column}_{key}": value for key, value in enumerate(clf.predict_proba(predict_sparse_vect).T)
            }
        else:
            col_dict = {prediction_column: clf.predict_proba(predict_sparse_vect)[:, 1]}
        return new_df.assign(**col_dict)
    p.__doc__ = learner_pred_fn_docstring('nlp_logistic_classification_learner')
    params: Dict[str, Any] = {'vectorizer_params': merged_vect_params, 'logistic_params': merged_logistic_params}
    log: LogType = {
        'nlp_logistic_classification_learner': {
            'features': text_feature_cols,
            'target': target,
            'prediction_column': prediction_column,
            'parameters': assoc(params, 'vocab_size', sparse_vect.shape[1]),
            'package': 'sklearn',
            'package_version': sk_version,
            'training_samples': len(df)
        },
        'object': clf
    }
    return (p, p(df), log)
nlp_logistic_classification_learner.__doc__ += learner_return_docstring('NLP Logistic Regression')


@curry
@log_learner_time(learner_name='lgbm_classification_learner')
def lgbm_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
    valid_sets: Optional[List[pd.DataFrame]] = None,
    valid_names: Optional[List[str]] = None,
    feval: Optional[Union[Callable, List[Callable]]] = None,
    init_model: Optional[Union[str, Path, 'Booster']] = None,
    feature_name: Union[List[str], str] = 'auto',
    categorical_feature: Union[List[str], List[int], str] = 'auto',
    keep_training_booster: bool = False,
    callbacks: Optional[List[Callable]] = None,
    dataset_init_score: Optional[Union[List[float], List[List[float]], np.ndarray, pd.Series, pd.DataFrame]] = None
) -> LearnerReturnType:
    """
    Fits an LGBM classifier to the dataset.

    It first generates a Dataset
    with the specified features and labels from `df`. Then, it fits a LGBM
    model to this Dataset. Return the predict function for the model and the
    predictions for the input dataset.

    Parameters
    ----------
    
    df : pandas.DataFrame
       A pandas DataFrame with features and target columns.
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
        See the learning_rate hyper-parameter in:
        https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst

    num_estimators : int
        Int in the range (0, inf)
        Number of boosted trees to fit.
        See the num_iterations hyper-parameter in:
        https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst

    extra_params : dict, optional
        Dictionary in the format {"hyperparameter_name" : hyperparameter_value}.
        Other parameters for the LGBM model. See the list in:
        https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
        If not passed, the default will be used.

    prediction_column : str
        The name of the column with the predictions from the model.

    weight_column : str, optional
        The name of the column with scores to weight the data.

    encode_extra_cols : bool (default: True)
        If True, treats all columns in `df` with name pattern fklearn_feat__col==val` as feature columns.

    valid_sets : list of pandas.DataFrame, optional (default=None)
        A list of datasets to be used for early-stopping during training.

    valid_names : list of strings, optional (default=None)
        A list of dataset names matching the list of datasets provided through the ``valid_sets`` parameter.

    feval : callable, list of callable, or None, optional (default=None)
        Customized evaluation function. Each evaluation function should accept two parameters: preds, eval_data, and
        return (eval_name, eval_result, is_higher_better) or list of such tuples.

    init_model : str, pathlib.Path, Booster or None, optional (default=None)
        Filename of LightGBM model or Booster instance used for continue training.

    feature_name : list of str, or 'auto', optional (default="auto")
        Feature names. If ‘auto’ and data is pandas DataFrame, data columns names are used.

    categorical_feature : list of str or int, or 'auto', optional (default="auto")
        Categorical features. If list of int, interpreted as indices. If list of str, interpreted as feature names (need
        to specify feature_name as well). If ‘auto’ and data is pandas DataFrame, pandas unordered categorical columns
        are used. All values in categorical features will be cast to int32 and thus should be less than int32 max value
        (2147483647). Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values. The output cannot be
        monotonically constrained with respect to a categorical feature. Floating point numbers in categorical features
        will be rounded towards 0.

    keep_training_booster : bool, optional (default=False)
        Whether the returned Booster will be used to keep training. If False, the returned value will be converted into
        _InnerPredictor before returning. This means you won’t be able to use eval, eval_train or eval_valid methods of
        the returned Booster. When your model is very large and cause the memory error, you can try to set this param to
        True to avoid the model conversion performed during the internal call of model_to_string. You can still use
        _InnerPredictor as init_model for future continue training.

    callbacks : list of callable, or None, optional (default=None)
        List of callback functions that are applied at each iteration. See Callbacks in LightGBM Python API for more
        information.

    dataset_init_score : list, list of lists (for multi-class task), numpy array, pandas Series, pandas DataFrame (for
        multi-class task), or None, optional (default=None)
        Init score for Dataset. It could be the prediction of the majority class or a prediction from any other model.
    """
    import lightgbm as lgbm
    LGBM_MULTICLASS_OBJECTIVES: set = {'multiclass', 'softmax', 'multiclassova', 'multiclass_ova', 'ova', 'ovr'}
    params: Dict[str, Any] = extra_params if extra_params else {}
    params = assoc(params, 'eta', learning_rate)
    params = params if 'objective' in params else assoc(params, 'objective', 'binary')
    is_multiclass_classification: bool = params['objective'] in LGBM_MULTICLASS_OBJECTIVES
    weights: Optional[np.ndarray] = df[weight_column].values if weight_column else None
    encoded_features: List[str] = features if not encode_extra_cols else expand_features_encoded(df, features)
    dtrain: lgbm.Dataset = lgbm.Dataset(
        df[encoded_features].values,
        label=df[target],
        feature_name=list(map(str, encoded_features)),
        weight=weights,
        init_score=dataset_init_score
    )
    bst: lgbm.Booster = lgbm.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        feval=feval,
        init_model=init_model,
        feature_name=feature_name,
        categorical_feature=categorical_feature,
        keep_training_booster=keep_training_booster,
        callbacks=callbacks
    )

    def p(new_df: pd.DataFrame, apply_shap: bool = False) -> pd.DataFrame:
        predictions: np.ndarray = bst.predict(new_df[encoded_features].values)
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        if is_multiclass_classification:
            col_dict: Dict[str, Any] = {
                f"{prediction_column}_{key}": value for key, value in enumerate(predictions.T)
            }
        else:
            col_dict = {prediction_column: predictions}
        if apply_shap:
            import shap
            explainer: shap.TreeExplainer = shap.TreeExplainer(bst)
            shap_values: Union[np.ndarray, List[np.ndarray]] = explainer.shap_values(new_df[encoded_features].values)
            shap_expected_value: Union[float, List[float]] = explainer.expected_value
            if is_multiclass_classification:
                shap_values_multiclass: Dict[str, List[float]] = {
                    f'shap_values_{class_index}': list(value) for class_index, value in enumerate(shap_values)
                }
                shap_expected_value_multiclass: Dict[str, Any] = {
                    f'shap_expected_value_{class_index}': np.repeat(expected_value, len(class_shap_values))
                    for class_index, (expected_value, shap_values_multiclass_list) in enumerate(zip(shap_expected_value, shap_values))
                }
                shap_output: Dict[str, Any] = merge(shap_values_multiclass, shap_expected_value_multiclass)
            else:
                shap_values_list: List[float] = list(shap_values[1])
                shap_output: Dict[str, Any] = {
                    'shap_values': shap_values_list,
                    'shap_expected_value': np.repeat(shap_expected_value, len(shap_values_list))
                }
            col_dict = merge(col_dict, shap_output)
        return new_df.assign(**col_dict)
    p.__doc__ = learner_pred_fn_docstring('lgbm_classification_learner', shap=True)
    log: LogType = {
        'lgbm_classification_learner': {
            'features': encoded_features,
            'target': target,
            'prediction_column': prediction_column,
            'package': 'lightgbm',
            'package_version': lgbm.__version__,
            'parameters': assoc(params, 'num_estimators', num_estimators),
            'feature_importance': dict(zip(encoded_features, bst.feature_importance().tolist())),
            'training_samples': len(df)
        },
        'object': bst
    }
    return (p, p(df), log)
lgbm_classification_learner.__doc__ += learner_return_docstring('LGBM Classifier')
