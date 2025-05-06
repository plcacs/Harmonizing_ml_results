from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from toolz import merge, curry, assoc
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn import __version__ as sk_version
from fklearn.common_docstrings import learner_pred_fn_docstring, learner_return_docstring
from fklearn.types import LearnerReturnType
from fklearn.training.utils import log_learner_time, expand_features_encoded

@curry
@log_learner_time(learner_name='linear_regression_learner')
def linear_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    """
    Fits an linear regressor to the dataset. Return the predict function
    for the model and the predictions for the input dataset.

    Parameters
    ----------
    ...
    """
    def_params = {'fit_intercept': True}
    params = def_params if not params else merge(def_params, params)
    weights = df[weight_column].values if weight_column else None
    features = features if not encode_extra_cols else expand_features_encoded(df, features)
    regr = LinearRegression(**params)
    regr.fit(df[features].values, df[target].values, sample_weight=weights)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return new_df.assign(**{prediction_column: regr.predict(new_df[features].values)})
    p.__doc__ = learner_pred_fn_docstring('linear_regression_learner')
    log = {
        'linear_regression_learner': {
            'features': features,
            'target': target,
            'parameters': params,
            'prediction_column': prediction_column,
            'package': 'sklearn',
            'package_version': sk_version,
            'feature_importance': dict(zip(features, regr.coef_.flatten())),
            'training_samples': len(df)
        },
        'object': regr
    }
    return (p, p(df), log)
linear_regression_learner.__doc__ += learner_return_docstring('Linear Regression')


@curry
@log_learner_time(learner_name='xgb_regression_learner')
def xgb_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True
) -> Tuple[Callable[[pd.DataFrame, bool], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    """
    Fits an XGBoost regressor to the dataset. It first generates a DMatrix
    with the specified features and labels from `df`. Then it fits a XGBoost
    model to this DMatrix. Return the predict function for the model and the
    predictions for the input dataset.

    Parameters
    ----------
    ...
    """
    import xgboost as xgb
    weights = df[weight_column].values if weight_column else None
    params = extra_params if extra_params else {}
    params = assoc(params, 'eta', learning_rate)
    params = params if 'objective' in params else assoc(params, 'objective', 'reg:linear')
    features = features if not encode_extra_cols else expand_features_encoded(df, features)
    dtrain = xgb.DMatrix(df[features].values, label=df[target].values, weight=weights, feature_names=list(map(str, features)))
    bst = xgb.train(params, dtrain, num_estimators)

    def p(new_df: pd.DataFrame, apply_shap: bool = False) -> pd.DataFrame:
        dtest = xgb.DMatrix(new_df[features].values, feature_names=list(map(str, features)))
        col_dict = {prediction_column: bst.predict(dtest)}
        if apply_shap:
            import shap
            explainer = shap.TreeExplainer(bst)
            shap_values = list(explainer.shap_values(new_df[features]))
            shap_expected_value = explainer.expected_value
            shap_output = {
                'shap_values': shap_values,
                'shap_expected_value': np.repeat(shap_expected_value, len(shap_values))
            }
            col_dict = merge(col_dict, shap_output)
        return new_df.assign(**col_dict)
    p.__doc__ = learner_pred_fn_docstring('xgb_regression_learner', shap=True)
    log = {
        'xgb_regression_learner': {
            'features': features,
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
xgb_regression_learner.__doc__ += learner_return_docstring('XGboost Regressor')


@curry
@log_learner_time(learner_name='catboost_regressor_learner')
def catboost_regressor_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None
) -> Tuple[Callable[[pd.DataFrame, bool], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    """
    Fits an CatBoost regressor to the dataset. It first generates a Pool
    with the specified features and labels from `df`. Then it fits a CatBoost
    model to this Pool. Return the predict function for the model and the
    predictions for the input dataset.

    Parameters
    ----------
    ...
    """
    from catboost import Pool, CatBoostRegressor
    import catboost
    weights = df[weight_column].values if weight_column else None
    params = extra_params if extra_params else {}
    params = assoc(params, 'eta', learning_rate)
    dtrain = Pool(df[features].values, df[target].values, weight=weights, feature_names=list(map(str, features)))
    cat_boost_regressor = CatBoostRegressor(iterations=num_estimators, **params)
    cbr = cat_boost_regressor.fit(dtrain, verbose=0)

    def p(new_df: pd.DataFrame, apply_shap: bool = False) -> pd.DataFrame:
        dtest = Pool(new_df[features].values, feature_names=list(map(str, features)))
        col_dict = {prediction_column: cbr.predict(dtest)}
        if apply_shap:
            import shap
            explainer = shap.TreeExplainer(cbr)
            shap_values = list(explainer.shap_values(new_df[features]))
            shap_expected_value = explainer.expected_value
            shap_output = {
                'shap_values': shap_values,
                'shap_expected_value': np.repeat(shap_expected_value, len(shap_values))
            }
            col_dict = merge(col_dict, shap_output)
        return new_df.assign(**col_dict)
    p.__doc__ = learner_pred_fn_docstring('CatBoostRegressor', shap=False)
    log = {
        'catboost_regression_learner': {
            'features': features,
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
catboost_regressor_learner.__doc__ += learner_return_docstring('CatBoostRegressor')


@curry
@log_learner_time(learner_name='gp_regression_learner')
def gp_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    kernel: Optional[kernels.Kernel] = None,
    alpha: float = 0.1,
    extra_variance: Union[float, str] = 'fit',
    return_std: bool = False,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    encode_extra_cols: bool = True
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    """
    Fits an gaussian process regressor to the dataset.

    Parameters
    ----------
    ...
    """
    params = extra_params if extra_params else {}
    params['alpha'] = alpha
    params['kernel'] = kernel
    features = features if not encode_extra_cols else expand_features_encoded(df, features)
    gp = GaussianProcessRegressor(**params)
    gp.fit(df[features], df[target])
    if extra_variance == 'fit':
        extra_variance_value = df[target].std()
    else:
        extra_variance_value = extra_variance if extra_variance else 1

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        if return_std:
            pred_mean, pred_std = gp.predict(new_df[features], return_std=True)
            pred_std *= extra_variance_value
            return new_df.assign(**{
                prediction_column: pred_mean,
                prediction_column + '_std': pred_std
            })
        else:
            return new_df.assign(**{prediction_column: gp.predict(new_df[features])})
    p.__doc__ = learner_pred_fn_docstring('gp_regression_learner')
    log = {
        'gp_regression_learner': {
            'features': features,
            'target': target,
            'parameters': merge(params, {'extra_variance': extra_variance_value, 'return_std': return_std}),
            'prediction_column': prediction_column,
            'package': 'sklearn',
            'package_version': sk_version,
            'training_samples': len(df)
        },
        'object': gp
    }
    return (p, p(df), log)
gp_regression_learner.__doc__ += learner_return_docstring('Gaussian Process Regressor')


@curry
@log_learner_time(learner_name='lgbm_regression_learner')
def lgbm_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True
) -> Tuple[Callable[[pd.DataFrame, bool], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    """
    Fits an LGBM regressor to the dataset.

    It first generates a Dataset with the specified features and labels
    from `df`. Then, it fits a LGBM model to this Dataset. Return the predict
    function for the model and the predictions for the input dataset.

    Parameters
    ----------
    ...
    """
    import lightgbm as lgbm
    params = extra_params if extra_params else {}
    params = assoc(params, 'eta', learning_rate)
    params = params if 'objective' in params else assoc(params, 'objective', 'regression')
    weights = df[weight_column].values if weight_column else None
    features = features if not encode_extra_cols else expand_features_encoded(df, features)
    dtrain = lgbm.Dataset(df[features].values, label=df[target], feature_name=list(map(str, features)), weight=weights)
    bst = lgbm.train(params, dtrain, num_estimators)

    def p(new_df: pd.DataFrame, apply_shap: bool = False) -> pd.DataFrame:
        col_dict = {prediction_column: bst.predict(new_df[features].values)}
        if apply_shap:
            import shap
            explainer = shap.TreeExplainer(bst)
            shap_values = list(explainer.shap_values(new_df[features]))
            shap_expected_value = explainer.expected_value
            shap_output = {
                'shap_values': shap_values,
                'shap_expected_value': np.repeat(shap_expected_value, len(shap_values))
            }
            col_dict = merge(col_dict, shap_output)
        return new_df.assign(**col_dict)
    p.__doc__ = learner_pred_fn_docstring('lgbm_regression_learner', shap=True)
    log = {
        'lgbm_regression_learner': {
            'features': features,
            'target': target,
            'prediction_column': prediction_column,
            'package': 'lightgbm',
            'package_version': lgbm.__version__,
            'parameters': assoc(params, 'num_estimators', num_estimators),
            'feature_importance': dict(zip(features, bst.feature_importance().tolist())),
            'training_samples': len(df)
        },
        'object': bst
    }
    return (p, p(df), log)
lgbm_regression_learner.__doc__ += learner_return_docstring('LGBM Regressor')


@curry
@log_learner_time(learner_name='custom_supervised_model_learner')
def custom_supervised_model_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    model: Any,
    supervised_type: str,
    log: Dict[str, Dict[str, Any]],
    prediction_column: str = 'prediction'
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    """
    Fits a custom model to the dataset.
    Return the predict function, the predictions for the input dataset and a log describing the model.

    Parameters
    ----------
    ...
    """
    if len(log) != 1:
        raise ValueError("'log' dictionary must start with model name")
    if supervised_type not in ('classification', 'regression'):
        raise TypeError("supervised_type options are: 'classification' or 'regression'")
    if not hasattr(model, 'fit'):
        raise AttributeError("'model' object must have 'fit' attribute")
    if supervised_type == 'classification' and (not hasattr(model, 'predict_proba')):
        raise AttributeError("'model' object for classification must have 'predict_proba' attribute")
    if supervised_type == 'regression' and (not hasattr(model, 'predict')):
        raise AttributeError("'model' object for regression must have 'predict' attribute")
    model.fit(df[features].values, df[target].values)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        if supervised_type == 'classification':
            pred = model.predict_proba(new_df[features].values)
            col_dict: Dict[str, Any] = {}
            for key, value in enumerate(pred.T):
                col_dict.update({f"{prediction_column}_{key}": value})
        elif supervised_type == 'regression':
            col_dict = {prediction_column: model.predict(new_df[features].values)}
        return new_df.assign(**col_dict)
    p.__doc__ = learner_pred_fn_docstring('custom_supervised_model_learner')
    log['object'] = model
    return (p, p(df), log)
custom_supervised_model_learner.__doc__ += learner_return_docstring('Custom Supervised Model Learner')


@curry
@log_learner_time(learner_name='elasticnet_regression_learner')
def elasticnet_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    """
    Fits an elastic net regressor to the dataset. Return the predict function
    for the model and the predictions for the input dataset.

    Parameters
    ----------
    ...
    """
    def_params = {'fit_intercept': True}
    params = def_params if not params else merge(def_params, params)
    weights = df[weight_column].values if weight_column else None
    features = features if not encode_extra_cols else expand_features_encoded(df, features)
    regr = ElasticNet(**params)
    regr.fit(df[features].values, df[target].values, sample_weight=weights)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return new_df.assign(**{prediction_column: regr.predict(new_df[features].values)})
    p.__doc__ = learner_pred_fn_docstring('elasticnet_regression_learner')
    log = {
        'elasticnet_regression_learner': {
            'features': features,
            'target': target,
            'parameters': params,
            'prediction_column': prediction_column,
            'package': 'sklearn',
            'package_version': sk_version,
            'feature_importance': dict(zip(features, regr.coef_.flatten())),
            'training_samples': len(df)
        },
        'object': regr
    }
    return (p, p(df), log)
elasticnet_regression_learner.__doc__ += learner_return_docstring('ElasticNet Regression')
