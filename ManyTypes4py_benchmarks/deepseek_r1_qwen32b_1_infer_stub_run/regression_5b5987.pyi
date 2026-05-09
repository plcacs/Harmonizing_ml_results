from typing import Any, Dict, List, Optional, Tuple, Union
from pandas import DataFrame
from numpy import ndarray
from toolz import Curry
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import Booster
from lightgbm import LGBMModel
from catboost import CatBoostRegressor

def linear_regression_learner(df: DataFrame, features: List[str], target: str, params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True) -> Tuple[Curry, DataFrame, Dict[str, Any]]:
    ...

def xgb_regression_learner(df: DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True) -> Tuple[Curry, DataFrame, Dict[str, Any]]:
    ...

def catboost_regressor_learner(df: DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None) -> Tuple[Curry, DataFrame, Dict[str, Any]]:
    ...

def gp_regression_learner(df: DataFrame, features: List[str], target: str, kernel: Optional[Any] = None, alpha: float = 0.1, extra_variance: Union[float, str] = 'fit', return_std: bool = False, extra_params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction', encode_extra_cols: bool = True) -> Tuple[Curry, DataFrame, Dict[str, Any]]:
    ...

def lgbm_regression_learner(df: DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True) -> Tuple[Curry, DataFrame, Dict[str, Any]]:
    ...

def custom_supervised_model_learner(df: DataFrame, features: List[str], target: str, model: Any, supervised_type: str, log: Dict[str, Dict[str, Any]], prediction_column: str = 'prediction') -> Tuple[Curry, DataFrame, Dict[str, Any]]:
    ...

def elasticnet_regression_learner(df: DataFrame, features: List[str], target: str, params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True) -> Tuple[Curry, DataFrame, Dict[str, Any]]:
    ...