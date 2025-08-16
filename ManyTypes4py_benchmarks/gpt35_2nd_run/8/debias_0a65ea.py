from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from statsmodels.formula.api import ols
from toolz import curry, merge

@curry
def debias_with_regression_formula(df: pd.DataFrame, treatment_column: str, outcome_column: str, confounder_formula: str, suffix: str='_debiased', denoise: bool=True) -> pd.DataFrame:

@curry
def debias_with_regression(df: pd.DataFrame, treatment_column: str, outcome_column: str, confounder_columns: List[str], suffix: str='_debiased', denoise: bool=True) -> pd.DataFrame:

@curry
def debias_with_fixed_effects(df: pd.DataFrame, treatment_column: str, outcome_column: str, confounder_columns: List[str], suffix: str='_debiased', denoise: bool=True) -> pd.DataFrame:

@curry
def debias_with_double_ml(df: pd.DataFrame, treatment_column: str, outcome_column: str, confounder_columns: List[str], ml_regressor: RegressorMixin, extra_params: Dict[str, Any]=None, cv: int=5, suffix: str='_debiased', denoise: bool=True, seed: int=123) -> pd.DataFrame:
