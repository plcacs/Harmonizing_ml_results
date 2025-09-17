from typing import List, Dict, Any, Optional, Type
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from statsmodels.formula.api import ols
from toolz import curry


@curry
def debias_with_regression_formula(
    df: DataFrame,
    treatment_column: str,
    outcome_column: str,
    confounder_formula: str,
    suffix: str = '_debiased',
    denoise: bool = True
) -> DataFrame:
    """
    Frisch-Waugh-Lovell style debiasing with linear regression. With R formula to define confounders.
    To debias, we
     1) fit a linear model to predict the treatment from the confounders and take the residuals from this fit
     (debias step)
     2) fit a linear model to predict the outcome from the confounders and take the residuals from this fit
     (denoise step).
    We then add back the average outcome and treatment so that their levels remain unchanged.

    Returns a dataframe with the debiased columns with suffix appended to the name
    """
    cols_to_debias: List[str] = [treatment_column, outcome_column] if denoise else [treatment_column]

    def get_resid(col_to_debias: str) -> Series:
        model = ols(f'{col_to_debias}~{confounder_formula}', data=df).fit()
        return model.resid + df[col_to_debias].mean()

    return df.assign(**{c + suffix: get_resid(c) for c in cols_to_debias})


@curry
def debias_with_regression(
    df: DataFrame,
    treatment_column: str,
    outcome_column: str,
    confounder_columns: List[str],
    suffix: str = '_debiased',
    denoise: bool = True
) -> DataFrame:
    """
    Frisch-Waugh-Lovell style debiasing with linear regression.
    To debias, we
     1) fit a linear model to predict the treatment from the confounders and take the residuals from this fit
     (debias step)
     2) fit a linear model to predict the outcome from the confounders and take the residuals from this fit
     (denoise step).
    We then add back the average outcome and treatment so that their levels remain unchanged.

    Returns a dataframe with the debiased columns with suffix appended to the name
    """
    model = LinearRegression()
    cols_to_debias: List[str] = [treatment_column, outcome_column] if denoise else [treatment_column]
    model.fit(df[confounder_columns], df[cols_to_debias])
    debiased: DataFrame = df[cols_to_debias] - model.predict(df[confounder_columns]) + df[cols_to_debias].mean()
    return df.assign(**{c + suffix: debiased[c] for c in cols_to_debias})


@curry
def debias_with_fixed_effects(
    df: DataFrame,
    treatment_column: str,
    outcome_column: str,
    confounder_columns: List[str],
    suffix: str = '_debiased',
    denoise: bool = True
) -> DataFrame:
    """
    Returns a dataframe with the debiased columns with suffix appended to the name

    This is equivalent of debiasing with regression where the forumla is "C(x1) + C(x2) + ...".
    However, it is much more eficient than runing such a dummy variable regression.
    """
    cols_to_debias: List[str] = [treatment_column, outcome_column] if denoise else [treatment_column]

    def debias_column(c: str) -> Dict[str, Series]:
        mu: Series = sum([df.groupby(x)[c].transform('mean') for x in confounder_columns])
        return {c + suffix: df[c] - mu + df[c].mean()}

    from toolz import merge
    return df.assign(**merge(*[debias_column(c) for c in cols_to_debias]))


@curry
def debias_with_double_ml(
    df: DataFrame,
    treatment_column: str,
    outcome_column: str,
    confounder_columns: List[str],
    ml_regressor: Type[RegressorMixin] = GradientBoostingRegressor,
    extra_params: Optional[Dict[str, Any]] = None,
    cv: int = 5,
    suffix: str = '_debiased',
    denoise: bool = True,
    seed: int = 123
) -> DataFrame:
    """
    Frisch-Waugh-Lovell style debiasing with ML model.
    To debias, we
     1) fit a regression ml model to predict the treatment from the confounders and take out of fold residuals from
      this fit (debias step)
     2) fit a regression ml model to predict the outcome from the confounders and take the out of fold residuals from
      this fit (denoise step).
    We then add back the average outcome and treatment so that their levels remain unchanged.

    Returns a dataframe with the debiased columns with suffix appended to the name
    """
    params: Dict[str, Any] = extra_params if extra_params is not None else {}
    cols_to_debias: List[str] = [treatment_column, outcome_column] if denoise else [treatment_column]
    np.random.seed(seed)

    def get_cv_resid(col_to_debias: str) -> Series:
        model = ml_regressor(**params)
        cv_pred: np.ndarray = cross_val_predict(estimator=model, X=df[confounder_columns], y=df[col_to_debias], cv=cv)
        return df[col_to_debias] - cv_pred + df[col_to_debias].mean()

    return df.assign(**{c + suffix: get_cv_resid(c) for c in cols_to_debias})