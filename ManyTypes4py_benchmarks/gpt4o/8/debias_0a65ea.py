from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from statsmodels.formula.api import ols
from toolz import curry, merge

@curry
def debias_with_regression_formula(
    df: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    confounder_formula: str,
    suffix: str = '_debiased',
    denoise: bool = True
) -> pd.DataFrame:
    cols_to_debias: List[str] = [treatment_column, outcome_column] if denoise else [treatment_column]

    def get_resid(col_to_debias: str) -> pd.Series:
        model = ols(f'{col_to_debias}~{confounder_formula}', data=df).fit()
        return model.resid + df[col_to_debias].mean()

    return df.assign(**{c + suffix: get_resid(c) for c in cols_to_debias})

@curry
def debias_with_regression(
    df: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    confounder_columns: List[str],
    suffix: str = '_debiased',
    denoise: bool = True
) -> pd.DataFrame:
    model = LinearRegression()
    cols_to_debias: List[str] = [treatment_column, outcome_column] if denoise else [treatment_column]
    model.fit(df[confounder_columns], df[cols_to_debias])
    debiased: pd.DataFrame = df[cols_to_debias] - model.predict(df[confounder_columns]) + df[cols_to_debias].mean()
    return df.assign(**{c + suffix: debiased[c] for c in cols_to_debias})

@curry
def debias_with_fixed_effects(
    df: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    confounder_columns: List[str],
    suffix: str = '_debiased',
    denoise: bool = True
) -> pd.DataFrame:
    cols_to_debias: List[str] = [treatment_column, outcome_column] if denoise else [treatment_column]

    def debias_column(c: str) -> Dict[str, pd.Series]:
        mu = sum([df.groupby(x)[c].transform('mean') for x in confounder_columns])
        return {c + suffix: df[c] - mu + df[c].mean()}

    return df.assign(**merge(*[debias_column(c) for c in cols_to_debias]))

@curry
def debias_with_double_ml(
    df: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    confounder_columns: List[str],
    ml_regressor: RegressorMixin = GradientBoostingRegressor,
    extra_params: Union[Dict[str, Any], None] = None,
    cv: int = 5,
    suffix: str = '_debiased',
    denoise: bool = True,
    seed: int = 123
) -> pd.DataFrame:
    params: Dict[str, Any] = extra_params if extra_params else {}
    cols_to_debias: List[str] = [treatment_column, outcome_column] if denoise else [treatment_column]
    np.random.seed(seed)

    def get_cv_resid(col_to_debias: str) -> pd.Series:
        model = ml_regressor(**params)
        cv_pred = cross_val_predict(estimator=model, X=df[confounder_columns], y=df[col_to_debias], cv=cv)
        return df[col_to_debias] - cv_pred + df[col_to_debias].mean()

    return df.assign(**{c + suffix: get_cv_resid(c) for c in cols_to_debias})
