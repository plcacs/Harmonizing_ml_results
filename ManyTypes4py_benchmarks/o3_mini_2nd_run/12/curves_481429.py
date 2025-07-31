from typing import Callable, Union
import numpy as np
import pandas as pd
from toolz import curry, partial
from fklearn.types import EffectFnType
from fklearn.causal.effects import linear_effect

@curry
def effect_by_segment(
    df: pd.DataFrame, 
    treatment: str, 
    outcome: str, 
    prediction: str, 
    segments: int = 10, 
    effect_fn: EffectFnType = linear_effect
) -> pd.Series:
    """
    Segments the dataset by a prediction's quantile and estimates the treatment effect by segment.

    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment : str
        The name of the treatment column in `df`.

    outcome : str
        The name of the outcome column in `df`.

    prediction : str
        The name of the prediction column in `df`.

    segments : int
        The number of the segments to create. Uses Pandas' qcut under the hood.

    effect_fn : function (df: pd.DataFrame, treatment: str, outcome: str) -> Union[int, np.ndarray]
        A function that computes the treatment effect given a dataframe, the name of the treatment column and the name
        of the outcome column.

    Returns
    ----------
    effect by band : pd.Series
        The effect stored in a Pandas' series where the indexes are the segments
    """
    effect_fn_partial = partial(effect_fn, treatment_column=treatment, outcome_column=outcome)
    return df.assign(**{f'{prediction}_band': pd.qcut(df[prediction], q=segments)}).groupby(f'{prediction}_band').apply(effect_fn_partial)

@curry
def cumulative_effect_curve(
    df: pd.DataFrame, 
    treatment: str, 
    outcome: str, 
    prediction: str, 
    min_rows: int = 30, 
    steps: int = 100, 
    effect_fn: EffectFnType = linear_effect
) -> np.ndarray:
    """
    Orders the dataset by prediction and computes the cumulative effect curve according to that ordering

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment : str
        The name of the treatment column in `df`.

    outcome : str
        The name of the outcome column in `df`.

    prediction : str
        The name of the prediction column in `df`.

    min_rows : int
        Minimum number of observations needed to have a valid result.

    steps : int
        The number of cumulative steps to iterate when accumulating the effect

    effect_fn : function (df: pd.DataFrame, treatment: str, outcome: str) -> Union[int, np.ndarray]
        A function that computes the treatment effect given a dataframe, the name of the treatment column and the name
        of the outcome column.

    Returns
    ----------
    cumulative effect curve: np.ndarray
        The cumulative treatment effect according to the predictions ordering.
    """
    size = df.shape[0]
    ordered_df = df.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_rows, size, size // steps)) + [size]
    return np.array([effect_fn(ordered_df.head(rows), treatment, outcome) for rows in n_rows])

@curry
def cumulative_gain_curve(
    df: pd.DataFrame, 
    treatment: str, 
    outcome: str, 
    prediction: str, 
    min_rows: int = 30, 
    steps: int = 100, 
    effect_fn: EffectFnType = linear_effect
) -> np.ndarray:
    """
    Orders the dataset by prediction and computes the cumulative gain (effect * proportional sample size) curve
    according to that ordering.

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment : str
        The name of the treatment column in `df`.

    outcome : str
        The name of the outcome column in `df`.

    prediction : str
        The name of the prediction column in `df`.

    min_rows : int
        Minimum number of observations needed to have a valid result.

    steps : int
        The number of cumulative steps to iterate when accumulating the effect

    effect_fn : function (df: pd.DataFrame, treatment: str, outcome: str) -> Union[int, np.ndarray]
        A function that computes the treatment effect given a dataframe, the name of the treatment column and the name
        of the outcome column.

    Returns
    ----------
    cumulative gain curve: np.ndarray
        The cumulative gain according to the predictions ordering.
    """
    size = df.shape[0]
    n_rows = list(range(min_rows, size, size // steps)) + [size]
    cum_effect = cumulative_effect_curve(
        df=df, 
        treatment=treatment, 
        outcome=outcome, 
        prediction=prediction, 
        min_rows=min_rows, 
        steps=steps, 
        effect_fn=effect_fn
    )
    return np.array([effect * (rows / size) for rows, effect in zip(n_rows, cum_effect)])

@curry
def relative_cumulative_gain_curve(
    df: pd.DataFrame, 
    treatment: str, 
    outcome: str, 
    prediction: str, 
    min_rows: int = 30, 
    steps: int = 100, 
    effect_fn: EffectFnType = linear_effect
) -> np.ndarray:
    """
    Orders the dataset by prediction and computes the relative cumulative gain curve according to that ordering.
    The relative gain is simply the cumulative effect minus the Average Treatment Effect (ATE) times the relative
    sample size.

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment : str
        The name of the treatment column in `df`.

    outcome : str
        The name of the outcome column in `df`.

    prediction : str
        The name of the prediction column in `df`.

    min_rows : int
        Minimum number of observations needed to have a valid result.

    steps : int
        The number of cumulative steps to iterate when accumulating the effect

    effect_fn : function (df: pd.DataFrame, treatment: str, outcome: str) -> Union[int, np.ndarray]
        A function that computes the treatment effect given a dataframe, the name of the treatment column and the name
        of the outcome column.

    Returns
    ----------
    relative cumulative gain curve: np.ndarray
        The relative cumulative gain according to the predictions ordering.
    """
    ate = effect_fn(df, treatment, outcome)
    size = df.shape[0]
    n_rows = list(range(min_rows, size, size // steps)) + [size]
    cum_effect = cumulative_effect_curve(
        df=df, 
        treatment=treatment, 
        outcome=outcome, 
        prediction=prediction, 
        min_rows=min_rows, 
        steps=steps, 
        effect_fn=effect_fn
    )
    return np.array([(effect - ate) * (rows / size) for rows, effect in zip(n_rows, cum_effect)])

@curry
def effect_curves(
    df: pd.DataFrame, 
    treatment: str, 
    outcome: str, 
    prediction: str, 
    min_rows: int = 30, 
    steps: int = 100, 
    effect_fn: EffectFnType = linear_effect
) -> pd.DataFrame:
    """
    Creates a dataset summarizing the effect curves: cumulative effect, cumulative gain and
    relative cumulative gain. The dataset also contains two columns referencing the data
    used to compute the curves at each step: number of samples and fraction of samples used.
    Moreover one column indicating the cumulative gain for a corresponding random model is
    also included as a benchmark.

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment : str
        The name of the treatment column in `df`.

    outcome : str
        The name of the outcome column in `df`.

    prediction : str
        The name of the prediction column in `df`.

    min_rows : int
        Minimum number of observations needed to have a valid result.

    steps : int
        The number of cumulative steps to iterate when accumulating the effect

    effect_fn : function (df: pd.DataFrame, treatment: str, outcome: str) -> Union[int, np.ndarray]
        A function that computes the treatment effect given a dataframe, the name of the treatment column and the name
        of the outcome column.

    Returns
    ----------
    summary curves dataset: pd.DataFrame
        The dataset with the results for multiple validation causal curves according to the predictions ordering.
    """
    size = df.shape[0]
    n_rows = list(range(min_rows, size, size // steps)) + [size]
    cum_effect = cumulative_effect_curve(
        df=df, 
        treatment=treatment, 
        outcome=outcome, 
        prediction=prediction, 
        min_rows=min_rows, 
        steps=steps, 
        effect_fn=effect_fn
    )
    ate = cum_effect[-1]
    return pd.DataFrame({
        'samples_count': n_rows, 
        'cumulative_effect_curve': cum_effect
    }).assign(
        samples_fraction=lambda x: x['samples_count'] / size,
        cumulative_gain_curve=lambda x: x['samples_fraction'] * x['cumulative_effect_curve'],
        random_model_cumulative_gain_curve=lambda x: x['samples_fraction'] * ate,
        relative_cumulative_gain_curve=lambda x: x['samples_fraction'] * x['cumulative_effect_curve'] - x['random_model_cumulative_gain_curve']
    )