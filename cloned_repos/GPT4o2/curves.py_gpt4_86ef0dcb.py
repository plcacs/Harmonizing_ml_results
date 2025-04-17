```python
from typing import List, Callable, Union

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
    effect_fn_partial: Callable[[pd.DataFrame], Union[int, np.ndarray]] = partial(
        effect_fn, treatment_column=treatment, outcome_column=outcome
    )
    return (
        df.assign(**{f"{prediction}_band": pd.qcut(df[prediction], q=segments)})
        .groupby(f"{prediction}_band")
        .apply(effect_fn_partial)
    )


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
    size: int = df.shape[0]
    ordered_df: pd.DataFrame = df.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows: List[int] = list(range(min_rows, size, size // steps)) + [size]
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
    size: int = df.shape[0]
    n_rows: List[int] = list(range(min_rows, size, size // steps)) + [size]

    cum_effect: np.ndarray = cumulative_effect_curve(
        df=df, treatment=treatment, outcome=outcome, prediction=prediction,
        min_rows=min_rows, steps=steps, effect_fn=effect_fn
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
    ate: Union[int, np.ndarray] = effect_fn(df, treatment, outcome)
    size: int = df.shape[0]
    n_rows: List[int] = list(range(min_rows, size, size // steps)) + [size]

    cum_effect: np.ndarray = cumulative_effect_curve(
        df=df, treatment=treatment, outcome=outcome, prediction=prediction,
        min_rows=min_rows, steps=steps, effect_fn=effect_fn
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
    effect_fn: EffectFnType = linear_effect,
) -> pd.DataFrame:
    size: int = df.shape[0]
    n_rows: List[int] = list(range(min_rows, size, size // steps)) + [size]

    cum_effect: np.ndarray = cumulative_effect_curve(
        df=df,
        treatment=treatment,
        outcome=outcome,
        prediction=prediction,
        min_rows=min_rows,
        steps=steps,
        effect_fn=effect_fn,
    )
    ate: float = cum_effect[-1]

    return pd.DataFrame({"samples_count": n_rows, "cumulative_effect_curve": cum_effect}).assign(
        samples_fraction=lambda x: x["samples_count"] / size,
        cumulative_gain_curve=lambda x: x["samples_fraction"] * x["cumulative_effect_curve"],
        random_model_cumulative_gain_curve=lambda x: x["samples_fraction"] * ate,
        relative_cumulative_gain_curve=lambda x: (
            x["samples_fraction"] * x["cumulative_effect_curve"] - x["random_model_cumulative_gain_curve"]
        ),
    )
```