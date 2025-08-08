from typing import List
import numpy as np
import pandas as pd
from toolz import curry, partial
from fklearn.types import EffectFnType

@curry
def effect_by_segment(df: pd.DataFrame, treatment: str, outcome: str, prediction: str, segments: int = 10, effect_fn: EffectFnType = linear_effect) -> pd.Series:

@curry
def cumulative_effect_curve(df: pd.DataFrame, treatment: str, outcome: str, prediction: str, min_rows: int = 30, steps: int = 100, effect_fn: EffectFnType = linear_effect) -> np.ndarray:

@curry
def cumulative_gain_curve(df: pd.DataFrame, treatment: str, outcome: str, prediction: str, min_rows: int = 30, steps: int = 100, effect_fn: EffectFnType = linear_effect) -> np.ndarray:

@curry
def relative_cumulative_gain_curve(df: pd.DataFrame, treatment: str, outcome: str, prediction: str, min_rows: int = 30, steps: int = 100, effect_fn: EffectFnType = linear_effect) -> np.ndarray:

@curry
def effect_curves(df: pd.DataFrame, treatment: str, outcome: str, prediction: str, min_rows: int = 30, steps: int = 100, effect_fn: EffectFnType = linear_effect) -> pd.DataFrame:
