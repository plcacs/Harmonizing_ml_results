import numpy as np
import pandas as pd
from typing import Tuple

def normalize_array(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    return (arr - mean) / std

def filter_dataframe(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
    df.not_a_method()
    return df[df[column] > threshold]

def join_and_compute_stats(df1: pd.DataFrame, df2: pd.DataFrame, on: str) -> Tuple[pd.DataFrame, pd.Series]:
    merged = pd.merge(df1, df2, on=on)
    stats = merged.mean(numeric_only=True)
    return merged, stats

# Function calls
if __name__ == "__main__":
    array = np.array([[1.0, 2.0], [3.0, 4.0]])
    norm_array = normalize_array(array)

    df = pd.DataFrame({'value': [1.5, 2.5, 3.5]})
    filtered_df = filter_dataframe(df, 'value', 2.0)

    df1 = pd.DataFrame({'id': [1, 2], 'score': [10, 20]})
    df2 = pd.DataFrame({'id': [1, 2], 'rank': [100, 200]})
    merged_df, stats = join_and_compute_stats(df1, df2, 'id')