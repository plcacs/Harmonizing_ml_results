from typing import List
import pandas as pd

def compute_expected(df_left: pd.DataFrame, df_right: pd.DataFrame, on: List[str] = None, left_on: List[str] = None, right_on: List[str] = None, how: str = None) -> pd.DataFrame:
    ...
