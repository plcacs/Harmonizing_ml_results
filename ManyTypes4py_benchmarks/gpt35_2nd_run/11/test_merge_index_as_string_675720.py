from typing import List
import pandas as pd

def compute_expected(df_left: pd.DataFrame, df_right: pd.DataFrame, on: List[str] = None, left_on: List[str] = None, right_on: List[str] = None, how: str = None) -> pd.DataFrame:
    ...

def test_merge_indexes_and_columns_on(left_df: pd.DataFrame, right_df: pd.DataFrame, on: List[str], how: str):
    ...

def test_merge_indexes_and_columns_lefton_righton(left_df: pd.DataFrame, right_df: pd.DataFrame, left_on: List[str], right_on: List[str], how: str):
    ...

def test_join_indexes_and_columns_on(df1: pd.DataFrame, df2: pd.DataFrame, left_index: str, join_type: str):
    ...
