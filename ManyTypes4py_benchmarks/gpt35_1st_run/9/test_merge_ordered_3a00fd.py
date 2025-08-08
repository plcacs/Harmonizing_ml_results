from pandas import DataFrame, merge_ordered
import pandas._testing as tm
import numpy as np
import pytest
import pandas as pd
from typing import List, Dict, Union

def test_list_type_by(left: Dict[str, List[Union[str, int]]], right: Dict[str, List[Union[str, int]]], on: Union[str, List[str]], left_by: Union[str, List[str]], right_by: Union[str, List[str]], expected: Dict[str, List[Union[str, int]]]):
def test_left_by_length_equals_to_right_shape0():
def test_elements_not_in_by_but_in_df():
def test_ffill_validate_fill_method(left: DataFrame, right: DataFrame, invalid_method: str):
def test_ffill_left_merge():
