from datetime import datetime
import re
import numpy as np
import pytest
from pandas import DataFrame, NaT, concat
import pandas._testing as tm
from typing import List, Union, Dict

def test_drop_duplicates_with_misspelled_column_name(subset: Union[str, List[str]]):
def test_drop_duplicates():
def test_drop_duplicates_with_duplicate_column_names():
def test_drop_duplicates_for_take_all():
def test_drop_duplicates_tuple():
def test_drop_duplicates_empty(df: DataFrame):
def test_drop_duplicates_NA():
def test_drop_duplicates_NA_for_take_all():
def test_drop_duplicates_inplace():
def test_drop_duplicates_ignore_index(inplace: bool, origin_dict: Dict[str, List], output_dict: Dict[str, List], ignore_index: bool, output_index: List):
def test_drop_duplicates_null_in_object_column(nulls_fixture):
def test_drop_duplicates_series_vs_dataframe(keep: Union[str, bool]):
def test_drop_duplicates_non_boolean_ignore_index(arg: Union[List, int, str]):
def test_drop_duplicates_set():
