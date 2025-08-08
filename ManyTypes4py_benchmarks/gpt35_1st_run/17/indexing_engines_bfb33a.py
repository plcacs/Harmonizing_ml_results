import numpy as np
from pandas._libs import index as libindex
from pandas.core.arrays import BaseMaskedArray
from typing import List, Tuple

def _get_numeric_engines() -> List[Tuple[type, type]]:
def _get_masked_engines() -> List[Tuple[type, str]]:
class NumericEngineIndexing:
    params: List[List] = [_get_numeric_engines(), ['monotonic_incr', 'monotonic_decr', 'non_monotonic'], [True, False], [10 ** 5, 2 * 10 ** 6]]
    param_names: List[str] = ['engine_and_dtype', 'index_type', 'unique', 'N']

    def setup(self, engine_and_dtype: Tuple[type, type], index_type: str, unique: bool, N: int) -> None:
    def time_get_loc(self, engine_and_dtype: Tuple[type, type], index_type: str, unique: bool, N: int) -> None:
    def time_get_loc_near_middle(self, engine_and_dtype: Tuple[type, type], index_type: str, unique: bool, N: int) -> None:

class MaskedNumericEngineIndexing:
    params: List[List] = [_get_masked_engines(), ['monotonic_incr', 'monotonic_decr', 'non_monotonic'], [True, False], [10 ** 5, 2 * 10 ** 6]]
    param_names: List[str] = ['engine_and_dtype', 'index_type', 'unique', 'N']

    def setup(self, engine_and_dtype: Tuple[type, str], index_type: str, unique: bool, N: int) -> None:
    def time_get_loc(self, engine_and_dtype: Tuple[type, str], index_type: str, unique: bool, N: int) -> None:
    def time_get_loc_near_middle(self, engine_and_dtype: Tuple[type, str], index_type: str, unique: bool, N: int) -> None:

class ObjectEngineIndexing:
    params: List[str] = [('monotonic_incr', 'monotonic_decr', 'non_monotonic')]
    param_names: List[str] = ['index_type']

    def setup(self, index_type: str) -> None:
    def time_get_loc(self, index_type: str) -> None:
