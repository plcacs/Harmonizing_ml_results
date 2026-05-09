from datetime import datetime, timedelta
from importlib import reload
import string
import sys
import numpy as np
import pandas as pd
from pandas._libs.tslibs import iNaT
from pandas._testing import tm
from typing import Any

class TestAstypeAPI:
    def test_astype_unitless_dt64_raises(self) -> None:
        # ...

    def test_arg_for_errors_in_astype(self) -> None:
        # ...

    @pytest.mark.parametrize('dtype_class', [dict, Series])
    def test_astype_dict_like(self, dtype_class: Any) -> None:
        # ...

    def test_astype_object_to_dt64_non_nano(self, tz: str) -> None:
        # ...

    def test_astype_no_pandas_dtype(self) -> None:
        # ...

    @pytest.mark.parametrize('dtype', np.typecodes['All'])
    def test_astype_empty_constructor_equality(self, dtype: str) -> None:
        # ...

    # ...

class TestAstypeString:
    def test_astype_string_to_extension_dtype_roundtrip(self, data: list, dtype: str, request: Any, nullable_string_dtype: str) -> None:
        # ...

    def test_astype_categorical_to_other(self) -> None:
        # ...

class TestAstypeCategorical:
    def test_astype_categorical_to_categorical(self, name: str, dtype_ordered: bool, series_ordered: bool) -> None:
        # ...

    @pytest.mark.parametrize('name', [None, 'foo'])
    @pytest.mark.parametrize('dtype_ordered', [True, False])
    @pytest.mark.parametrize('series_ordered', [True, False])
    def test_astype_categorical_to_categorical(self, name: str, dtype_ordered: bool, series_ordered: bool) -> None:
        # ...
