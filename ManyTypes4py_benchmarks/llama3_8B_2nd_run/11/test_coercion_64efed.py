from __future__ import annotations
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class CoercionBase:
    klasses: list[str]
    dtypes: list[str]

    @property
    def method(self) -> str:
        raise NotImplementedError

class TestSetitemCoercion(CoercionBase):
    klasses: list[str] = ['index', 'series']
    dtypes: list[str] = ['object', 'int64', 'float64', 'complex128', 'bool', 'datetime64', 'datetime64tz', 'timedelta64', 'period']

    # ... rest of the class ...

class TestFillnaSeriesCoercion(CoercionBase):
    klasses: list[str] = ['series']
    method: str = 'fillna'
    rep: dict[str, list] = {}

    # ... rest of the class ...

    def test_replace_series(self, how: str, to_key: str, from_key: str, replacer: dict[str, str]) -> None:
        # ... rest of the function ...

    @pytest.mark.parametrize('to_key', ['timedelta64[ns]', 'bool', 'object', 'complex128', 'float64', 'int64'], indirect=True)
    @pytest.mark.parametrize('from_key', ['datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]'], indirect=True)
    def test_replace_series_datetime_tz(self, how: str, to_key: str, from_key: str, replacer: dict[str, str], using_infer_string: bool) -> None:
        # ... rest of the function ...

    @pytest.mark.parametrize('to_key', ['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]'], indirect=True)
    @pytest.mark.parametrize('from_key', ['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]'], indirect=True)
    def test_replace_series_datetime_datetime(self, how: str, to_key: str, from_key: str, replacer: dict[str, str]) -> None:
        # ... rest of the function ...
