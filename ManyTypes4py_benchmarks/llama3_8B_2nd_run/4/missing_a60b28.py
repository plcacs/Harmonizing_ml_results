import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm

class BaseMissingTests:
    def test_isna(self, data_missing: pd.Series) -> None:
        # ...

    @pytest.mark.parametrize('na_func', ['isna', 'notna'])
    def test_isna_returns_copy(self, data_missing: pd.Series, na_func: str) -> None:
        # ...

    def test_dropna_array(self, data_missing: pd.Series) -> None:
        # ...

    def test_dropna_series(self, data_missing: pd.Series) -> None:
        # ...

    def test_dropna_frame(self, data_missing: pd.Series) -> None:
        # ...

    def test_fillna_scalar(self, data_missing: pd.Series) -> None:
        # ...

    def test_fillna_with_none(self, data_missing: pd.Series) -> None:
        # ...

    def test_fillna_limit_pad(self, data_missing: pd.Series) -> None:
        # ...

    @pytest.mark.parametrize('limit_area, input_ilocs, expected_ilocs', [('outside', [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]), ('outside', [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]), ('outside', [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]), ('outside', [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]), ('inside', [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]), ('inside', [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]), ('inside', [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]), ('inside', [0, 1, 0, 1, 0], [0, 1, 1, 1, 0])])
    def test_ffill_limit_area(self, data_missing: pd.Series, limit_area: str, input_ilocs: list, expected_ilocs: list) -> None:
        # ...

    def test_fillna_limit_backfill(self, data_missing: pd.Series) -> None:
        # ...

    def test_fillna_no_op_returns_copy(self, data: pd.Series) -> None:
        # ...

    def test_fillna_series(self, data_missing: pd.Series) -> None:
        # ...

    def test_fillna_series_method(self, data_missing: pd.Series, fillna_method: str) -> None:
        # ...

    def test_fillna_frame(self, data_missing: pd.Series) -> None:
        # ...

    def test_fillna_fill_other(self, data: pd.Series) -> None:
        # ...
