from datetime import datetime
from typing import Any
import zoneinfo
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under18p0, pa_version_under19p0
import pandas as pd
import pandas._testing as tm
from pandas.io.feather_format import read_feather, to_feather
pytestmark: Any = pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')
pa: Any = pytest.importorskip('pyarrow')

class TestFeather:

    def check_error_on_write(self, df: Any, exc: Any, err_msg: str) -> None:
        ...

    def check_external_error_on_write(self, df: Any) -> None:
        ...

    def check_round_trip(self, df: Any, expected: Any = None, write_kwargs: Any = None, **read_kwargs: Any) -> None:
        ...

    def test_error(self) -> None:
        ...

    def test_basic(self) -> None:
        ...

    def test_duplicate_columns(self) -> None:
        ...

    def test_read_columns(self) -> None:
        ...

    def test_read_columns_different_order(self) -> None:
        ...

    def test_unsupported_other(self) -> None:
        ...

    def test_rw_use_threads(self) -> None:
        ...

    def test_path_pathlib(self) -> None:
        ...

    def test_passthrough_keywords(self) -> None:
        ...

    def test_http_path(self, feather_file: Any, httpserver: Any) -> None:
        ...

    def test_read_feather_dtype_backend(self, string_storage: Any, dtype_backend: Any, using_infer_string: Any) -> None:
        ...

    def test_int_columns_and_index(self) -> None:
        ...

    def test_invalid_dtype_backend(self) -> None:
        ...

    def test_string_inference(self, tmp_path: Any, using_infer_string: Any) -> None:
        ...

    def test_string_inference_string_view_type(self, tmp_path: Any) -> None:
        ...

    def test_out_of_bounds_datetime_to_feather(self) -> None:
        ...
