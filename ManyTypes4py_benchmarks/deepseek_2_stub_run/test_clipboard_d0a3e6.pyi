```python
from typing import Any, Callable, Iterator, Literal, overload
from typing_extensions import Self
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pandas._testing as tm
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch

def build_kwargs(
    sep: str = "default", excel: str = "default"
) -> dict[str, Any]: ...

@pytest.fixture
def df(request: FixtureRequest) -> DataFrame: ...

@pytest.fixture
def mock_ctypes(monkeypatch: MonkeyPatch) -> Iterator[None]: ...

@pytest.mark.usefixtures("mock_ctypes")
def test_checked_call_with_bad_call(monkeypatch: MonkeyPatch) -> None: ...

@pytest.mark.usefixtures("mock_ctypes")
def test_checked_call_with_valid_call(monkeypatch: MonkeyPatch) -> None: ...

@pytest.mark.parametrize("text", ["String_test", True, 1, 1.0, 1j])
def test_stringify_text(text: Any) -> None: ...

@pytest.fixture
def set_pyqt_clipboard(monkeypatch: MonkeyPatch) -> Iterator[None]: ...

@pytest.fixture
def clipboard(qapp: Any) -> Iterator[Any]: ...

@pytest.mark.single_cpu
@pytest.mark.clipboard
@pytest.mark.usefixtures("set_pyqt_clipboard")
@pytest.mark.usefixtures("clipboard")
class TestClipboard:
    @pytest.mark.parametrize("sep", [None, "\t", ",", "|"])
    @pytest.mark.parametrize("encoding", [None, "UTF-8", "utf-8", "utf8"])
    def test_round_trip_frame_sep(
        self, df: DataFrame, sep: str | None, encoding: str | None
    ) -> None: ...

    def test_round_trip_frame_string(self, df: DataFrame) -> None: ...

    def test_excel_sep_warning(self, df: DataFrame) -> None: ...

    def test_copy_delim_warning(self, df: DataFrame) -> None: ...

    @pytest.mark.parametrize("sep", ["\t", None, "default"])
    @pytest.mark.parametrize("excel", [True, None, "default"])
    def test_clipboard_copy_tabs_default(
        self,
        sep: str | None,
        excel: bool | None | Literal["default"],
        df: DataFrame,
        clipboard: Any,
    ) -> None: ...

    @pytest.mark.parametrize("sep", [None, "default"])
    def test_clipboard_copy_strings(
        self, sep: str | None, df: DataFrame
    ) -> None: ...

    def test_read_clipboard_infer_excel(self, clipboard: Any) -> None: ...

    def test_infer_excel_with_nulls(self, clipboard: Any) -> None: ...

    @pytest.mark.parametrize(
        "multiindex",
        [
            (
                "\n".join(
                    [
                        "\t\t\tcol1\tcol2",
                        "A\t0\tTrue\t1\tred",
                        "A\t1\tTrue\t\tblue",
                        "B\t0\tFalse\t2\tgreen",
                    ]
                ),
                [["A", "A", "B"], [0, 1, 0], [True, True, False]],
            ),
            (
                "\n".join(
                    [
                        "\t\tcol1\tcol2",
                        "A\t0\t1\tred",
                        "A\t1\t\tblue",
                        "B\t0\t2\tgreen",
                    ]
                ),
                [["A", "A", "B"], [0, 1, 0]],
            ),
        ],
    )
    def test_infer_excel_with_multiindex(
        self, clipboard: Any, multiindex: tuple[str, list[Any]]
    ) -> None: ...

    def test_invalid_encoding(self, df: DataFrame) -> None: ...

    @pytest.mark.parametrize("data", ["👍...", "Ωœ∑`...", "abcd..."])
    def test_raw_roundtrip(self, data: str) -> None: ...

    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_read_clipboard_dtype_backend(
        self,
        clipboard: Any,
        string_storage: Any,
        dtype_backend: Any,
        engine: str,
        using_infer_string: Any,
    ) -> None: ...

    def test_invalid_dtype_backend(self) -> None: ...
```