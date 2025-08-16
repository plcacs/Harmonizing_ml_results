from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, List, Tuple
import pandas as pd
import pytest

def _expand_user(filename: str) -> str:
    ...

def stringify_path(path: Any) -> str:
    ...

def infer_compression(path: Any, compression: str) -> str:
    ...

def get_handle(path: Any, mode: str, is_text: bool = False) -> Any:
    ...

def is_fsspec_url(url: str) -> bool:
    ...

def get_writer_reader(encoding: str) -> Tuple[Any, Any]:
    ...

def read_csv_chained_url_no_error(compression: str) -> pd.DataFrame:
    ...

def pickle_reader(reader: Any) -> None:
    ...

def pyarrow_read_csv_datetime_dtype(data: str) -> pd.DataFrame:
    ...
