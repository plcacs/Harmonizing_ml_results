from io import BytesIO, StringIO
import os
import platform
from urllib.error import URLError
import uuid
import numpy as np
import pytest
from pandas.compat import WASM
from pandas.errors import EmptyDataError, ParserError
import pandas.util._test_decorators as td
from pandas import DataFrame, Index
import pandas._testing as tm
from typing import Any, Dict, List, Tuple, Union

def test_url(all_parsers: Any, csv_dir_path: str, httpserver: Any) -> None:
    ...

def test_local_file(all_parsers: Any, csv_dir_path: str) -> None:
    ...

def test_path_path_lib(all_parsers: Any) -> None:
    ...

def test_nonexistent_path(all_parsers: Any) -> None:
    ...

def test_no_permission(all_parsers: Any) -> None:
    ...

def test_eof_states(all_parsers: Any, data: str, kwargs: Dict[str, Any], expected: Any, msg: str, request: Any) -> None:
    ...

def test_temporary_file(all_parsers: Any, temp_file: Any) -> None:
    ...

def test_internal_eof_byte(all_parsers: Any) -> None:
    ...

def test_internal_eof_byte_to_file(all_parsers: Any) -> None:
    ...

def test_file_handle_string_io(all_parsers: Any) -> None:
    ...

def test_file_handles_with_open(all_parsers: Any, csv1: str) -> None:
    ...

def test_invalid_file_buffer_class(all_parsers: Any) -> None:
    ...

def test_invalid_file_buffer_mock(all_parsers: Any) -> None:
    ...

def test_valid_file_buffer_seems_invalid(all_parsers: Any) -> None:
    ...

def test_read_csv_file_handle(all_parsers: Any, io_class: Any, encoding: Union[None, str]) -> None:
    ...

def test_memory_map_compression(all_parsers: Any, compression: str) -> None:
    ...

def test_context_manager(all_parsers: Any, datapath: Any) -> None:
    ...

def test_context_manageri_user_provided(all_parsers: Any, datapath: Any) -> None:
    ...

def test_file_descriptor_leak(all_parsers: Any) -> None:
    ...

def test_memory_map(all_parsers: Any, csv_dir_path: str) -> None:
    ...
