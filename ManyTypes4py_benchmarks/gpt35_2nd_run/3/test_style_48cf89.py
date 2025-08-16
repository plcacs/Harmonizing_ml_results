from pathlib import Path
from typing import List, Dict, Union
import contextlib
import time
import uuid
import numpy as np
import pytest
from pandas import DataFrame, MultiIndex, Timestamp, period_range, read_excel
from pandas.io.excel import ExcelWriter
from pandas.io.formats.excel import ExcelFormatter

def assert_equal_cell_styles(cell1: object, cell2: object) -> None:
    assert cell1.alignment.__dict__ == cell2.alignment.__dict__
    assert cell1.border.__dict__ == cell2.border.__dict__
    assert cell1.fill.__dict__ == cell2.fill.__dict__
    assert cell1.font.__dict__ == cell2.font.__dict__
    assert cell1.number_format == cell2.number_format
    assert cell1.protection.__dict__ == cell2.protection.__dict__

def test_styler_default_values(tmp_excel: str) -> None:
    ...

def test_styler_to_excel_unstyled(engine: str, tmp_excel: str) -> None:
    ...

def test_styler_custom_style(tmp_excel: str) -> None:
    ...

def test_styler_to_excel_basic(engine: str, css: str, attrs: List[str], expected: Union[str, Dict[str, str]], tmp_excel: str) -> None:
    ...

def test_styler_to_excel_basic_indexes(engine: str, css: str, attrs: List[str], expected: Union[str, Dict[str, str]], tmp_excel: str) -> None:
    ...

def test_styler_to_excel_border_style(engine: str, border_style: str, tmp_excel: str) -> None:
    ...

def test_styler_custom_converter(tmp_excel: str) -> None:
    ...

def test_styler_to_s3(s3_public_bucket, s3so) -> None:
    ...

def test_format_hierarchical_rows_periodindex(merge_cells: Union[bool, str]) -> None:
    ...
