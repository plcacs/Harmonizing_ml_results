from __future__ import annotations
from datetime import datetime
import struct
from typing import TYPE_CHECKING, List, Dict, Union
import warnings
import numpy as np
import pandas as pd
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas._typing import CompressionOptions, DatetimeNaTType, FilePath, ReadBuffer

_correct_line1: str = 'HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!000000000000000000000000000000  '
_correct_header1: str = 'HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!000000000000000001600000000'
_correct_header2: str = 'HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!000000000000000000000000000000  '
_correct_obs_header: str = 'HEADER RECORD*******OBS     HEADER RECORD!!!!!!!000000000000000000000000000000  '
_fieldkeys: List[str] = ['ntype', 'nhfun', 'field_length', 'nvar0', 'name', 'label', 'nform', 'nfl', 'num_decimals', 'nfj', 'nfill', 'niform', 'nifl', 'nifd', 'npos', '_']
_base_params_doc: str = 'Parameters\n----------\nfilepath_or_buffer : str or file-like object\n    Path to SAS file or object implementing binary read method.'
_params2_doc: str = 'index : identifier of index column\n    Identifier of column that should be used as index of the DataFrame.\nencoding : str\n    Encoding for text data.\nchunksize : int\n    Read file `chunksize` lines at a time, returns iterator.'
_format_params_doc: str = 'format : str\n    File format, only `xport` is currently supported.'
_iterator_doc: str = 'iterator : bool, default False\n    Return XportReader object for reading file incrementally.'
_read_sas_doc: str = f"Read a SAS file into a DataFrame.\n\n{_base_params_doc}\n{_format_params_doc}\n{_params2_doc}\n{_iterator_doc}\n\nReturns\n-------\nDataFrame or XportReader\n\nExamples\n--------\nRead a SAS Xport file:\n\n>>> df = pd.read_sas('filename.XPT')\n\nRead a Xport file in 10,000 line chunks:\n\n>>> itr = pd.read_sas('filename.XPT', chunksize=10000)\n>>> for chunk in itr:\n>>>     do_something(chunk)\n\n"
_xport_reader_doc: str = f'Class for reading SAS Xport files.\n\n{_base_params_doc}\n{_params2_doc}\n\nAttributes\n----------\nmember_info : list\n    Contains information about the file\nfields : list\n    Contains information about the variables in the file\n'
_read_method_doc: str = 'Read observations from SAS Xport file, returning as data frame.\n\nParameters\n----------\nnrows : int\n    Number of rows to read from data file; if None, read whole\n    file.\n\nReturns\n-------\nA DataFrame.\n'

def _parse_date(datestr: str) -> Union[datetime, pd.NaT]:
    ...

def _split_line(s: str, parts: List[Tuple[str, int]]) -> Dict[str, str]:
    ...

def _handle_truncated_float_vec(vec: np.ndarray, nbytes: int) -> np.ndarray:
    ...

def _parse_float_vec(vec: np.ndarray) -> np.ndarray:
    ...

class XportReader(SASReader):
    __doc__: str = _xport_reader_doc

    def __init__(self, filepath_or_buffer: FilePath, index: str = None, encoding: str = 'ISO-8859-1', chunksize: int = None, compression: CompressionOptions = 'infer'):
        ...

    def close(self) -> None:
        ...

    def _get_row(self) -> str:
        ...

    def _read_header(self) -> None:
        ...

    def __next__(self) -> pd.DataFrame:
        ...

    def _record_count(self) -> int:
        ...

    def get_chunk(self, size: int = None) -> pd.DataFrame:
        ...

    def _missing_double(self, vec: np.ndarray) -> np.ndarray:
        ...

    @Appender(_read_method_doc)
    def read(self, nrows: int = None) -> pd.DataFrame:
        ...
