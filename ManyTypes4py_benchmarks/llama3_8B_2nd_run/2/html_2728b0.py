import re
from collections import abc
import errno
import numbers
import os
from re import Pattern
from typing import TYPE_CHECKING, Literal, cast
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError, EmptyDataError
from pandas.util._decorators import doc
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import is_list_like
from pandas import isna
from pandas.core.indexes.base import Index
from pandas.core.indexes.multi import MultiIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle, is_url, stringify_path, validate_header_arg, check_dtype_backend
from pandas.io.formats.printing import pprint_thing
from pandas.io.parsers import TextParser

def read_html(io: str | path | file_like, 
              *, 
              match: str | Pattern, 
              flavor: None | str | list_like, 
              header: int | list_like, 
              index_col: int | list_like, 
              skiprows: int | list_like | slice, 
              attrs: dict, 
              parse_dates: bool, 
              thousands: str, 
              encoding: str, 
              decimal: str, 
              converters: dict, 
              na_values: list_like, 
              keep_default_na: bool, 
              displayed_only: bool, 
              extract_links: Literal[None, 'all', 'header', 'body', 'footer'], 
              dtype_backend: DtypeBackend, 
              storage_options: StorageOptions = None) -> list[DataFrame]:
    ...
