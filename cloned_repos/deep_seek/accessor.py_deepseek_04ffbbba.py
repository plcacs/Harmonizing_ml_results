from __future__ import annotations

import codecs
from functools import wraps
import re
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
    Union,
    Optional,
    Callable,
    List,
    Tuple,
    Dict,
    Any,
    Pattern,
    Hashable,
    Iterator,
    overload,
    Sequence,
    TypeVar,
    Collection,
)
import warnings

import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas._typing import (
    TYPE_CHECKING,
    AlignJoin,
    DtypeObj,
    F,
    Scalar,
    npt,
    NpDtype,
)
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_extension_array_dtype,
    is_integer,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_re,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import ExtensionArray
from pandas.core.base import NoNewAttributesMixin
from pandas.core.construction import extract_array
from pandas import (
    Series,
    Index,
    DataFrame,
    MultiIndex,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterator,
    )
    from pandas._typing import NpDtype
    from pandas import (
        DataFrame,
        Index,
        Series,
    )

_shared_docs: dict[str, str] = {}
_cpython_optimized_encoders = (
    "utf-8",
    "utf8",
    "latin-1",
    "latin1",
    "iso-8859-1",
    "mbcs",
    "ascii",
)
_cpython_optimized_decoders = _cpython_optimized_encoders + ("utf-16", "utf-32")

F = TypeVar('F', bound=Callable[..., Any])

def forbid_nonstring_types(
    forbidden: list[str] | None, name: str | None = None
) -> Callable[[F], F]:
    """
    Decorator to forbid specific types for a method of StringMethods.
    """
    # Implementation remains the same

def _map_and_wrap(name: str | None, docstring: str | None) -> Callable[[StringMethods], Union[Series, Index]]:
    # Implementation remains the same

class StringMethods(NoNewAttributesMixin):
    _inferred_dtype: str
    _is_categorical: bool
    _is_string: bool
    _data: Union[Series, Index]
    _index: Optional[Index]
    _name: Optional[Hashable]
    _parent: Union[Series, Index]
    _orig: Union[Series, Index]

    def __init__(self, data: Union[Series, Index]) -> None:
        # Implementation remains the same

    @staticmethod
    def _validate(data: Union[Series, Index]) -> str:
        # Implementation remains the same

    def __getitem__(self, key: Union[int, slice]) -> Union[Series, Index]:
        # Implementation remains the same

    def __iter__(self) -> Iterator:
        # Implementation remains the same

    def _wrap_result(
        self,
        result: Union[Series, Index, DataFrame, np.ndarray],
        name: Optional[Hashable] = None,
        expand: Optional[bool] = None,
        fill_value: Any = np.nan,
        returns_string: bool = True,
        dtype: Optional[DtypeObj] = None,
    ) -> Union[Series, Index, DataFrame]:
        # Implementation remains the same

    def _get_series_list(self, others: Any) -> List[Series]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes", "mixed", "mixed-integer"])
    def cat(
        self,
        others: Optional[Union[Series, Index, DataFrame, np.ndarray, List]] = None,
        sep: Optional[str] = None,
        na_rep: Optional[str] = None,
        join: AlignJoin = "left",
    ) -> Union[str, Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["str_split"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def split(
        self,
        pat: Union[str, Pattern] = None,
        *,
        n: int = -1,
        expand: bool = False,
        regex: Optional[bool] = None,
    ) -> Union[Series, Index, DataFrame]:
        # Implementation remains the same

    @Appender(_shared_docs["str_split"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def rsplit(
        self,
        pat: Optional[str] = None,
        *,
        n: int = -1,
        expand: bool = False,
    ) -> Union[Series, Index, DataFrame]:
        # Implementation remains the same

    @Appender(_shared_docs["str_partition"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def partition(
        self,
        sep: str = " ",
        expand: bool = True,
    ) -> Union[Series, Index, DataFrame]:
        # Implementation remains the same

    @Appender(_shared_docs["str_partition"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def rpartition(
        self,
        sep: str = " ",
        expand: bool = True,
    ) -> Union[Series, Index, DataFrame]:
        # Implementation remains the same

    def get(self, i: Union[int, Hashable]) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def join(self, sep: str) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def contains(
        self,
        pat: Union[str, Pattern],
        case: bool = True,
        flags: int = 0,
        na: Union[Scalar, lib.NoDefault] = lib.no_default,
        regex: bool = True,
    ) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def match(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Union[Scalar, lib.NoDefault] = lib.no_default,
    ) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def fullmatch(
        self,
        pat: Union[str, Pattern],
        case: bool = True,
        flags: int = 0,
        na: Union[Scalar, lib.NoDefault] = lib.no_default,
    ) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def replace(
        self,
        pat: Union[str, Pattern, Dict],
        repl: Union[str, Callable, None] = None,
        n: int = -1,
        case: Optional[bool] = None,
        flags: int = 0,
        regex: bool = False,
    ) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def repeat(self, repeats: Union[int, Sequence[int]]) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ) -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["str_pad"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def center(self, width: int, fillchar: str = " ") -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["str_pad"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def ljust(self, width: int, fillchar: str = " ") -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["str_pad"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def rjust(self, width: int, fillchar: str = " ") -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def zfill(self, width: int) -> Union[Series, Index]:
        # Implementation remains the same

    def slice(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def slice_replace(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        repl: Optional[str] = None,
    ) -> Union[Series, Index]:
        # Implementation remains the same

    def decode(self, encoding: str, errors: str = "strict") -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def encode(self, encoding: str, errors: str = "strict") -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["str_strip"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def strip(self, to_strip: Optional[str] = None) -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["str_strip"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def lstrip(self, to_strip: Optional[str] = None) -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["str_strip"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def rstrip(self, to_strip: Optional[str] = None) -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["str_removefix"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def removeprefix(self, prefix: str) -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["str_removefix"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def removesuffix(self, suffix: str) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def wrap(
        self,
        width: int,
        expand_tabs: bool = True,
        tabsize: int = 8,
        replace_whitespace: bool = True,
        drop_whitespace: bool = True,
        initial_indent: str = "",
        subsequent_indent: str = "",
        fix_sentence_endings: bool = False,
        break_long_words: bool = True,
        break_on_hyphens: bool = True,
        max_lines: Optional[int] = None,
        placeholder: str = " [...]",
    ) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def get_dummies(
        self,
        sep: str = "|",
        dtype: Optional[NpDtype] = None,
    ) -> DataFrame:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def translate(self, table: Dict[int, Union[int, str, None]]) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def count(
        self,
        pat: Union[str, Pattern],
        flags: int = 0,
    ) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def startswith(
        self,
        pat: Union[str, Tuple[str, ...]],
        na: Union[Scalar, lib.NoDefault] = lib.no_default,
    ) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def endswith(
        self,
        pat: Union[str, Tuple[str, ...]],
        na: Union[Scalar, lib.NoDefault] = lib.no_default,
    ) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def findall(
        self,
        pat: Union[str, Pattern],
        flags: int = 0,
    ) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def extract(
        self,
        pat: str,
        flags: int = 0,
        expand: bool = True,
    ) -> Union[DataFrame, Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def extractall(self, pat: str, flags: int = 0) -> DataFrame:
        # Implementation remains the same

    @Appender(_shared_docs["find"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def find(
        self,
        sub: str,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["find"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def rfind(
        self,
        sub: str,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def normalize(self, form: str) -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["index"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def index(
        self,
        sub: str,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["index"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def rindex(
        self,
        sub: str,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Union[Series, Index]:
        # Implementation remains the same

    def len(self) -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["casemethods"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def lower(self) -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["casemethods"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def upper(self) -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["casemethods"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def title(self) -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["casemethods"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def capitalize(self) -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["casemethods"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def swapcase(self) -> Union[Series, Index]:
        # Implementation remains the same

    @Appender(_shared_docs["casemethods"] % { ... })
    @forbid_nonstring_types(["bytes"])
    def casefold(self) -> Union[Series, Index]:
        # Implementation remains the same

    # Boolean methods with type annotations
    @forbid_nonstring_types(["bytes"])
    def isalnum(self) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def isalpha(self) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def isdigit(self) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def isspace(self) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def islower(self) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def isupper(self) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def istitle(self) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def isnumeric(self) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def isdecimal(self) -> Union[Series, Index]:
        # Implementation remains the same

    @forbid_nonstring_types(["bytes"])
    def isascii(self) -> Union[Series, Index]:
        # Implementation remains the same

def cat_safe(
    list_of_columns: List[npt.NDArray[np.object_]],
    sep: str
) -> npt.NDArray[np.object_]:
    # Implementation remains the same

def cat_core(
    list_of_columns: List[npt.NDArray[np.object_]],
    sep: str
) -> npt.NDArray[np.object_]:
    # Implementation remains the same

def str_extractall(
    arr: Union[Series, Index],
    pat: str,
    flags: int = 0
) -> DataFrame:
    # Implementation remains the same

def _result_dtype(arr: Union[Series, Index]) -> DtypeObj:
    # Implementation remains the same

def _get_single_group_name(regex: Pattern) -> Optional[Hashable]:
    # Implementation remains the same

def _get_group_names(regex: Pattern) -> Union[List[Hashable], range