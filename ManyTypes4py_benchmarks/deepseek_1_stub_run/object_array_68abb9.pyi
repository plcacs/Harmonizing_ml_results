```python
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    overload,
    Union,
    Callable,
    Sequence,
    Pattern,
)
import numpy as np
import pandas._libs.lib as lib
import pandas._libs.missing as libmissing
import pandas._libs.ops as libops
from pandas.core.strings.base import BaseStringArrayMethods
from pandas._typing import NpDtype, Scalar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pandas._typing import NpDtype, Scalar

class ObjectStringArrayMixin(BaseStringArrayMethods):
    def __len__(self) -> int: ...
    
    def _str_map(
        self,
        f: Callable[[Any], Any],
        na_value: Any = ...,
        dtype: Any = None,
        convert: bool = True
    ) -> Any: ...
    
    def _str_count(self, pat: str, flags: int = 0) -> Any: ...
    
    def _str_pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " "
    ) -> Any: ...
    
    def _str_contains(
        self,
        pat: str | Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: Any = ...,
        regex: bool = True
    ) -> Any: ...
    
    def _str_startswith(self, pat: str, na: Any = ...) -> Any: ...
    
    def _str_endswith(self, pat: str, na: Any = ...) -> Any: ...
    
    def _str_replace(
        self,
        pat: str | Pattern[str],
        repl: str | Callable[[Any], str],
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True
    ) -> Any: ...
    
    def _str_repeat(self, repeats: int | Sequence[int]) -> Any: ...
    
    def _str_match(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Any = ...
    ) -> Any: ...
    
    def _str_fullmatch(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Any = ...
    ) -> Any: ...
    
    def _str_encode(
        self,
        encoding: str,
        errors: str = "strict"
    ) -> Any: ...
    
    def _str_find(
        self,
        sub: str,
        start: int = 0,
        end: int | None = None
    ) -> Any: ...
    
    def _str_rfind(
        self,
        sub: str,
        start: int = 0,
        end: int | None = None
    ) -> Any: ...
    
    def _str_find_(
        self,
        sub: str,
        start: int,
        end: int | None,
        side: Literal["left", "right"]
    ) -> Any: ...
    
    def _str_findall(self, pat: str, flags: int = 0) -> Any: ...
    
    def _str_get(self, i: Any) -> Any: ...
    
    def _str_index(
        self,
        sub: str,
        start: int = 0,
        end: int | None = None
    ) -> Any: ...
    
    def _str_rindex(
        self,
        sub: str,
        start: int = 0,
        end: int | None = None
    ) -> Any: ...
    
    def _str_join(self, sep: str) -> Any: ...
    
    def _str_partition(self, sep: str, expand: bool) -> Any: ...
    
    def _str_rpartition(self, sep: str, expand: bool) -> Any: ...
    
    def _str_len(self) -> Any: ...
    
    def _str_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None
    ) -> Any: ...
    
    def _str_slice_replace(
        self,
        start: int | None = None,
        stop: int | None = None,
        repl: str | None = None
    ) -> Any: ...
    
    def _str_split(
        self,
        pat: str | None = None,
        n: int = -1,
        expand: bool = False,
        regex: bool | None = None
    ) -> Any: ...
    
    def _str_rsplit(
        self,
        pat: str | None = None,
        n: int = -1
    ) -> Any: ...
    
    def _str_translate(self, table: Any) -> Any: ...
    
    def _str_wrap(self, width: int, **kwargs: Any) -> Any: ...
    
    def _str_get_dummies(
        self,
        sep: str = "|",
        dtype: Any = None
    ) -> tuple[Any, Any]: ...
    
    def _str_upper(self) -> Any: ...
    
    def _str_isalnum(self) -> Any: ...
    
    def _str_isalpha(self) -> Any: ...
    
    def _str_isascii(self) -> Any: ...
    
    def _str_isdecimal(self) -> Any: ...
    
    def _str_isdigit(self) -> Any: ...
    
    def _str_islower(self) -> Any: ...
    
    def _str_isnumeric(self) -> Any: ...
    
    def _str_isspace(self) -> Any: ...
    
    def _str_istitle(self) -> Any: ...
    
    def _str_isupper(self) -> Any: ...
    
    def _str_capitalize(self) -> Any: ...
    
    def _str_casefold(self) -> Any: ...
    
    def _str_title(self) -> Any: ...
    
    def _str_swapcase(self) -> Any: ...
    
    def _str_lower(self) -> Any: ...
    
    def _str_normalize(self, form: str) -> Any: ...
    
    def _str_strip(self, to_strip: str | None = None) -> Any: ...
    
    def _str_lstrip(self, to_strip: str | None = None) -> Any: ...
    
    def _str_rstrip(self, to_strip: str | None = None) -> Any: ...
    
    def _str_removeprefix(self, prefix: str) -> Any: ...
    
    def _str_removesuffix(self, suffix: str) -> Any: ...
    
    def _str_extract(
        self,
        pat: str,
        flags: int = 0,
        expand: bool = True
    ) -> Any: ...
```