from collections.abc import Callable, Sequence
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence as Seq,
    Set,
    Tuple,
    Union,
)
from pandas._typing import NpDtype, Scalar
import numpy as np
import re
import textwrap

class ObjectStringArrayMixin:
    def __len__(self) -> int: ...
    
    def _str_map(
        self,
        f: Callable,
        na_value: Scalar = ...,
        dtype: Optional[NpDtype] = ...,
        convert: bool = ...,
    ) -> np.ndarray: ...

    def _str_count(self, pat: str, flags: int = ...) -> np.ndarray[np.int64]: ...

    def _str_pad(
        self,
        width: int,
        side: Literal['left', 'right', 'both'] = ...,
        fillchar: str = ...,
    ) -> np.ndarray: ...

    def _str_contains(
        self,
        pat: str,
        case: bool = ...,
        flags: int = ...,
        na: Scalar = ...,
        regex: bool = ...,
    ) -> np.ndarray[np.bool_]: ...

    def _str_startswith(self, pat: str, na: Scalar = ...) -> np.ndarray[np.bool_]: ...

    def _str_endswith(self, pat: str, na: Scalar = ...) -> np.ndarray[np.bool_]: ...

    def _str_replace(
        self,
        pat: Union[str, re.Pattern],
        repl: Union[str, Callable],
        n: int = ...,
        case: bool = ...,
        flags: int = ...,
        regex: bool = ...,
    ) -> np.ndarray: ...

    def _str_repeat(self, repeats: Union[int, np.ndarray]) -> np.ndarray: ...

    def _str_match(
        self,
        pat: str,
        case: bool = ...,
        flags: int = ...,
        na: Scalar = ...,
    ) -> np.ndarray[np.bool_]: ...

    def _str_fullmatch(
        self,
        pat: str,
        case: bool = ...,
        flags: int = ...,
        na: Scalar = ...,
    ) -> np.ndarray[np.bool_]: ...

    def _str_encode(self, encoding: str, errors: str = ...) -> np.ndarray: ...

    def _str_find(
        self,
        sub: str,
        start: int = ...,
        end: Optional[int] = ...,
    ) -> np.ndarray[np.int64]: ...

    def _str_rfind(
        self,
        sub: str,
        start: int = ...,
        end: Optional[int] = ...,
    ) -> np.ndarray[np.int64]: ...

    def _str_find_(self, sub: str, start: int, end: Optional[int], side: str) -> np.ndarray[np.int64]: ...

    def _str_findall(self, pat: str, flags: int = ...) -> np.ndarray: ...

    def _str_get(self, i: Union[int, str]) -> np.ndarray: ...

    def _str_index(
        self,
        sub: str,
        start: int = ...,
        end: Optional[int] = ...,
    ) -> np.ndarray[np.int64]: ...

    def _str_rindex(
        self,
        sub: str,
        start: int = ...,
        end: Optional[int] = ...,
    ) -> np.ndarray[np.int64]: ...

    def _str_join(self, sep: str) -> np.ndarray: ...

    def _str_partition(self, sep: str, expand: bool) -> np.ndarray: ...

    def _str_rpartition(self, sep: str, expand: bool) -> np.ndarray: ...

    def _str_len(self) -> np.ndarray[np.int64]: ...

    def _str_slice(self, start: Optional[int] = ..., stop: Optional[int] = ..., step: Optional[int] = ...) -> np.ndarray: ...

    def _str_slice_replace(
        self,
        start: Optional[int] = ...,
        stop: Optional[int] = ...,
        repl: str = ...,
    ) -> np.ndarray: ...

    def _str_split(
        self,
        pat: Optional[str] = ...,
        n: int = ...,
        expand: bool = ...,
        regex: Optional[bool] = ...,
    ) -> np.ndarray: ...

    def _str_rsplit(self, pat: Optional[str] = ..., n: int = ...) -> np.ndarray: ...

    def _str_translate(self, table: Dict[int, Union[str, int, None]]) -> np.ndarray: ...

    def _str_wrap(self, width: int, **kwargs: Any) -> np.ndarray: ...

    def _str_get_dummies(
        self,
        sep: str = ...,
        dtype: Optional[NpDtype] = ...,
    ) -> Tuple[np.ndarray, List[str]]: ...

    def _str_upper(self) -> np.ndarray: ...

    def _str_isalnum(self) -> np.ndarray[np.bool_]: ...

    def _str_isalpha(self) -> np.ndarray[np.bool_]: ...

    def _str_isascii(self) -> np.ndarray[np.bool_]: ...

    def _str_isdecimal(self) -> np.ndarray[np.bool_]: ...

    def _str_isdigit(self) -> np.ndarray[np.bool_]: ...

    def _str_islower(self) -> np.ndarray[np.bool_]: ...

    def _str_isnumeric(self) -> np.ndarray[np.bool_]: ...

    def _str_isspace(self) -> np.ndarray[np.bool_]: ...

    def _str_istitle(self) -> np.ndarray[np.bool_]: ...

    def _str_isupper(self) -> np.ndarray[np.bool_]: ...

    def _str_capitalize(self) -> np.ndarray: ...

    def _str_casefold(self) -> np.ndarray: ...

    def _str_title(self) -> np.ndarray: ...

    def _str_swapcase(self) -> np.ndarray: ...

    def _str_lower(self) -> np.ndarray: ...

    def _str_normalize(self, form: str) -> np.ndarray: ...

    def _str_strip(self, to_strip: Optional[str] = ...) -> np.ndarray: ...

    def _str_lstrip(self, to_strip: Optional[str] = ...) -> np.ndarray: ...

    def _str_rstrip(self, to_strip: Optional[str] = ...) -> np.ndarray: ...

    def _str_removeprefix(self, prefix: str) -> np.ndarray: ...

    def _str_removesuffix(self, suffix: str) -> np.ndarray: ...

    def _str_extract(
        self,
        pat: str,
        flags: int = ...,
        expand: bool = ...,
    ) -> List[List[Optional[str]]]: ...