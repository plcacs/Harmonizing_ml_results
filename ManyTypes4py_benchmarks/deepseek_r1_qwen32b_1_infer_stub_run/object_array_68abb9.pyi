import numpy as np
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from numpy import (
    ndarray,
    dtype,
    int64,
    bool_,
    object_,
    str_,
)
from pandas._typing import NpDtype, Scalar

class ObjectStringArrayMixin:
    def __len__(self) -> int:
        ...

    def _str_map(
        self,
        f: Callable,
        na_value: Scalar = ...,
        dtype: Optional[NpDtype] = ...,
        convert: bool = ...,
    ) -> Union[ndarray, Scalar]:
        ...

    def _str_count(self, pat: str, flags: int = ...) -> ndarray[int64]:
        ...

    def _str_pad(
        self,
        width: int,
        side: Literal['left', 'right', 'both'] = ...,
        fillchar: str = ...,
    ) -> ndarray[object_]:
        ...

    def _str_contains(
        self,
        pat: str,
        case: bool = ...,
        flags: int = ...,
        na: Scalar = ...,
        regex: bool = ...,
    ) -> ndarray[bool_]:
        ...

    def _str_startswith(self, pat: str, na: Scalar = ...) -> ndarray[bool_]:
        ...

    def _str_endswith(self, pat: str, na: Scalar = ...) -> ndarray[bool_]:
        ...

    def _str_replace(
        self,
        pat: str,
        repl: Union[str, Callable],
        n: int = ...,
        case: bool = ...,
        flags: int = ...,
        regex: bool = ...,
    ) -> ndarray[object_]:
        ...

    def _str_repeat(
        self,
        repeats: Union[int, ndarray[object_]],
    ) -> Union[ndarray[str_], ndarray[object_]]:
        ...

    def _str_match(
        self,
        pat: str,
        case: bool = ...,
        flags: int = ...,
        na: Scalar = ...,
    ) -> ndarray[bool_]:
        ...

    def _str_fullmatch(
        self,
        pat: str,
        case: bool = ...,
        flags: int = ...,
        na: Scalar = ...,
    ) -> ndarray[bool_]:
        ...

    def _str_encode(
        self,
        encoding: str,
        errors: str = ...,
    ) -> ndarray[object_]:
        ...

    def _str_find(
        self,
        sub: str,
        start: int = ...,
        end: Optional[int] = ...,
    ) -> ndarray[int64]:
        ...

    def _str_rfind(
        self,
        sub: str,
        start: int = ...,
        end: Optional[int] = ...,
    ) -> ndarray[int64]:
        ...

    def _str_get(self, i: int) -> ndarray[object_]:
        ...

    def _str_index(
        self,
        sub: str,
        start: int = ...,
        end: Optional[int] = ...,
    ) -> ndarray[int64]:
        ...

    def _str_rindex(
        self,
        sub: str,
        start: int = ...,
        end: Optional[int] = ...,
    ) -> ndarray[int64]:
        ...

    def _str_join(self, sep: str) -> ndarray[object_]:
        ...

    def _str_split(
        self,
        pat: Optional[str] = ...,
        n: int = ...,
        expand: bool = ...,
        regex: Optional[bool] = ...,
    ) -> ndarray[object_]:
        ...

    def _str_rsplit(self, pat: Optional[str] = ..., n: int = ...) -> ndarray[object_]:
        ...

    def _str_translate(self, table: Any) -> ndarray[object_]:
        ...

    def _str_wrap(
        self,
        width: int,
        **kwargs: Any,
    ) -> ndarray[object_]:
        ...

    def _str_get_dummies(
        self,
        sep: str = ...,
        dtype: Optional[NpDtype] = ...,
    ) -> Tuple[ndarray[Any], List[str]]:
        ...

    def _str_upper(self) -> ndarray[object_]:
        ...

    def _str_isalnum(self) -> ndarray[bool_]:
        ...

    def _str_isalpha(self) -> ndarray[bool_]:
        ...

    def _str_isascii(self) -> ndarray[bool_]:
        ...

    def _str_isdecimal(self) -> ndarray[bool_]:
        ...

    def _str_isdigit(self) -> ndarray[bool_]:
        ...

    def _str_islower(self) -> ndarray[bool_]:
        ...

    def _str_isnumeric(self) -> ndarray[bool_]:
        ...

    def _str_isspace(self) -> ndarray[bool_]:
        ...

    def _str_istitle(self) -> ndarray[bool_]:
        ...

    def _str_isupper(self) -> ndarray[bool_]:
        ...

    def _str_capitalize(self) -> ndarray[object_]:
        ...

    def _str_casefold(self) -> ndarray[object_]:
        ...

    def _str_title(self) -> ndarray[object_]:
        ...

    def _str_swapcase(self) -> ndarray[object_]:
        ...

    def _str_lower(self) -> ndarray[object_]:
        ...

    def _str_normalize(self, form: str) -> ndarray[object_]:
        ...

    def _str_strip(self, to_strip: Optional[str] = ...) -> ndarray[object_]:
        ...

    def _str_lstrip(self, to_strip: Optional[str] = ...) -> ndarray[object_]:
        ...

    def _str_rstrip(self, to_strip: Optional[str] = ...) -> ndarray[object_]:
        ...

    def _str_removeprefix(self, prefix: str) -> ndarray[object_]:
        ...

    def _str_removesuffix(self, suffix: str) -> ndarray[object_]:
        ...

    def _str_extract(
        self,
        pat: str,
        flags: int = ...,
        expand: bool = ...,
    ) -> List[List[Optional[str]]]:
        ...