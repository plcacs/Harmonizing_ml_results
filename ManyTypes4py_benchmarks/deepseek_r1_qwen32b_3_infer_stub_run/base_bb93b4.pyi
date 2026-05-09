from __future__ import annotations
import abc
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from pandas._libs import lib, ExtensionArray
from pandas._typing import NpDtype, Scalar, Self

class BaseStringArrayMethods(abc.ABC):
    def _str_getitem(self, key: Union[int, slice]) -> ExtensionArray:
        ...

    @abc.abstractmethod
    def _str_count(self, pat: str, flags: int = 0) -> ExtensionArray[int]:
        ...

    @abc.abstractmethod
    def _str_pad(
        self, width: int, side: Literal["left", "right", "both"] = "left", fillchar: str = " "
    ) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_contains(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Optional[Scalar] = None,
        regex: bool = True,
    ) -> ExtensionArray[bool]:
        ...

    @abc.abstractmethod
    def _str_startswith(self, pat: str, na: Optional[Scalar] = None) -> ExtensionArray[bool]:
        ...

    @abc.abstractmethod
    def _str_endswith(self, pat: str, na: Optional[Scalar] = None) -> ExtensionArray[bool]:
        ...

    @abc.abstractmethod
    def _str_replace(
        self,
        pat: str,
        repl: str,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_repeat(self, repeats: int) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_match(
        self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default
    ) -> ExtensionArray[Optional[Scalar]]:
        ...

    @abc.abstractmethod
    def _str_fullmatch(
        self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default
    ) -> ExtensionArray[Optional[Scalar]]:
        ...

    @abc.abstractmethod
    def _str_encode(self, encoding: str, errors: str = "strict") -> ExtensionArray[bytes]:
        ...

    @abc.abstractmethod
    def _str_find(self, sub: str, start: int = 0, end: Optional[int] = None) -> ExtensionArray[int]:
        ...

    @abc.abstractmethod
    def _str_rfind(self, sub: str, start: int = 0, end: Optional[int] = None) -> ExtensionArray[int]:
        ...

    @abc.abstractmethod
    def _str_findall(self, pat: str, flags: int = 0) -> ExtensionArray[List[str]]:
        ...

    @abc.abstractmethod
    def _str_get(self, i: int) -> ExtensionArray:
        ...

    @abc.abstractmethod
    def _str_index(self, sub: str, start: int = 0, end: Optional[int] = None) -> ExtensionArray[int]:
        ...

    @abc.abstractmethod
    def _str_rindex(self, sub: str, start: int = 0, end: Optional[int] = None) -> ExtensionArray[int]:
        ...

    @abc.abstractmethod
    def _str_join(self, sep: str) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_partition(self, sep: str, expand: bool) -> ExtensionArray[Tuple[str, str, str]]:
        ...

    @abc.abstractmethod
    def _str_rpartition(self, sep: str, expand: bool) -> ExtensionArray[Tuple[str, str, str]]:
        ...

    @abc.abstractmethod
    def _str_len(self) -> ExtensionArray[int]:
        ...

    @abc.abstractmethod
    def _str_slice(
        self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None
    ) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_slice_replace(
        self, start: Optional[int] = None, stop: Optional[int] = None, repl: Optional[str] = None
    ) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_translate(self, table: Dict[int, Union[str, int, None]]) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_wrap(self, width: int, **kwargs: Any) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_get_dummies(self, sep: str = "|", dtype: Optional[NpDtype] = None) -> ExtensionArray[dict]:
        ...

    @abc.abstractmethod
    def _str_isalnum(self) -> ExtensionArray[bool]:
        ...

    @abc.abstractmethod
    def _str_isalpha(self) -> ExtensionArray[bool]:
        ...

    @abc.abstractmethod
    def _str_isascii(self) -> ExtensionArray[bool]:
        ...

    @abc.abstractmethod
    def _str_isdecimal(self) -> ExtensionArray[bool]:
        ...

    @abc.abstractmethod
    def _str_isdigit(self) -> ExtensionArray[bool]:
        ...

    @abc.abstractmethod
    def _str_islower(self) -> ExtensionArray[bool]:
        ...

    @abc.abstractmethod
    def _str_isnumeric(self) -> ExtensionArray[bool]:
        ...

    @abc.abstractmethod
    def _str_isspace(self) -> ExtensionArray[bool]:
        ...

    @abc.abstractmethod
    def _str_istitle(self) -> ExtensionArray[bool]:
        ...

    @abc.abstractmethod
    def _str_isupper(self) -> ExtensionArray[bool]:
        ...

    @abc.abstractmethod
    def _str_capitalize(self) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_casefold(self) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_title(self) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_swapcase(self) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_lower(self) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_upper(self) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_normalize(self, form: str) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_strip(self, to_strip: Optional[str] = None) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_lstrip(self, to_strip: Optional[str] = None) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_rstrip(self, to_strip: Optional[str] = None) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_removeprefix(self, prefix: str) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_removesuffix(self, suffix: str) -> ExtensionArray[str]:
        ...

    @abc.abstractmethod
    def _str_split(
        self,
        pat: Optional[str] = None,
        n: int = -1,
        expand: bool = False,
        regex: Optional[bool] = None,
    ) -> ExtensionArray[List[str]]:
        ...

    @abc.abstractmethod
    def _str_rsplit(self, pat: Optional[str] = None, n: int = -1) -> ExtensionArray[List[str]]:
        ...

    @abc.abstractmethod
    def _str_extract(
        self, pat: str, flags: int = 0, expand: bool = True
    ) -> ExtensionArray[Union[List[str], str]]:
        ...