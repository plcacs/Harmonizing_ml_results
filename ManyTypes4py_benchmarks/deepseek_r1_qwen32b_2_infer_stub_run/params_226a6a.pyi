import copy
from itertools import chain
import json
import logging
import os
import zlib
from collections import OrderedDict
from collections.abc import MutableMapping
from os import PathLike
from typing import (
    Any,
    Dict,
    List,
    Union,
    Optional,
    TypeVar,
    Iterable,
    Set,
    OrderedDict,
    Tuple,
    overload,
    Type,
)

T = TypeVar("T", dict, list)

class Params(MutableMapping):
    DEFAULT: Any

    def __init__(self, params: Union[dict, list, Any], history: str = "") -> None:
        ...

    @overload
    def pop(self, key: str, default: Any = ..., keep_as_dict: bool = False) -> Any:
        ...

    @overload
    def pop(self, key: str, default: None, keep_as_dict: bool = False) -> Optional[Any]:
        ...

    def pop(self, key: str, default: Any = ..., keep_as_dict: bool = False) -> Any:
        ...

    def pop_int(self, key: str, default: Any = ...) -> Optional[int]:
        ...

    def pop_float(self, key: str, default: Any = ...) -> Optional[float]:
        ...

    def pop_bool(self, key: str, default: Any = ...) -> Optional[bool]:
        ...

    @overload
    def get(self, key: str, default: Any = ...) -> Any:
        ...

    @overload
    def get(self, key: str, default: None) -> Optional[Any]:
        ...

    def get(self, key: str, default: Any = ...) -> Any:
        ...

    def pop_choice(
        self,
        key: str,
        choices: List[Any],
        default_to_first_choice: bool = False,
        allow_class_names: bool = True,
    ) -> Any:
        ...

    def as_dict(
        self, quiet: bool = False, infer_type_and_cast: bool = False
    ) -> Dict[str, Any]:
        ...

    def as_flat_dict(self) -> Dict[str, Any]:
        ...

    def duplicate(self) -> "Params":
        ...

    def assert_empty(self, class_name: str) -> None:
        ...

    def __getitem__(self, key: str) -> Any:
        ...

    def __setitem__(self, key: str, value: Any) -> None:
        ...

    def __delitem__(self, key: str) -> None:
        ...

    def __iter__(self) -> Iterable[str]:
        ...

    def __len__(self) -> int:
        ...

    def _check_is_dict(self, new_history: str, value: Any) -> Any:
        ...

    @classmethod
    def from_file(
        cls,
        params_file: Union[str, PathLike],
        params_overrides: Union[str, Dict[str, Any]] = "",
        ext_vars: Optional[Dict[str, Any]] = None,
    ) -> "Params":
        ...

    def to_file(
        self, params_file: Union[str, PathLike], preference_orders: Optional[List[List[str]]] = None
    ) -> None:
        ...

    def as_ordered_dict(
        self, preference_orders: Optional[List[List[str]]] = None
    ) -> OrderedDict:
        ...

    def get_hash(self) -> str:
        ...

    def __str__(self) -> str:
        ...

def infer_and_cast(value: Any) -> Union[int, float, bool, str, list, dict]:
    ...

def _is_encodable(value: Any) -> bool:
    ...

def _environment_variables() -> Dict[str, str]:
    ...

def with_overrides(
    original: T, overrides_dict: Dict[str, Any], prefix: str = ""
) -> T:
    ...

def parse_overrides(
    serialized_overrides: str, ext_vars: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    ...

def _is_dict_free(obj: Any) -> bool:
    ...

def pop_choice(
    params: Union[dict, list, Any],
    key: str,
    choices: List[Any],
    default_to_first_choice: bool = False,
    history: str = "?.",
    allow_class_names: bool = True,
) -> Any:
    ...

def _replace_none(params: Any) -> Union[dict, list, Any]:
    ...

def remove_keys_from_params(
    params: Union[dict, "Params"], keys: List[str] = ["pretrained_file", "initializer"]
) -> None:
    ...