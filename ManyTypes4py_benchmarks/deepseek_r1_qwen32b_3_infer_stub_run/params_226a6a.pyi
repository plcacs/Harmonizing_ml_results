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
    Tuple,
    overload,
)

T = TypeVar("T", dict, list)

class Params(MutableMapping):
    DEFAULT: Any

    def __init__(self, params: Dict[str, Any], history: str = "") -> None:
        ...

    def pop(self, key: str, default: Any = ..., keep_as_dict: bool = ...) -> Any:
        ...

    def pop_int(self, key: str, default: Any = ...) -> Union[int, None]:
        ...

    def pop_float(self, key: str, default: Any = ...) -> Union[float, None]:
        ...

    def pop_bool(self, key: str, default: Any = ...) -> Union[bool, None]:
        ...

    def get(self, key: str, default: Any = ...) -> Any:
        ...

    def pop_choice(
        self, key: str, choices: List[Any], default_to_first_choice: bool = ..., allow_class_names: bool = ...
    ) -> Any:
        ...

    def as_dict(self, quiet: bool = ..., infer_type_and_cast: bool = ...) -> Dict[Any, Any]:
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
        cls, params_file: Union[str, PathLike], params_overrides: Union[str, Dict[str, Any]] = ..., ext_vars: Optional[Dict[str, Any]] = ...
    ) -> "Params":
        ...

    def to_file(self, params_file: Union[str, PathLike], preference_orders: Optional[List[List[str]]] = ...) -> None:
        ...

    def as_ordered_dict(self, preference_orders: Optional[List[List[str]]] = ...) -> OrderedDict:
        ...

    def get_hash(self) -> str:
        ...

    def __str__(self) -> str:
        ...

def infer_and_cast(value: Any) -> Union[int, float, bool, list, dict, str]:
    ...

def _is_encodable(value: str) -> bool:
    ...

def _environment_variables() -> Dict[str, str]:
    ...

def with_overrides(
    original: T, overrides_dict: Dict[str, Any], prefix: str = ...
) -> T:
    ...

def parse_overrides(
    serialized_overrides: str, ext_vars: Optional[Dict[str, Any]] = ...
) -> Dict[Any, Any]:
    ...

def _is_dict_free(obj: Any) -> bool:
    ...

def pop_choice(
    params: Dict[str, Any],
    key: str,
    choices: List[Any],
    default_to_first_choice: bool = ...,
    history: str = ...,
    allow_class_names: bool = ...,
) -> Any:
    ...