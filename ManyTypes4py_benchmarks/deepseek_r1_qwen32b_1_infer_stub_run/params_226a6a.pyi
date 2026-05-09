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
    Callable,
    overload,
)

logger = logging.getLogger(__name__)
T = TypeVar("T", dict, list)

def infer_and_cast(value: Any) -> Union[int, float, bool, str]:
    ...

def _is_encodable(value: str) -> bool:
    ...

def _environment_variables() -> Dict[str, str]:
    ...

def with_overrides(
    original: Union[dict, list], overrides_dict: dict, prefix: str = ""
) -> Union[dict, list]:
    ...

def parse_overrides(
    serialized_overrides: str, ext_vars: Optional[dict] = None
) -> Dict[Any, Any]:
    ...

def _is_dict_free(obj: Any) -> bool:
    ...

class Params(MutableMapping):
    DEFAULT: object

    def __init__(self, params: Union[dict, list], history: str = "") -> None:
        ...

    def pop(
        self, key: str, default: Union[Any, object] = DEFAULT, keep_as_dict: bool = False
    ) -> Any:
        ...

    def pop_int(self, key: str, default: Union[Any, object] = DEFAULT) -> Optional[int]:
        ...

    def pop_float(
        self, key: str, default: Union[Any, object] = DEFAULT
    ) -> Optional[float]:
        ...

    def pop_bool(
        self, key: str, default: Union[Any, object] = DEFAULT
    ) -> Optional[bool]:
        ...

    def get(self, key: str, default: Union[Any, object] = DEFAULT) -> Any:
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
    ) -> Dict[Any, Any]:
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
        params_overrides: Union[str, dict] = "",
        ext_vars: Optional[dict] = None,
    ) -> "Params":
        ...

    def to_file(self, params_file: Union[str, PathLike], preference_orders: Optional[List[List[str]]] = None) -> None:
        ...

    def as_ordered_dict(
        self, preference_orders: Optional[List[List[str]]] = None
    ) -> OrderedDict:
        ...

    def get_hash(self) -> str:
        ...

    def __str__(self) -> str:
        ...

def pop_choice(
    params: Union[dict, "Params"],
    key: str,
    choices: List[Any],
    default_to_first_choice: bool = False,
    history: str = "?.",
    allow_class_names: bool = True,
) -> Any:
    ...

def _replace_none(params: Union[dict, list, Any]) -> Union[None, dict, list, Any]:
    ...

def remove_keys_from_params(
    params: Union[dict, "Params"], keys: List[str] = ["pretrained_file", "initializer"]
) -> None:
    ...