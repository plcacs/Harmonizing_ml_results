import copy
import json
import logging
import os
import zlib
from collections import OrderedDict
from collections.abc import MutableMapping
from os import PathLike
from typing import Any, Dict, List, Union, Optional, TypeVar, Iterable, Set

try:
    from _jsonnet import evaluate_file, evaluate_snippet
except ImportError:

    def evaluate_file(filename: str, **_kwargs: Any) -> str:
        ...

    def evaluate_snippet(_filename: str, expr: str, **_kwargs: Any) -> str:
        ...

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path

logger = logging.getLogger(__name__)

def infer_and_cast(value: Any) -> Any:
    ...

def _is_encodable(value: str) -> bool:
    ...

def _environment_variables() -> Dict[str, str]:
    ...

T = TypeVar('T', dict, list)

def with_overrides(original: T, overrides_dict: Dict[str, Any], prefix: str = '') -> T:
    ...

def parse_overrides(serialized_overrides: str, ext_vars: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    ...

def _is_dict_free(obj: Any) -> bool:
    ...

class Params(MutableMapping):
    DEFAULT: object = object()

    def __init__(self, params: Dict[str, Any], history: str = '') -> None:
        ...

    def pop(self, key: str, default: Any = DEFAULT, keep_as_dict: bool = False) -> Any:
        ...

    def pop_int(self, key: str, default: Any = DEFAULT) -> Optional[int]:
        ...

    def pop_float(self, key: str, default: Any = DEFAULT) -> Optional[float]:
        ...

    def pop_bool(self, key: str, default: Any = DEFAULT) -> Optional[bool]:
        ...

    def get(self, key: str, default: Any = DEFAULT) -> Any:
        ...

    def pop_choice(self, key: str, choices: List[Any], default_to_first_choice: bool = False, allow_class_names: bool = True) -> Any:
        ...

    def as_dict(self, quiet: bool = False, infer_type_and_cast: bool = False) -> Dict[str, Any]:
        ...

    def as_flat_dict(self) -> Dict[str, Any]:
        ...

    def duplicate(self) -> 'Params':
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
    def from_file(cls, params_file: str, params_overrides: Union[str, Dict[str, Any]] = '', ext_vars: Optional[Dict[str, str]] = None) -> 'Params':
        ...

    def to_file(self, params_file: str, preference_orders: Optional[List[List[str]]] = None) -> None:
        ...

    def as_ordered_dict(self, preference_orders: Optional[List[List[str]]] = None) -> OrderedDict:
        ...

    def get_hash(self) -> str:
        ...

    def __str__(self) -> str:
        ...

def pop_choice(params: Dict[str, Any], key: str, choices: List[Any], default_to_first_choice: bool = False, history: str = '?.', allow_class_names: bool = True) -> Any:
    ...

def _replace_none(params: Any) -> Any:
    ...

def remove_keys_from_params(params: Params, keys: List[str] = ['pretrained_file', 'initializer']) -> None:
    ...
