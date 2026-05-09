import copy
from itertools import chain
import json
import logging
import os
import zlib
from collections import OrderedDict
from collections.abc import MutableMapping
from os import PathLike
from typing import Any, Dict, List, Union, Optional, TypeVar, Iterable, Set, Tuple, overload

T = TypeVar('T', dict, list)

def infer_and_cast(value: Any) -> Union[int, float, bool, str, list, dict]:
    pass

def _is_encodable(value: Any) -> bool:
    pass

def _environment_variables() -> Dict[str, str]:
    pass

def with_overrides(original: T, overrides_dict: Dict[str, Any], prefix: str = '') -> T:
    pass

def parse_overrides(serialized_overrides: Union[str, dict, None], ext_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    pass

def _is_dict_free(obj: Any) -> bool:
    pass

class Params(MutableMapping[str, Any]):
    DEFAULT = object()

    def __init__(self, params: Union[dict, list], history: str = ''):
        pass

    def pop(self, key: str, default: Union[DEFAULT, Any] = DEFAULT, keep_as_dict: bool = False) -> Any:
        pass

    def pop_int(self, key: str, default: Union[DEFAULT, Any] = DEFAULT) -> Optional[int]:
        pass

    def pop_float(self, key: str, default: Union[DEFAULT, Any] = DEFAULT) -> Optional[float]:
        pass

    def pop_bool(self, key: str, default: Union[DEFAULT, Any] = DEFAULT) -> Optional[bool]:
        pass

    def get(self, key: str, default: Union[DEFAULT, Any] = DEFAULT) -> Any:
        pass

    def pop_choice(self, key: str, choices: List[Any], default_to_first_choice: bool = False, allow_class_names: bool = True) -> Any:
        pass

    def as_dict(self, quiet: bool = False, infer_type_and_cast: bool = False) -> Dict[str, Any]:
        pass

    def as_flat_dict(self) -> Dict[str, Any]:
        pass

    def duplicate(self) -> 'Params':
        pass

    def assert_empty(self, class_name: str) -> None:
        pass

    def __getitem__(self, key: str) -> Any:
        pass

    def __setitem__(self, key: str, value: Any) -> None:
        pass

    def __delitem__(self, key: str) -> None:
        pass

    def __iter__(self) -> Iterable[str]:
        pass

    def __len__(self) -> int:
        pass

    def _check_is_dict(self, new_history: str, value: Any) -> Any:
        pass

    @classmethod
    def from_file(cls, params_file: Union[str, PathLike], params_overrides: Union[str, dict, None] = '', ext_vars: Optional[Dict[str, Any]] = None) -> 'Params':
        pass

    def to_file(self, params_file: Union[str, PathLike], preference_orders: Optional[List[List[str]]] = None) -> None:
        pass

    def as_ordered_dict(self, preference_orders: Optional[List[List[str]]] = None) -> OrderedDict[str, Any]:
        pass

    def get_hash(self) -> str:
        pass

    def __str__(self) -> str:
        pass

def pop_choice(params: Union[dict, list], key: str, choices: List[Any], default_to_first_choice: bool = False, history: str = '?.', allow_class_names: bool = True) -> Any:
    pass

def _replace_none(params: Any) -> Any:
    pass

def remove_keys_from_params(params: Union[dict, 'Params'], keys: List[str] = ['pretrained_file', 'initializer']) -> None:
    pass