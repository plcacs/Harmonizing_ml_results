from __future__ import annotations
from contextlib import contextmanager
import re
from typing import TYPE_CHECKING, Any, NamedTuple, Callable, Generator, Sequence

class OptionError(AttributeError, KeyError):
    ...

def _get_single_key(pat: str) -> str:
    ...

def get_option(pat: str) -> Any:
    ...

def set_option(*args: str | object) -> None:
    ...

def describe_option(pat: str = '', _print_desc: bool = True) -> None | str:
    ...

def reset_option(pat: str) -> None:
    ...

def get_default_val(pat: str) -> Any:
    ...

class DictWrapper:
    ...

@contextmanager
def option_context(*args: str | object) -> None:
    ...

def register_option(key: str, defval: object, doc: str = '', validator: Callable | None = None, cb: Callable | None = None) -> None:
    ...

def deprecate_option(key: str, msg: str | None = None, rkey: str | None = None, removal_ver: str | None = None) -> None:
    ...

def _select_options(pat: str) -> list[str]:
    ...

def _get_root(key: str) -> tuple[dict, str]:
    ...

def _get_deprecated_option(key: str) -> DeprecatedOption | None:
    ...

def _get_registered_option(key: str) -> RegisteredOption | None:
    ...

def _translate_key(key: str) -> str:
    ...

def _warn_if_deprecated(key: str) -> bool:
    ...

def _build_option_description(k: str) -> str:
    ...

@contextmanager
def config_prefix(prefix: str) -> Generator:
    ...

def is_type_factory(_type: type) -> Callable[[Any], None]:
    ...

def is_instance_factory(_type: type) -> Callable[[Any], None]:
    ...

def is_one_of_factory(legal_values: Sequence[Any]) -> Callable[[Any], None]:
    ...

def is_nonnegative_int(value: Any) -> None:
    ...

is_int: Callable[[Any], None] = is_type_factory(int)
is_bool: Callable[[Any], None] = is_type_factory(bool)
is_float: Callable[[Any], None] = is_type_factory(float)
is_str: Callable[[Any], None] = is_type_factory(str)
is_text: Callable[[Any], None] = is_instance_factory((str, bytes))

def is_callable(obj: Any) -> None:
    ...
