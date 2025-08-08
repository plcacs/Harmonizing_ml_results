from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple, cast

class DeprecatedOption(NamedTuple):
    pass

class RegisteredOption(NamedTuple):
    pass

_deprecated_options: dict[str, DeprecatedOption] = {}
_registered_options: dict[str, RegisteredOption] = {}
_global_config: dict[str, Any] = {}
_reserved_keys: list[str] = ['all']

class OptionError(AttributeError, KeyError):
    ...

def _get_single_key(pat: str) -> str:
    ...

def get_option(pat: str) -> Any:
    ...

def set_option(*args: str) -> None:
    ...

def describe_option(pat: str = '', _print_desc: bool = True) -> None:
    ...

def reset_option(pat: str) -> None:
    ...

def get_default_val(pat: str) -> Any:
    ...

class DictWrapper:
    def __init__(self, d: dict[str, Any], prefix: str = ''):
        ...

    def __setattr__(self, key: str, val: Any) -> None:
        ...

    def __getattr__(self, key: str) -> Any:
        ...

    def __dir__(self) -> list[str]:
        ...

options: DictWrapper = DictWrapper(_global_config)

@contextmanager
def option_context(*args: str) -> Generator[None, None, None]:
    ...

def register_option(key: str, defval: Any, doc: str = '', validator: Callable = None, cb: Callable = None) -> None:
    ...

def deprecate_option(key: str, msg: str = None, rkey: str = None, removal_ver: str = None) -> None:
    ...

def _select_options(pat: str) -> list[str]:
    ...

def _get_root(key: str) -> tuple[dict[str, Any], str]:
    ...

def _get_deprecated_option(key: str) -> DeprecatedOption:
    ...

def _get_registered_option(key: str) -> RegisteredOption:
    ...

def _translate_key(key: str) -> str:
    ...

def _warn_if_deprecated(key: str) -> bool:
    ...

def _build_option_description(k: str) -> str:
    ...

@contextmanager
def config_prefix(prefix: str) -> Generator[None, None, None]:
    ...

def is_type_factory(_type: type) -> Callable[[Any], None]:
    ...

def is_instance_factory(_type: type) -> Callable[[Any], None]:
    ...

def is_one_of_factory(legal_values: list[Any]) -> Callable[[Any], None]:
    ...

def is_nonnegative_int(value: int) -> None:
    ...

def is_int(value: Any) -> None:
    ...

def is_bool(value: Any) -> None:
    ...

def is_float(value: Any) -> None:
    ...

def is_str(value: Any) -> None:
    ...

def is_text(value: Any) -> None:
    ...

def is_callable(obj: Any) -> bool:
    ...
