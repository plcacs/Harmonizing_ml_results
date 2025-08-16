from contextlib import contextmanager
import json
from typing import Union, Any, Tuple, Callable, List, Dict

class Option:
    def __init__(self, *, key: str, doc: str, default: Any, types: Union[Tuple[type, ...], type] = str, check_func: Tuple[Callable[[Any], bool], str] = (lambda v: True, '')) -> None:
    def validate(self, v: Any) -> None:

def get_option(key: str, default: Any = _NoValue) -> Any:

def set_option(key: str, value: Any) -> None:

def reset_option(key: str) -> None:

@contextmanager
def option_context(*args: Any) -> None:

def _check_option(key: str) -> None:

class DictWrapper:
    def __init__(self, d: Dict, prefix: str = '') -> None:
    def __setattr__(self, key: str, val: Any) -> None:
    def __getattr__(self, key: str) -> Any:
    def __dir__(self) -> List[str]:
