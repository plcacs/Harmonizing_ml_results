from functools import wraps
from typing import Callable, Any, Union

_registry: dict = {}

def convert(pinyin: str, style: str, strict: bool, default: Any = None, **kwargs: Any) -> Union[str, Any]:
    if style in _registry:
        return _registry[style](pinyin, strict=strict, **kwargs)
    return default

def register(style: str, func: Callable = None) -> Callable:
    if func is not None:
        _registry[style] = func
        return

    def decorator(func: Callable) -> Callable:
        _registry[style] = func

        @wraps(func)
        def wrapper(pinyin: str, **kwargs: Any) -> Any:
            return func(pinyin, **kwargs)
        return wrapper
    return decorator

def auto_discover() -> None:
    from pypinyin.style import initials, tone, finals, bopomofo, cyrillic, wadegiles, others
