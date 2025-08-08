from functools import wraps
from typing import Callable, Any, Union

_registry: dict[str, Callable[..., Union[str, None]]] = {}

def convert(pinyin: str, style: str, strict: bool, default: Any = None, **kwargs) -> Union[str, None]:
    if style in _registry:
        return _registry[style](pinyin, strict=strict, **kwargs)
    return default

def register(style: str, func: Union[Callable[..., Union[str, None]], None] = None) -> Union[Callable[..., Callable[..., Union[str, None]]], Callable[..., Union[str, None]]]:
    if func is not None:
        _registry[style] = func
        return

    def decorator(func: Callable[..., Union[str, None]]) -> Callable[..., Union[str, None]]:
        _registry[style] = func

        @wraps(func)
        def wrapper(pinyin: str, **kwargs) -> Union[str, None]:
            return func(pinyin, **kwargs)
        return wrapper
    return decorator

def auto_discover() -> None:
    from pypinyin.style import initials, tone, finals, bopomofo, cyrillic, wadegiles, others
