import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar, Union, cast
from pydantic.v1.parse import load_file, load_str_bytes
from pydantic.v1.types import StrBytes
from pydantic.v1.typing import display_as_type
if __import__("typing").TYPE_CHECKING:
    from pydantic.v1.typing import DictStrAny

__all__ = ('parse_file_as', 'parse_obj_as', 'parse_raw_as', 'schema_of', 'schema_json_of')

T = TypeVar('T')
NameFactory = Union[str, Callable[[Type[Any]], str]]


def _generate_parsing_type_name(type_: Type[Any]) -> str:
    return f'ParsingModel[{display_as_type(type_)}]'


@lru_cache(maxsize=2048)
def _get_parsing_type(
    type_: Type[Any],
    *,
    type_name: Optional[Union[str, Callable[[Type[Any]], str]]] = None
) -> Type[Any]:
    from pydantic.v1.main import create_model
    if type_name is None:
        type_name = _generate_parsing_type_name
    if not isinstance(type_name, str):
        type_name = type_name(type_)
    return create_model(type_name, __root__=(type_, ...))


def parse_obj_as(
    type_: Type[T],
    obj: Any,
    *,
    type_name: Optional[Union[str, Callable[[Type[Any]], str]]] = None
) -> T:
    model_type = _get_parsing_type(type_, type_name=type_name)
    return cast(T, model_type(__root__=obj).__root__)


def parse_file_as(
    type_: Type[T],
    path: Union[str, Path],
    *,
    content_type: Optional[str] = None,
    encoding: str = 'utf8',
    proto: Optional[Any] = None,
    allow_pickle: bool = False,
    json_loads: Callable[[str], Any] = json.loads,
    type_name: Optional[Union[str, Callable[[Type[Any]], str]]] = None
) -> T:
    obj = load_file(path, proto=proto, content_type=content_type, encoding=encoding, allow_pickle=allow_pickle, json_loads=json_loads)
    return parse_obj_as(type_, obj, type_name=type_name)


def parse_raw_as(
    type_: Type[T],
    b: Union[str, bytes, bytearray, StrBytes],
    *,
    content_type: Optional[str] = None,
    encoding: str = 'utf8',
    proto: Optional[Any] = None,
    allow_pickle: bool = False,
    json_loads: Callable[[str], Any] = json.loads,
    type_name: Optional[Union[str, Callable[[Type[Any]], str]]] = None
) -> T:
    obj = load_str_bytes(b, proto=proto, content_type=content_type, encoding=encoding, allow_pickle=allow_pickle, json_loads=json_loads)
    return parse_obj_as(type_, obj, type_name=type_name)


def schema_of(
    type_: Type[Any],
    *,
    title: Optional[Union[str, Callable[[Type[Any]], str]]] = None,
    **schema_kwargs: Any
) -> 'DictStrAny':
    return _get_parsing_type(type_, type_name=title).schema(**schema_kwargs)


def schema_json_of(
    type_: Type[Any],
    *,
    title: Optional[Union[str, Callable[[Type[Any]], str]]] = None,
    **schema_json_kwargs: Any
) -> str:
    return _get_parsing_type(type_, type_name=title).schema_json(**schema_json_kwargs)