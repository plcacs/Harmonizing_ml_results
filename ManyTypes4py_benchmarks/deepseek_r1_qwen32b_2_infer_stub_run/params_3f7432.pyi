import warnings
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TypeVar, overload
from fastapi.openapi.models import Example
from pydantic.fields import FieldInfo
from typing_extensions import Annotated, deprecated
from ._compat import PYDANTIC_V2, PYDANTIC_VERSION_MINOR_TUPLE, Undefined
_Unset = Undefined

class ParamTypes(Enum):
    query: str
    header: str
    path: str
    cookie: str

class Param(FieldInfo):
    example: Any
    include_in_schema: bool
    openapi_examples: Optional[List[Example]]
    deprecated: Optional[bool]

    def __init__(self, default: Any = Undefined, *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Optional[Any] = None, alias: Optional[str] = None, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Optional[Union[str, Sequence[str]]] = None, serialization_alias: Optional[Union[str, Sequence[str]]] = None, title: Optional[str] = None, description: Optional[str] = None, gt: Union[int, float, _Unset] = _Unset, ge: Union[int, float, _Unset] = _Unset, lt: Union[int, float, _Unset] = _Unset, le: Union[int, float, _Unset] = _Unset, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[Dict[str, str]] = None, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[int, float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Optional[List[Example]] = None, example: Union[Any, _Unset] = _Unset, openapi_examples: Optional[List[Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None: ...

class Path(Param):
    in_: ParamTypes

    def __init__(self, default: Any = ..., *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Optional[Any] = None, alias: Optional[str] = None, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Optional[Union[str, Sequence[str]]] = None, serialization_alias: Optional[Union[str, Sequence[str]]] = None, title: Optional[str] = None, description: Optional[str] = None, gt: Union[int, float, _Unset] = _Unset, ge: Union[int, float, _Unset] = _Unset, lt: Union[int, float, _Unset] = _Unset, le: Union[int, float, _Unset] = _Unset, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[Dict[str, str]] = None, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[int, float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Optional[List[Example]] = None, example: Union[Any, _Unset] = _Unset, openapi_examples: Optional[List[Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None: ...

class Query(Param):
    in_: ParamTypes

    def __init__(self, default: Any = Undefined, *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Optional[Any] = None, alias: Optional[str] = None, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Optional[Union[str, Sequence[str]]] = None, serialization_alias: Optional[Union[str, Sequence[str]]] = None, title: Optional[str] = None, description: Optional[str] = None, gt: Union[int, float, _Unset] = _Unset, ge: Union[int, float, _Unset] = _Unset, lt: Union[int, float, _Unset] = _Unset, le: Union[int, float, _Unset] = _Unset, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[Dict[str, str]] = None, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[int, float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Optional[List[Example]] = None, example: Union[Any, _Unset] = _Unset, openapi_examples: Optional[List[Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None: ...

class Header(Param):
    in_: ParamTypes
    convert_underscores: bool

    def __init__(self, default: Any = Undefined, *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Optional[Any] = None, alias: Optional[str] = None, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Optional[Union[str, Sequence[str]]] = None, serialization_alias: Optional[Union[str, Sequence[str]]] = None, convert_underscores: bool = True, title: Optional[str] = None, description: Optional[str] = None, gt: Union[int, float, _Unset] = _Unset, ge: Union[int, float, _Unset] = _Unset, lt: Union[int, float, _Unset] = _Unset, le: Union[int, float, _Unset] = _Unset, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[Dict[str, str]] = None, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[int, float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Optional[List[Example]] = None, example: Union[Any, _Unset] = _Unset, openapi_examples: Optional[List[Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None: ...

class Cookie(Param):
    in_: ParamTypes

    def __init__(self, default: Any = Undefined, *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Optional[Any] = None, alias: Optional[str] = None, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Optional[Union[str, Sequence[str]]] = None, serialization_alias: Optional[Union[str, Sequence[str]]] = None, title: Optional[str] = None, description: Optional[str] = None, gt: Union[int, float, _Unset] = _Unset, ge: Union[int, float, _Unset] = _Unset, lt: Union[int, float, _Unset] = _Unset, le: Union[int, float, _Unset] = _Unset, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[Dict[str, str]] = None, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[int, float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Optional[List[Example]] = None, example: Union[Any, _Unset] = _Unset, openapi_examples: Optional[List[Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None: ...

class Body(FieldInfo):
    embed: Optional[bool]
    media_type: str
    example: Any
    include_in_schema: bool
    openapi_examples: Optional[List[Example]]

    def __init__(self, default: Any = Undefined, *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Optional[Any] = None, embed: Optional[bool] = None, media_type: str = 'application/json', alias: Optional[str] = None, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Optional[Union[str, Sequence[str]]] = None, serialization_alias: Optional[Union[str, Sequence[str]]] = None, title: Optional[str] = None, description: Optional[str] = None, gt: Union[int, float, _Unset] = _Unset, ge: Union[int, float, _Unset] = _Unset, lt: Union[int, float, _Unset] = _Unset, le: Union[int, float, _Unset] = _Unset, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[Dict[str, str]] = None, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[int, float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Optional[List[Example]] = None, example: Union[Any, _Unset] = _Unset, openapi_examples: Optional[List[Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None: ...

class Form(Body):
    def __init__(self, default: Any = Undefined, *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Optional[Any] = None, media_type: str = 'application/x-www-form-urlencoded', alias: Optional[str] = None, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Optional[Union[str, Sequence[str]]] = None, serialization_alias: Optional[Union[str, Sequence[str]]] = None, title: Optional[str] = None, description: Optional[str] = None, gt: Union[int, float, _Unset] = _Unset, ge: Union[int, float, _Unset] = _Unset, lt: Union[int, float, _Unset] = _Unset, le: Union[int, float, _Unset] = _Unset, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[Dict[str, str]] = None, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[int, float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Optional[List[Example]] = None, example: Union[Any, _Unset] = _Unset, openapi_examples: Optional[List[Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None: ...

class File(Form):
    def __init__(self, default: Any = Undefined, *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Optional[Any] = None, media_type: str = 'multipart/form-data', alias: Optional[str] = None, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Optional[Union[str, Sequence[str]]] = None, serialization_alias: Optional[Union[str, Sequence[str]]] = None, title: Optional[str] = None, description: Optional[str] = None, gt: Union[int, float, _Unset] = _Unset, ge: Union[int, float, _Unset] = _Unset, lt: Union[int, float, _Unset] = _Unset, le: Union[int, float, _Unset] = _Unset, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[Dict[str, str]] = None, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[int, float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Optional[List[Example]] = None, example: Union[Any, _Unset] = _Unset, openapi_examples: Optional[List[Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None: ...

class Depends:
    dependency: Optional[Callable]
    use_cache: bool

    def __init__(self, dependency: Optional[Callable] = None, *, use_cache: bool = True) -> None: ...

class Security(Depends):
    scopes: List[str]

    def __init__(self, dependency: Optional[Callable] = None, *, scopes: Optional[List[str]] = None, use_cache: bool = True) -> None: ...