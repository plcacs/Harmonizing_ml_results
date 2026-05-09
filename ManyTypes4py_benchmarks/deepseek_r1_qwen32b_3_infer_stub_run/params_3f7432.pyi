import warnings
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
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
    def __init__(self, default: Any = Undefined, *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Union[Any, _Unset] = _Unset, alias: Union[str, _Unset] = _Unset, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Union[str, _Unset] = _Unset, serialization_alias: Union[str, _Unset] = _Unset, title: Union[str, _Unset] = _Unset, description: Union[str, _Unset] = _Unset, gt: Union[float, _Unset] = _Unset, ge: Union[float, _Unset] = _Unset, lt: Union[float, _Unset] = _Unset, le: Union[float, _Unset] = _Unset, min_length: Union[int, _Unset] = _Unset, max_length: Union[int, _Unset] = _Unset, pattern: Union[str, _Unset] = _Unset, regex: Union[str, _Unset] = _Unset, discriminator: Union[str, _Unset] = _Unset, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Union[List[Example], _Unset] = _Unset, example: Union[Any, _Unset] = _Unset, openapi_examples: Union[List[Example], _Unset] = _Unset, deprecated: Union[bool, _Unset] = _Unset, include_in_schema: bool = True, json_schema_extra: Union[Dict[str, Any], _Unset] = _Unset, **extra: Any) -> None: ...
    example: Any
    include_in_schema: bool
    openapi_examples: Optional[List[Example]]

class Path(Param):
    in_: ParamTypes
    def __init__(self, default: Any = ..., *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Union[Any, _Unset] = _Unset, alias: Union[str, _Unset] = _Unset, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Union[str, _Unset] = _Unset, serialization_alias: Union[str, _Unset] = _Unset, title: Union[str, _Unset] = _Unset, description: Union[str, _Unset] = _Unset, gt: Union[float, _Unset] = _Unset, ge: Union[float, _Unset] = _Unset, lt: Union[float, _Unset] = _Unset, le: Union[float, _Unset] = _Unset, min_length: Union[int, _Unset] = _Unset, max_length: Union[int, _Unset] = _Unset, pattern: Union[str, _Unset] = _Unset, regex: Union[str, _Unset] = _Unset, discriminator: Union[str, _Unset] = _Unset, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Union[List[Example], _Unset] = _Unset, example: Union[Any, _Unset] = _Unset, openapi_examples: Union[List[Example], _Unset] = _Unset, deprecated: Union[bool, _Unset] = _Unset, include_in_schema: bool = True, json_schema_extra: Union[Dict[str, Any], _Unset] = _Unset, **extra: Any) -> None: ...

class Query(Param):
    in_: ParamTypes
    def __init__(self, default: Any = Undefined, *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Union[Any, _Unset] = _Unset, alias: Union[str, _Unset] = _Unset, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Union[str, _Unset] = _Unset, serialization_alias: Union[str, _Unset] = _Unset, title: Union[str, _Unset] = _Unset, description: Union[str, _Unset] = _Unset, gt: Union[float, _Unset] = _Unset, ge: Union[float, _Unset] = _Unset, lt: Union[float, _Unset] = _Unset, le: Union[float, _Unset] = _Unset, min_length: Union[int, _Unset] = _Unset, max_length: Union[int, _Unset] = _Unset, pattern: Union[str, _Unset] = _Unset, regex: Union[str, _Unset] = _Unset, discriminator: Union[str, _Unset] = _Unset, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Union[List[Example], _Unset] = _Unset, example: Union[Any, _Unset] = _Unset, openapi_examples: Union[List[Example], _Unset] = _Unset, deprecated: Union[bool, _Unset] = _Unset, include_in_schema: bool = True, json_schema_extra: Union[Dict[str, Any], _Unset] = _Unset, **extra: Any) -> None: ...

class Header(Param):
    in_: ParamTypes
    convert_underscores: bool
    def __init__(self, default: Any = Undefined, *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Union[Any, _Unset] = _Unset, alias: Union[str, _Unset] = _Unset, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Union[str, _Unset] = _Unset, serialization_alias: Union[str, _Unset] = _Unset, convert_underscores: bool = True, title: Union[str, _Unset] = _Unset, description: Union[str, _Unset] = _Unset, gt: Union[float, _Unset] = _Unset, ge: Union[float, _Unset] = _Unset, lt: Union[float, _Unset] = _Unset, le: Union[float, _Unset] = _Unset, min_length: Union[int, _Unset] = _Unset, max_length: Union[int, _Unset] = _Unset, pattern: Union[str, _Unset] = _Unset, regex: Union[str, _Unset] = _Unset, discriminator: Union[str, _Unset] = _Unset, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Union[List[Example], _Unset] = _Unset, example: Union[Any, _Unset] = _Unset, openapi_examples: Union[List[Example], _Unset] = _Unset, deprecated: Union[bool, _Unset] = _Unset, include_in_schema: bool = True, json_schema_extra: Union[Dict[str, Any], _Unset] = _Unset, **extra: Any) -> None: ...

class Cookie(Param):
    in_: ParamTypes
    def __init__(self, default: Any = Undefined, *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Union[Any, _Unset] = _Unset, alias: Union[str, _Unset] = _Unset, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Union[str, _Unset] = _Unset, serialization_alias: Union[str, _Unset] = _Unset, title: Union[str, _Unset] = _Unset, description: Union[str, _Unset] = _Unset, gt: Union[float, _Unset] = _Unset, ge: Union[float, _Unset] = _Unset, lt: Union[float, _Unset] = _Unset, le: Union[float, _Unset] = _Unset, min_length: Union[int, _Unset] = _Unset, max_length: Union[int, _Unset] = _Unset, pattern: Union[str, _Unset] = _Unset, regex: Union[str, _Unset] = _Unset, discriminator: Union[str, _Unset] = _Unset, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Union[List[Example], _Unset] = _Unset, example: Union[Any, _Unset] = _Unset, openapi_examples: Union[List[Example], _Unset] = _Unset, deprecated: Union[bool, _Unset] = _Unset, include_in_schema: bool = True, json_schema_extra: Union[Dict[str, Any], _Unset] = _Unset, **extra: Any) -> None: ...

class Body(FieldInfo):
    embed: Optional[bool]
    media_type: str
    example: Any
    include_in_schema: bool
    openapi_examples: Optional[List[Example]]
    def __init__(self, default: Any = Undefined, *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Union[Any, _Unset] = _Unset, embed: Optional[bool] = _Unset, media_type: str = 'application/json', alias: Union[str, _Unset] = _Unset, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Union[str, _Unset] = _Unset, serialization_alias: Union[str, _Unset] = _Unset, title: Union[str, _Unset] = _Unset, description: Union[str, _Unset] = _Unset, gt: Union[float, _Unset] = _Unset, ge: Union[float, _Unset] = _Unset, lt: Union[float, _Unset] = _Unset, le: Union[float, _Unset] = _Unset, min_length: Union[int, _Unset] = _Unset, max_length: Union[int, _Unset] = _Unset, pattern: Union[str, _Unset] = _Unset, regex: Union[str, _Unset] = _Unset, discriminator: Union[str, _Unset] = _Unset, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Union[List[Example], _Unset] = _Unset, example: Union[Any, _Unset] = _Unset, openapi_examples: Union[List[Example], _Unset] = _Unset, deprecated: Union[bool, _Unset] = _Unset, include_in_schema: bool = True, json_schema_extra: Union[Dict[str, Any], _Unset] = _Unset, **extra: Any) -> None: ...

class Form(Body):
    def __init__(self, default: Any = Undefined, *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Union[Any, _Unset] = _Unset, media_type: str = 'application/x-www-form-urlencoded', alias: Union[str, _Unset] = _Unset, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Union[str, _Unset] = _Unset, serialization_alias: Union[str, _Unset] = _Unset, title: Union[str, _Unset] = _Unset, description: Union[str, _Unset] = _Unset, gt: Union[float, _Unset] = _Unset, ge: Union[float, _Unset] = _Unset, lt: Union[float, _Unset] = _Unset, le: Union[float, _Unset] = _Unset, min_length: Union[int, _Unset] = _Unset, max_length: Union[int, _Unset] = _Unset, pattern: Union[str, _Unset] = _Unset, regex: Union[str, _Unset] = _Unset, discriminator: Union[str, _Unset] = _Unset, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Union[List[Example], _Unset] = _Unset, example: Union[Any, _Unset] = _Unset, openapi_examples: Union[List[Example], _Unset] = _Unset, deprecated: Union[bool, _Unset] = _Unset, include_in_schema: bool = True, json_schema_extra: Union[Dict[str, Any], _Unset] = _Unset, **extra: Any) -> None: ...

class File(Form):
    def __init__(self, default: Any = Undefined, *, default_factory: Union[Callable, _Unset] = _Unset, annotation: Union[Any, _Unset] = _Unset, media_type: str = 'multipart/form-data', alias: Union[str, _Unset] = _Unset, alias_priority: Union[int, _Unset] = _Unset, validation_alias: Union[str, _Unset] = _Unset, serialization_alias: Union[str, _Unset] = _Unset, title: Union[str, _Unset] = _Unset, description: Union[str, _Unset] = _Unset, gt: Union[float, _Unset] = _Unset, ge: Union[float, _Unset] = _Unset, lt: Union[float, _Unset] = _Unset, le: Union[float, _Unset] = _Unset, min_length: Union[int, _Unset] = _Unset, max_length: Union[int, _Unset] = _Unset, pattern: Union[str, _Unset] = _Unset, regex: Union[str, _Unset] = _Unset, discriminator: Union[str, _Unset] = _Unset, strict: Union[bool, _Unset] = _Unset, multiple_of: Union[float, _Unset] = _Unset, allow_inf_nan: Union[bool, _Unset] = _Unset, max_digits: Union[int, _Unset] = _Unset, decimal_places: Union[int, _Unset] = _Unset, examples: Union[List[Example], _Unset] = _Unset, example: Union[Any, _Unset] = _Unset, openapi_examples: Union[List[Example], _Unset] = _Unset, deprecated: Union[bool, _Unset] = _Unset, include_in_schema: bool = True, json_schema_extra: Union[Dict[str, Any], _Unset] = _Unset, **extra: Any) -> None: ...

class Depends:
    dependency: Optional[Callable]
    use_cache: bool
    def __init__(self, dependency: Optional[Callable] = None, *, use_cache: bool = True) -> None: ...
    def __repr__(self) -> str: ...

class Security(Depends):
    scopes: Optional[List[str]]
    def __init__(self, dependency: Optional[Callable] = None, *, scopes: Optional[List[str]] = None, use_cache: bool = True) -> None: ...