from fastapi.openapi.models import Example
from pydantic.fields import FieldInfo
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from typing_extensions import Annotated, deprecated

_Unset: Any = Undefined

class ParamTypes(Enum):
    query: str = 'query'
    header: str = 'header'
    path: str = 'path'
    cookie: str = 'cookie'

class Param(FieldInfo):
    def __init__(self, default: Any = Undefined, *, default_factory: Any = _Unset, annotation: Any = None, alias: str = None, alias_priority: Any = _Unset, validation_alias: Any = None, serialization_alias: Any = None, title: str = None, description: str = None, gt: Any = None, ge: Any = None, lt: Any = None, le: Any = None, min_length: int = None, max_length: int = None, pattern: str = None, regex: str = None, discriminator: Any = None, strict: Any = _Unset, multiple_of: Any = _Unset, allow_inf_nan: Any = _Unset, max_digits: Any = _Unset, decimal_places: Any = _Unset, examples: Any = None, example: Any = _Unset, openapi_examples: Any = None, deprecated: Any = None, include_in_schema: bool = True, json_schema_extra: Any = None, **extra: Any) -> None:
        ...

class Path(Param):
    in_: str = ParamTypes.path
    def __init__(self, default: Any = ..., *, default_factory: Any = _Unset, annotation: Any = None, alias: str = None, alias_priority: Any = _Unset, validation_alias: Any = None, serialization_alias: Any = None, title: str = None, description: str = None, gt: Any = None, ge: Any = None, lt: Any = None, le: Any = None, min_length: int = None, max_length: int = None, pattern: str = None, regex: str = None, discriminator: Any = None, strict: Any = _Unset, multiple_of: Any = _Unset, allow_inf_nan: Any = _Unset, max_digits: Any = _Unset, decimal_places: Any = _Unset, examples: Any = None, example: Any = _Unset, openapi_examples: Any = None, deprecated: Any = None, include_in_schema: bool = True, json_schema_extra: Any = None, **extra: Any) -> None:
        ...

class Query(Param):
    in_: str = ParamTypes.query
    def __init__(self, default: Any = Undefined, *, default_factory: Any = _Unset, annotation: Any = None, alias: str = None, alias_priority: Any = _Unset, validation_alias: Any = None, serialization_alias: Any = None, title: str = None, description: str = None, gt: Any = None, ge: Any = None, lt: Any = None, le: Any = None, min_length: int = None, max_length: int = None, pattern: str = None, regex: str = None, discriminator: Any = None, strict: Any = _Unset, multiple_of: Any = _Unset, allow_inf_nan: Any = _Unset, max_digits: Any = _Unset, decimal_places: Any = _Unset, examples: Any = None, example: Any = _Unset, openapi_examples: Any = None, deprecated: Any = None, include_in_schema: bool = True, json_schema_extra: Any = None, **extra: Any) -> None:
        ...

class Header(Param):
    in_: str = ParamTypes.header
    def __init__(self, default: Any = Undefined, *, default_factory: Any = _Unset, annotation: Any = None, alias: str = None, alias_priority: Any = _Unset, validation_alias: Any = None, serialization_alias: Any = None, convert_underscores: bool = True, title: str = None, description: str = None, gt: Any = None, ge: Any = None, lt: Any = None, le: Any = None, min_length: int = None, max_length: int = None, pattern: str = None, regex: str = None, discriminator: Any = None, strict: Any = _Unset, multiple_of: Any = _Unset, allow_inf_nan: Any = _Unset, max_digits: Any = _Unset, decimal_places: Any = _Unset, examples: Any = None, example: Any = _Unset, openapi_examples: Any = None, deprecated: Any = None, include_in_schema: bool = True, json_schema_extra: Any = None, **extra: Any) -> None:
        ...

class Cookie(Param):
    in_: str = ParamTypes.cookie
    def __init__(self, default: Any = Undefined, *, default_factory: Any = _Unset, annotation: Any = None, alias: str = None, alias_priority: Any = _Unset, validation_alias: Any = None, serialization_alias: Any = None, title: str = None, description: str = None, gt: Any = None, ge: Any = None, lt: Any = None, le: Any = None, min_length: int = None, max_length: int = None, pattern: str = None, regex: str = None, discriminator: Any = None, strict: Any = _Unset, multiple_of: Any = _Unset, allow_inf_nan: Any = _Unset, max_digits: Any = _Unset, decimal_places: Any = _Unset, examples: Any = None, example: Any = _Unset, openapi_examples: Any = None, deprecated: Any = None, include_in_schema: bool = True, json_schema_extra: Any = None, **extra: Any) -> None:
        ...

class Body(FieldInfo):
    def __init__(self, default: Any = Undefined, *, default_factory: Any = _Unset, annotation: Any = None, embed: Any = None, media_type: str = 'application/json', alias: str = None, alias_priority: Any = _Unset, validation_alias: Any = None, serialization_alias: Any = None, title: str = None, description: str = None, gt: Any = None, ge: Any = None, lt: Any = None, le: Any = None, min_length: int = None, max_length: int = None, pattern: str = None, regex: str = None, discriminator: Any = None, strict: Any = _Unset, multiple_of: Any = _Unset, allow_inf_nan: Any = _Unset, max_digits: Any = _Unset, decimal_places: Any = _Unset, examples: Any = None, example: Any = _Unset, openapi_examples: Any = None, deprecated: Any = None, include_in_schema: bool = True, json_schema_extra: Any = None, **extra: Any) -> None:
        ...

class Form(Body):
    def __init__(self, default: Any = Undefined, *, default_factory: Any = _Unset, annotation: Any = None, media_type: str = 'application/x-www-form-urlencoded', alias: str = None, alias_priority: Any = _Unset, validation_alias: Any = None, serialization_alias: Any = None, title: str = None, description: str = None, gt: Any = None, ge: Any = None, lt: Any = None, le: Any = None, min_length: int = None, max_length: int = None, pattern: str = None, regex: str = None, discriminator: Any = None, strict: Any = _Unset, multiple_of: Any = _Unset, allow_inf_nan: Any = _Unset, max_digits: Any = _Unset, decimal_places: Any = _Unset, examples: Any = None, example: Any = _Unset, openapi_examples: Any = None, deprecated: Any = None, include_in_schema: bool = True, json_schema_extra: Any = None, **extra: Any) -> None:
        ...

class File(Form):
    def __init__(self, default: Any = Undefined, *, default_factory: Any = _Unset, annotation: Any = None, media_type: str = 'multipart/form-data', alias: str = None, alias_priority: Any = _Unset, validation_alias: Any = None, serialization_alias: Any = None, title: str = None, description: str = None, gt: Any = None, ge: Any = None, lt: Any = None, le: Any = None, min_length: int = None, max_length: int = None, pattern: str = None, regex: str = None, discriminator: Any = None, strict: Any = _Unset, multiple_of: Any = _Unset, allow_inf_nan: Any = _Unset, max_digits: Any = _Unset, decimal_places: Any = _Unset, examples: Any = None, example: Any = _Unset, openapi_examples: Any = None, deprecated: Any = None, include_in_schema: bool = True, json_schema_extra: Any = None, **extra: Any) -> None:
        ...

class Depends:
    def __init__(self, dependency: Any = None, *, use_cache: bool = True) -> None:
        ...

class Security(Depends):
    def __init__(self, dependency: Any = None, *, scopes: List[str] = None, use_cache: bool = True) -> None:
        ...
