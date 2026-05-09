import warnings
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, overload
from fastapi.openapi.models import Example
from pydantic.fields import FieldInfo
from typing_extensions import Annotated
from ._compat import Undefined

_Unset: Any = Undefined

class ParamTypes(Enum):
    query = 'query'
    header = 'header'
    path = 'path'
    cookie = 'cookie'

class Param(FieldInfo):
    example: Any
    include_in_schema: bool
    openapi_examples: Optional[Union[List[Example], Example]]
    deprecated: Optional[bool]

    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Optional[Callable[[], Any]] = _Unset,
        annotation: Any = None,
        alias: Optional[str] = None,
        alias_priority: Optional[int] = _Unset,
        validation_alias: Optional[str] = None,
        serialization_alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[Union[int, float]] = None,
        ge: Optional[Union[int, float]] = None,
        lt: Optional[Union[int, float]] = None,
        le: Optional[Union[int, float]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Optional[str] = None,
        strict: Optional[bool] = _Unset,
        multiple_of: Optional[Union[int, float]] = _Unset,
        allow_inf_nan: Optional[bool] = _Unset,
        max_digits: Optional[int] = _Unset,
        decimal_places: Optional[int] = _Unset,
        examples: Optional[Union[List[Example], Example]] = None,
        example: Any = _Unset,
        openapi_examples: Optional[Union[List[Example], Example]] = None,
        deprecated: Optional[bool] = None,
        include_in_schema: bool = True,
        json_schema_extra: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None: ...

    def __repr__(self) -> str: ...

class Path(Param):
    in_: ParamTypes

    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Optional[Callable[[], Any]] = _Unset,
        annotation: Any = None,
        alias: Optional[str] = None,
        alias_priority: Optional[int] = _Unset,
        validation_alias: Optional[str] = None,
        serialization_alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[Union[int, float]] = None,
        ge: Optional[Union[int, float]] = None,
        lt: Optional[Union[int, float]] = None,
        le: Optional[Union[int, float]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Optional[str] = None,
        strict: Optional[bool] = _Unset,
        multiple_of: Optional[Union[int, float]] = _Unset,
        allow_inf_nan: Optional[bool] = _Unset,
        max_digits: Optional[int] = _Unset,
        decimal_places: Optional[int] = _Unset,
        examples: Optional[Union[List[Example], Example]] = None,
        example: Any = _Unset,
        openapi_examples: Optional[Union[List[Example], Example]] = None,
        deprecated: Optional[bool] = None,
        include_in_schema: bool = True,
        json_schema_extra: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None: ...

class Query(Param):
    in_: ParamTypes

    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Optional[Callable[[], Any]] = _Unset,
        annotation: Any = None,
        alias: Optional[str] = None,
        alias_priority: Optional[int] = _Unset,
        validation_alias: Optional[str] = None,
        serialization_alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[Union[int, float]] = None,
        ge: Optional[Union[int, float]] = None,
        lt: Optional[Union[int, float]] = None,
        le: Optional[Union[int, float]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Optional[str] = None,
        strict: Optional[bool] = _Unset,
        multiple_of: Optional[Union[int, float]] = _Unset,
        allow_inf_nan: Optional[bool] = _Unset,
        max_digits: Optional[int] = _Unset,
        decimal_places: Optional[int] = _Unset,
        examples: Optional[Union[List[Example], Example]] = None,
        example: Any = _Unset,
        openapi_examples: Optional[Union[List[Example], Example]] = None,
        deprecated: Optional[bool] = None,
        include_in_schema: bool = True,
        json_schema_extra: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None: ...

class Header(Param):
    in_: ParamTypes
    convert_underscores: bool

    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Optional[Callable[[], Any]] = _Unset,
        annotation: Any = None,
        alias: Optional[str] = None,
        alias_priority: Optional[int] = _Unset,
        validation_alias: Optional[str] = None,
        serialization_alias: Optional[str] = None,
        convert_underscores: bool = True,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[Union[int, float]] = None,
        ge: Optional[Union[int, float]] = None,
        lt: Optional[Union[int, float]] = None,
        le: Optional[Union[int, float]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Optional[str] = None,
        strict: Optional[bool] = _Unset,
        multiple_of: Optional[Union[int, float]] = _Unset,
        allow_inf_nan: Optional[bool] = _Unset,
        max_digits: Optional[int] = _Unset,
        decimal_places: Optional[int] = _Unset,
        examples: Optional[Union[List[Example], Example]] = None,
        example: Any = _Unset,
        openapi_examples: Optional[Union[List[Example], Example]] = None,
        deprecated: Optional[bool] = None,
        include_in_schema: bool = True,
        json_schema_extra: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None: ...

class Cookie(Param):
    in_: ParamTypes

    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Optional[Callable[[], Any]] = _Unset,
        annotation: Any = None,
        alias: Optional[str] = None,
        alias_priority: Optional[int] = _Unset,
        validation_alias: Optional[str] = None,
        serialization_alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[Union[int, float]] = None,
        ge: Optional[Union[int, float]] = None,
        lt: Optional[Union[int, float]] = None,
        le: Optional[Union[int, float]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Optional[str] = None,
        strict: Optional[bool] = _Unset,
        multiple_of: Optional[Union[int, float]] = _Unset,
        allow_inf_nan: Optional[bool] = _Unset,
        max_digits: Optional[int] = _Unset,
        decimal_places: Optional[int] = _Unset,
        examples: Optional[Union[List[Example], Example]] = None,
        example: Any = _Unset,
        openapi_examples: Optional[Union[List[Example], Example]] = None,
        deprecated: Optional[bool] = None,
        include_in_schema: bool = True,
        json_schema_extra: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None: ...

class Body(FieldInfo):
    embed: Optional[bool]
    media_type: str
    example: Any
    include_in_schema: bool
    openapi_examples: Optional[Union[List[Example], Example]]
    deprecated: Optional[bool]

    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Optional[Callable[[], Any]] = _Unset,
        annotation: Any = None,
        embed: Optional[bool] = None,
        media_type: str = 'application/json',
        alias: Optional[str] = None,
        alias_priority: Optional[int] = _Unset,
        validation_alias: Optional[str] = None,
        serialization_alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[Union[int, float]] = None,
        ge: Optional[Union[int, float]] = None,
        lt: Optional[Union[int, float]] = None,
        le: Optional[Union[int, float]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Optional[str] = None,
        strict: Optional[bool] = _Unset,
        multiple_of: Optional[Union[int, float]] = _Unset,
        allow_inf_nan: Optional[bool] = _Unset,
        max_digits: Optional[int] = _Unset,
        decimal_places: Optional[int] = _Unset,
        examples: Optional[Union[List[Example], Example]] = None,
        example: Any = _Unset,
        openapi_examples: Optional[Union[List[Example], Example]] = None,
        deprecated: Optional[bool] = None,
        include_in_schema: bool = True,
        json_schema_extra: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None: ...

    def __repr__(self) -> str: ...

class Form(Body):
    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Optional[Callable[[], Any]] = _Unset,
        annotation: Any = None,
        media_type: str = 'application/x-www-form-urlencoded',
        alias: Optional[str] = None,
        alias_priority: Optional[int] = _Unset,
        validation_alias: Optional[str] = None,
        serialization_alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[Union[int, float]] = None,
        ge: Optional[Union[int, float]] = None,
        lt: Optional[Union[int, float]] = None,
        le: Optional[Union[int, float]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Optional[str] = None,
        strict: Optional[bool] = _Unset,
        multiple_of: Optional[Union[int, float]] = _Unset,
        allow_inf_nan: Optional[bool] = _Unset,
        max_digits: Optional[int] = _Unset,
        decimal_places: Optional[int] = _Unset,
        examples: Optional[Union[List[Example], Example]] = None,
        example: Any = _Unset,
        openapi_examples: Optional[Union[List[Example], Example]] = None,
        deprecated: Optional[bool] = None,
        include_in_schema: bool = True,
        json_schema_extra: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None: ...

class File(Form):
    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Optional[Callable[[], Any]] = _Unset,
        annotation: Any = None,
        media_type: str = 'multipart/form-data',
        alias: Optional[str] = None,
        alias_priority: Optional[int] = _Unset,
        validation_alias: Optional[str] = None,
        serialization_alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[Union[int, float]] = None,
        ge: Optional[Union[int, float]] = None,
        lt: Optional[Union[int, float]] = None,
        le: Optional[Union[int, float]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Optional[str] = None,
        strict: Optional[bool] = _Unset,
        multiple_of: Optional[Union[int, float]] = _Unset,
        allow_inf_nan: Optional[bool] = _Unset,
        max_digits: Optional[int] = _Unset,
        decimal_places: Optional[int] = _Unset,
        examples: Optional[Union[List[Example], Example]] = None,
        example: Any = _Unset,
        openapi_examples: Optional[Union[List[Example], Example]] = None,
        deprecated: Optional[bool] = None,
        include_in_schema: bool = True,
        json_schema_extra: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None: ...

class Depends:
    dependency: Any
    use_cache: bool

    def __init__(self, dependency: Any = None, *, use_cache: bool = True) -> None: ...

    def __repr__(self) -> str: ...

class Security(Depends):
    scopes: List[str]

    def __init__(self, dependency: Any = None, *, scopes: Optional[List[str]] = None, use_cache: bool = True) -> None: ...