from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from fastapi.openapi.models import Example
from pydantic.fields import FieldInfo

from ._compat import Undefined


class ParamTypes(Enum):
    query = 'query'
    header = 'header'
    path = 'path'
    cookie = 'cookie'


class Param(FieldInfo):
    example: Any
    include_in_schema: bool
    openapi_examples: Optional[Union[List[Example], Dict[str, Example]]]

    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        alias: Optional[str] = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[Sequence[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Union[List[Example], Dict[str, Example]]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Dict[str, Any]] = ...,
        **extra: Any,
    ) -> None: ...
    def __repr__(self) -> str: ...


class Path(Param):
    in_: ParamTypes

    def __init__(
        self,
        default=...,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        alias: Optional[str] = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[Sequence[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Union[List[Example], Dict[str, Example]]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Dict[str, Any]] = ...,
        **extra: Any,
    ) -> None: ...


class Query(Param):
    in_: ParamTypes

    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        alias: Optional[str] = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[Sequence[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Union[List[Example], Dict[str, Example]]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Dict[str, Any]] = ...,
        **extra: Any,
    ) -> None: ...


class Header(Param):
    in_: ParamTypes
    convert_underscores: bool

    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        alias: Optional[str] = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        convert_underscores: bool = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[Sequence[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Union[List[Example], Dict[str, Example]]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Dict[str, Any]] = ...,
        **extra: Any,
    ) -> None: ...


class Cookie(Param):
    in_: ParamTypes

    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        alias: Optional[str] = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[Sequence[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Union[List[Example], Dict[str, Example]]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Dict[str, Any]] = ...,
        **extra: Any,
    ) -> None: ...


class Body(FieldInfo):
    embed: Optional[bool]
    media_type: str
    example: Any
    include_in_schema: bool
    openapi_examples: Optional[Union[List[Example], Dict[str, Example]]]

    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        embed: Optional[bool] = ...,
        media_type: str = ...,
        alias: Optional[str] = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[Sequence[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Union[List[Example], Dict[str, Example]]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Dict[str, Any]] = ...,
        **extra: Any,
    ) -> None: ...
    def __repr__(self) -> str: ...


class Form(Body):
    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        media_type: str = ...,
        alias: Optional[str] = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[Sequence[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Union[List[Example], Dict[str, Example]]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Dict[str, Any]] = ...,
        **extra: Any,
    ) -> None: ...


class File(Form):
    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        media_type: str = ...,
        alias: Optional[str] = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[Sequence[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Union[List[Example], Dict[str, Example]]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Dict[str, Any]] = ...,
        **extra: Any,
    ) -> None: ...


class Depends:
    dependency: Optional[Callable[..., Any]]
    use_cache: bool

    def __init__(self, dependency: Optional[Callable[..., Any]] = ..., *, use_cache: bool = ...) -> None: ...
    def __repr__(self) -> str: ...


class Security(Depends):
    scopes: List[str]

    def __init__(
        self,
        dependency: Optional[Callable[..., Any]] = ...,
        *,
        scopes: Optional[Sequence[str]] = ...,
        use_cache: bool = ...,
    ) -> None: ...