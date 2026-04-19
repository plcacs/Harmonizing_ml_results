from enum import Enum
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence

from fastapi.openapi.models import Example
from pydantic.fields import FieldInfo


class ParamTypes(Enum):
    query: "ParamTypes"
    header: "ParamTypes"
    path: "ParamTypes"
    cookie: "ParamTypes"


class Param(FieldInfo):
    example: Any
    include_in_schema: bool
    openapi_examples: Optional[Dict[str, Example]]

    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Callable[[], Any] | object = ...,
        annotation: Any | None = None,
        alias: str | None = None,
        alias_priority: int | object = ...,
        validation_alias: Any | None = None,
        serialization_alias: str | None = None,
        title: str | None = None,
        description: str | None = None,
        gt: int | float | None = None,
        ge: int | float | None = None,
        lt: int | float | None = None,
        le: int | float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        regex: str | None = None,
        discriminator: str | None = None,
        strict: bool | object = ...,
        multiple_of: int | float | object = ...,
        allow_inf_nan: bool | object = ...,
        max_digits: int | object = ...,
        decimal_places: int | object = ...,
        examples: List[Any] | Dict[str, Any] | None = None,
        example: Any | object = ...,
        openapi_examples: Dict[str, Example] | None = None,
        deprecated: bool | str | None = None,
        include_in_schema: bool = True,
        json_schema_extra: Dict[str, Any] | None = None,
        **extra: Any,
    ) -> None: ...
    def __repr__(self) -> str: ...


class Path(Param):
    in_: ClassVar[ParamTypes]

    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Callable[[], Any] | object = ...,
        annotation: Any | None = None,
        alias: str | None = None,
        alias_priority: int | object = ...,
        validation_alias: Any | None = None,
        serialization_alias: str | None = None,
        title: str | None = None,
        description: str | None = None,
        gt: int | float | None = None,
        ge: int | float | None = None,
        lt: int | float | None = None,
        le: int | float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        regex: str | None = None,
        discriminator: str | None = None,
        strict: bool | object = ...,
        multiple_of: int | float | object = ...,
        allow_inf_nan: bool | object = ...,
        max_digits: int | object = ...,
        decimal_places: int | object = ...,
        examples: List[Any] | Dict[str, Any] | None = None,
        example: Any | object = ...,
        openapi_examples: Dict[str, Example] | None = None,
        deprecated: bool | str | None = None,
        include_in_schema: bool = True,
        json_schema_extra: Dict[str, Any] | None = None,
        **extra: Any,
    ) -> None: ...


class Query(Param):
    in_: ClassVar[ParamTypes]

    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Callable[[], Any] | object = ...,
        annotation: Any | None = None,
        alias: str | None = None,
        alias_priority: int | object = ...,
        validation_alias: Any | None = None,
        serialization_alias: str | None = None,
        title: str | None = None,
        description: str | None = None,
        gt: int | float | None = None,
        ge: int | float | None = None,
        lt: int | float | None = None,
        le: int | float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        regex: str | None = None,
        discriminator: str | None = None,
        strict: bool | object = ...,
        multiple_of: int | float | object = ...,
        allow_inf_nan: bool | object = ...,
        max_digits: int | object = ...,
        decimal_places: int | object = ...,
        examples: List[Any] | Dict[str, Any] | None = None,
        example: Any | object = ...,
        openapi_examples: Dict[str, Example] | None = None,
        deprecated: bool | str | None = None,
        include_in_schema: bool = True,
        json_schema_extra: Dict[str, Any] | None = None,
        **extra: Any,
    ) -> None: ...


class Header(Param):
    in_: ClassVar[ParamTypes]
    convert_underscores: bool

    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Callable[[], Any] | object = ...,
        annotation: Any | None = None,
        alias: str | None = None,
        alias_priority: int | object = ...,
        validation_alias: Any | None = None,
        serialization_alias: str | None = None,
        convert_underscores: bool = True,
        title: str | None = None,
        description: str | None = None,
        gt: int | float | None = None,
        ge: int | float | None = None,
        lt: int | float | None = None,
        le: int | float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        regex: str | None = None,
        discriminator: str | None = None,
        strict: bool | object = ...,
        multiple_of: int | float | object = ...,
        allow_inf_nan: bool | object = ...,
        max_digits: int | object = ...,
        decimal_places: int | object = ...,
        examples: List[Any] | Dict[str, Any] | None = None,
        example: Any | object = ...,
        openapi_examples: Dict[str, Example] | None = None,
        deprecated: bool | str | None = None,
        include_in_schema: bool = True,
        json_schema_extra: Dict[str, Any] | None = None,
        **extra: Any,
    ) -> None: ...


class Cookie(Param):
    in_: ClassVar[ParamTypes]

    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Callable[[], Any] | object = ...,
        annotation: Any | None = None,
        alias: str | None = None,
        alias_priority: int | object = ...,
        validation_alias: Any | None = None,
        serialization_alias: str | None = None,
        title: str | None = None,
        description: str | None = None,
        gt: int | float | None = None,
        ge: int | float | None = None,
        lt: int | float | None = None,
        le: int | float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        regex: str | None = None,
        discriminator: str | None = None,
        strict: bool | object = ...,
        multiple_of: int | float | object = ...,
        allow_inf_nan: bool | object = ...,
        max_digits: int | object = ...,
        decimal_places: int | object = ...,
        examples: List[Any] | Dict[str, Any] | None = None,
        example: Any | object = ...,
        openapi_examples: Dict[str, Example] | None = None,
        deprecated: bool | str | None = None,
        include_in_schema: bool = True,
        json_schema_extra: Dict[str, Any] | None = None,
        **extra: Any,
    ) -> None: ...


class Body(FieldInfo):
    embed: Optional[bool]
    media_type: str
    example: Any
    include_in_schema: bool
    openapi_examples: Optional[Dict[str, Example]]

    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Callable[[], Any] | object = ...,
        annotation: Any | None = None,
        embed: bool | None = None,
        media_type: str = "application/json",
        alias: str | None = None,
        alias_priority: int | object = ...,
        validation_alias: Any | None = None,
        serialization_alias: str | None = None,
        title: str | None = None,
        description: str | None = None,
        gt: int | float | None = None,
        ge: int | float | None = None,
        lt: int | float | None = None,
        le: int | float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        regex: str | None = None,
        discriminator: str | None = None,
        strict: bool | object = ...,
        multiple_of: int | float | object = ...,
        allow_inf_nan: bool | object = ...,
        max_digits: int | object = ...,
        decimal_places: int | object = ...,
        examples: List[Any] | Dict[str, Any] | None = None,
        example: Any | object = ...,
        openapi_examples: Dict[str, Example] | None = None,
        deprecated: bool | str | None = None,
        include_in_schema: bool = True,
        json_schema_extra: Dict[str, Any] | None = None,
        **extra: Any,
    ) -> None: ...
    def __repr__(self) -> str: ...


class Form(Body):
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Callable[[], Any] | object = ...,
        annotation: Any | None = None,
        media_type: str = "application/x-www-form-urlencoded",
        alias: str | None = None,
        alias_priority: int | object = ...,
        validation_alias: Any | None = None,
        serialization_alias: str | None = None,
        title: str | None = None,
        description: str | None = None,
        gt: int | float | None = None,
        ge: int | float | None = None,
        lt: int | float | None = None,
        le: int | float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        regex: str | None = None,
        discriminator: str | None = None,
        strict: bool | object = ...,
        multiple_of: int | float | object = ...,
        allow_inf_nan: bool | object = ...,
        max_digits: int | object = ...,
        decimal_places: int | object = ...,
        examples: List[Any] | Dict[str, Any] | None = None,
        example: Any | object = ...,
        openapi_examples: Dict[str, Example] | None = None,
        deprecated: bool | str | None = None,
        include_in_schema: bool = True,
        json_schema_extra: Dict[str, Any] | None = None,
        **extra: Any,
    ) -> None: ...


class File(Form):
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Callable[[], Any] | object = ...,
        annotation: Any | None = None,
        media_type: str = "multipart/form-data",
        alias: str | None = None,
        alias_priority: int | object = ...,
        validation_alias: Any | None = None,
        serialization_alias: str | None = None,
        title: str | None = None,
        description: str | None = None,
        gt: int | float | None = None,
        ge: int | float | None = None,
        lt: int | float | None = None,
        le: int | float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        regex: str | None = None,
        discriminator: str | None = None,
        strict: bool | object = ...,
        multiple_of: int | float | object = ...,
        allow_inf_nan: bool | object = ...,
        max_digits: int | object = ...,
        decimal_places: int | object = ...,
        examples: List[Any] | Dict[str, Any] | None = None,
        example: Any | object = ...,
        openapi_examples: Dict[str, Example] | None = None,
        deprecated: bool | str | None = None,
        include_in_schema: bool = True,
        json_schema_extra: Dict[str, Any] | None = None,
        **extra: Any,
    ) -> None: ...


class Depends:
    dependency: Optional[Callable[..., Any]]
    use_cache: bool

    def __init__(self, dependency: Callable[..., Any] | None = None, *, use_cache: bool = True) -> None: ...
    def __repr__(self) -> str: ...


class Security(Depends):
    scopes: List[str]

    def __init__(
        self,
        dependency: Callable[..., Any] | None = None,
        *,
        scopes: Optional[Sequence[str]] = None,
        use_cache: bool = True,
    ) -> None: ...