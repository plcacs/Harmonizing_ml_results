```pyi
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from fastapi.openapi.models import Example
from pydantic.fields import FieldInfo
from typing_extensions import Annotated

_Unset: Any

class ParamTypes(Enum):
    query: str
    header: str
    path: str
    cookie: str

class Param(FieldInfo):
    example: Any
    include_in_schema: bool
    openapi_examples: Optional[Dict[str, Example]]
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Optional[Any] = ...,
        alias: Optional[str] = ...,
        alias_priority: Optional[int] = ...,
        validation_alias: Optional[Union[str, Any]] = ...,
        serialization_alias: Optional[str] = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Optional[float] = ...,
        ge: Optional[float] = ...,
        lt: Optional[float] = ...,
        le: Optional[float] = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Optional[Union[str, Any]] = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[List[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Dict[str, Example]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any], type], None]]] = ...,
        **extra: Any,
    ) -> None: ...
    def __repr__(self) -> str: ...

class Path(Param):
    in_: ParamTypes
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Optional[Any] = ...,
        alias: Optional[str] = ...,
        alias_priority: Optional[int] = ...,
        validation_alias: Optional[Union[str, Any]] = ...,
        serialization_alias: Optional[str] = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Optional[float] = ...,
        ge: Optional[float] = ...,
        lt: Optional[float] = ...,
        le: Optional[float] = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Optional[Union[str, Any]] = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[List[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Dict[str, Example]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any], type], None]]] = ...,
        **extra: Any,
    ) -> None: ...

class Query(Param):
    in_: ParamTypes
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Optional[Any] = ...,
        alias: Optional[str] = ...,
        alias_priority: Optional[int] = ...,
        validation_alias: Optional[Union[str, Any]] = ...,
        serialization_alias: Optional[str] = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Optional[float] = ...,
        ge: Optional[float] = ...,
        lt: Optional[float] = ...,
        le: Optional[float] = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Optional[Union[str, Any]] = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[List[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Dict[str, Example]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any], type], None]]] = ...,
        **extra: Any,
    ) -> None: ...

class Header(Param):
    in_: ParamTypes
    convert_underscores: bool
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Optional[Any] = ...,
        alias: Optional[str] = ...,
        alias_priority: Optional[int] = ...,
        validation_alias: Optional[Union[str, Any]] = ...,
        serialization_alias: Optional[str] = ...,
        convert_underscores: bool = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Optional[float] = ...,
        ge: Optional[float] = ...,
        lt: Optional[float] = ...,
        le: Optional[float] = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Optional[Union[str, Any]] = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[List[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Dict[str, Example]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any], type], None]]] = ...,
        **extra: Any,
    ) -> None: ...

class Cookie(Param):
    in_: ParamTypes
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Optional[Any] = ...,
        alias: Optional[str] = ...,
        alias_priority: Optional[int] = ...,
        validation_alias: Optional[Union[str, Any]] = ...,
        serialization_alias: Optional[str] = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Optional[float] = ...,
        ge: Optional[float] = ...,
        lt: Optional[float] = ...,
        le: Optional[float] = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Optional[Union[str, Any]] = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[List[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Dict[str, Example]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any], type], None]]] = ...,
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
        default_factory: Any = ...,
        annotation: Optional[Any] = ...,
        embed: Optional[bool] = ...,
        media_type: str = ...,
        alias: Optional[str] = ...,
        alias_priority: Optional[int] = ...,
        validation_alias: Optional[Union[str, Any]] = ...,
        serialization_alias: Optional[str] = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Optional[float] = ...,
        ge: Optional[float] = ...,
        lt: Optional[float] = ...,
        le: Optional[float] = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Optional[Union[str, Any]] = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[List[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Dict[str, Example]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any], type], None]]] = ...,
        **extra: Any,
    ) -> None: ...
    def __repr__(self) -> str: ...

class Form(Body):
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Optional[Any] = ...,
        media_type: str = ...,
        alias: Optional[str] = ...,
        alias_priority: Optional[int] = ...,
        validation_alias: Optional[Union[str, Any]] = ...,
        serialization_alias: Optional[str] = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Optional[float] = ...,
        ge: Optional[float] = ...,
        lt: Optional[float] = ...,
        le: Optional[float] = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Optional[Union[str, Any]] = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[List[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Dict[str, Example]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any], type], None]]] = ...,
        **extra: Any,
    ) -> None: ...

class File(Form):
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Optional[Any] = ...,
        media_type: str = ...,
        alias: Optional[str] = ...,
        alias_priority: Optional[int] = ...,
        validation_alias: Optional[Union[str, Any]] = ...,
        serialization_alias: Optional[str] = ...,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        gt: Optional[float] = ...,
        ge: Optional[float] = ...,
        lt: Optional[float] = ...,
        le: Optional[float] = ...,
        min_length: Optional[int] = ...,
        max_length: Optional[int] = ...,
        pattern: Optional[str] = ...,
        regex: Optional[str] = ...,
        discriminator: Optional[Union[str, Any]] = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Optional[List[Any]] = ...,
        example: Any = ...,
        openapi_examples: Optional[Dict[str, Example]] = ...,
        deprecated: Optional[Union[bool, str]] = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any], type], None]]] = ...,
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
        scopes: Optional[List[str]] = ...,
        use_cache: bool = ...,
    ) -> None: ...
```