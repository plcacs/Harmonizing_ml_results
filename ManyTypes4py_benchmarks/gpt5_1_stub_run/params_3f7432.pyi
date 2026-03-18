from typing import Any
from enum import Enum
from pydantic.fields import FieldInfo

class ParamTypes(Enum):
    query: "ParamTypes"
    header: "ParamTypes"
    path: "ParamTypes"
    cookie: "ParamTypes"

class Param(FieldInfo):
    example: Any
    include_in_schema: bool
    openapi_examples: Any

    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        alias: Any = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        title: Any = ...,
        description: Any = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Any = ...,
        max_length: Any = ...,
        pattern: Any = ...,
        regex: Any = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = ...,
        example: Any = ...,
        openapi_examples: Any = ...,
        deprecated: Any = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Any = ...,
        **extra: Any
    ) -> None: ...
    def __repr__(self) -> str: ...

class Path(Param):
    in_: ParamTypes

    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        alias: Any = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        title: Any = ...,
        description: Any = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Any = ...,
        max_length: Any = ...,
        pattern: Any = ...,
        regex: Any = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = ...,
        example: Any = ...,
        openapi_examples: Any = ...,
        deprecated: Any = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Any = ...,
        **extra: Any
    ) -> None: ...

class Query(Param):
    in_: ParamTypes

    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        alias: Any = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        title: Any = ...,
        description: Any = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Any = ...,
        max_length: Any = ...,
        pattern: Any = ...,
        regex: Any = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = ...,
        example: Any = ...,
        openapi_examples: Any = ...,
        deprecated: Any = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Any = ...,
        **extra: Any
    ) -> None: ...

class Header(Param):
    in_: ParamTypes
    convert_underscores: bool

    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        alias: Any = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        convert_underscores: bool = ...,
        title: Any = ...,
        description: Any = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Any = ...,
        max_length: Any = ...,
        pattern: Any = ...,
        regex: Any = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = ...,
        example: Any = ...,
        openapi_examples: Any = ...,
        deprecated: Any = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Any = ...,
        **extra: Any
    ) -> None: ...

class Cookie(Param):
    in_: ParamTypes

    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        alias: Any = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        title: Any = ...,
        description: Any = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Any = ...,
        max_length: Any = ...,
        pattern: Any = ...,
        regex: Any = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = ...,
        example: Any = ...,
        openapi_examples: Any = ...,
        deprecated: Any = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Any = ...,
        **extra: Any
    ) -> None: ...

class Body(FieldInfo):
    embed: Any
    media_type: str
    example: Any
    include_in_schema: bool
    openapi_examples: Any

    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        embed: Any = ...,
        media_type: str = ...,
        alias: Any = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        title: Any = ...,
        description: Any = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Any = ...,
        max_length: Any = ...,
        pattern: Any = ...,
        regex: Any = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = ...,
        example: Any = ...,
        openapi_examples: Any = ...,
        deprecated: Any = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Any = ...,
        **extra: Any
    ) -> None: ...
    def __repr__(self) -> str: ...

class Form(Body):
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        media_type: str = ...,
        alias: Any = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        title: Any = ...,
        description: Any = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Any = ...,
        max_length: Any = ...,
        pattern: Any = ...,
        regex: Any = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = ...,
        example: Any = ...,
        openapi_examples: Any = ...,
        deprecated: Any = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Any = ...,
        **extra: Any
    ) -> None: ...

class File(Form):
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = ...,
        media_type: str = ...,
        alias: Any = ...,
        alias_priority: Any = ...,
        validation_alias: Any = ...,
        serialization_alias: Any = ...,
        title: Any = ...,
        description: Any = ...,
        gt: Any = ...,
        ge: Any = ...,
        lt: Any = ...,
        le: Any = ...,
        min_length: Any = ...,
        max_length: Any = ...,
        pattern: Any = ...,
        regex: Any = ...,
        discriminator: Any = ...,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = ...,
        example: Any = ...,
        openapi_examples: Any = ...,
        deprecated: Any = ...,
        include_in_schema: bool = ...,
        json_schema_extra: Any = ...,
        **extra: Any
    ) -> None: ...

class Depends:
    dependency: Any
    use_cache: bool

    def __init__(self, dependency: Any = ..., *, use_cache: bool = ...) -> None: ...
    def __repr__(self) -> str: ...

class Security(Depends):
    scopes: Any

    def __init__(self, dependency: Any = ..., *, scopes: Any = ..., use_cache: bool = ...) -> None: ...