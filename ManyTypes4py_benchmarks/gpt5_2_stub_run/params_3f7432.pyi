from enum import Enum
from typing import Any, List
from pydantic.fields import FieldInfo

Example: Any = ...
Annotated: Any = ...
deprecated: Any = ...
PYDANTIC_V2: Any = ...
PYDANTIC_VERSION_MINOR_TUPLE: Any = ...
Undefined: Any = ...
_Unset: Any = ...

class ParamTypes(Enum):
    query: ParamTypes
    header: ParamTypes
    path: ParamTypes
    cookie: ParamTypes
    ...

class Param(FieldInfo):
    example: Any
    include_in_schema: bool
    openapi_examples: Any
    deprecated: Any
    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Any = _Unset,
        annotation: Any = None,
        alias: Any = None,
        alias_priority: Any = _Unset,
        validation_alias: Any = None,
        serialization_alias: Any = None,
        title: Any = None,
        description: Any = None,
        gt: Any = None,
        ge: Any = None,
        lt: Any = None,
        le: Any = None,
        min_length: Any = None,
        max_length: Any = None,
        pattern: Any = None,
        regex: Any = None,
        discriminator: Any = None,
        strict: Any = _Unset,
        multiple_of: Any = _Unset,
        allow_inf_nan: Any = _Unset,
        max_digits: Any = _Unset,
        decimal_places: Any = _Unset,
        examples: Any = None,
        example: Any = _Unset,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...
    def __repr__(self) -> str: ...

class Path(Param):
    in_: ParamTypes
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = _Unset,
        annotation: Any = None,
        alias: Any = None,
        alias_priority: Any = _Unset,
        validation_alias: Any = None,
        serialization_alias: Any = None,
        title: Any = None,
        description: Any = None,
        gt: Any = None,
        ge: Any = None,
        lt: Any = None,
        le: Any = None,
        min_length: Any = None,
        max_length: Any = None,
        pattern: Any = None,
        regex: Any = None,
        discriminator: Any = None,
        strict: Any = _Unset,
        multiple_of: Any = _Unset,
        allow_inf_nan: Any = _Unset,
        max_digits: Any = _Unset,
        decimal_places: Any = _Unset,
        examples: Any = None,
        example: Any = _Unset,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...

class Query(Param):
    in_: ParamTypes
    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Any = _Unset,
        annotation: Any = None,
        alias: Any = None,
        alias_priority: Any = _Unset,
        validation_alias: Any = None,
        serialization_alias: Any = None,
        title: Any = None,
        description: Any = None,
        gt: Any = None,
        ge: Any = None,
        lt: Any = None,
        le: Any = None,
        min_length: Any = None,
        max_length: Any = None,
        pattern: Any = None,
        regex: Any = None,
        discriminator: Any = None,
        strict: Any = _Unset,
        multiple_of: Any = _Unset,
        allow_inf_nan: Any = _Unset,
        max_digits: Any = _Unset,
        decimal_places: Any = _Unset,
        examples: Any = None,
        example: Any = _Unset,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...

class Header(Param):
    in_: ParamTypes
    convert_underscores: bool
    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Any = _Unset,
        annotation: Any = None,
        alias: Any = None,
        alias_priority: Any = _Unset,
        validation_alias: Any = None,
        serialization_alias: Any = None,
        convert_underscores: bool = True,
        title: Any = None,
        description: Any = None,
        gt: Any = None,
        ge: Any = None,
        lt: Any = None,
        le: Any = None,
        min_length: Any = None,
        max_length: Any = None,
        pattern: Any = None,
        regex: Any = None,
        discriminator: Any = None,
        strict: Any = _Unset,
        multiple_of: Any = _Unset,
        allow_inf_nan: Any = _Unset,
        max_digits: Any = _Unset,
        decimal_places: Any = _Unset,
        examples: Any = None,
        example: Any = _Unset,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...

class Cookie(Param):
    in_: ParamTypes
    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Any = _Unset,
        annotation: Any = None,
        alias: Any = None,
        alias_priority: Any = _Unset,
        validation_alias: Any = None,
        serialization_alias: Any = None,
        title: Any = None,
        description: Any = None,
        gt: Any = None,
        ge: Any = None,
        lt: Any = None,
        le: Any = None,
        min_length: Any = None,
        max_length: Any = None,
        pattern: Any = None,
        regex: Any = None,
        discriminator: Any = None,
        strict: Any = _Unset,
        multiple_of: Any = _Unset,
        allow_inf_nan: Any = _Unset,
        max_digits: Any = _Unset,
        decimal_places: Any = _Unset,
        examples: Any = None,
        example: Any = _Unset,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...

class Body(FieldInfo):
    embed: Any
    media_type: str
    example: Any
    include_in_schema: bool
    openapi_examples: Any
    deprecated: Any
    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Any = _Unset,
        annotation: Any = None,
        embed: Any = None,
        media_type: str = 'application/json',
        alias: Any = None,
        alias_priority: Any = _Unset,
        validation_alias: Any = None,
        serialization_alias: Any = None,
        title: Any = None,
        description: Any = None,
        gt: Any = None,
        ge: Any = None,
        lt: Any = None,
        le: Any = None,
        min_length: Any = None,
        max_length: Any = None,
        pattern: Any = None,
        regex: Any = None,
        discriminator: Any = None,
        strict: Any = _Unset,
        multiple_of: Any = _Unset,
        allow_inf_nan: Any = _Unset,
        max_digits: Any = _Unset,
        decimal_places: Any = _Unset,
        examples: Any = None,
        example: Any = _Unset,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...
    def __repr__(self) -> str: ...

class Form(Body):
    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Any = _Unset,
        annotation: Any = None,
        media_type: str = 'application/x-www-form-urlencoded',
        alias: Any = None,
        alias_priority: Any = _Unset,
        validation_alias: Any = None,
        serialization_alias: Any = None,
        title: Any = None,
        description: Any = None,
        gt: Any = None,
        ge: Any = None,
        lt: Any = None,
        le: Any = None,
        min_length: Any = None,
        max_length: Any = None,
        pattern: Any = None,
        regex: Any = None,
        discriminator: Any = None,
        strict: Any = _Unset,
        multiple_of: Any = _Unset,
        allow_inf_nan: Any = _Unset,
        max_digits: Any = _Unset,
        decimal_places: Any = _Unset,
        examples: Any = None,
        example: Any = _Unset,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...

class File(Form):
    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Any = _Unset,
        annotation: Any = None,
        media_type: str = 'multipart/form-data',
        alias: Any = None,
        alias_priority: Any = _Unset,
        validation_alias: Any = None,
        serialization_alias: Any = None,
        title: Any = None,
        description: Any = None,
        gt: Any = None,
        ge: Any = None,
        lt: Any = None,
        le: Any = None,
        min_length: Any = None,
        max_length: Any = None,
        pattern: Any = None,
        regex: Any = None,
        discriminator: Any = None,
        strict: Any = _Unset,
        multiple_of: Any = _Unset,
        allow_inf_nan: Any = _Unset,
        max_digits: Any = _Unset,
        decimal_places: Any = _Unset,
        examples: Any = None,
        example: Any = _Unset,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...

class Depends:
    dependency: Any
    use_cache: bool
    def __init__(self, dependency: Any = None, *, use_cache: bool = True) -> None: ...
    def __repr__(self) -> str: ...

class Security(Depends):
    scopes: List[str]
    def __init__(self, dependency: Any = None, *, scopes: Any = None, use_cache: bool = True) -> None: ...