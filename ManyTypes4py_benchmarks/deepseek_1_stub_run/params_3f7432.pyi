```python
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from fastapi.openapi.models import Example
from pydantic.fields import FieldInfo
from typing_extensions import Annotated, deprecated

class ParamTypes(Enum):
    query: str = ...
    header: str = ...
    path: str = ...
    cookie: str = ...

class Param(FieldInfo):
    in_: ParamTypes = ...
    example: Any = ...
    include_in_schema: bool = ...
    openapi_examples: Any = ...
    deprecated: Any = ...
    
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = None,
        alias: Optional[str] = None,
        alias_priority: Any = ...,
        validation_alias: Any = None,
        serialization_alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[float] = None,
        ge: Optional[float] = None,
        lt: Optional[float] = None,
        le: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Any = None,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = None,
        example: Any = ...,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...
    
    def __repr__(self) -> str: ...

class Path(Param):
    in_: ParamTypes = ...
    
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = None,
        alias: Optional[str] = None,
        alias_priority: Any = ...,
        validation_alias: Any = None,
        serialization_alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[float] = None,
        ge: Optional[float] = None,
        lt: Optional[float] = None,
        le: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Any = None,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = None,
        example: Any = ...,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...

class Query(Param):
    in_: ParamTypes = ...
    
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = None,
        alias: Optional[str] = None,
        alias_priority: Any = ...,
        validation_alias: Any = None,
        serialization_alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[float] = None,
        ge: Optional[float] = None,
        lt: Optional[float] = None,
        le: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Any = None,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = None,
        example: Any = ...,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...

class Header(Param):
    in_: ParamTypes = ...
    convert_underscores: bool = ...
    
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = None,
        alias: Optional[str] = None,
        alias_priority: Any = ...,
        validation_alias: Any = None,
        serialization_alias: Optional[str] = None,
        convert_underscores: bool = True,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[float] = None,
        ge: Optional[float] = None,
        lt: Optional[float] = None,
        le: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Any = None,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = None,
        example: Any = ...,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...

class Cookie(Param):
    in_: ParamTypes = ...
    
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = None,
        alias: Optional[str] = None,
        alias_priority: Any = ...,
        validation_alias: Any = None,
        serialization_alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[float] = None,
        ge: Optional[float] = None,
        lt: Optional[float] = None,
        le: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Any = None,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = None,
        example: Any = ...,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...

class Body(FieldInfo):
    embed: Any = ...
    media_type: str = ...
    example: Any = ...
    include_in_schema: bool = ...
    openapi_examples: Any = ...
    deprecated: Any = ...
    
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = None,
        embed: Any = None,
        media_type: str = "application/json",
        alias: Optional[str] = None,
        alias_priority: Any = ...,
        validation_alias: Any = None,
        serialization_alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[float] = None,
        ge: Optional[float] = None,
        lt: Optional[float] = None,
        le: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Any = None,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = None,
        example: Any = ...,
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
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = None,
        media_type: str = "application/x-www-form-urlencoded",
        alias: Optional[str] = None,
        alias_priority: Any = ...,
        validation_alias: Any = None,
        serialization_alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[float] = None,
        ge: Optional[float] = None,
        lt: Optional[float] = None,
        le: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Any = None,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = None,
        example: Any = ...,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...

class File(Form):
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Any = ...,
        annotation: Any = None,
        media_type: str = "multipart/form-data",
        alias: Optional[str] = None,
        alias_priority: Any = ...,
        validation_alias: Any = None,
        serialization_alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[float] = None,
        ge: Optional[float] = None,
        lt: Optional[float] = None,
        le: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        regex: Optional[str] = None,
        discriminator: Any = None,
        strict: Any = ...,
        multiple_of: Any = ...,
        allow_inf_nan: Any = ...,
        max_digits: Any = ...,
        decimal_places: Any = ...,
        examples: Any = None,
        example: Any = ...,
        openapi_examples: Any = None,
        deprecated: Any = None,
        include_in_schema: bool = True,
        json_schema_extra: Any = None,
        **extra: Any
    ) -> None: ...

class Depends:
    dependency: Any = ...
    use_cache: bool = ...
    
    def __init__(self, dependency: Any = None, *, use_cache: bool = True) -> None: ...
    def __repr__(self) -> str: ...

class Security(Depends):
    scopes: List[Any] = ...
    
    def __init__(self, dependency: Any = None, *, scopes: Any = None, use_cache: bool = True) -> None: ...

_Unset: Any = ...
```