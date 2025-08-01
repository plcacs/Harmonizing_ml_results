import warnings
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Set, Type
from fastapi.openapi.models import Example
from pydantic.fields import FieldInfo
from typing_extensions import Annotated, deprecated
from ._compat import PYDANTIC_V2, PYDANTIC_VERSION_MINOR_TUPLE, Undefined
_Unset = Undefined

class ParamTypes(Enum):
    query = 'query'
    header = 'header'
    path = 'path'
    cookie = 'cookie'

class Param(FieldInfo):

    def __init__(
        self, 
        default: Any = Undefined, 
        *, 
        default_factory: Optional[Callable[[], Any]] = _Unset, 
        annotation: Optional[Any] = None, 
        alias: Optional[str] = None, 
        alias_priority: Any = _Unset, 
        validation_alias: Optional[str] = None, 
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
        discriminator: Optional[str] = None, 
        strict: Any = _Unset, 
        multiple_of: Any = _Unset, 
        allow_inf_nan: Any = _Unset, 
        max_digits: Any = _Unset, 
        decimal_places: Any = _Unset, 
        examples: Optional[Dict[str, Any]] = None, 
        example: Any = _Unset, 
        openapi_examples: Optional[Dict[str, Example]] = None, 
        deprecated: Optional[bool] = None, 
        include_in_schema: bool = True, 
        json_schema_extra: Optional[Dict[str, Any]] = None, 
        **extra: Any
    ) -> None:
        if example is not _Unset:
            warnings.warn('`example` has been deprecated, please use `examples` instead', category=DeprecationWarning, stacklevel=4)
        self.example: Any = example
        self.include_in_schema: bool = include_in_schema
        self.openapi_examples: Optional[Dict[str, Example]] = openapi_examples
        kwargs: Dict[str, Any] = dict(default=default, default_factory=default_factory, alias=alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, discriminator=discriminator, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, **extra)
        if examples is not None:
            kwargs['examples'] = examples
        if regex is not None:
            warnings.warn('`regex` has been deprecated, please use `pattern` instead', category=DeprecationWarning, stacklevel=4)
        current_json_schema_extra: Optional[Dict[str, Any]] = json_schema_extra or extra
        if PYDANTIC_VERSION_MINOR_TUPLE < (2, 7):
            self.deprecated: Optional[bool] = deprecated
        else:
            kwargs['deprecated'] = deprecated
        if PYDANTIC_V2:
            kwargs.update({'annotation': annotation, 'alias_priority': alias_priority, 'validation_alias': validation_alias, 'serialization_alias': serialization_alias, 'strict': strict, 'json_schema_extra': current_json_schema_extra})
            kwargs['pattern'] = pattern or regex
        else:
            kwargs['regex'] = pattern or regex
            kwargs.update(**current_json_schema_extra)
        use_kwargs: Dict[str, Any] = {k: v for k, v in kwargs.items() if v is not _Unset}
        super().__init__(**use_kwargs)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.default})'

class Path(Param):
    in_: ParamTypes = ParamTypes.path

    def __init__(
        self, 
        default: Any = ..., 
        *, 
        default_factory: Optional[Callable[[], Any]] = _Unset, 
        annotation: Optional[Any] = None, 
        alias: Optional[str] = None, 
        alias_priority: Any = _Unset, 
        validation_alias: Optional[str] = None, 
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
        discriminator: Optional[str] = None, 
        strict: Any = _Unset, 
        multiple_of: Any = _Unset, 
        allow_inf_nan: Any = _Unset, 
        max_digits: Any = _Unset, 
        decimal_places: Any = _Unset, 
        examples: Optional[Dict[str, Any]] = None, 
        example: Any = _Unset, 
        openapi_examples: Optional[Dict[str, Example]] = None, 
        deprecated: Optional[bool] = None, 
        include_in_schema: bool = True, 
        json_schema_extra: Optional[Dict[str, Any]] = None, 
        **extra: Any
    ) -> None:
        assert default is ..., 'Path parameters cannot have a default value'
        self.in_: ParamTypes = self.in_
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Query(Param):
    in_: ParamTypes = ParamTypes.query

    def __init__(
        self, 
        default: Any = Undefined, 
        *, 
        default_factory: Optional[Callable[[], Any]] = _Unset, 
        annotation: Optional[Any] = None, 
        alias: Optional[str] = None, 
        alias_priority: Any = _Unset, 
        validation_alias: Optional[str] = None, 
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
        discriminator: Optional[str] = None, 
        strict: Any = _Unset, 
        multiple_of: Any = _Unset, 
        allow_inf_nan: Any = _Unset, 
        max_digits: Any = _Unset, 
        decimal_places: Any = _Unset, 
        examples: Optional[Dict[str, Any]] = None, 
        example: Any = _Unset, 
        openapi_examples: Optional[Dict[str, Example]] = None, 
        deprecated: Optional[bool] = None, 
        include_in_schema: bool = True, 
        json_schema_extra: Optional[Dict[str, Any]] = None, 
        **extra: Any
    ) -> None:
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Header(Param):
    in_: ParamTypes = ParamTypes.header

    def __init__(
        self, 
        default: Any = Undefined, 
        *, 
        default_factory: Optional[Callable[[], Any]] = _Unset, 
        annotation: Optional[Any] = None, 
        alias: Optional[str] = None, 
        alias_priority: Any = _Unset, 
        validation_alias: Optional[str] = None, 
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
        discriminator: Optional[str] = None, 
        strict: Any = _Unset, 
        multiple_of: Any = _Unset, 
        allow_inf_nan: Any = _Unset, 
        max_digits: Any = _Unset, 
        decimal_places: Any = _Unset, 
        examples: Optional[Dict[str, Any]] = None, 
        example: Any = _Unset, 
        openapi_examples: Optional[Dict[str, Example]] = None, 
        deprecated: Optional[bool] = None, 
        include_in_schema: bool = True, 
        json_schema_extra: Optional[Dict[str, Any]] = None, 
        **extra: Any
    ) -> None:
        self.convert_underscores: bool = convert_underscores
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Cookie(Param):
    in_: ParamTypes = ParamTypes.cookie

    def __init__(
        self, 
        default: Any = Undefined, 
        *, 
        default_factory: Optional[Callable[[], Any]] = _Unset, 
        annotation: Optional[Any] = None, 
        alias: Optional[str] = None, 
        alias_priority: Any = _Unset, 
        validation_alias: Optional[str] = None, 
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
        discriminator: Optional[str] = None, 
        strict: Any = _Unset, 
        multiple_of: Any = _Unset, 
        allow_inf_nan: Any = _Unset, 
        max_digits: Any = _Unset, 
        decimal_places: Any = _Unset, 
        examples: Optional[Dict[str, Any]] = None, 
        example: Any = _Unset, 
        openapi_examples: Optional[Dict[str, Example]] = None, 
        deprecated: Optional[bool] = None, 
        include_in_schema: bool = True, 
        json_schema_extra: Optional[Dict[str, Any]] = None, 
        **extra: Any
    ) -> None:
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Body(FieldInfo):

    def __init__(
        self, 
        default: Any = Undefined, 
        *, 
        default_factory: Optional[Callable[[], Any]] = _Unset, 
        annotation: Optional[Any] = None, 
        embed: Optional[bool] = None, 
        media_type: str = 'application/json', 
        alias: Optional[str] = None, 
        alias_priority: Any = _Unset, 
        validation_alias: Optional[str] = None, 
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
        discriminator: Optional[str] = None, 
        strict: Any = _Unset, 
        multiple_of: Any = _Unset, 
        allow_inf_nan: Any = _Unset, 
        max_digits: Any = _Unset, 
        decimal_places: Any = _Unset, 
        examples: Optional[Dict[str, Any]] = None, 
        example: Any = _Unset, 
        openapi_examples: Optional[Dict[str, Example]] = None, 
        deprecated: Optional[bool] = None, 
        include_in_schema: bool = True, 
        json_schema_extra: Optional[Dict[str, Any]] = None, 
        **extra: Any
    ) -> None:
        self.embed: Optional[bool] = embed
        self.media_type: str = media_type
        if example is not _Unset:
            warnings.warn('`example` has been deprecated, please use `examples` instead', category=DeprecationWarning, stacklevel=4)
        self.example: Any = example
        self.include_in_schema: bool = include_in_schema
        self.openapi_examples: Optional[Dict[str, Example]] = openapi_examples
        kwargs: Dict[str, Any] = dict(default=default, default_factory=default_factory, alias=alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, discriminator=discriminator, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, **extra)
        if examples is not None:
            kwargs['examples'] = examples
        if regex is not None:
            warnings.warn('`regex` has been deprecated, please use `pattern` instead', category=DeprecationWarning, stacklevel=4)
        current_json_schema_extra: Optional[Dict[str, Any]] = json_schema_extra or extra
        if PYDANTIC_VERSION_MINOR_TUPLE < (2, 7):
            self.deprecated: Optional[bool] = deprecated
        else:
            kwargs['deprecated'] = deprecated
        if PYDANTIC_V2:
            kwargs.update({'annotation': annotation, 'alias_priority': alias_priority, 'validation_alias': validation_alias, 'serialization_alias': serialization_alias, 'strict': strict, 'json_schema_extra': current_json_schema_extra})
            kwargs['pattern'] = pattern or regex
        else:
            kwargs['regex'] = pattern or regex
            kwargs.update(**current_json_schema_extra)
        use_kwargs: Dict[str, Any] = {k: v for k, v in kwargs.items() if v is not _Unset}
        super().__init__(**use_kwargs)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.default})'

class Form(Body):

    def __init__(
        self, 
        default: Any = Undefined, 
        *, 
        default_factory: Optional[Callable[[], Any]] = _Unset, 
        annotation: Optional[Any] = None, 
        media_type: str = 'application/x-www-form-urlencoded', 
        alias: Optional[str] = None, 
        alias_priority: Any = _Unset, 
        validation_alias: Optional[str] = None, 
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
        discriminator: Optional[str] = None, 
        strict: Any = _Unset, 
        multiple_of: Any = _Unset, 
        allow_inf_nan: Any = _Unset, 
        max_digits: Any = _Unset, 
        decimal_places: Any = _Unset, 
        examples: Optional[Dict[str, Any]] = None, 
        example: Any = _Unset, 
        openapi_examples: Optional[Dict[str, Example]] = None, 
        deprecated: Optional[bool] = None, 
        include_in_schema: bool = True, 
        json_schema_extra: Optional[Dict[str, Any]] = None, 
        **extra: Any
    ) -> None:
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class File(Form):

    def __init__(
        self, 
        default: Any = Undefined, 
        *, 
        default_factory: Optional[Callable[[], Any]] = _Unset, 
        annotation: Optional[Any] = None, 
        media_type: str = 'multipart/form-data', 
        alias: Optional[str] = None, 
        alias_priority: Any = _Unset, 
        validation_alias: Optional[str] = None, 
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
        discriminator: Optional[str] = None, 
        strict: Any = _Unset, 
        multiple_of: Any = _Unset, 
        allow_inf_nan: Any = _Unset, 
        max_digits: Any = _Unset, 
        decimal_places: Any = _Unset, 
        examples: Optional[Dict[str, Any]] = None, 
        example: Any = _Unset, 
        openapi_examples: Optional[Dict[str, Example]] = None, 
        deprecated: Optional[bool] = None, 
        include_in_schema: bool = True, 
        json_schema_extra: Optional[Dict[str, Any]] = None, 
        **extra: Any
    ) -> None:
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Depends:

    def __init__(self, dependency: Optional[Callable[..., Any]] = None, *, use_cache: bool = True) -> None:
        self.dependency: Optional[Callable[..., Any]] = dependency
        self.use_cache: bool = use_cache

    def __repr__(self) -> str:
        attr = getattr(self.dependency, '__name__', type(self.dependency).__name__)
        cache = '' if self.use_cache else ', use_cache=False'
        return f'{self.__class__.__name__}({attr}{cache})'

class Security(Depends):

    def __init__(self, dependency: Optional[Callable[..., Any]] = None, *, scopes: Optional[List[str]] = None, use_cache: bool = True) -> None:
        super().__init__(dependency=dependency, use_cache=use_cache)
        self.scopes: List[str] = scopes or []
