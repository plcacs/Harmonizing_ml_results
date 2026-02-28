import warnings
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from fastapi.openapi.models import Example
from pydantic.fields import FieldInfo
from typing_extensions import Annotated, deprecated
from ._compat import PYDANTIC_V2, PYDANTIC_VERSION_MINOR_TUPLE, Undefined

_Unset = Undefined

class ParamTypes(Enum):
    query: str = 'query'
    header: str = 'header'
    path: str = 'path'
    cookie: str = 'cookie'

class Param(FieldInfo):
    def __init__(self, default: Any = Undefined, *, default_factory: Optional[Callable[[], Any]] = _Unset, annotation: Any = None, alias: str = None, alias_priority: Optional[int] = _Unset, validation_alias: str = None, serialization_alias: str = None, title: str = None, description: str = None, gt: Optional[float] = None, ge: Optional[float] = None, lt: Optional[float] = None, le: Optional[float] = None, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[str] = None, strict: Optional[bool] = _Unset, multiple_of: Optional[float] = _Unset, allow_inf_nan: Optional[bool] = _Unset, max_digits: Optional[int] = _Unset, decimal_places: Optional[int] = _Unset, examples: Optional[List[Example]] = None, example: Any = _Unset, openapi_examples: Optional[Dict[str, Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None:
        if example is not _Unset:
            warnings.warn('`example` has been deprecated, please use `examples` instead', category=DeprecationWarning, stacklevel=4)
        self.example = example
        self.include_in_schema = include_in_schema
        self.openapi_examples = openapi_examples
        kwargs = dict(default=default, default_factory=default_factory, alias=alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, discriminator=discriminator, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, **extra)
        if examples is not None:
            kwargs['examples'] = examples
        if regex is not None:
            warnings.warn('`regex` has been deprecated, please use `pattern` instead', category=DeprecationWarning, stacklevel=4)
        current_json_schema_extra = json_schema_extra or extra
        if PYDANTIC_VERSION_MINOR_TUPLE < (2, 7):
            self.deprecated = deprecated
        else:
            kwargs['deprecated'] = deprecated
        if PYDANTIC_V2:
            kwargs.update({'annotation': annotation, 'alias_priority': alias_priority, 'validation_alias': validation_alias, 'serialization_alias': serialization_alias, 'strict': strict, 'json_schema_extra': current_json_schema_extra})
            kwargs['pattern'] = pattern or regex
        else:
            kwargs['regex'] = pattern or regex
            kwargs.update(**current_json_schema_extra)
        use_kwargs = {k: v for k, v in kwargs.items() if v is not _Unset}
        super().__init__(**use_kwargs)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.default})'

class Path(Param):
    in_: ParamTypes = ParamTypes.path

    def __init__(self, default: Any = ..., *, default_factory: Optional[Callable[[], Any]] = _Unset, annotation: Any = None, alias: str = None, alias_priority: Optional[int] = _Unset, validation_alias: str = None, serialization_alias: str = None, title: str = None, description: str = None, gt: Optional[float] = None, ge: Optional[float] = None, lt: Optional[float] = None, le: Optional[float] = None, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[str] = None, strict: Optional[bool] = _Unset, multiple_of: Optional[float] = _Unset, allow_inf_nan: Optional[bool] = _Unset, max_digits: Optional[int] = _Unset, decimal_places: Optional[int] = _Unset, examples: Optional[List[Example]] = None, example: Any = _Unset, openapi_examples: Optional[Dict[str, Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None:
        assert default is ..., 'Path parameters cannot have a default value'
        self.in_ = self.in_
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Query(Param):
    in_: ParamTypes = ParamTypes.query

    def __init__(self, default: Any = Undefined, *, default_factory: Optional[Callable[[], Any]] = _Unset, annotation: Any = None, alias: str = None, alias_priority: Optional[int] = _Unset, validation_alias: str = None, serialization_alias: str = None, title: str = None, description: str = None, gt: Optional[float] = None, ge: Optional[float] = None, lt: Optional[float] = None, le: Optional[float] = None, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[str] = None, strict: Optional[bool] = _Unset, multiple_of: Optional[float] = _Unset, allow_inf_nan: Optional[bool] = _Unset, max_digits: Optional[int] = _Unset, decimal_places: Optional[int] = _Unset, examples: Optional[List[Example]] = None, example: Any = _Unset, openapi_examples: Optional[Dict[str, Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None:
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Header(Param):
    in_: ParamTypes = ParamTypes.header

    def __init__(self, default: Any = Undefined, *, default_factory: Optional[Callable[[], Any]] = _Unset, annotation: Any = None, alias: str = None, alias_priority: Optional[int] = _Unset, validation_alias: str = None, serialization_alias: str = None, convert_underscores: bool = True, title: str = None, description: str = None, gt: Optional[float] = None, ge: Optional[float] = None, lt: Optional[float] = None, le: Optional[float] = None, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[str] = None, strict: Optional[bool] = _Unset, multiple_of: Optional[float] = _Unset, allow_inf_nan: Optional[bool] = _Unset, max_digits: Optional[int] = _Unset, decimal_places: Optional[int] = _Unset, examples: Optional[List[Example]] = None, example: Any = _Unset, openapi_examples: Optional[Dict[str, Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None:
        self.convert_underscores = convert_underscores
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Cookie(Param):
    in_: ParamTypes = ParamTypes.cookie

    def __init__(self, default: Any = Undefined, *, default_factory: Optional[Callable[[], Any]] = _Unset, annotation: Any = None, alias: str = None, alias_priority: Optional[int] = _Unset, validation_alias: str = None, serialization_alias: str = None, title: str = None, description: str = None, gt: Optional[float] = None, ge: Optional[float] = None, lt: Optional[float] = None, le: Optional[float] = None, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[str] = None, strict: Optional[bool] = _Unset, multiple_of: Optional[float] = _Unset, allow_inf_nan: Optional[bool] = _Unset, max_digits: Optional[int] = _Unset, decimal_places: Optional[int] = _Unset, examples: Optional[List[Example]] = None, example: Any = _Unset, openapi_examples: Optional[Dict[str, Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None:
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Body(FieldInfo):

    def __init__(self, default: Any = Undefined, *, default_factory: Optional[Callable[[], Any]] = _Unset, annotation: Any = None, embed: Optional[str] = None, media_type: str = 'application/json', alias: str = None, alias_priority: Optional[int] = _Unset, validation_alias: str = None, serialization_alias: str = None, title: str = None, description: str = None, gt: Optional[float] = None, ge: Optional[float] = None, lt: Optional[float] = None, le: Optional[float] = None, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[str] = None, strict: Optional[bool] = _Unset, multiple_of: Optional[float] = _Unset, allow_inf_nan: Optional[bool] = _Unset, max_digits: Optional[int] = _Unset, decimal_places: Optional[int] = _Unset, examples: Optional[List[Example]] = None, example: Any = _Unset, openapi_examples: Optional[Dict[str, Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None:
        self.embed = embed
        self.media_type = media_type
        if example is not _Unset:
            warnings.warn('`example` has been deprecated, please use `examples` instead', category=DeprecationWarning, stacklevel=4)
        self.example = example
        self.include_in_schema = include_in_schema
        self.openapi_examples = openapi_examples
        kwargs = dict(default=default, default_factory=default_factory, alias=alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, discriminator=discriminator, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, **extra)
        if examples is not None:
            kwargs['examples'] = examples
        if regex is not None:
            warnings.warn('`regex` has been deprecated, please use `pattern` instead', category=DeprecationWarning, stacklevel=4)
        current_json_schema_extra = json_schema_extra or extra
        if PYDANTIC_VERSION_MINOR_TUPLE < (2, 7):
            self.deprecated = deprecated
        else:
            kwargs['deprecated'] = deprecated
        if PYDANTIC_V2:
            kwargs.update({'annotation': annotation, 'alias_priority': alias_priority, 'validation_alias': validation_alias, 'serialization_alias': serialization_alias, 'strict': strict, 'json_schema_extra': current_json_schema_extra})
            kwargs['pattern'] = pattern or regex
        else:
            kwargs['regex'] = pattern or regex
            kwargs.update(**current_json_schema_extra)
        use_kwargs = {k: v for k, v in kwargs.items() if v is not _Unset}
        super().__init__(**use_kwargs)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.default})'

class Form(Body):

    def __init__(self, default: Any = Undefined, *, default_factory: Optional[Callable[[], Any]] = _Unset, annotation: Any = None, media_type: str = 'application/x-www-form-urlencoded', alias: str = None, alias_priority: Optional[int] = _Unset, validation_alias: str = None, serialization_alias: str = None, title: str = None, description: str = None, gt: Optional[float] = None, ge: Optional[float] = None, lt: Optional[float] = None, le: Optional[float] = None, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[str] = None, strict: Optional[bool] = _Unset, multiple_of: Optional[float] = _Unset, allow_inf_nan: Optional[bool] = _Unset, max_digits: Optional[int] = _Unset, decimal_places: Optional[int] = _Unset, examples: Optional[List[Example]] = None, example: Any = _Unset, openapi_examples: Optional[Dict[str, Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None:
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class File(Form):

    def __init__(self, default: Any = Undefined, *, default_factory: Optional[Callable[[], Any]] = _Unset, annotation: Any = None, media_type: str = 'multipart/form-data', alias: str = None, alias_priority: Optional[int] = _Unset, validation_alias: str = None, serialization_alias: str = None, title: str = None, description: str = None, gt: Optional[float] = None, ge: Optional[float] = None, lt: Optional[float] = None, le: Optional[float] = None, min_length: Optional[int] = None, max_length: Optional[int] = None, pattern: Optional[str] = None, regex: Optional[str] = None, discriminator: Optional[str] = None, strict: Optional[bool] = _Unset, multiple_of: Optional[float] = _Unset, allow_inf_nan: Optional[bool] = _Unset, max_digits: Optional[int] = _Unset, decimal_places: Optional[int] = _Unset, examples: Optional[List[Example]] = None, example: Any = _Unset, openapi_examples: Optional[Dict[str, Example]] = None, deprecated: Optional[bool] = None, include_in_schema: bool = True, json_schema_extra: Optional[Dict[str, Any]] = None, **extra: Any) -> None:
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Depends:

    def __init__(self, dependency: Optional[Callable] = None, *, use_cache: bool = True) -> None:
        self.dependency = dependency
        self.use_cache = use_cache

    def __repr__(self) -> str:
        attr = getattr(self.dependency, '__name__', type(self.dependency).__name__)
        cache = '' if self.use_cache else ', use_cache=False'
        return f'{self.__class__.__name__}({attr}{cache})'

class Security(Depends):

    def __init__(self, dependency: Optional[Callable] = None, *, scopes: Optional[List[str]] = None, use_cache: bool = True) -> None:
        super().__init__(dependency=dependency, use_cache=use_cache)
        self.scopes = scopes or []
