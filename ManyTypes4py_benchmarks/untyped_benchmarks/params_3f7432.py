import warnings
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
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

    def __init__(self, default=Undefined, *, default_factory=_Unset, annotation=None, alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
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

    def __repr__(self):
        return f'{self.__class__.__name__}({self.default})'

class Path(Param):
    in_ = ParamTypes.path

    def __init__(self, default=..., *, default_factory=_Unset, annotation=None, alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
        assert default is ..., 'Path parameters cannot have a default value'
        self.in_ = self.in_
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Query(Param):
    in_ = ParamTypes.query

    def __init__(self, default=Undefined, *, default_factory=_Unset, annotation=None, alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Header(Param):
    in_ = ParamTypes.header

    def __init__(self, default=Undefined, *, default_factory=_Unset, annotation=None, alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, convert_underscores=True, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
        self.convert_underscores = convert_underscores
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Cookie(Param):
    in_ = ParamTypes.cookie

    def __init__(self, default=Undefined, *, default_factory=_Unset, annotation=None, alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Body(FieldInfo):

    def __init__(self, default=Undefined, *, default_factory=_Unset, annotation=None, embed=None, media_type='application/json', alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
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

    def __repr__(self):
        return f'{self.__class__.__name__}({self.default})'

class Form(Body):

    def __init__(self, default=Undefined, *, default_factory=_Unset, annotation=None, media_type='application/x-www-form-urlencoded', alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class File(Form):

    def __init__(self, default=Undefined, *, default_factory=_Unset, annotation=None, media_type='multipart/form-data', alias=None, alias_priority=_Unset, validation_alias=None, serialization_alias=None, title=None, description=None, gt=None, ge=None, lt=None, le=None, min_length=None, max_length=None, pattern=None, regex=None, discriminator=None, strict=_Unset, multiple_of=_Unset, allow_inf_nan=_Unset, max_digits=_Unset, decimal_places=_Unset, examples=None, example=_Unset, openapi_examples=None, deprecated=None, include_in_schema=True, json_schema_extra=None, **extra):
        super().__init__(default=default, default_factory=default_factory, annotation=annotation, media_type=media_type, alias=alias, alias_priority=alias_priority, validation_alias=validation_alias, serialization_alias=serialization_alias, title=title, description=description, gt=gt, ge=ge, lt=lt, le=le, min_length=min_length, max_length=max_length, pattern=pattern, regex=regex, discriminator=discriminator, strict=strict, multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits, decimal_places=decimal_places, deprecated=deprecated, example=example, examples=examples, openapi_examples=openapi_examples, include_in_schema=include_in_schema, json_schema_extra=json_schema_extra, **extra)

class Depends:

    def __init__(self, dependency=None, *, use_cache=True):
        self.dependency = dependency
        self.use_cache = use_cache

    def __repr__(self):
        attr = getattr(self.dependency, '__name__', type(self.dependency).__name__)
        cache = '' if self.use_cache else ', use_cache=False'
        return f'{self.__class__.__name__}({attr}{cache})'

class Security(Depends):

    def __init__(self, dependency=None, *, scopes=None, use_cache=True):
        super().__init__(dependency=dependency, use_cache=use_cache)
        self.scopes = scopes or []