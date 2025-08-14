from __future__ import annotations as _annotations

import operator
import sys
import types
import typing
import warnings
from collections.abc import Generator, Mapping
from copy import copy, deepcopy
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Literal,
    TypeVar,
    Union,
    cast,
    overload,
)

import pydantic_core
import typing_extensions
from pydantic_core import PydanticUndefined, ValidationError
from typing_extensions import Self, TypeAlias, Unpack

from . import PydanticDeprecatedSince20, PydanticDeprecatedSince211
from ._internal import (
    _config,
    _decorators,
    _fields,
    _forward_ref,
    _generics,
    _mock_val_ser,
    _model_construction,
    _namespace_utils,
    _repr,
    _typing_extra,
    _utils,
)
from ._migration import getattr_migration
from .aliases import AliasChoices, AliasPath
from .annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from .config import ConfigDict
from .errors import PydanticUndefinedAnnotation, PydanticUserError
from .json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaMode, JsonSchemaValue, model_json_schema
from .plugin._schema_validator import PluggableSchemaValidator

if TYPE_CHECKING:
    from inspect import Signature
    from pathlib import Path

    from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator

    from ._internal._namespace_utils import MappingNamespace
    from ._internal._utils import AbstractSetIntStr, MappingIntStrAny
    from .deprecated.parse import Protocol as DeprecatedParseProtocol
    from .fields import ComputedFieldInfo, FieldInfo, ModelPrivateAttr
else:
    DeprecationWarning = PydanticDeprecatedSince20

__all__ = ('BaseModel', 'create_model')

TupleGenerator: TypeAlias = Generator[tuple[str, Any], None, None]
IncEx: TypeAlias = Union[set[int], set[str], Mapping[int, Union['IncEx', bool]], Mapping[str, Union['IncEx', bool]]]

def _check_frozen(model_cls: type['BaseModel'], name: str, value: Any) -> None:
    ...

def _model_field_setattr_handler(model: 'BaseModel', name: str, val: Any) -> None:
    ...

_SIMPLE_SETATTR_HANDLERS: Mapping[str, Callable[['BaseModel', str, Any], None]] = {
    'model_field': _model_field_setattr_handler,
    'validate_assignment': lambda model, name, val: model.__pydantic_validator__.validate_assignment(model, name, val),
    'private': lambda model, name, val: model.__pydantic_private__.__setitem__(name, val),
    'cached_property': lambda model, name, val: model.__dict__.__setitem__(name, val),
    'extra_known': lambda model, name, val: _object_setattr(model, name, val),
}

class BaseModel(metaclass=_model_construction.ModelMetaclass):
    model_config: ClassVar[ConfigDict] = ConfigDict()
    __class_vars__: ClassVar[set[str]]
    __private_attributes__: ClassVar[Dict[str, 'ModelPrivateAttr']]
    __signature__: ClassVar['Signature']
    __pydantic_complete__: ClassVar[bool] = False
    __pydantic_core_schema__: ClassVar['CoreSchema']
    __pydantic_custom_init__: ClassVar[bool]
    __pydantic_decorators__: ClassVar[_decorators.DecoratorInfos] = _decorators.DecoratorInfos()
    __pydantic_generic_metadata__: ClassVar[_generics.PydanticGenericMetadata]
    __pydantic_parent_namespace__: ClassVar[Dict[str, Any] | None] = None
    __pydantic_post_init__: ClassVar[None | Literal['model_post_init']]
    __pydantic_root_model__: ClassVar[bool] = False
    __pydantic_serializer__: ClassVar['SchemaSerializer']
    __pydantic_validator__: ClassVar['SchemaValidator | PluggableSchemaValidator']
    __pydantic_fields__: ClassVar[Dict[str, 'FieldInfo']]
    __pydantic_setattr_handlers__: ClassVar[Dict[str, Callable[['BaseModel', str, Any], None]]]
    __pydantic_computed_fields__: ClassVar[Dict[str, 'ComputedFieldInfo']]
    __pydantic_extra__: dict[str, Any] | None = _model_construction.NoInitField(init=False)
    __pydantic_fields_set__: set[str] = _model_construction.NoInitField(init=False)
    __pydantic_private__: dict[str, Any] | None = _model_construction.NoInitField(init=False)

    def __init__(self, /, **data: Any) -> None:
        ...

    @_utils.deprecated_instance_property
    @classmethod
    def model_fields(cls) -> dict[str, 'FieldInfo']:
        ...

    @_utils.deprecated_instance_property
    @classmethod
    def model_computed_fields(cls) -> dict[str, 'ComputedFieldInfo']:
        ...

    @property
    def model_extra(self) -> dict[str, Any] | None:
        ...

    @property
    def model_fields_set(self) -> set[str]:
        ...

    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: Any) -> Self:
        ...

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        ...

    def model_dump(
        self,
        *,
        mode: Literal['json', 'python'] | str = 'python',
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        context: Any | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal['none', 'warn', 'error'] = True,
        fallback: Callable[[Any], Any] | None = None,
        serialize_as_any: bool = False,
    ) -> dict[str, Any]:
        ...

    def model_dump_json(
        self,
        *,
        indent: int | None = None,
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        context: Any | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal['none', 'warn', 'error'] = True,
        fallback: Callable[[Any], Any] | None = None,
        serialize_as_any: bool = False,
    ) -> str:
        ...

    @classmethod
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = 'validation',
    ) -> dict[str, Any]:
        ...

    @classmethod
    def model_parametrized_name(cls, params: tuple[type[Any], ...]) -> str:
        ...

    def model_post_init(self, context: Any, /) -> None:
        ...

    @classmethod
    def model_rebuild(
        cls,
        *,
        force: bool = False,
        raise_errors: bool = True,
        _parent_namespace_depth: int = 2,
        _types_namespace: 'MappingNamespace | None' = None,
    ) -> bool | None:
        ...

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
    ) -> Self:
        ...

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        context: Any | None = None,
    ) -> Self:
        ...

    @classmethod
    def model_validate_strings(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        context: Any | None = None,
    ) -> Self:
        ...

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type['BaseModel'], handler: GetCoreSchemaHandler, /) -> 'CoreSchema':
        ...

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: 'CoreSchema',
        handler: GetJsonSchemaHandler,
        /,
    ) -> JsonSchemaValue:
        ...

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        ...

    def __class_getitem__(
        cls, typevar_values: type[Any] | tuple[type[Any], ...]
    ) -> type['BaseModel'] | _forward_ref.PydanticRecursiveRef:
        ...

    def __copy__(self) -> Self:
        ...

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        ...

    def __getstate__(self) -> dict[Any, Any]:
        ...

    def __setstate__(self, state: dict[Any, Any]) -> None:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __iter__(self) -> TupleGenerator:
        ...

    def __repr__(self) -> str:
        ...

    def __repr_args__(self) -> '_repr.ReprArgs':
        ...

    def __str__(self) -> str:
        ...

    # Deprecated methods
    @property
    def __fields__(self) -> dict[str, 'FieldInfo']:
        ...

    @property
    def __fields_set__(self) -> set[str]:
        ...

    def dict(
        self,
        *,
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> Dict[str, Any]:
        ...

    def json(
        self,
        *,
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        encoder: Callable[[Any], Any] | None = PydanticUndefined,
        models_as_dict: bool = PydanticUndefined,
        **dumps_kwargs: Any,
    ) -> str:
        ...

    @classmethod
    def parse_obj(cls, obj: Any) -> Self:
        ...

    @classmethod
    def parse_raw(
        cls,
        b: str | bytes,
        *,
        content_type: str | None = None,
        encoding: str = 'utf8',
        proto: 'DeprecatedParseProtocol | None' = None,
        allow_pickle: bool = False,
    ) -> Self:
        ...

    @classmethod
    def parse_file(
        cls,
        path: str | 'Path',
        *,
        content_type: str | None = None,
        encoding: str = 'utf8',
        proto: 'DeprecatedParseProtocol | None' = None,
        allow_pickle: bool = False,
    ) -> Self:
        ...

    @classmethod
    def from_orm(cls, obj: Any) -> Self:
        ...

    @classmethod
    def construct(cls, _fields_set: set[str] | None = None, **values: Any) -> Self:
        ...

    def copy(
        self,
        *,
        include: 'AbstractSetIntStr | MappingIntStrAny | None' = None,
        exclude: 'AbstractSetIntStr | MappingIntStrAny | None' = None,
        update: Dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        ...

    @classmethod
    def schema(cls, by_alias: bool = True, ref_template: str = DEFAULT_REF_TEMPLATE) -> Dict[str, Any]:
        ...

    @classmethod
    def schema_json(
        cls, *, by_alias: bool = True, ref_template: str = DEFAULT_REF_TEMPLATE, **dumps_kwargs: Any
    ) -> str:
        ...

    @classmethod
    def validate(cls, value: Any) -> Self:
        ...

    @classmethod
    def update_forward_refs(cls, **localns: Any) -> None:
        ...

    def _iter(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def _copy_and_set_values(self, *args: Any, **kwargs: Any) -> Any:
        ...

    @classmethod
    def _get_value(cls, *args: Any, **kwargs: Any) -> Any:
        ...

    def _calculate_keys(self, *args: Any, **kwargs: Any) -> Any:
        ...

ModelT = TypeVar('ModelT', bound=BaseModel)

@overload
def create_model(
    model_name: str,
    /,
    *,
    __config__: ConfigDict | None = None,
    __doc__: str | None = None,
    __base__: None = None,
    __module__: str = __name__,
    __validators__: dict[str, Callable[..., Any]] | None = None,
    __cls_kwargs__: dict[str, Any] | None = None,
    **field_definitions: Any | tuple[str, Any],
) -> type[BaseModel]: ...

@overload
def create_model(
    model_name: str,
    /,
    *,
    __config__: ConfigDict | None = None,
    __doc__: str | None = None,
    __base__: type[ModelT] | tuple[type[ModelT], ...],
    __module__: str = __name__,
    __validators__: dict[str, Callable[..., Any]] | None = None,
    __cls_kwargs__: dict[str, Any] | None = None,
    **field_definitions: Any | tuple[str, Any],
) -> type[ModelT]: ...

def create_model(
    model_name: str,
    /,
    *,
    __config__: ConfigDict | None = None,
    __doc__: str | None = None,
    __base__: type[ModelT] | tuple[type[ModelT], ...] | None = None,
    __module__: str | None = None,
    __validators__: dict[str, Callable[..., Any]] | None = None,
    __cls_kwargs__: dict[str, Any] | None = None,
    **field_definitions: Any | tuple[str, Any],
) -> type[ModelT]:
    ...

__getattr__ = getattr_migration(__name__)
