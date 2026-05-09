from __future__ import annotations
import builtins
import operator
import sys
import typing
import warnings
import weakref
from abc import ABCMeta
from functools import cache, partial, wraps
from types import FunctionType
from typing import Any, Callable, Generic, Literal, NoReturn, cast
from pydantic_core import PydanticUndefined, SchemaSerializer
from typing_extensions import TypeAliasType, dataclass_transform, deprecated, get_args
from ..errors import PydanticUndefinedAnnotation, PydanticUserError
from ..plugin._schema_validator import create_schema_validator
from ..warnings import GenericBeforeBaseModelWarning, PydanticDeprecatedSince20
from ._config import ConfigWrapper
from ._decorators import DecoratorInfos, PydanticDescriptorProxy, get_attribute_from_bases, unwrap_wrapped_function
from ._fields import collect_model_fields, is_valid_field_name, is_valid_privateattr_name
from ._generate_schema import GenerateSchema, InvalidSchemaError
from ._generics import PydanticGenericMetadata, get_model_typevars_map
from ._import_utils import import_cached_base_model, import_cached_field_info
from ._mock_val_ser import set_model_mocks
from ._namespace_utils import NsResolver
from ._signature import generate_pydantic_signature
from ._typing_extra import _make_forward_ref, eval_type_backport, is_annotated, is_classvar_annotation, parent_frame_namespace
from ._utils import LazyClassAttribute, SafeGetItemProxy
if typing.TYPE_CHECKING:
    from ..fields import Field as PydanticModelField
    from ..fields import FieldInfo, ModelPrivateAttr
    from ..fields import PrivateAttr as PydanticModelPrivateAttr
    from ..main import BaseModel
else:
    DeprecationWarning = PydanticDeprecatedSince20
    PydanticModelField = object()
    PydanticModelPrivateAttr = object()
object_setattr = object.__setattr__

class _ModelNamespaceDict(dict):
    """A dictionary subclass that intercepts attribute setting on model classes and warns about overriding of decorators."""
    def __setitem__(self, k: str, v: Any) -> None:
        existing = self.get(k, None)
        if existing and v is not existing and isinstance(existing, PydanticDescriptorProxy):
            warnings.warn(f'{k} overrides an existing Pydantic `{existing.decorator_info.decorator_repr}` decorator')
        return super().__setitem__(k, v)

def NoInitField(*, init: bool = False) -> NoReturn:
    """Only for typing purposes. Used as default value of `__pydantic_fields_set__`, `__pydantic_extra__`, `__pydantic_private__`, so they could be ignored when synthesizing the `__init__` signature."""
    pass

class ModelMetaclass(ABCMeta):
    def __new__(cls, cls_name: str, bases: tuple, namespace: dict, __pydantic_generic_metadata__: Any = None, __pydantic_reset_parent_namespace__: bool = True, _create_model_module: Any = None, **kwargs: Any) -> type[BaseModel]:
        # ... rest of the code ...

    @classmethod
    def __prepare__(cls, *args, **kwargs) -> _ModelNamespaceDict:
        return _ModelNamespaceDict()

    # ... rest of the code ...

def init_private_attributes(self: BaseModel, context: Any, /) -> None:
    # ... rest of the code ...

def get_model_post_init(namespace: dict, bases: tuple) -> Callable[[BaseModel], None]:
    # ... rest of the code ...

def inspect_namespace(namespace: dict, ignored_types: tuple, base_class_vars: set, base_class_fields: set) -> dict:
    # ... rest of the code ...

def set_model_fields(cls: type[BaseModel], config_wrapper: ConfigWrapper, ns_resolver: NsResolver) -> None:
    # ... rest of the code ...

def complete_model_class(cls: type[BaseModel], config_wrapper: ConfigWrapper, *, raise_errors: bool, ns_resolver: NsResolver, create_model_module: str) -> bool:
    # ... rest of the code ...

def set_deprecated_descriptors(cls: type[BaseModel]) -> None:
    # ... rest of the code ...

class _DeprecatedFieldDescriptor:
    def __init__(self, msg: str, wrapped_property: Any = None) -> None:
        self.msg = msg
        self.wrapped_property = wrapped_property

    # ... rest of the code ...

class _PydanticWeakRef:
    def __init__(self, obj: Any) -> None:
        if obj is None:
            self._wr = None
        else:
            self._wr = weakref.ref(obj)

    # ... rest of the code ...

def build_lenient_weakvaluedict(d: dict) -> dict:
    # ... rest of the code ...

def unpack_lenient_weakvaluedict(d: dict) -> dict:
    # ... rest of the code ...

def default_ignored_types() -> tuple:
    # ... rest of the code ...
