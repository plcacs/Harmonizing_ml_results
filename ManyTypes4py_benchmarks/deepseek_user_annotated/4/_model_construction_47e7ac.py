"""Private logic for creating models."""

from __future__ import annotations as _annotations

import builtins
import operator
import sys
import typing
import warnings
import weakref
from abc import ABCMeta
from functools import cache, partial, wraps
from types import FunctionType
from typing import Any, Callable, Dict, Generic, Literal, NoReturn, Optional, Set, Tuple, Type, TypeVar, Union, cast

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
from ._typing_extra import (
    _make_forward_ref,
    eval_type_backport,
    is_annotated,
    is_classvar_annotation,
    parent_frame_namespace,
)
from ._utils import LazyClassAttribute, SafeGetItemProxy

if typing.TYPE_CHECKING:
    from ..fields import Field as PydanticModelField
    from ..fields import FieldInfo, ModelPrivateAttr
    from ..fields import PrivateAttr as PydanticModelPrivateAttr
    from ..main import BaseModel
else:
    # See PyCharm issues https://youtrack.jetbrains.com/issue/PY-21915
    # and https://youtrack.jetbrains.com/issue/PY-51428
    DeprecationWarning = PydanticDeprecatedSince20
    PydanticModelField = object()
    PydanticModelPrivateAttr = object()

object_setattr = object.__setattr__


class _ModelNamespaceDict(dict):
    """A dictionary subclass that intercepts attribute setting on model classes and
    warns about overriding of decorators.
    """

    def __setitem__(self, k: str, v: object) -> None:
        existing: Any = self.get(k, None)
        if existing and v is not existing and isinstance(existing, PydanticDescriptorProxy):
            warnings.warn(f'`{k}` overrides an existing Pydantic `{existing.decorator_info.decorator_repr}` decorator')

        return super().__setitem__(k, v)


def NoInitField(
    *,
    init: Literal[False] = False,
) -> Any:
    """Only for typing purposes. Used as default value of `__pydantic_fields_set__`,
    `__pydantic_extra__`, `__pydantic_private__`, so they could be ignored when
    synthesizing the `__init__` signature.
    """


@dataclass_transform(kw_only_default=True, field_specifiers=(PydanticModelField, PydanticModelPrivateAttr, NoInitField))
class ModelMetaclass(ABCMeta):
    def __new__(
        mcs,
        cls_name: str,
        bases: Tuple[Type[Any], ...],
        namespace: Dict[str, Any],
        __pydantic_generic_metadata__: Optional[PydanticGenericMetadata] = None,
        __pydantic_reset_parent_namespace__: bool = True,
        _create_model_module: Optional[str] = None,
        **kwargs: Any,
    ) -> Type:
        # ... [rest of the __new__ method implementation remains the same]
        pass

    if not typing.TYPE_CHECKING:  # pragma: no branch
        def __getattr__(self, item: str) -> Any:
            """This is necessary to keep attribute access working for class attribute access."""
            private_attributes = self.__dict__.get('__private_attributes__')
            if private_attributes and item in private_attributes:
                return private_attributes[item]
            raise AttributeError(item)

    @classmethod
    def __prepare__(cls, *args: Any, **kwargs: Any) -> Dict[str, object]:
        return _ModelNamespaceDict()

    def __instancecheck__(self, instance: Any) -> bool:
        """Avoid calling ABC _abc_instancecheck unless we're pretty sure."""
        return hasattr(instance, '__pydantic_decorators__') and super().__instancecheck__(instance)

    def __subclasscheck__(self, subclass: Type[Any]) -> bool:
        """Avoid calling ABC _abc_subclasscheck unless we're pretty sure."""
        return hasattr(subclass, '__pydantic_decorators__') and super().__subclasscheck__(subclass)

    @staticmethod
    def _collect_bases_data(bases: Tuple[Type[Any], ...]) -> Tuple[Set[str], Set[str], Dict[str, 'ModelPrivateAttr']]:
        # ... [implementation remains the same]
        pass

    @property
    @deprecated('The `__fields__` attribute is deprecated, use `model_fields` instead.', category=None)
    def __fields__(self) -> Dict[str, 'FieldInfo']:
        warnings.warn(
            'The `__fields__` attribute is deprecated, use `model_fields` instead.',
            PydanticDeprecatedSince20,
            stacklevel=2,
        )
        return getattr(self, '__pydantic_fields__', {})

    @property
    def __pydantic_fields_complete__(self) -> bool:
        """Whether the fields where successfully collected (i.e. type hints were successfully resolves)."""
        if not hasattr(self, '__pydantic_fields__'):
            return False

        field_infos = cast('Dict[str, FieldInfo]', self.__pydantic_fields__)
        return all(field_info._complete for field_info in field_infos.values())

    def __dir__(self) -> List[str]:
        attributes = list(super().__dir__())
        if '__fields__' in attributes:
            attributes.remove('__fields__')
        return attributes


def init_private_attributes(self: 'BaseModel', context: Any, /) -> None:
    """Initialize private attributes."""
    # ... [implementation remains the same]
    pass


def get_model_post_init(namespace: Dict[str, Any], bases: Tuple[Type[Any], ...]) -> Optional[Callable[..., Any]]:
    """Get the `model_post_init` method."""
    # ... [implementation remains the same]
    pass


def inspect_namespace(
    namespace: Dict[str, Any],
    ignored_types: Tuple[Type[Any], ...],
    base_class_vars: Set[str],
    base_class_fields: Set[str],
) -> Dict[str, 'ModelPrivateAttr']:
    """Inspect the namespace for private attributes."""
    # ... [implementation remains the same]
    pass


def set_default_hash_func(cls: Type['BaseModel'], bases: Tuple[Type[Any], ...]) -> None:
    """Set default hash function for frozen models."""
    # ... [implementation remains the same]
    pass


def make_hash_func(cls: Type['BaseModel']) -> Callable[[Any], int]:
    """Create a hash function for the model."""
    # ... [implementation remains the same]
    pass


def set_model_fields(
    cls: Type['BaseModel'],
    config_wrapper: ConfigWrapper,
    ns_resolver: Optional[NsResolver],
) -> None:
    """Collect and set model fields."""
    # ... [implementation remains the same]
    pass


def complete_model_class(
    cls: Type['BaseModel'],
    config_wrapper: ConfigWrapper,
    *,
    raise_errors: bool = True,
    ns_resolver: Optional[NsResolver] = None,
    create_model_module: Optional[str] = None,
) -> bool:
    """Finish building a model class."""
    # ... [implementation remains the same]
    pass


def set_deprecated_descriptors(cls: Type['BaseModel']) -> None:
    """Set data descriptors on the class for deprecated fields."""
    # ... [implementation remains the same]
    pass


class _DeprecatedFieldDescriptor:
    """Descriptor for deprecated fields."""
    field_name: str

    def __init__(self, msg: str, wrapped_property: Optional[property] = None) -> None:
        self.msg = msg
        self.wrapped_property = wrapped_property

    def __set_name__(self, cls: Type['BaseModel'], name: str) -> None:
        self.field_name = name

    def __get__(self, obj: Optional['BaseModel'], obj_type: Optional[Type['BaseModel']] = None) -> Any:
        # ... [implementation remains the same]
        pass

    def __set__(self, obj: Any, value: Any) -> NoReturn:
        raise AttributeError(self.field_name)


class _PydanticWeakRef:
    """Wrapper for weakref.ref that enables pickle serialization."""
    def __init__(self, obj: Any) -> None:
        # ... [implementation remains the same]
        pass

    def __call__(self) -> Any:
        # ... [implementation remains the same]
        pass

    def __reduce__(self) -> Tuple[Callable, Tuple[Optional[weakref.ReferenceType]]]:
        return _PydanticWeakRef, (self(),)


def build_lenient_weakvaluedict(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Create a dictionary with weakref values."""
    # ... [implementation remains the same]
    pass


def unpack_lenient_weakvaluedict(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Unpack a dictionary with weakref values."""
    # ... [implementation remains the same]
    pass


@cache
def default_ignored_types() -> Tuple[Type[Any], ...]:
    """Get default ignored types."""
    # ... [implementation remains the same]
    pass
