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
from typing import Any, Callable, Generic, Literal, NoReturn, cast, Dict, Set, Tuple, Type, Optional, Union, TypeVar, ClassVar
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
    """A dictionary subclass that intercepts attribute setting on model classes and
    warns about overriding of decorators.
    """

    def __setitem__(self, k: str, v: Any) -> None:
        existing = self.get(k, None)
        if existing and v is not existing and isinstance(existing, PydanticDescriptorProxy):
            warnings.warn(f'`{k}` overrides an existing Pydantic `{existing.decorator_info.decorator_repr}` decorator')
        return super().__setitem__(k, v)

def NoInitField(*, init: bool = False) -> Any:
    """Only for typing purposes. Used as default value of `__pydantic_fields_set__`,
    `__pydantic_extra__`, `__pydantic_private__`, so they could be ignored when
    synthesizing the `__init__` signature.
    """

@dataclass_transform(kw_only_default=True, field_specifiers=(PydanticModelField, PydanticModelPrivateAttr, NoInitField))
class ModelMetaclass(ABCMeta):

    def __new__(
        mcs: Type[ModelMetaclass], 
        cls_name: str, 
        bases: Tuple[Type, ...], 
        namespace: Dict[str, Any], 
        __pydantic_generic_metadata__: Optional[Dict[str, Any]] = None, 
        __pydantic_reset_parent_namespace__: bool = True, 
        _create_model_module: Optional[str] = None, 
        **kwargs: Any
    ) -> Type:
        """Metaclass for creating Pydantic models.

        Args:
            cls_name: The name of the class to be created.
            bases: The base classes of the class to be created.
            namespace: The attribute dictionary of the class to be created.
            __pydantic_generic_metadata__: Metadata for generic models.
            __pydantic_reset_parent_namespace__: Reset parent namespace.
            _create_model_module: The module of the class to be created, if created by `create_model`.
            **kwargs: Catch-all for any other keyword arguments.

        Returns:
            The new class created by the metaclass.
        """
        if bases:
            base_field_names, class_vars, base_private_attributes = mcs._collect_bases_data(bases)
            config_wrapper = ConfigWrapper.for_model(bases, namespace, kwargs)
            namespace['model_config'] = config_wrapper.config_dict
            private_attributes = inspect_namespace(namespace, config_wrapper.ignored_types, class_vars, base_field_names)
            if private_attributes or base_private_attributes:
                original_model_post_init = get_model_post_init(namespace, bases)
                if original_model_post_init is not None:

                    @wraps(original_model_post_init)
                    def wrapped_model_post_init(self: Any, context: Any, /) -> None:
                        """We need to both initialize private attributes and call the user-defined model_post_init
                        method.
                        """
                        init_private_attributes(self, context)
                        original_model_post_init(self, context)
                    namespace['model_post_init'] = wrapped_model_post_init
                else:
                    namespace['model_post_init'] = init_private_attributes
            namespace['__class_vars__'] = class_vars
            namespace['__private_attributes__'] = {**base_private_attributes, **private_attributes}
            cls = cast('type[BaseModel]', super().__new__(mcs, cls_name, bases, namespace, **kwargs))
            BaseModel_ = import_cached_base_model()
            mro = cls.__mro__
            if Generic in mro and mro.index(Generic) < mro.index(BaseModel_):
                warnings.warn(GenericBeforeBaseModelWarning('Classes should inherit from `BaseModel` before generic classes (e.g. `typing.Generic[T]`) for pydantic generics to work properly.'), stacklevel=2)
            cls.__pydantic_custom_init__ = not getattr(cls.__init__, '__pydantic_base_init__', False)
            cls.__pydantic_post_init__ = None if cls.model_post_init is BaseModel_.model_post_init else 'model_post_init'
            cls.__pydantic_setattr_handlers__ = {}
            cls.__pydantic_decorators__ = DecoratorInfos.build(cls)
            if __pydantic_generic_metadata__:
                cls.__pydantic_generic_metadata__ = __pydantic_generic_metadata__
            else:
                parent_parameters = getattr(cls, '__pydantic_generic_metadata__', {}).get('parameters', ())
                parameters = getattr(cls, '__parameters__', None) or parent_parameters
                if parameters and parent_parameters and (not all((x in parameters for x in parent_parameters))):
                    from ..root_model import RootModelRootType
                    missing_parameters = tuple((x for x in parameters if x not in parent_parameters))
                    if RootModelRootType in parent_parameters and RootModelRootType not in parameters:
                        parameters_str = ', '.join([x.__name__ for x in missing_parameters])
                        error_message = f'{cls.__name__} is a subclass of `RootModel`, but does not include the generic type identifier(s) {parameters_str} in its parameters. You should parametrize RootModel directly, e.g., `class {cls.__name__}(RootModel[{parameters_str}]): ...`.'
                    else:
                        combined_parameters = parent_parameters + missing_parameters
                        parameters_str = ', '.join([str(x) for x in combined_parameters])
                        generic_type_label = f'typing.Generic[{parameters_str}]'
                        error_message = f'All parameters must be present on typing.Generic; you should inherit from {generic_type_label}.'
                        if Generic not in bases:
                            bases_str = ', '.join([x.__name__ for x in bases] + [generic_type_label])
                            error_message += f' Note: `typing.Generic` must go last: `class {cls.__name__}({bases_str}): ...`)'
                    raise TypeError(error_message)
                cls.__pydantic_generic_metadata__ = {'origin': None, 'args': (), 'parameters': parameters}
            cls.__pydantic_complete__ = False
            for name, obj in private_attributes.items():
                obj.__set_name__(cls, name)
            if __pydantic_reset_parent_namespace__:
                cls.__pydantic_parent_namespace__ = build_lenient_weakvaluedict(parent_frame_namespace())
            parent_namespace = getattr(cls, '__pydantic_parent_namespace__', None)
            if isinstance(parent_namespace, dict):
                parent_namespace = unpack_lenient_weakvaluedict(parent_namespace)
            ns_resolver = NsResolver(parent_namespace=parent_namespace)
            set_model_fields(cls, config_wrapper=config_wrapper, ns_resolver=ns_resolver)
            cls.__pydantic_computed_fields__ = {k: v.info for k, v in cls.__pydantic_decorators__.computed_fields.items()}
            if config_wrapper.defer_build:
                set_model_mocks(cls)
            else:
                complete_model_class(cls, config_wrapper, raise_errors=False, ns_resolver=ns_resolver, create_model_module=_create_model_module)
            if config_wrapper.frozen and '__hash__' not in namespace:
                set_default_hash_func(cls, bases)
            super(cls, cls).__pydantic_init_subclass__(**kwargs)
            return cls
        else:
            for instance_slot in ('__pydantic_fields_set__', '__pydantic_extra__', '__pydantic_private__'):
                namespace.pop(instance_slot, None)
            namespace.get('__annotations__', {}).clear()
            return super().__new__(mcs, cls_name, bases, namespace, **kwargs)
    if not typing.TYPE_CHECKING:

        def __getattr__(self, item: str) -> Any:
            """This is necessary to keep attribute access working for class attribute access."""
            private_attributes = self.__dict__.get('__private_attributes__')
            if private_attributes and item in private_attributes:
                return private_attributes[item]
            raise AttributeError(item)

    @classmethod
    def __prepare__(cls, *args: Any, **kwargs: Any) -> _ModelNamespaceDict:
        return _ModelNamespaceDict()

    def __instancecheck__(self, instance: Any) -> bool:
        """Avoid calling ABC _abc_instancecheck unless we're pretty sure.

        See #3829 and python/cpython#92810
        """
        return hasattr(instance, '__pydantic_decorators__') and super().__instancecheck__(instance)

    def __subclasscheck__(self, subclass: Any) -> bool:
        """Avoid calling ABC _abc_subclasscheck unless we're pretty sure.

        See #3829 and python/cpython#92810
        """
        return hasattr(subclass, '__pydantic_decorators__') and super().__subclasscheck__(subclass)

    @staticmethod
    def _collect_bases_data(bases: Tuple[Type, ...]) -> Tuple[Set[str], Set[str], Dict[str, Any]]:
        BaseModel = import_cached_base_model()
        field_names = set()
        class_vars = set()
        private_attributes = {}
        for base in bases:
            if issubclass(base, BaseModel) and base is not BaseModel:
                field_names.update(getattr(base, '__pydantic_fields__', {}).keys())
                class_vars.update(base.__class_vars__)
                private_attributes.update(base.__private_attributes__)
        return (field_names, class_vars, private_attributes)

    @property
    @deprecated('The `__fields__` attribute is deprecated, use `model_fields` instead.', category=None)
    def __fields__(self) -> Dict[str, Any]:
        warnings.warn('The `__fields__` attribute is deprecated, use `model_fields` instead.', PydanticDeprecatedSince20, stacklevel=2)
        return getattr(self, '__pydantic_fields__', {})

    @property
    def __pydantic_fields_complete__(self) -> bool:
        """Whether the fields where successfully collected (i.e. type hints were successfully resolves).

        This is a private attribute, not meant to be used outside Pydantic.
        """
        if not hasattr(self, '__pydantic_fields__'):
            return False
        field_infos = cast('dict[str, FieldInfo]', self.__pydantic_fields__)
        return all((field_info._complete for field_info in field_infos.values()))

    def __dir__(self) -> List[str]:
        attributes = list(super().__dir__())
        if '__fields__' in attributes:
            attributes.remove('__fields__')
        return attributes

def init_private_attributes(self: Any, context: Any, /) -> None:
    """This function is meant to behave like a BaseModel method to initialise private attributes.

    It takes context as an argument since that's what pydantic-core passes when calling it.

    Args:
        self: The BaseModel instance.
        context: The context.
    """
    if getattr(self, '__pydantic_private__', None) is None:
        pydantic_private = {}
        for name, private_attr in self.__private_attributes__.items():
            default = private_attr.get_default()
            if default is not PydanticUndefined:
                pydantic_private[name] = default
        object_setattr(self, '__pydantic_private__', pydantic_private)

def get_model_post_init(namespace: Dict[str, Any], bases: Tuple[Type, ...]) -> Optional[Callable]:
    """Get the `model_post_init` method from the namespace or the class bases, or `None` if not defined."""
    if 'model_post_init' in namespace:
        return namespace['model_post_init']
    BaseModel = import_cached_base_model()
    model_post_init = get_attribute_from_bases(bases, 'model_post_init')
    if model_post_init is not BaseModel.model_post_init:
        return model_post_init
    return None

def inspect_namespace(
    namespace: Dict[str, Any], 
    ignored_types: Tuple[Type, ...], 
    base_class_vars: Set[str], 
    base_class_fields: Set[str]
) -> Dict[str, Any]:
    """Iterate over the namespace and:
    * gather private attributes
    * check for items which look like fields but are not (e.g. have no annotation) and warn.

    Args:
        namespace: The attribute dictionary of the class to be created.
        ignored_types: A tuple of ignore types.
        base_class_vars: A set of base class class variables.
        base_class_fields: A set of base class fields.

    Returns:
        A dict contains private attributes info.

    Raises:
        TypeError: If there is a `__root__` field in model.
        NameError: If private attribute name is invalid.
        PydanticUserError:
            - If a field does not have a type annotation.
            - If a field on base class was overridden by a non-annotated attribute.
    """
    from ..fields import ModelPrivateAttr, PrivateAttr
    FieldInfo = import_cached_field_info()
    all_ignored_types = ignored_types + default_ignored_types()
    private_attributes: Dict[str, Any] = {}
    raw_annotations = namespace.get('__annotations__', {})
    if '__root__' in raw_annotations or '__root__' in namespace:
        raise TypeError("To define root models, use `pydantic.RootModel` rather than a field called '__root__'")
    ignored_names: Set[str] = set()
    for var_name, value in list(namespace.items()):
        if var_name == 'model_config' or var_name == '__pydantic_extra__':
            continue
        elif isinstance(value, type) and value.__module__ == namespace['__module__'] and ('__qualname__' in namespace) and value.__qualname__.startswith(namespace['__qualname__']):
            continue
        elif isinstance(value, all_ignored_types) or value.__class__.__module__ == 'functools':
            ignored_names.add(var_name)
            continue
        elif isinstance(value, ModelPrivateAttr):
            if var_name.startswith('__'):
                raise NameError(f'Private attributes must not use dunder names; use a single underscore prefix instead of {var_name!r}.')
            elif is_valid_field_name(var_name):
                raise NameError(f'Private attributes must not use valid field names; use sunder names, e.g. {'_' + var_name!r} instead of {var_name!r}.')
            private_attributes[var_name] = value
            del namespace[var_name]
        elif isinstance(value, FieldInfo) and (not is_valid_field_name(var_name)):
            suggested_name = var_name.lstrip('_') or 'my_field'
            raise NameError(f'Fields must not use names with leading underscores; e.g., use {suggested_name!r} instead of {var_name!r}.')
        elif var_name.startswith('__'):
            continue
        elif is_valid_privateattr_name(var_name):
            if var_name not in raw_annotations or not is_classvar_annotation(raw_annotations[var_name]):
                private_attributes[var_name] = cast(ModelPrivateAttr, PrivateAttr(default=value))
                del namespace[var_name]
        elif var_name in base_class_vars:
            continue
        elif var_name not in raw_annotations:
            if var_name in base_class_fields:
                raise PydanticUserError(f'Field {var_name!r} defined on a base class was overridden by a non-annotated attribute. All field definitions, including overrides, require a type annotation.', code='model-field-overridden')
            elif isinstance(value, FieldInfo):
                raise PydanticUserError(f'Field {var_name!r} requires a type annotation', code='model-field-missing-annotation')
            else:
                raise PydanticUserError(f"A non-annotated attribute was detected: `{var_name} = {value!r}`. All model fields require a type annotation; if `{var_name}` is not meant to be a field, you may be able to resolve this error by annotating it as a `ClassVar` or updating `model_config['ignored_types']`.", code='model-field-missing-annotation')
    for ann_name, ann_type in raw_annotations.items():
        if is_valid_privateattr_name(ann_name) and ann_name not in private_attributes and (ann_name not in ignored_names) and (not is_classvar_annotation(ann_type)) and (ann_type not in all_ignored_types) and (getattr(ann_type, '__module__', None) != 'functools'):
            if isinstance(ann_type, str):
                frame = sys._getframe(2)
                if frame is not None:
                    try:
                        ann_type = eval_type_backport(_make_forward_ref(ann_type, is_argument=False, is_class=True), globalns=frame.f_globals, localns=frame.f_locals)
                    except (NameError, TypeError):
                        pass
            if is_annotated(ann_type):
                _, *metadata = get_args(ann_type)
                private_attr = next((v for v in metadata if isinstance(v, ModelPrivateAttr)), None)
                if private_attr is not None:
                    private_attributes[ann_name] = private_attr
                    continue
            private_attributes[ann_name] = PrivateAttr()
    return private_attributes

def set_default_hash_func(cls: Type, bases: Tuple[Type, ...]) -> None:
    base_hash_func = get_attribute_from_bases(bases, '__hash__')
    new_hash_func = make_hash_func(cls)
    if base_hash_func in {None, object.__hash__} or getattr(base_hash_func, '__code__', None) == new_hash_func.__code__:
        cls.__hash__ = new_hash_func

def make_hash_func(cls: Type) -> Callable[[Any], int]:
    getter = operator.itemgetter(*cls.__pydantic_fields__.keys()) if cls.__pydantic_fields__ else lambda _: 0

    def hash_func(self: Any) -> int:
        try:
            return hash(getter(self.__dict__))
        except KeyError:
            return hash(getter(SafeGetItemProxy(self.__dict__)))
    return hash_func

def set_model_fields(cls: Type, config_wrapper: ConfigWrapper, ns_resolver: NsResolver) -> None:
    """Collect and set `cls.__pydantic_fields__` and `cls.__class_vars__`.

    Args:
        cls: BaseModel or dataclass.
        config_wrapper: The config wrapper instance.
        ns_resolver: Namespace resolver to use when getting model annotations.
    """
    typevars_map = get_model_typevars_map(cls)
    fields, class_vars = collect_model_fields(cls, config_wrapper, ns_resolver, typevars_map=typevars_map)
    cls.__pydantic_fields__ = fields
    cls.__class_vars__.update(class_vars)
    for k in class_vars:
        value = cls.__private_attributes__.pop(k, None)
        if value is not None and value.default is not PydanticUndefined:
            setattr(cls, k, value.default)

def complete_model_class(
    cls: Type, 
    config_wrapper: ConfigWrapper, 
    *, 
    raise_errors: bool = True, 
    ns_resolver: Optional[NsResolver] = None, 
    create_model_module: Optional[str] = None
) -> bool:
    """Finish building a model class.

    This logic must be called after class has been created since validation functions must be bound
    and `get_type_hints` requires a class object.

    Args:
        cls: BaseModel or dataclass.
        config_wrapper: The config wrapper instance.
        raise_errors: Whether to raise errors.
        ns_resolver: The namespace resolver instance to use during schema building.
        create_model_module: The module of the class to be created, if created by `create_model`.

    Returns:
        `True` if the model is successfully completed, else `False`.

    Raises:
        PydanticUndefinedAnnotation: If `PydanticUndefinedAnnotation` occurs in`__get_pydantic_core_schema__`
            and `raise_errors=True`.
    """
    typevars_map = get_model_typevars_map(cls)
    gen_schema = GenerateSchema(config_wrapper, ns_resolver, typevars_map)
    try:
        schema = gen_schema.generate_schema(cls)
    except PydanticUndefinedAnnotation as e:
        if raise_errors:
            raise
        set_model_mocks(cls, f'`{e.name}`')
        return False
    core_config = config_wrapper.core_config(title=cls.__name__)
    try:
        schema = gen_schema.clean_schema(schema)
    except InvalidSchemaError:
        set_model_mocks(cls)
        return False
    cls.__pydantic_computed_fields__ = {k: v.info for k, v in cls.__pydantic_decorators__.computed_fields.items()}
    set_deprecated_descriptors(cls)
    cls.__pydantic_core_schema__ = schema
    cls.__pydantic_validator__ = create_schema_validator(schema, cls, create_model_module or cls.__module__, cls.__qualname__, 'create_model' if create_model_module else 'BaseModel', core_config, config_wrapper.plugin_settings)
    cls.__pydantic_serializer__ = SchemaSerializer(schema, core_config)
    cls.__pydantic_complete__ = True
    cls.__signature__ = LazyClassAttribute('__signature__', partial(generate_pydantic_signature, init=cls.__init__, fields=cls.__pydantic_fields__, populate_by_name=config_wrapper.populate_by_name, extra=config_wrapper.extra))
    return True

def set_deprecated_descriptors(cls: Type) -> None:
    """Set data descriptors on the class for deprecated fields."""
    for field, field_info in cls.__pydantic_fields__.items():
        if (msg := field_info.deprecation_message) is not None:
            desc = _DeprecatedFieldDescriptor(msg)
            desc.__set_name__(cls, field)
            setattr(cls, field, desc)
    for field, computed_field_info in cls.__pydantic_computed_fields__.items():
        if (msg := computed_field_info.deprecation_message) is not None and (not hasattr(unwrap_wrapped_function(computed_field_info.wrapped_property), '__deprecated__')):
            desc = _DeprecatedFieldDescriptor(msg, computed_field_info.wrapped_property)
            desc.__set_name__(cls, field)
            setattr(cls, field, desc)

class _DeprecatedFieldDescriptor:
    """Read-only data descriptor used to emit a runtime deprecation warning before accessing a deprecated field.

    Attributes:
        msg: The deprecation message to be emitted.
        wrapped_property: The property instance if the deprecated field is a computed field, or `None`.
        field_name: The name of the field being deprecated.
    """

    def __init__(self, msg: str, wrapped_property: Optional[property] = None):
        self.msg = msg
        self.wrapped_property = wrapped_property
        self.field_name: str = ""

    def __set_name__(self, cls: Type, name: str) -> None:
        self.field_name = name

    def __get__(self, obj: Any, obj_type: Optional[Type] = None) -> Any:
        if obj is None:
            if self.wrapped_property is not None:
                return self.wrapped_property.__get__(None, obj_type)
            raise AttributeError(self.field_name)
        warnings.warn(self.msg, builtins.DeprecationWarning, stacklevel=2)
        if self.wrapped_property is not None:
            return self.wrapped_property.__get__(obj, obj_type)
        return obj.__dict__[self.field_name]

    def __set__(self, obj: Any, value: Any) -> NoReturn:
        raise AttributeError(self.field_name)

class _PydanticWeakRef:
    """Wrapper for `weakref.ref` that enables `pickle` serialization.

    Cloudpickle fails to serialize `weakref.ref` objects due to an arcane error related
    to abstract base classes (`abc.ABC`). This class works around the issue by wrapping
    `weakref.ref` instead of subclassing it.

    See https://github.com/pydantic/pydantic/issues/6763 for context.

    Semantics:
        - If not pickled, behaves the same as a `weakref.ref`.
        - If pickled along with the referenced object, the same `weakref.ref` behavior
          will be maintained between them after unpickling.
        - If pickled without the referenced object, after unpickling the underlying
          reference will be cleared (`__call__` will always return `None`).
    """

    def __init__(self, obj: Optional[Any]):
        if obj is None:
            self._wr = None
        else:
            self._wr = weakref.ref(obj)

    def __call__(self) -> Optional[Any]:
        if self._wr is None:
            return None
        else:
            return self._wr()

    def __reduce__(self) -> Tuple[Type[_PydanticWeakRef], Tuple[Optional[Any]]]:
        return (_PydanticWeakRef, (self(),))

def build_lenient_weakvaluedict(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Takes an input dictionary, and produces a new value that (invertibly) replaces the values with weakrefs.

    We can't just use a WeakValueDictionary because many types (including int, str, etc.) can't be stored as values
    in a WeakValueDictionary.

    The `unpack_lenient_weakvaluedict` function can be used to reverse this operation.
    """
    if d is None:
        return None
    result: Dict[str, Any] = {}
    for k, v in d.items():
        try:
            proxy = _PydanticWeakRef(v)
        except TypeError:
            proxy = v
        result[k] = proxy
    return result

def unpack_lenient_weakvaluedict(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Inverts the transform performed by `build_lenient_weakvaluedict`."""
    if d is None:
        return None
    result: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, _PydanticWeakRef):
            v = v()
            if v is not None:
                result[k] = v
        else:
            result[k] = v
    return result

@cache
def default_ignored_types() -> Tuple[Type, ...]:
    from ..fields import ComputedFieldInfo
    ignored_types = [FunctionType, property, classmethod, staticmethod, PydanticDescriptorProxy, ComputedFieldInfo, TypeAliasType]
    if sys.version_info >= (3, 12):
        ignored_types.append(typing.TypeAliasType)
    return tuple(ignored_types)
