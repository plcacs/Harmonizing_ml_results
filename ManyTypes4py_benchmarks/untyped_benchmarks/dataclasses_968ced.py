"""Provide an enhanced dataclass that performs validation."""
from __future__ import annotations as _annotations
import dataclasses
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, NoReturn, TypeVar, overload
from warnings import warn
from typing_extensions import TypeGuard, dataclass_transform
from ._internal import _config, _decorators, _namespace_utils, _typing_extra
from ._internal import _dataclasses as _pydantic_dataclasses
from ._migration import getattr_migration
from .config import ConfigDict
from .errors import PydanticUserError
from .fields import Field, FieldInfo, PrivateAttr
if TYPE_CHECKING:
    from ._internal._dataclasses import PydanticDataclass
    from ._internal._namespace_utils import MappingNamespace
__all__ = ('dataclass', 'rebuild_dataclass')
_T = TypeVar('_T')
if sys.version_info >= (3, 10):

    @dataclass_transform(field_specifiers=(dataclasses.field, Field, PrivateAttr))
    @overload
    def dataclass(*, init=False, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, config=None, validate_on_init=None, kw_only=..., slots=...):
        ...

    @dataclass_transform(field_specifiers=(dataclasses.field, Field, PrivateAttr))
    @overload
    def dataclass(_cls, *, init=False, repr=True, eq=True, order=False, unsafe_hash=False, frozen=None, config=None, validate_on_init=None, kw_only=..., slots=...):
        ...
else:

    @dataclass_transform(field_specifiers=(dataclasses.field, Field, PrivateAttr))
    @overload
    def dataclass(*, init=False, repr=True, eq=True, order=False, unsafe_hash=False, frozen=None, config=None, validate_on_init=None):
        ...

    @dataclass_transform(field_specifiers=(dataclasses.field, Field, PrivateAttr))
    @overload
    def dataclass(_cls, *, init=False, repr=True, eq=True, order=False, unsafe_hash=False, frozen=None, config=None, validate_on_init=None):
        ...

@dataclass_transform(field_specifiers=(dataclasses.field, Field, PrivateAttr))
def dataclass(_cls=None, *, init=False, repr=True, eq=True, order=False, unsafe_hash=False, frozen=None, config=None, validate_on_init=None, kw_only=False, slots=False):
    """!!! abstract "Usage Documentation"
        [`dataclasses`](../concepts/dataclasses.md)

    A decorator used to create a Pydantic-enhanced dataclass, similar to the standard Python `dataclass`,
    but with added validation.

    This function should be used similarly to `dataclasses.dataclass`.

    Args:
        _cls: The target `dataclass`.
        init: Included for signature compatibility with `dataclasses.dataclass`, and is passed through to
            `dataclasses.dataclass` when appropriate. If specified, must be set to `False`, as pydantic inserts its
            own  `__init__` function.
        repr: A boolean indicating whether to include the field in the `__repr__` output.
        eq: Determines if a `__eq__` method should be generated for the class.
        order: Determines if comparison magic methods should be generated, such as `__lt__`, but not `__eq__`.
        unsafe_hash: Determines if a `__hash__` method should be included in the class, as in `dataclasses.dataclass`.
        frozen: Determines if the generated class should be a 'frozen' `dataclass`, which does not allow its
            attributes to be modified after it has been initialized. If not set, the value from the provided `config` argument will be used (and will default to `False` otherwise).
        config: The Pydantic config to use for the `dataclass`.
        validate_on_init: A deprecated parameter included for backwards compatibility; in V2, all Pydantic dataclasses
            are validated on init.
        kw_only: Determines if `__init__` method parameters must be specified by keyword only. Defaults to `False`.
        slots: Determines if the generated class should be a 'slots' `dataclass`, which does not allow the addition of
            new attributes after instantiation.

    Returns:
        A decorator that accepts a class as its argument and returns a Pydantic `dataclass`.

    Raises:
        AssertionError: Raised if `init` is not `False` or `validate_on_init` is `False`.
    """
    assert init is False, 'pydantic.dataclasses.dataclass only supports init=False'
    assert validate_on_init is not False, 'validate_on_init=False is no longer supported'
    if sys.version_info >= (3, 10):
        kwargs = {'kw_only': kw_only, 'slots': slots}
    else:
        kwargs = {}

    def make_pydantic_fields_compatible(cls):
        """Make sure that stdlib `dataclasses` understands `Field` kwargs like `kw_only`
        To do that, we simply change
          `x: int = pydantic.Field(..., kw_only=True)`
        into
          `x: int = dataclasses.field(default=pydantic.Field(..., kw_only=True), kw_only=True)`
        """
        for annotation_cls in cls.__mro__:
            annotations = getattr(annotation_cls, '__annotations__', {})
            for field_name in annotations:
                field_value = getattr(cls, field_name, None)
                if not isinstance(field_value, FieldInfo):
                    continue
                field_args = {'default': field_value}
                if sys.version_info >= (3, 10) and field_value.kw_only:
                    field_args['kw_only'] = True
                if field_value.repr is not True:
                    field_args['repr'] = field_value.repr
                setattr(cls, field_name, dataclasses.field(**field_args))
                if cls.__dict__.get('__annotations__') is None:
                    cls.__annotations__ = {}
                cls.__annotations__[field_name] = annotations[field_name]

    def create_dataclass(cls):
        """Create a Pydantic dataclass from a regular dataclass.

        Args:
            cls: The class to create the Pydantic dataclass from.

        Returns:
            A Pydantic dataclass.
        """
        from ._internal._utils import is_model_class
        if is_model_class(cls):
            raise PydanticUserError(f'Cannot create a Pydantic dataclass from {cls.__name__} as it is already a Pydantic model', code='dataclass-on-model')
        original_cls = cls
        has_dataclass_base = any((dataclasses.is_dataclass(base) for base in cls.__bases__))
        if not has_dataclass_base and config is not None and hasattr(cls, '__pydantic_config__'):
            warn(f'`config` is set via both the `dataclass` decorator and `__pydantic_config__` for dataclass {cls.__name__}. The `config` specification from `dataclass` decorator will take priority.', category=UserWarning, stacklevel=2)
        config_dict = config if config is not None else getattr(cls, '__pydantic_config__', None)
        config_wrapper = _config.ConfigWrapper(config_dict)
        decorators = _decorators.DecoratorInfos.build(cls)
        original_doc = cls.__doc__
        if _pydantic_dataclasses.is_builtin_dataclass(cls):
            original_doc = None
            bases = (cls,)
            if issubclass(cls, Generic):
                generic_base = Generic[cls.__parameters__]
                bases = bases + (generic_base,)
            cls = types.new_class(cls.__name__, bases)
        make_pydantic_fields_compatible(cls)
        if frozen is not None:
            frozen_ = frozen
            if config_wrapper.frozen:
                warn(f'`frozen` is set via both the `dataclass` decorator and `config` for dataclass {cls.__name__!r}.This is not recommended. The `frozen` specification on `dataclass` will take priority.', category=UserWarning, stacklevel=2)
        else:
            frozen_ = config_wrapper.frozen or False
        cls = dataclasses.dataclass(cls, init=True, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, frozen=frozen_, **kwargs)
        cls.__is_pydantic_dataclass__ = True
        cls.__pydantic_decorators__ = decorators
        cls.__doc__ = original_doc
        cls.__module__ = original_cls.__module__
        cls.__qualname__ = original_cls.__qualname__
        cls.__pydantic_complete__ = False
        _pydantic_dataclasses.complete_dataclass(cls, config_wrapper, raise_errors=False)
        return cls
    return create_dataclass if _cls is None else create_dataclass(_cls)
__getattr__ = getattr_migration(__name__)
if sys.version_info < (3, 11):

    def _call_initvar(*args, **kwargs):
        """This function does nothing but raise an error that is as similar as possible to what you'd get
        if you were to try calling `InitVar[int]()` without this monkeypatch. The whole purpose is just
        to ensure typing._type_check does not error if the type hint evaluates to `InitVar[<parameter>]`.
        """
        raise TypeError("'InitVar' object is not callable")
    dataclasses.InitVar.__call__ = _call_initvar

def rebuild_dataclass(cls, *, force=False, raise_errors=True, _parent_namespace_depth=2, _types_namespace=None):
    """Try to rebuild the pydantic-core schema for the dataclass.

    This may be necessary when one of the annotations is a ForwardRef which could not be resolved during
    the initial attempt to build the schema, and automatic rebuilding fails.

    This is analogous to `BaseModel.model_rebuild`.

    Args:
        cls: The class to rebuild the pydantic-core schema for.
        force: Whether to force the rebuilding of the schema, defaults to `False`.
        raise_errors: Whether to raise errors, defaults to `True`.
        _parent_namespace_depth: The depth level of the parent namespace, defaults to 2.
        _types_namespace: The types namespace, defaults to `None`.

    Returns:
        Returns `None` if the schema is already "complete" and rebuilding was not required.
        If rebuilding _was_ required, returns `True` if rebuilding was successful, otherwise `False`.
    """
    if not force and cls.__pydantic_complete__:
        return None
    for attr in ('__pydantic_core_schema__', '__pydantic_validator__', '__pydantic_serializer__'):
        if attr in cls.__dict__:
            delattr(cls, attr)
    cls.__pydantic_complete__ = False
    if _types_namespace is not None:
        rebuild_ns = _types_namespace
    elif _parent_namespace_depth > 0:
        rebuild_ns = _typing_extra.parent_frame_namespace(parent_depth=_parent_namespace_depth, force=True) or {}
    else:
        rebuild_ns = {}
    ns_resolver = _namespace_utils.NsResolver(parent_namespace=rebuild_ns)
    return _pydantic_dataclasses.complete_dataclass(cls, _config.ConfigWrapper(cls.__pydantic_config__, check=False), raise_errors=raise_errors, ns_resolver=ns_resolver, _force_build=True)

def is_pydantic_dataclass(class_, /):
    """Whether a class is a pydantic dataclass.

    Args:
        class_: The class.

    Returns:
        `True` if the class is a pydantic dataclass, `False` otherwise.
    """
    try:
        return '__is_pydantic_dataclass__' in class_.__dict__ and dataclasses.is_dataclass(class_)
    except AttributeError:
        return False