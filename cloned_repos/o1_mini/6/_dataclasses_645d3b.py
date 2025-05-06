"""Private logic for creating pydantic dataclasses."""
from __future__ import annotations as _annotations
import dataclasses
import typing
import warnings
from functools import partial, wraps
from typing import Any, ClassVar, Optional, Type, TypeVar

from pydantic_core import ArgsKwargs, SchemaSerializer, SchemaValidator, core_schema
from typing_extensions import TypeGuard

from ..errors import PydanticUndefinedAnnotation
from ..plugin._schema_validator import PluggableSchemaValidator, create_schema_validator
from ..warnings import PydanticDeprecatedSince20
from . import _config, _decorators
from ._fields import collect_dataclass_fields
from ._generate_schema import GenerateSchema, InvalidSchemaError
from ._generics import get_standard_typevars_map
from ._mock_val_ser import set_dataclass_mocks
from ._namespace_utils import NsResolver
from ._signature import generate_pydantic_signature
from ._utils import LazyClassAttribute

if typing.TYPE_CHECKING:
    from _typeshed import DataclassInstance as StandardDataclass
    from ..config import ConfigDict
    from ..fields import FieldInfo

    class PydanticDataclass(StandardDataclass, typing.Protocol):
        """A protocol containing attributes only available once a class has been decorated as a Pydantic dataclass.

        Attributes:
            __pydantic_config__: Pydantic-specific configuration settings for the dataclass.
            __pydantic_complete__: Whether dataclass building is completed, or if there are still undefined fields.
            __pydantic_core_schema__: The pydantic-core schema used to build the SchemaValidator and SchemaSerializer.
            __pydantic_decorators__: Metadata containing the decorators defined on the dataclass.
            __pydantic_fields__: Metadata about the fields defined on the dataclass.
            __pydantic_serializer__: The pydantic-core SchemaSerializer used to dump instances of the dataclass.
            __pydantic_validator__: The pydantic-core SchemaValidator used to validate instances of the dataclass.
        """
else:
    DeprecationWarning = PydanticDeprecatedSince20

def set_dataclass_fields(
    cls: Type[Any],
    ns_resolver: Optional[NsResolver] = None,
    config_wrapper: Optional[_config.ConfigWrapper] = None
) -> None:
    """Collect and set `cls.__pydantic_fields__`.

    Args:
        cls: The class.
        ns_resolver: Namespace resolver to use when getting dataclass annotations.
        config_wrapper: The config wrapper instance, defaults to `None`.
    """
    typevars_map = get_standard_typevars_map(cls)
    fields = collect_dataclass_fields(
        cls, ns_resolver=ns_resolver, typevars_map=typevars_map, config_wrapper=config_wrapper
    )
    cls.__pydantic_fields__ = fields

def complete_dataclass(
    cls: Type[Any],
    config_wrapper: _config.ConfigWrapper,
    *,
    raise_errors: bool = True,
    ns_resolver: Optional[NsResolver] = None,
    _force_build: bool = False
) -> bool:
    """Finish building a pydantic dataclass.

    This logic is called on a class which has already been wrapped in `dataclasses.dataclass()`.

    This is somewhat analogous to `pydantic._internal._model_construction.complete_model_class`.

    Args:
        cls: The class.
        config_wrapper: The config wrapper instance.
        raise_errors: Whether to raise errors, defaults to `True`.
        ns_resolver: The namespace resolver instance to use when collecting dataclass fields
            and during schema building.
        _force_build: Whether to force building the dataclass, no matter if
            [`defer_build`][pydantic.config.ConfigDict.defer_build] is set.

    Returns:
        `True` if building a pydantic dataclass is successfully completed, `False` otherwise.

    Raises:
        PydanticUndefinedAnnotation: If `raise_error` is `True` and there is an undefined annotations.
    """
    original_init = cls.__init__

    def __init__(__dataclass_self__: Any, *args: Any, **kwargs: Any) -> None:
        __tracebackhide__ = True
        s = __dataclass_self__
        s.__pydantic_validator__.validate_python(ArgsKwargs(args, kwargs), self_instance=s)
    __init__.__qualname__ = f'{cls.__qualname__}.__init__'
    cls.__init__ = __init__
    cls.__pydantic_config__ = config_wrapper.config_dict
    set_dataclass_fields(cls, ns_resolver, config_wrapper=config_wrapper)
    if not _force_build and config_wrapper.defer_build:
        set_dataclass_mocks(cls)
        return False
    if hasattr(cls, '__post_init_post_parse__'):
        warnings.warn(
            'Support for `__post_init_post_parse__` has been dropped, the method will not be called',
            DeprecationWarning
        )
    typevars_map = get_standard_typevars_map(cls)
    gen_schema = GenerateSchema(config_wrapper, ns_resolver=ns_resolver, typevars_map=typevars_map)
    cls.__signature__ = LazyClassAttribute(
        '__signature__',
        partial(
            generate_pydantic_signature,
            init=original_init,
            fields=cls.__pydantic_fields__,
            populate_by_name=config_wrapper.populate_by_name,
            extra=config_wrapper.extra,
            is_dataclass=True
        )
    )
    try:
        schema = gen_schema.generate_schema(cls)
    except PydanticUndefinedAnnotation as e:
        if raise_errors:
            raise
        set_dataclass_mocks(cls, f'`{e.name}`')
        return False
    core_config = config_wrapper.core_config(title=cls.__name__)
    try:
        schema = gen_schema.clean_schema(schema)
    except InvalidSchemaError:
        set_dataclass_mocks(cls)
        return False
    cls = typing.cast('Type[PydanticDataclass]', cls)
    cls.__pydantic_core_schema__ = schema
    cls.__pydantic_validator__ = validator = create_schema_validator(
        schema,
        cls,
        cls.__module__,
        cls.__qualname__,
        'dataclass',
        core_config,
        config_wrapper.plugin_settings
    )
    cls.__pydantic_serializer__ = SchemaSerializer(schema, core_config)
    if config_wrapper.validate_assignment:

        @wraps(cls.__setattr__)
        def validated_setattr(instance: Any, field: str, value: Any, /) -> None:
            validator.validate_assignment(instance, field, value)
        cls.__setattr__ = validated_setattr.__get__(None, cls)
    cls.__pydantic_complete__ = True
    return True

def is_builtin_dataclass(_cls: Type[Any]) -> bool:
    """Returns True if a class is a stdlib dataclass and *not* a pydantic dataclass.

    We check that
    - `_cls` is a dataclass
    - `_cls` does not inherit from a processed pydantic dataclass (and thus have a `__pydantic_validator__`)
    - `_cls` does not have any annotations that are not dataclass fields
    e.g.
    