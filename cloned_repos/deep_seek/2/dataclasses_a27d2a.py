from __future__ import annotations
import copy
import dataclasses
import sys
from contextlib import contextmanager
from functools import wraps
try:
    from functools import cached_property
except ImportError:
    pass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Generator, Optional, Type, TypeVar, Union, overload, cast
from typing_extensions import dataclass_transform
from pydantic.v1.class_validators import gather_all_validators
from pydantic.v1.config import BaseConfig, ConfigDict, Extra, get_config
from pydantic.v1.error_wrappers import ValidationError
from pydantic.v1.errors import DataclassTypeError
from pydantic.v1.fields import Field, FieldInfo, Required, Undefined
from pydantic.v1.main import create_model, validate_model
from pydantic.v1.utils import ClassAttribute

if TYPE_CHECKING:
    from pydantic.v1.main import BaseModel
    from pydantic.v1.typing import CallableGenerator, NoArgAnyCallable
    from typing import ContextManager

__all__ = ['dataclass', 'set_validation', 'create_pydantic_model_from_dataclass', 'is_builtin_dataclass', 'make_dataclass_validator']

_T = TypeVar('_T')
DataclassT = TypeVar('DataclassT', bound='Dataclass')
DataclassClassOrWrapper = Union[Type['Dataclass'], 'DataclassProxy']

class Dataclass:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    def __get_validators__(cls) -> Generator[Callable[..., Any], None, None]:
        pass

    @classmethod
    def __validate__(cls, v: Any) -> Any:
        pass

if sys.version_info >= (3, 10):
    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    @overload
    def dataclass(
        *,
        init: bool = True,
        repr: bool = True,
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        config: Optional[Union[ConfigDict, Type[BaseConfig]]] = None,
        validate_on_init: Optional[bool] = None,
        use_proxy: Optional[bool] = None,
        kw_only: bool = ...,
    ) -> Callable[[Type[_T]], Type[_T]]:
        ...

    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    @overload
    def dataclass(
        _cls: Type[_T],
        *,
        init: bool = True,
        repr: bool = True,
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        config: Optional[Union[ConfigDict, Type[BaseConfig]]] = None,
        validate_on_init: Optional[bool] = None,
        use_proxy: Optional[bool] = None,
        kw_only: bool = ...,
    ) -> Type[_T]:
        ...
else:
    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    @overload
    def dataclass(
        *,
        init: bool = True,
        repr: bool = True,
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        config: Optional[Union[ConfigDict, Type[BaseConfig]]] = None,
        validate_on_init: Optional[bool] = None,
        use_proxy: Optional[bool] = None,
    ) -> Callable[[Type[_T]], Type[_T]]:
        ...

    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    @overload
    def dataclass(
        _cls: Type[_T],
        *,
        init: bool = True,
        repr: bool = True,
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        config: Optional[Union[ConfigDict, Type[BaseConfig]]] = None,
        validate_on_init: Optional[bool] = None,
        use_proxy: Optional[bool] = None,
    ) -> Type[_T]:
        ...

@dataclass_transform(field_specifiers=(dataclasses.field, Field))
def dataclass(
    _cls: Optional[Type[_T]] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    config: Optional[Union[ConfigDict, Type[BaseConfig]]] = None,
    validate_on_init: Optional[bool] = None,
    use_proxy: Optional[bool] = None,
    kw_only: bool = False,
) -> Union[Type[_T], Callable[[Type[_T]], Type[_T]]]:
    the_config = get_config(config)

    def wrap(cls: Type[_T]) -> Type[_T]:
        should_use_proxy = (
            use_proxy if use_proxy is not None 
            else is_builtin_dataclass(cls) and 
                (cls.__bases__[0] is object or set(dir(cls)) == set(dir(cls.__bases__[0])))
        if should_use_proxy:
            dc_cls_doc = ''
            dc_cls = cast(Type[_T], DataclassProxy(cls))
            default_validate_on_init = False
        else:
            dc_cls_doc = cls.__doc__ or ''
            if sys.version_info >= (3, 10):
                dc_cls = dataclasses.dataclass(
                    cls, init=init, repr=repr, eq=eq, order=order,
                    unsafe_hash=unsafe_hash, frozen=frozen, kw_only=kw_only)
            else:
                dc_cls = dataclasses.dataclass(
                    cls, init=init, repr=repr, eq=eq, order=order,
                    unsafe_hash=unsafe_hash, frozen=frozen)
            default_validate_on_init = True
        should_validate_on_init = default_validate_on_init if validate_on_init is None else validate_on_init
        _add_pydantic_validation_attributes(cls, the_config, should_validate_on_init, dc_cls_doc)
        dc_cls.__pydantic_model__.__try_update_forward_refs__(**{cls.__name__: cls})
        return dc_cls

    if _cls is None:
        return wrap
    return wrap(_cls)

@contextmanager
def set_validation(cls: Type[Any], value: bool) -> Generator[Type[Any], None, None]:
    original_run_validation = cls.__pydantic_run_validation__
    try:
        cls.__pydantic_run_validation__ = value
        yield cls
    finally:
        cls.__pydantic_run_validation__ = original_run_validation

class DataclassProxy:
    __slots__ = '__dataclass__'

    def __init__(self, dc_cls: Type[Any]) -> None:
        object.__setattr__(self, '__dataclass__', dc_cls)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with set_validation(self.__dataclass__, True):
            return self.__dataclass__(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dataclass__, name)

    def __setattr__(self, name: str, value: Any) -> None:
        return setattr(self.__dataclass__, name, value)

    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, self.__dataclass__)

    def __copy__(self) -> DataclassProxy:
        return DataclassProxy(copy.copy(self.__dataclass__))

    def __deepcopy__(self, memo: Dict[int, Any]) -> DataclassProxy:
        return DataclassProxy(copy.deepcopy(self.__dataclass__, memo))

def _add_pydantic_validation_attributes(
    dc_cls: Type[Any],
    config: Union[ConfigDict, Type[BaseConfig]],
    validate_on_init: bool,
    dc_cls_doc: Optional[str],
) -> None:
    init = dc_cls.__init__

    @wraps(init)
    def handle_extra_init(self: Any, *args: Any, **kwargs: Any) -> None:
        if config.extra == Extra.ignore:
            init(self, *args, **{k: v for k, v in kwargs.items() if k in self.__dataclass_fields__})
        elif config.extra == Extra.allow:
            for k, v in kwargs.items():
                self.__dict__.setdefault(k, v)
            init(self, *args, **{k: v for k, v in kwargs.items() if k in self.__dataclass_fields__})
        else:
            init(self, *args, **kwargs)

    if hasattr(dc_cls, '__post_init__'):
        try:
            post_init = dc_cls.__post_init__.__wrapped__
        except AttributeError:
            post_init = dc_cls.__post_init__

        @wraps(post_init)
        def new_post_init(self: Any, *args: Any, **kwargs: Any) -> None:
            if config.post_init_call == 'before_validation':
                post_init(self, *args, **kwargs)
            if self.__class__.__pydantic_run_validation__:
                self.__pydantic_validate_values__()
                if hasattr(self, '__post_init_post_parse__'):
                    self.__post_init_post_parse__(*args, **kwargs)
            if config.post_init_call == 'after_validation':
                post_init(self, *args, **kwargs)

        setattr(dc_cls, '__init__', handle_extra_init)
        setattr(dc_cls, '__post_init__', new_post_init)
    else:
        @wraps(init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            handle_extra_init(self, *args, **kwargs)
            if self.__class__.__pydantic_run_validation__:
                self.__pydantic_validate_values__()
            if hasattr(self, '__post_init_post_parse__'):
                initvars_and_values = {}
                for i, f in enumerate(self.__class__.__dataclass_fields__.values()):
                    if f._field_type is dataclasses._FIELD_INITVAR:
                        try:
                            initvars_and_values[f.name] = args[i]
                        except IndexError:
                            initvars_and_values[f.name] = kwargs.get(f.name, f.default)
                self.__post_init_post_parse__(**initvars_and_values)

        setattr(dc_cls, '__init__', new_init)

    setattr(dc_cls, '__pydantic_run_validation__', ClassAttribute('__pydantic_run_validation__', validate_on_init))
    setattr(dc_cls, '__pydantic_initialised__', False)
    setattr(dc_cls, '__pydantic_model__', create_pydantic_model_from_dataclass(dc_cls, config, dc_cls_doc))
    setattr(dc_cls, '__pydantic_validate_values__', _dataclass_validate_values)
    setattr(dc_cls, '__validate__', classmethod(_validate_dataclass))
    setattr(dc_cls, '__get_validators__', classmethod(_get_validators))
    if dc_cls.__pydantic_model__.__config__.validate_assignment and (not dc_cls.__dataclass_params__.frozen):
        setattr(dc_cls, '__setattr__', _dataclass_validate_assignment_setattr)

def _get_validators(cls: Type[Any]) -> Generator[Callable[..., Any], None, None]:
    yield cls.__validate__

def _validate_dataclass(cls: Type[Any], v: Any) -> Any:
    with set_validation(cls, True):
        if isinstance(v, cls):
            v.__pydantic_validate_values__()
            return v
        elif isinstance(v, (list, tuple)):
            return cls(*v)
        elif isinstance(v, dict):
            return cls(**v)
        else:
            raise DataclassTypeError(class_name=cls.__name__)

def create_pydantic_model_from_dataclass(
    dc_cls: Type[Any],
    config: Union[ConfigDict, Type[BaseConfig]] = BaseConfig,
    dc_cls_doc: Optional[str] = None,
) -> Type[BaseModel]:
    field_definitions: Dict[str, Any] = {}
    for field in dataclasses.fields(dc_cls):
        default = Undefined
        default_factory = None
        if field.default is not dataclasses.MISSING:
            default = field.default
        elif field.default_factory is not dataclasses.MISSING:
            default_factory = field.default_factory
        else:
            default = Required

        if isinstance(default, FieldInfo):
            field_info = default
            dc_cls.__pydantic_has_field_info_default__ = True
        else:
            field_info = Field(default=default, default_factory=default_factory, **field.metadata)
        field_definitions[field.name] = (field.type, field_info)

    validators = gather_all_validators(dc_cls)
    model = create_model(
        dc_cls.__name__,
        __config__=config,
        __module__=dc_cls.__module__,
        __validators__=validators,
        __cls_kwargs__={'__resolve_forward_refs__': False},
        **field_definitions
    )
    model.__doc__ = dc_cls_doc if dc_cls_doc is not None else dc_cls.__doc__ or ''
    return model

if sys.version_info >= (3, 8):
    def _is_field_cached_property(obj: Any, k: str) -> bool:
        return isinstance(getattr(type(obj), k, None), cached_property)
else:
    def _is_field_cached_property(obj: Any, k: str) -> bool:
        return False

def _dataclass_validate_values(self: Any) -> None:
    if getattr(self, '__pydantic_initialised__'):
        return
    if getattr(self, '__pydantic_has_field_info_default__', False):
        input_data = {
            k: v for k, v in self.__dict__.items()
            if not (isinstance(v, FieldInfo) or _is_field_cached_property(self, k))
        }
    else:
        input_data = {
            k: v for k, v in self.__dict__.items()
            if not _is_field_cached_property(self, k)
        }
    d, _, validation_error = validate_model(self.__pydantic_model__, input_data, cls=self.__class__)
    if validation_error:
        raise validation_error
    self.__dict__.update(d)
    object.__setattr__(self, '__pydantic_initialised__', True)

def _dataclass_validate_assignment_setattr(self: Any, name: str, value: Any) -> None:
    if self.__pydantic_initialised__:
        d = dict(self.__dict__)
        d.pop(name, None)
        known_field = self.__pydantic_model__.__fields__.get(name, None)
        if known_field:
            value, error_ = known_field.validate(value, d, loc=name, cls=self.__class__)
            if error_:
                raise ValidationError([error_], self.__class__)
    object.__setattr__(self, name, value)

def is_builtin_dataclass(_cls: Type[Any]) -> bool:
    return (
        dataclasses.is_dataclass(_cls) and
        (not hasattr(_cls, '__pydantic_model__')) and
        set(_cls.__dataclass_fields__).issuperset(set(getattr(_cls, '__annotations__', {})))
    )

def make_dataclass_validator(
    dc_cls: Type[Any],
    config: Union[ConfigDict, Type[BaseConfig]],
) -> Generator[Callable[..., Any], None, None]:
    yield from _get_validators(dataclass(dc_cls, config=config, use_proxy=True))
