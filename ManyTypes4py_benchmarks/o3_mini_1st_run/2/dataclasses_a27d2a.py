#!/usr/bin/env python3
"""
The main purpose is to enhance stdlib dataclasses by adding validation
A pydantic dataclass can be generated from scratch or from a stdlib one.

Behind the scene, a pydantic dataclass is just like a regular one on which we attach
a `BaseModel` and magic methods to trigger the validation of the data.
`__init__` and `__post_init__` are hence overridden and have extra logic to be
able to validate input data.

When a pydantic dataclass is generated from scratch, it's just a plain dataclass
with validation triggered at initialization

The tricky part if for stdlib dataclasses that are converted after into pydantic ones e.g.

```py
@dataclasses.dataclass
class M:
    x: int

ValidatedM = pydantic.dataclasses.dataclass(M)
```

We indeed still want to support equality, hashing, repr, ... as if it was the stdlib one!

```py
assert isinstance(ValidatedM(x=1), M)
assert ValidatedM(x=1) == M(x=1)
```

This means we **don't want to create a new dataclass that inherits from it**
The trick is to create a wrapper around `M` that will act as a proxy to trigger
validation without altering default `M` behaviour.
"""
import copy
import dataclasses
import sys
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Generator, Optional, Type, TypeVar, Union, cast
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
    DataclassT = TypeVar('DataclassT', bound='Dataclass')
    DataclassClassOrWrapper = Union[Type['Dataclass'], 'DataclassProxy']

    class Dataclass:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        @classmethod
        def __get_validators__(cls) -> Generator[Callable[[Any], Any], None, None]:
            pass

        @classmethod
        def __validate__(cls, v: Any) -> Any:
            pass

__all__ = ['dataclass', 'set_validation', 'create_pydantic_model_from_dataclass', 'is_builtin_dataclass', 'make_dataclass_validator']
_T = TypeVar('_T')

if sys.version_info >= (3, 10):

    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    @overload
    def dataclass(*, init: bool = True, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool = False, config: Optional[Any] = None, validate_on_init: Optional[bool] = None, use_proxy: Optional[bool] = None, kw_only: Any = ...) -> Callable[[Type[Any]], Type[Any]]:
        ...

    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    @overload
    def dataclass(_cls: Type[Any], *, init: bool = True, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool = False, config: Optional[Any] = None, validate_on_init: Optional[bool] = None, use_proxy: Optional[bool] = None, kw_only: Any = ...) -> Type[Any]:
        ...
else:

    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    @overload
    def dataclass(*, init: bool = True, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool = False, config: Optional[Any] = None, validate_on_init: Optional[bool] = None, use_proxy: Optional[bool] = None) -> Callable[[Type[Any]], Type[Any]]:
        ...

    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    @overload
    def dataclass(_cls: Type[Any], *, init: bool = True, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool = False, config: Optional[Any] = None, validate_on_init: Optional[bool] = None, use_proxy: Optional[bool] = None) -> Type[Any]:
        ...

@dataclass_transform(field_specifiers=(dataclasses.field, Field))
def dataclass(
    _cls: Optional[Type[Any]] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    config: Optional[Any] = None,
    validate_on_init: Optional[bool] = None,
    use_proxy: Optional[bool] = None,
    kw_only: bool = False
) -> Union[Callable[[Type[Any]], Type[Any]], Type[Any]]:
    """
    Like the python standard lib dataclasses but with type validation.
    The result is either a pydantic dataclass that will validate input data
    or a wrapper that will trigger validation around a stdlib dataclass
    to avoid modifying it directly
    """
    the_config: Any = get_config(config)

    def wrap(cls: Type[Any]) -> Type[Any]:
        should_use_proxy: bool = use_proxy if use_proxy is not None else is_builtin_dataclass(cls) and (cls.__bases__[0] is object or set(dir(cls)) == set(dir(cls.__bases__[0])))
        if should_use_proxy:
            dc_cls_doc: str = ''
            dc_cls: Any = DataclassProxy(cls)
            default_validate_on_init: bool = False
        else:
            dc_cls_doc = cls.__doc__ or ''
            if sys.version_info >= (3, 10):
                dc_cls = dataclasses.dataclass(cls, init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, frozen=frozen, kw_only=kw_only)
            else:
                dc_cls = dataclasses.dataclass(cls, init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, frozen=frozen)
            default_validate_on_init = True
        should_validate_on_init: bool = default_validate_on_init if validate_on_init is None else validate_on_init
        _add_pydantic_validation_attributes(cls, the_config, should_validate_on_init, dc_cls_doc)
        dc_cls.__pydantic_model__.__try_update_forward_refs__(**{cls.__name__: cls})
        return dc_cls
    if _cls is None:
        return wrap
    return wrap(_cls)

@contextmanager
def set_validation(cls: Type[Any], value: bool) -> Generator[Type[Any], None, None]:
    original_run_validation: Any = cls.__pydantic_run_validation__
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

    def __setattr__(self, __name: str, __value: Any) -> None:
        setattr(self.__dataclass__, __name, __value)

    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, self.__dataclass__)

    def __copy__(self) -> 'DataclassProxy':
        return DataclassProxy(copy.copy(self.__dataclass__))

    def __deepcopy__(self, memo: Dict[int, Any]) -> 'DataclassProxy':
        return DataclassProxy(copy.deepcopy(self.__dataclass__, memo))

def _add_pydantic_validation_attributes(dc_cls: Type[Any], config: Any, validate_on_init: bool, dc_cls_doc: str) -> None:
    """
    We need to replace the right method. If no `__post_init__` has been set in the stdlib dataclass
    it won't even exist (code is generated on the fly by `dataclasses`)
    By default, we run validation after `__init__` or `__post_init__` if defined
    """
    init: Callable[..., Any] = dc_cls.__init__

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
            post_init = dc_cls.__post_init__.__wrapped__  # type: ignore
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
                initvars_and_values: Dict[str, Any] = {}
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

def _get_validators(cls: Type[Any]) -> Generator[Callable[[Any], Any], None, None]:
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

def create_pydantic_model_from_dataclass(dc_cls: Type[Any], config: Any = BaseConfig, dc_cls_doc: Optional[str] = None) -> Any:
    field_definitions: Dict[str, Any] = {}
    for field in dataclasses.fields(dc_cls):
        default: Any = Undefined
        default_factory: Optional[Callable[[], Any]] = None
        if field.default is not dataclasses.MISSING:
            default = field.default
        elif field.default_factory is not dataclasses.MISSING:
            default_factory = field.default_factory  # type: ignore
        else:
            default = Required
        if isinstance(default, FieldInfo):
            field_info: FieldInfo = default
            dc_cls.__pydantic_has_field_info_default__ = True  # type: ignore
        else:
            field_info = Field(default=default, default_factory=default_factory, **field.metadata)
        field_definitions[field.name] = (field.type, field_info)
    validators = gather_all_validators(dc_cls)
    model: Any = create_model(
        dc_cls.__name__,
        __config__=config,
        __module__=dc_cls.__module__,
        __validators__=validators,
        __cls_kwargs__={'__resolve_forward_refs__': False},
        **field_definitions
    )
    model.__doc__ = dc_cls_doc if dc_cls_doc is not None else (dc_cls.__doc__ or '')
    return model

if sys.version_info >= (3, 8):

    def _is_field_cached_property(obj: Any, k: str) -> bool:
        return isinstance(getattr(type(obj), k, None),  getattr(sys.modules[__name__], "cached_property", object))
else:

    def _is_field_cached_property(obj: Any, k: str) -> bool:
        return False

def _dataclass_validate_values(self: Any) -> None:
    if getattr(self, '__pydantic_initialised__'):
        return
    if getattr(self, '__pydantic_has_field_info_default__', False):
        input_data: Dict[str, Any] = {k: v for k, v in self.__dict__.items() if not (isinstance(v, FieldInfo) or _is_field_cached_property(self, k))}
    else:
        input_data = {k: v for k, v in self.__dict__.items() if not _is_field_cached_property(self, k)}
    d, _, validation_error = validate_model(self.__pydantic_model__, input_data, cls=self.__class__)
    if validation_error:
        raise validation_error
    self.__dict__.update(d)
    object.__setattr__(self, '__pydantic_initialised__', True)

def _dataclass_validate_assignment_setattr(self: Any, name: str, value: Any) -> None:
    if self.__pydantic_initialised__:
        d: Dict[str, Any] = dict(self.__dict__)
        d.pop(name, None)
        known_field: Optional[Any] = self.__pydantic_model__.__fields__.get(name, None)
        if known_field:
            value, error_ = known_field.validate(value, d, loc=name, cls=self.__class__)
            if error_:
                raise ValidationError([error_], self.__class__)
    object.__setattr__(self, name, value)

def is_builtin_dataclass(_cls: Any) -> bool:
    """
    Whether a class is a stdlib dataclass
    (useful to discriminated a pydantic dataclass that is actually a wrapper around a stdlib dataclass)

    we check that
    - `_cls` is a dataclass
    - `_cls` is not a processed pydantic dataclass (with a basemodel attached)
    - `_cls` is not a pydantic dataclass inheriting directly from a stdlib dataclass
    e.g.
    ```
    @dataclasses.dataclass
    class A:
        x: int

    @pydantic.dataclasses.dataclass
    class B(A):
        y: int
    ```
    In this case, when we first check `B`, we make an extra check and look at the annotations ('y'),
    which won't be a superset of all the dataclass fields (only the stdlib fields i.e. 'x')
    """
    return dataclasses.is_dataclass(_cls) and (not hasattr(_cls, '__pydantic_model__')) and set(_cls.__dataclass_fields__).issuperset(set(getattr(_cls, '__annotations__', {})))

def make_dataclass_validator(dc_cls: Any, config: Any) -> Generator[Any, None, None]:
    """
    Create a pydantic.dataclass from a builtin dataclass to add type validation
    and yield the validators
    It retrieves the parameters of the dataclass and forwards them to the newly created dataclass
    """
    yield from _get_validators(dataclass(dc_cls, config=config, use_proxy=True))
