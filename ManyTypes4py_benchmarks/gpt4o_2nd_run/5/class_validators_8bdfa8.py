import warnings
from collections import ChainMap
from functools import partial, partialmethod, wraps
from itertools import chain
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union, overload
from pydantic.v1.errors import ConfigError
from pydantic.v1.typing import AnyCallable
from pydantic.v1.utils import ROOT_KEY, in_ipython

if TYPE_CHECKING:
    from pydantic.v1.typing import AnyClassMethod

class Validator:
    __slots__ = ('func', 'pre', 'each_item', 'always', 'check_fields', 'skip_on_failure')

    def __init__(self, func: Callable[..., Any], pre: bool = False, each_item: bool = False, always: bool = False, check_fields: bool = False, skip_on_failure: bool = False) -> None:
        self.func = func
        self.pre = pre
        self.each_item = each_item
        self.always = always
        self.check_fields = check_fields
        self.skip_on_failure = skip_on_failure

if TYPE_CHECKING:
    from inspect import Signature
    from pydantic.v1.config import BaseConfig
    from pydantic.v1.fields import ModelField
    from pydantic.v1.types import ModelOrDc
    ValidatorCallable = Callable[[Optional[ModelOrDc], Any, Dict[str, Any], ModelField, Type[BaseConfig]], Any]
    ValidatorsList = List[ValidatorCallable]
    ValidatorListDict = Dict[str, List[Validator]]

_FUNCS: Set[str] = set()
VALIDATOR_CONFIG_KEY = '__validator_config__'
ROOT_VALIDATOR_CONFIG_KEY = '__root_validator_config__'

def validator(*fields: str, pre: bool = False, each_item: bool = False, always: bool = False, check_fields: bool = True, whole: Optional[bool] = None, allow_reuse: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if not fields:
        raise ConfigError('validator with no fields specified')
    elif isinstance(fields[0], FunctionType):
        raise ConfigError("validators should be used with fields and keyword arguments, not bare. E.g. usage should be `@validator('<field_name>', ...)`")
    elif not all((isinstance(field, str) for field in fields)):
        raise ConfigError("validator fields should be passed as separate string args. E.g. usage should be `@validator('<field_name_1>', '<field_name_2>', ...)`")
    if whole is not None:
        warnings.warn('The "whole" keyword argument is deprecated, use "each_item" (inverse meaning, default False) instead', DeprecationWarning)
        assert each_item is False, '"each_item" and "whole" conflict, remove "whole"'
        each_item = not whole

    def dec(f: Callable[..., Any]) -> Callable[..., Any]:
        f_cls = _prepare_validator(f, allow_reuse)
        setattr(f_cls, VALIDATOR_CONFIG_KEY, (fields, Validator(func=f_cls.__func__, pre=pre, each_item=each_item, always=always, check_fields=check_fields)))
        return f_cls
    return dec

@overload
def root_validator(_func: Callable[..., Any]) -> Callable[..., Any]:
    ...

@overload
def root_validator(*, pre: bool = False, allow_reuse: bool = False, skip_on_failure: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...

def root_validator(_func: Optional[Callable[..., Any]] = None, *, pre: bool = False, allow_reuse: bool = False, skip_on_failure: bool = False) -> Union[Callable[..., Any], Callable[[Callable[..., Any]], Callable[..., Any]]]:
    if _func:
        f_cls = _prepare_validator(_func, allow_reuse)
        setattr(f_cls, ROOT_VALIDATOR_CONFIG_KEY, Validator(func=f_cls.__func__, pre=pre, skip_on_failure=skip_on_failure))
        return f_cls

    def dec(f: Callable[..., Any]) -> Callable[..., Any]:
        f_cls = _prepare_validator(f, allow_reuse)
        setattr(f_cls, ROOT_VALIDATOR_CONFIG_KEY, Validator(func=f_cls.__func__, pre=pre, skip_on_failure=skip_on_failure))
        return f_cls
    return dec

def _prepare_validator(function: Callable[..., Any], allow_reuse: bool) -> AnyClassMethod:
    f_cls = function if isinstance(function, classmethod) else classmethod(function)
    if not in_ipython() and (not allow_reuse):
        ref = getattr(f_cls.__func__, '__module__', '<No __module__>') + '.' + getattr(f_cls.__func__, '__qualname__', f'<No __qualname__: id:{id(f_cls.__func__)}>')
        if ref in _FUNCS:
            raise ConfigError(f'duplicate validator function "{ref}"; if this is intended, set `allow_reuse=True`')
        _FUNCS.add(ref)
    return f_cls

class ValidatorGroup:
    def __init__(self, validators: Dict[str, List[Validator]]) -> None:
        self.validators = validators
        self.used_validators: Set[str] = {'*'}

    def get_validators(self, name: str) -> Optional[Dict[str, Validator]]:
        self.used_validators.add(name)
        validators = self.validators.get(name, [])
        if name != ROOT_KEY:
            validators += self.validators.get('*', [])
        if validators:
            return {getattr(v.func, '__name__', f'<No __name__: id:{id(v.func)}>'): v for v in validators}
        else:
            return None

    def check_for_unused(self) -> None:
        unused_validators = set(chain.from_iterable(((getattr(v.func, '__name__', f'<No __name__: id:{id(v.func)}>') for v in self.validators[f] if v.check_fields) for f in self.validators.keys() - self.used_validators)))
        if unused_validators:
            fn = ', '.join(unused_validators)
            raise ConfigError(f"Validators defined with incorrect fields: {fn} (use check_fields=False if you're inheriting from the model and intended this)")

def extract_validators(namespace: Dict[str, Any]) -> Dict[str, List[Validator]]:
    validators: Dict[str, List[Validator]] = {}
    for var_name, value in namespace.items():
        validator_config = getattr(value, VALIDATOR_CONFIG_KEY, None)
        if validator_config:
            fields, v = validator_config
            for field in fields:
                if field in validators:
                    validators[field].append(v)
                else:
                    validators[field] = [v]
    return validators

def extract_root_validators(namespace: Dict[str, Any]) -> Tuple[List[Callable[..., Any]], List[Tuple[bool, Callable[..., Any]]]]:
    from inspect import signature
    pre_validators: List[Callable[..., Any]] = []
    post_validators: List[Tuple[bool, Callable[..., Any]]] = []
    for name, value in namespace.items():
        validator_config = getattr(value, ROOT_VALIDATOR_CONFIG_KEY, None)
        if validator_config:
            sig = signature(validator_config.func)
            args = list(sig.parameters.keys())
            if args[0] == 'self':
                raise ConfigError(f'Invalid signature for root validator {name}: {sig}, "self" not permitted as first argument, should be: (cls, values).')
            if len(args) != 2:
                raise ConfigError(f'Invalid signature for root validator {name}: {sig}, should be: (cls, values).')
            if validator_config.pre:
                pre_validators.append(validator_config.func)
            else:
                post_validators.append((validator_config.skip_on_failure, validator_config.func))
    return (pre_validators, post_validators)

def inherit_validators(base_validators: Dict[str, List[Validator]], validators: Dict[str, List[Validator]]) -> Dict[str, List[Validator]]:
    for field, field_validators in base_validators.items():
        if field not in validators:
            validators[field] = []
        validators[field] += field_validators
    return validators

def make_generic_validator(validator: Callable[..., Any]) -> Callable[..., Any]:
    from inspect import signature
    if not isinstance(validator, (partial, partialmethod)):
        sig = signature(validator)
        args = list(sig.parameters.keys())
    else:
        sig = signature(validator.func)
        args = [k for k in signature(validator.func).parameters.keys() if k not in validator.args | validator.keywords.keys()]
    first_arg = args.pop(0)
    if first_arg == 'self':
        raise ConfigError(f'Invalid signature for validator {validator}: {sig}, "self" not permitted as first argument, should be: (cls, value, values, config, field), "values", "config" and "field" are all optional.')
    elif first_arg == 'cls':
        return wraps(validator)(_generic_validator_cls(validator, sig, set(args[1:])))
    else:
        return wraps(validator)(_generic_validator_basic(validator, sig, set(args)))

def prep_validators(v_funcs: Iterable[Callable[..., Any]]) -> List[Callable[..., Any]]:
    return [make_generic_validator(f) for f in v_funcs if f]

all_kwargs: Set[str] = {'values', 'field', 'config'}

def _generic_validator_cls(validator: Callable[..., Any], sig: Signature, args: Set[str]) -> Callable[..., Any]:
    has_kwargs = False
    if 'kwargs' in args:
        has_kwargs = True
        args -= {'kwargs'}
    if not args.issubset(all_kwargs):
        raise ConfigError(f'Invalid signature for validator {validator}: {sig}, should be: (cls, value, values, config, field), "values", "config" and "field" are all optional.')
    if has_kwargs:
        return lambda cls, v, values, field, config: validator(cls, v, values=values, field=field, config=config)
    elif args == set():
        return lambda cls, v, values, field, config: validator(cls, v)
    elif args == {'values'}:
        return lambda cls, v, values, field, config: validator(cls, v, values=values)
    elif args == {'field'}:
        return lambda cls, v, values, field, config: validator(cls, v, field=field)
    elif args == {'config'}:
        return lambda cls, v, values, field, config: validator(cls, v, config=config)
    elif args == {'values', 'field'}:
        return lambda cls, v, values, field, config: validator(cls, v, values=values, field=field)
    elif args == {'values', 'config'}:
        return lambda cls, v, values, field, config: validator(cls, v, values=values, config=config)
    elif args == {'field', 'config'}:
        return lambda cls, v, values, field, config: validator(cls, v, field=field, config=config)
    else:
        return lambda cls, v, values, field, config: validator(cls, v, values=values, field=field, config=config)

def _generic_validator_basic(validator: Callable[..., Any], sig: Signature, args: Set[str]) -> Callable[..., Any]:
    has_kwargs = False
    if 'kwargs' in args:
        has_kwargs = True
        args -= {'kwargs'}
    if not args.issubset(all_kwargs):
        raise ConfigError(f'Invalid signature for validator {validator}: {sig}, should be: (value, values, config, field), "values", "config" and "field" are all optional.')
    if has_kwargs:
        return lambda cls, v, values, field, config: validator(v, values=values, field=field, config=config)
    elif args == set():
        return lambda cls, v, values, field, config: validator(v)
    elif args == {'values'}:
        return lambda cls, v, values, field, config: validator(v, values=values)
    elif args == {'field'}:
        return lambda cls, v, values, field, config: validator(v, field=field)
    elif args == {'config'}:
        return lambda cls, v, values, field, config: validator(v, config=config)
    elif args == {'values', 'field'}:
        return lambda cls, v, values, field, config: validator(v, values=values, field=field)
    elif args == {'values', 'config'}:
        return lambda cls, v, values, field, config: validator(v, values=values, config=config)
    elif args == {'field', 'config'}:
        return lambda cls, v, values, field, config: validator(v, field=field, config=config)
    else:
        return lambda cls, v, values, field, config: validator(v, values=values, field=field, config=config)

def gather_all_validators(type_: Type[Any]) -> Dict[str, Any]:
    all_attributes = ChainMap(*[cls.__dict__ for cls in type_.__mro__])
    return {k: v for k, v in all_attributes.items() if hasattr(v, VALIDATOR_CONFIG_KEY) or hasattr(v, ROOT_VALIDATOR_CONFIG_KEY)}
