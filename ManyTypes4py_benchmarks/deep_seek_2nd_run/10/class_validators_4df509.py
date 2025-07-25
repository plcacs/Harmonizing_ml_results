"""Old `@validator` and `@root_validator` function validators from V1."""
from __future__ import annotations as _annotations
from functools import partial, partialmethod
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union, overload
from warnings import warn
from typing_extensions import Protocol, TypeAlias, deprecated
from .._internal import _decorators, _decorators_v1
from ..errors import PydanticUserError
from ..warnings import PydanticDeprecatedSince20

_ALLOW_REUSE_WARNING_MESSAGE: str = '`allow_reuse` is deprecated and will be ignored; it should no longer be necessary'

if TYPE_CHECKING:
    class _OnlyValueValidatorClsMethod(Protocol):
        def __call__(self, __cls: Any, __value: Any) -> Any: ...
    
    class _V1ValidatorWithValuesClsMethod(Protocol):
        def __call__(self, __cls: Any, __value: Any, values: dict[str, Any]) -> Any: ...
    
    class _V1ValidatorWithValuesKwOnlyClsMethod(Protocol):
        def __call__(self, __cls: Any, __value: Any, *, values: dict[str, Any]) -> Any: ...
    
    class _V1ValidatorWithKwargsClsMethod(Protocol):
        def __call__(self, __cls: Any, **kwargs: Any) -> Any: ...
    
    class _V1ValidatorWithValuesAndKwargsClsMethod(Protocol):
        def __call__(self, __cls: Any, values: dict[str, Any], **kwargs: Any) -> Any: ...
    
    class _V1RootValidatorClsMethod(Protocol):
        def __call__(self, __cls: Any, __values: dict[str, Any]) -> dict[str, Any]: ...
    
    V1Validator: TypeAlias = Union[
        _OnlyValueValidatorClsMethod,
        _V1ValidatorWithValuesClsMethod,
        _V1ValidatorWithValuesKwOnlyClsMethod,
        _V1ValidatorWithKwargsClsMethod,
        _V1ValidatorWithValuesAndKwargsClsMethod,
        _decorators_v1.V1ValidatorWithValues,
        _decorators_v1.V1ValidatorWithValuesKwOnly,
        _decorators_v1.V1ValidatorWithKwargs,
        _decorators_v1.V1ValidatorWithValuesAndKwargs
    ]
    V1RootValidator: TypeAlias = Union[_V1RootValidatorClsMethod, _decorators_v1.V1RootValidatorFunction]
    _PartialClsOrStaticMethod: TypeAlias = Union[classmethod[Any, Any, Any], staticmethod[Any, Any], partialmethod[Any]]
    _V1ValidatorType = TypeVar('_V1ValidatorType', bound=V1Validator)
    _V1RootValidatorFunctionType = TypeVar('_V1RootValidatorFunctionType', bound=Union[_decorators_v1.V1RootValidatorFunction, _V1RootValidatorClsMethod, _PartialClsOrStaticMethod])
else:
    DeprecationWarning = PydanticDeprecatedSince20

@deprecated('Pydantic V1 style `@validator` validators are deprecated. You should migrate to Pydantic V2 style `@field_validator` validators, see the migration guide for more details', category=None)
def validator(
    __field: str,
    *fields: str,
    pre: bool = False,
    each_item: bool = False,
    always: bool = False,
    check_fields: bool | None = None,
    allow_reuse: bool = False
) -> Callable[[_V1ValidatorType], _V1ValidatorType]:
    warn('Pydantic V1 style `@validator` validators are deprecated. You should migrate to Pydantic V2 style `@field_validator` validators, see the migration guide for more details', DeprecationWarning, stacklevel=2)
    if allow_reuse is True:
        warn(_ALLOW_REUSE_WARNING_MESSAGE, DeprecationWarning)
    fields = (__field, *fields)
    if isinstance(fields[0], FunctionType):
        raise PydanticUserError("`@validator` should be used with fields and keyword arguments, not bare. E.g. usage should be `@validator('<field_name>', ...)`", code='validator-no-fields')
    elif not all((isinstance(field, str) for field in fields)):
        raise PydanticUserError("`@validator` fields should be passed as separate string args. E.g. usage should be `@validator('<field_name_1>', '<field_name_2>', ...)`", code='validator-invalid-fields')
    mode: Literal['before', 'after'] = 'before' if pre is True else 'after'

    def dec(f: _V1ValidatorType) -> _V1ValidatorType:
        if _decorators.is_instance_method_from_sig(f):
            raise PydanticUserError('`@validator` cannot be applied to instance methods', code='validator-instance-method')
        f = _decorators.ensure_classmethod_based_on_signature(f)
        wrap = _decorators_v1.make_generic_v1_field_validator
        validator_wrapper_info = _decorators.ValidatorDecoratorInfo(fields=fields, mode=mode, each_item=each_item, always=always, check_fields=check_fields)
        return _decorators.PydanticDescriptorProxy(f, validator_wrapper_info, shim=wrap)
    return dec

@overload
def root_validator(*, skip_on_failure: bool, allow_reuse: bool = ...) -> Callable[[_V1RootValidatorFunctionType], _V1RootValidatorFunctionType]: ...

@overload
def root_validator(*, pre: bool, allow_reuse: bool = ...) -> Callable[[_V1RootValidatorFunctionType], _V1RootValidatorFunctionType]: ...

@overload
def root_validator(*, pre: bool, skip_on_failure: bool, allow_reuse: bool = ...) -> Callable[[_V1RootValidatorFunctionType], _V1RootValidatorFunctionType]: ...

@deprecated('Pydantic V1 style `@root_validator` validators are deprecated. You should migrate to Pydantic V2 style `@model_validator` validators, see the migration guide for more details', category=None)
def root_validator(
    *__args: Any,
    pre: bool = False,
    skip_on_failure: bool = False,
    allow_reuse: bool = False
) -> Callable[[_V1RootValidatorFunctionType], _V1RootValidatorFunctionType] | Callable[..., Any]:
    warn('Pydantic V1 style `@root_validator` validators are deprecated. You should migrate to Pydantic V2 style `@model_validator` validators, see the migration guide for more details', DeprecationWarning, stacklevel=2)
    if __args:
        return root_validator()(*__args)
    if allow_reuse is True:
        warn(_ALLOW_REUSE_WARNING_MESSAGE, DeprecationWarning)
    mode: Literal['before', 'after'] = 'before' if pre is True else 'after'
    if pre is False and skip_on_failure is not True:
        raise PydanticUserError('If you use `@root_validator` with pre=False (the default) you MUST specify `skip_on_failure=True`. Note that `@root_validator` is deprecated and should be replaced with `@model_validator`.', code='root-validator-pre-skip')
    wrap = partial(_decorators_v1.make_v1_generic_root_validator, pre=pre)

    def dec(f: _V1RootValidatorFunctionType) -> _V1RootValidatorFunctionType:
        if _decorators.is_instance_method_from_sig(f):
            raise TypeError('`@root_validator` cannot be applied to instance methods')
        res = _decorators.ensure_classmethod_based_on_signature(f)
        dec_info = _decorators.RootValidatorDecoratorInfo(mode=mode)
        return _decorators.PydanticDescriptorProxy(res, dec_info, shim=wrap)
    return dec
