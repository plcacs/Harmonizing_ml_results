from __future__ import annotations
from functools import partial, partialmethod
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union, Optional, Tuple
from warnings import warn
from typing_extensions import Protocol, TypeAlias, deprecated
from .._internal import _decorators, _decorators_v1
from ..errors import PydanticUserError
from ..warnings import PydanticDeprecatedSince20

_ALLOW_REUSE_WARNING_MESSAGE: str = '`allow_reuse` is deprecated and will be ignored; it should no longer be necessary'

F = TypeVar("F", bound=Callable[..., Any])

if TYPE_CHECKING:

    class _OnlyValueValidatorClsMethod(Protocol):

        def __call__(self, __cls: Any, __value: Any) -> Any:
            ...

    class _V1ValidatorWithValuesClsMethod(Protocol):

        def __call__(self, __cls: Any, __value: Any, values: Any) -> Any:
            ...

    class _V1ValidatorWithValuesKwOnlyClsMethod(Protocol):

        def __call__(self, __cls: Any, __value: Any, *, values: Any) -> Any:
            ...

    class _V1ValidatorWithKwargsClsMethod(Protocol):

        def __call__(self, __cls: Any, **kwargs: Any) -> Any:
            ...

    class _V1ValidatorWithValuesAndKwargsClsMethod(Protocol):

        def __call__(self, __cls: Any, values: Any, **kwargs: Any) -> Any:
            ...

    class _V1RootValidatorClsMethod(Protocol):

        def __call__(self, __cls: Any, __values: Any) -> Any:
            ...

    V1Validator: TypeAlias = Union[
        _OnlyValueValidatorClsMethod,
        _V1ValidatorWithValuesClsMethod,
        _V1ValidatorWithValuesKwOnlyClsMethod,
        _V1ValidatorWithKwargsClsMethod,
        _V1ValidatorWithValuesAndKwargsClsMethod,
        _decorators_v1.V1ValidatorWithValues,
        _decorators_v1.V1ValidatorWithValuesKwOnly,
        _decorators_v1.V1ValidatorWithKwargs,
        _decorators_v1.V1ValidatorWithValuesAndKwargs,
    ]
    V1RootValidator: TypeAlias = Union[
        _V1RootValidatorClsMethod, _decorators_v1.V1RootValidatorFunction
    ]
    _PartialClsOrStaticMethod: TypeAlias = Union[
        classmethod[Any, Any, Any],
        staticmethod[Any, Any],
        partialmethod[Any]
    ]
    _V1ValidatorType = TypeVar('_V1ValidatorType', V1Validator, _PartialClsOrStaticMethod)
    _V1RootValidatorFunctionType = TypeVar('_V1RootValidatorFunctionType', _decorators_v1.V1RootValidatorFunction, _V1RootValidatorClsMethod, _PartialClsOrStaticMethod)
else:
    DeprecationWarning = PydanticDeprecatedSince20

@deprecated(
    'Pydantic V1 style `@validator` validators are deprecated. You should migrate to Pydantic V2 style `@field_validator` validators, see the migration guide for more details',
    category=None
)
def validator(
    __field: str,
    *fields: str,
    pre: bool = False,
    each_item: bool = False,
    always: bool = False,
    check_fields: Optional[bool] = None,
    allow_reuse: bool = False
) -> Callable[[F], F]:
    """Decorate methods on the class indicating that they should be used to validate fields.

    Args:
        __field (str): The first field the validator should be called on; this is separate
            from `fields` to ensure an error is raised if you don't pass at least one.
        *fields (str): Additional field(s) the validator should be called on.
        pre (bool, optional): Whether this validator should be called before the standard
            validators (else after). Defaults to False.
        each_item (bool, optional): For complex objects (sets, lists etc.) whether to validate
            individual elements rather than the whole object. Defaults to False.
        always (bool, optional): Whether this method and other validators should be called even if
            the value is missing. Defaults to False.
        check_fields (bool | None, optional): Whether to check that the fields actually exist on the model.
            Defaults to None.
        allow_reuse (bool, optional): Whether to track and raise an error if another validator refers to
            the decorated function. Defaults to False.

    Returns:
        Callable: A decorator that can be used to decorate a function to be used as a validator.
    """
    warn(
        'Pydantic V1 style `@validator` validators are deprecated. You should migrate to Pydantic V2 style `@field_validator` validators, see the migration guide for more details',
        DeprecationWarning,
        stacklevel=2
    )
    if allow_reuse is True:
        warn(_ALLOW_REUSE_WARNING_MESSAGE, DeprecationWarning)
    fields = (__field, *fields)
    if isinstance(fields[0], FunctionType):
        raise PydanticUserError(
            "`@validator` should be used with fields and keyword arguments, not bare. E.g. usage should be `@validator('<field_name>', ...)`",
            code='validator-no-fields'
        )
    elif not all(isinstance(field, str) for field in fields):
        raise PydanticUserError(
            "`@validator` fields should be passed as separate string args. E.g. usage should be `@validator('<field_name_1>', '<field_name_2>', ...)`",
            code='validator-invalid-fields'
        )
    mode: Literal["before", "after"] = 'before' if pre is True else 'after'

    def dec(f: F) -> F:
        if _decorators.is_instance_method_from_sig(f):
            raise PydanticUserError(
                '`@validator` cannot be applied to instance methods',
                code='validator-instance-method'
            )
        f_converted: Any = _decorators.ensure_classmethod_based_on_signature(f)
        wrap = _decorators_v1.make_generic_v1_field_validator
        validator_wrapper_info = _decorators.ValidatorDecoratorInfo(
            fields=fields,
            mode=mode,
            each_item=each_item,
            always=always,
            check_fields=check_fields
        )
        return _decorators.PydanticDescriptorProxy(f_converted, validator_wrapper_info, shim=wrap)  # type: ignore
    return dec

@overload
def root_validator(*, skip_on_failure: bool, allow_reuse: bool = ...) -> Callable[[F], F]:
    ...

@overload
def root_validator(*, pre: bool, allow_reuse: bool = ...) -> Callable[[F], F]:
    ...

@overload
def root_validator(*, pre: bool, skip_on_failure: bool, allow_reuse: bool = ...) -> Callable[[F], F]:
    ...

@deprecated(
    'Pydantic V1 style `@root_validator` validators are deprecated. You should migrate to Pydantic V2 style `@model_validator` validators, see the migration guide for more details',
    category=None
)
def root_validator(
    *__args: Any,
    pre: bool = False,
    skip_on_failure: bool = False,
    allow_reuse: bool = False
) -> Union[Callable[[F], F], F]:
    """Decorate methods on a model indicating that they should be used to validate (and perhaps
    modify) data either before or after standard model parsing/validation is performed.

    Args:
        pre (bool, optional): Whether this validator should be called before the standard
            validators (else after). Defaults to False.
        skip_on_failure (bool, optional): Whether to stop validation and return as soon as a
            failure is encountered. Defaults to False.
        allow_reuse (bool, optional): Whether to track and raise an error if another validator
            refers to the decorated function. Defaults to False.

    Returns:
        Any: A decorator that can be used to decorate a function to be used as a root_validator.
    """
    warn(
        'Pydantic V1 style `@root_validator` validators are deprecated. You should migrate to Pydantic V2 style `@model_validator` validators, see the migration guide for more details',
        DeprecationWarning,
        stacklevel=2
    )
    if __args:
        return root_validator(pre=pre, skip_on_failure=skip_on_failure, allow_reuse=allow_reuse)(*__args)
    if allow_reuse is True:
        warn(_ALLOW_REUSE_WARNING_MESSAGE, DeprecationWarning)
    mode: Literal["before", "after"] = 'before' if pre is True else 'after'
    if pre is False and skip_on_failure is not True:
        raise PydanticUserError(
            'If you use `@root_validator` with pre=False (the default) you MUST specify `skip_on_failure=True`. Note that `@root_validator` is deprecated and should be replaced with `@model_validator`.',
            code='root-validator-pre-skip'
        )
    wrap: Callable[..., Any] = partial(_decorators_v1.make_v1_generic_root_validator, pre=pre)

    def dec(f: F) -> F:
        if _decorators.is_instance_method_from_sig(f):
            raise TypeError('`@root_validator` cannot be applied to instance methods')
        res: Any = _decorators.ensure_classmethod_based_on_signature(f)
        dec_info = _decorators.RootValidatorDecoratorInfo(mode=mode)
        return _decorators.PydanticDescriptorProxy(res, dec_info, shim=wrap)  # type: ignore
    return dec