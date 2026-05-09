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

if TYPE_CHECKING:
    _OnlyValueValidatorClsMethod: Protocol = callable[[Any, Any], Any]
    _V1ValidatorWithValuesClsMethod: Protocol = callable[[Any, Any, dict], Any]
    _V1ValidatorWithValuesKwOnlyClsMethod: Protocol = callable[[Any, Any, dict], Any]
    _V1ValidatorWithKwargsClsMethod: Protocol = callable[[Any, **dict], Any]
    _V1ValidatorWithValuesAndKwargsClsMethod: Protocol = callable[[Any, dict, **dict], Any]
    _V1RootValidatorClsMethod: Protocol = callable[[Any, dict], Any]
    V1Validator: Union[_OnlyValueValidatorClsMethod, _V1ValidatorWithValuesClsMethod, _V1ValidatorWithValuesKwOnlyClsMethod, _V1ValidatorWithKwargsClsMethod, _V1ValidatorWithValuesAndKwargsClsMethod, type, type, type, type]
    V1RootValidator: Union[_V1RootValidatorClsMethod, type]
    _PartialClsOrStaticMethod: Union[classmethod[Any, Any, Any], staticmethod[Any, Any], partialmethod[Any]]
    _V1ValidatorType: TypeVar('_V1ValidatorType', V1Validator, _PartialClsOrStaticMethod)
    _V1RootValidatorFunctionType: TypeVar('_V1RootValidatorFunctionType', type, _V1RootValidatorClsMethod, _PartialClsOrStaticMethod)
else:
    DeprecationWarning = PydanticDeprecatedSince20

@deprecated('Pydantic V1 style `@validator` validators are deprecated. You should migrate to Pydantic V2 style `@field_validator` validators, see the migration guide for more details', category=None)
def validator(__field: str, *fields: str, pre: bool = False, each_item: bool = False, always: bool = False, check_fields: bool | None = None, allow_reuse: bool = False) -> Callable[[FunctionType], _V1ValidatorType]:
    ...

@overload
def root_validator(*, skip_on_failure: bool, allow_reuse: bool = ...) -> Callable[[FunctionType], _V1RootValidatorFunctionType]:
    ...

@overload
def root_validator(*, pre: bool, allow_reuse: bool = ...) -> Callable[[FunctionType], _V1RootValidatorFunctionType]:
    ...

@overload
def root_validator(*, pre: bool, skip_on_failure: bool, allow_reuse: bool = ...) -> Callable[[FunctionType], _V1RootValidatorFunctionType]:
    ...

@deprecated('Pydantic V1 style `@root_validator` validators are deprecated. You should migrate to Pydantic V2 style `@model_validator` validators, see the migration guide for more details', category=None)
def root_validator(*__args, pre: bool = False, skip_on_failure: bool = False, allow_reuse: bool = False) -> Any:
    ...
