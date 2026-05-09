from __future__ import annotations
import datetime
import operator
import re
import sys
from collections import deque
from collections.abc import Container
from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property, partial
from re import Pattern
from typing import TYPE_CHECKING, Annotated, Any, Callable, Generic, Protocol, TypeVar, Union, overload
import annotated_types
if TYPE_CHECKING:
    from pydantic_core import core_schema as cs
    from pydantic import GetCoreSchemaHandler
from pydantic._internal._internal_dataclass import slots_true as _slots_true
if sys.version_info < (3, 10):
    EllipsisType = type(Ellipsis)
else:
    from types import EllipsisType

__all__ = ['validate_as', 'validate_as_deferred', 'transform']

_slots_frozen = {**_slots_true, 'frozen': True}

@dataclass(**_slots_frozen)
class _ValidateAs(Generic[_InT, _OutT]):
    strict: bool

@dataclass
class _ValidateAsDefer(Generic[_InT, _OutT]):
    tp: TypeVar('_NewOutT')

@dataclass(**_slots_frozen)
class _Transform(Generic[_InT, _OutT]):
    pass

@dataclass(**_slots_frozen)
class _PipelineOr(Generic[_InT, _OutT]):
    pass

@dataclass(**_slots_frozen)
class _PipelineAnd(Generic[_InT, _OutT]):
    pass

@dataclass(**_slots_frozen)
class _Eq(Generic[_InT]):
    pass

@dataclass(**_slots_frozen)
class _NotEq(Generic[_InT]):
    pass

@dataclass(**_slots_frozen)
class _In(Generic[_InT]):
    pass

@dataclass(**_slots_frozen)
class _NotIn(Generic[_InT]):
    pass

_ConstraintAnnotation = Union[Annotated[TypeVar('_NewOutGt'), annotated_types.Le], 
                              Annotated[TypeVar('_NewOutGe'), annotated_types.Ge], 
                              Annotated[TypeVar('_NewOutLt'), annotated_types.Lt], 
                              Annotated[TypeVar('_NewOutLe'), annotated_types.Ge], 
                              Annotated[TypeVar('_NewOutLen'), _SupportsLen], 
                              Annotated[TypeVar('_NewOutDiv'), annotated_types.SupportsDiv], 
                              Annotated[TypeVar('_NewOutMod'), annotated_types.SupportsMod], 
                              Annotated[TypeVar('_NewOutDatetime'), datetime.datetime], 
                              Annotated[TypeVar('_NewOutInterval'), _SupportsRange], 
                              Pattern[str], 
                              Annotated[TypeVar('_OtherIn'), _OtherIn], 
                              Annotated[TypeVar('_OtherOut'), _OtherOut]]

@dataclass(**_slots_frozen)
class _Constraint(Generic[_InT, _OutT]):
    pass

_InT = TypeVar('_InT')
_OutT = TypeVar('_OutT')
_NewOutT = TypeVar('_NewOutT')

class _FieldTypeMarker:
    pass

@dataclass(**_slots_true)
class _Pipeline(Generic[_InT, _OutT]):
    # ... rest of the code ...
