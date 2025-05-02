import contextlib
import re
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from functools import partial, partialmethod
from itertools import product
from os.path import normcase
from typing import Annotated, Any, Callable, Dict, List, Literal, NamedTuple, Optional, Tuple, Type, TypeVar, Union
from unittest.mock import MagicMock
import pytest
from dirty_equals import HasRepr, IsInstance
from pydantic_core import core_schema
from typing_extensions import TypedDict
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    PlainSerializer,
    PydanticDeprecatedSince20,
    PydanticUserError,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    errors,
    field_validator,
    model_validator,
    root_validator,
    validate_call,
    validator,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.functional_validators import (
    AfterValidator,
    BeforeValidator,
    PlainValidator,
    WrapValidator,
)

V1_VALIDATOR_DEPRECATION_MATCH: str = 'Pydantic V1 style `@validator` validators are deprecated'

def test_annotated_validator_after() -> None:
    MyInt: Type[int] = Annotated[int, AfterValidator(lambda x, _info: x if x != -1 else 0)]

    class Model(BaseModel):
        x: MyInt

    assert Model(x=0).x == 0
    assert Model(x=-1).x == 0
    assert Model(x=-2).x == -2
    assert Model(x=1).x == 1
    assert Model(x='-1').x == 0

def test_annotated_validator_before() -> None:
    FloatMaybeInf: Type[float] = Annotated[float, BeforeValidator(lambda x, _info: x if x != 'zero' else 0.0)]

    class Model(BaseModel):
        x: FloatMaybeInf

    assert Model(x='zero').x == 0.0
    assert Model(x=1.0).x == 1.0
    assert Model(x='1.0').x == 1.0

def test_annotated_validator_builtin() -> None:
    """https://github.com/pydantic/pydantic/issues/6752"""
    TruncatedFloat: Type[float] = Annotated[float, BeforeValidator(int)]
    DateTimeFromIsoFormat: Type[datetime] = Annotated[datetime, BeforeValidator(datetime.fromisoformat)]

    class Model(BaseModel):
        x: TruncatedFloat
        y: DateTimeFromIsoFormat

    m: Model = Model(x=1.234, y='2011-11-04T00:05:23')
    assert m.x == 1
    assert m.y == datetime(2011, 11, 4, 0, 5, 23)

def test_annotated_validator_plain() -> None:
    MyInt: Type[int] = Annotated[int, PlainValidator(lambda x, _info: x if x != -1 else 0)]

    class Model(BaseModel):
        x: MyInt

    assert Model(x=0).x == 0
    assert Model(x=-1).x == 0
    assert Model(x=-2).x == -2

def test_annotated_validator_wrap() -> None:
    def sixties_validator(val: Any, handler: Callable[[Any], Any], info: ValidationInfo) -> date:
        if val == 'epoch':
            return date.fromtimestamp(0)
        newval: date = handler(val)
        if not date.fromisoformat('1960-01-01') <= newval < date.fromisoformat('1970-01-01'):
            raise ValueError(f'{val} is not in the sixties!')
        return newval

    SixtiesDateTime: Type[date] = Annotated[date, WrapValidator(sixties_validator)]

    class Model(BaseModel):
        x: SixtiesDateTime

    assert Model(x='epoch').x == date.fromtimestamp(0)
    assert Model(x='1962-01-13').x == date(year=1962, month=1, day=13)
    assert Model(x=datetime(year=1962, month=1, day=13)).x == date(year=1962, month=1, day=13)
    with pytest.raises(ValidationError) as exc_info:
        Model(x=date(year=1970, month=4, day=17))
    assert exc_info.value.errors(include_url=False) == [{'ctx': {'error': HasRepr(repr(ValueError('1970-04-17 is not in the sixties!')))}, 'input': date(1970, 4, 17), 'loc': ('x',), 'msg': 'Value error, 1970-04-17 is not in the sixties!', 'type': 'value_error'}]

def test_annotated_validator_nested() -> None:
    MyInt: Type[int] = Annotated[int, AfterValidator(lambda x: x if x != -1 else 0)]

    def non_decreasing_list(data: List[int]) -> List[int]:
        for prev, cur in zip(data, data[1:]):
            assert cur >= prev
        return data

    class Model(BaseModel):
        x: List[MyInt]

    assert Model(x=[0, -1, 2]).x == [0, 0, 2]
    with pytest.raises(ValidationError) as exc_info:
        Model(x=[0, -1, -2])
    assert exc_info.value.errors(include_url=False) == [{'ctx': {'error': HasRepr(repr(AssertionError('assert -2 >= 0')))}, 'input': [0, -1, -2], 'loc': ('x',), 'msg': 'Assertion failed, assert -2 >= 0', 'type': 'assertion_error'}]

def test_annotated_validator_runs_before_field_validators() -> None:
    MyInt: Type[int] = Annotated[int, AfterValidator(lambda x: x if x != -1 else 0)]

    class Model(BaseModel):
        x: MyInt

        @field_validator('x')
        @classmethod
        def val_x(cls, v: int) -> int:
            assert v != -1
            return v

    assert Model(x=-1).x == 0

@pytest.mark.parametrize('validator, func', [
    (PlainValidator, lambda x: x if x != -1 else 0),
    (WrapValidator, lambda x, nxt: x if x != -1 else 0),
    (BeforeValidator, lambda x: x if x != -1 else 0),
    (AfterValidator, lambda x: x if x != -1 else 0)
])
def test_annotated_validator_typing_cache(validator: Any, func: Callable[[Any], Any]) -> None:
    FancyInt: Type[int] = Annotated[int, validator(func)]

    class FancyIntModel(BaseModel):
        x: FancyInt

    assert FancyIntModel(x=1234).x == 1234
    assert FancyIntModel(x=-1).x == 0
    assert FancyIntModel(x=0).x == 0

def test_simple() -> None:
    class Model(BaseModel):
        a: str

        @field_validator('a')
        @classmethod
        def check_a(cls, v: str) -> str:
            if 'foobar' not in v:
                raise ValueError('"foobar" not found in a')
            return v

    assert Model(a='this is foobar good').a == 'this is foobar good'
    with pytest.raises(ValidationError) as exc_info:
        Model(a='snap')
    assert exc_info.value.errors(include_url=False) == [{'ctx': {'error': HasRepr(repr(ValueError('"foobar" not found in a')))}, 'input': 'snap', 'loc': ('a',), 'msg': 'Value error, "foobar" not found in a', 'type': 'value_error'}]

def test_int_validation() -> None:
    class Model(BaseModel):
        a: int

    with pytest.raises(ValidationError) as exc_info:
        Model(a='snap')
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': ('a',), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'snap'}]
    assert Model(a=3).a == 3
    assert Model(a=True).a == 1
    assert Model(a=False).a == 0
    with pytest.raises(ValidationError) as exc_info:
        Model(a=4.5)
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_from_float', 'loc': ('a',), 'msg': 'Input should be a valid integer, got a number with a fractional part', 'input': 4.5}]
    assert Model(a=2 ** 63 + 100).a == 2 ** 63 + 100

@pytest.mark.parametrize('value', [1e309, float('nan'), float('inf')])
def test_int_overflow_validation(value: float) -> None:
    class Model(BaseModel):
        a: int

    with pytest.raises(ValidationError) as exc_info:
        Model(a=value)
    assert exc_info.value.errors(include_url=False) == [{'type': 'finite_number', 'loc': ('a',), 'msg': 'Input should be a finite number', 'input': value}]

def test_frozenset_validation() -> None:
    class Model(BaseModel):
        a: frozenset[int]

    with pytest.raises(ValidationError) as exc_info:
        Model(a='snap')
    assert exc_info.value.errors(include_url=False) == [{'type': 'frozen_set_type', 'loc': ('a',), 'msg': 'Input should be a valid frozenset', 'input': 'snap'}]
    assert Model(a={1, 2, 3}).a == frozenset({1, 2, 3})
    assert Model(a=frozenset({1, 2, 3})).a == frozenset({1, 2, 3})
    assert Model(a=[4, 5]).a == frozenset({4, 5})
    assert Model(a=(6,)).a == frozenset({6})
    assert Model(a={'1', '2', '3'}).a == frozenset({1, 2, 3})

def test_deque_validation() -> None:
    class Model(BaseModel):
        a: deque[int]

    with pytest.raises(ValidationError) as exc_info:
        Model(a='snap')
    assert exc_info.value.errors(include_url=False) == [{'type': 'list_type', 'loc': ('a',), 'msg': 'Input should be a valid list', 'input': 'snap'}]
    with pytest.raises(ValidationError) as exc_info:
        Model(a=['a'])
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': ('a', 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'a'}]
    with pytest.raises(ValidationError) as exc_info:
        Model(a=('a',))
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': ('a', 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'a'}]
    assert Model(a={'1'}).a == deque([1])
    assert Model(a=[4, 5]).a == deque([4, 5])
    assert Model(a=(6,)).a == deque([6])

def test_validate_whole() -> None:
    class Model(BaseModel):
        a: List[int]

        @field_validator('a', mode='before')
        @classmethod
        def check_a1(cls, v: List[int]) -> List[int]:
            v.append('123')
            return v

        @field_validator('a')
        @classmethod
        def check_a2(cls, v: List[int]) -> List[int]:
            v.append(456)
            return v

    assert Model(a=[1, 2]).a == [1, 2, 123, 456]

def test_validate_pre_error() -> None:
    calls: List[str] = []

    class Model(BaseModel):
        a: List[int]

        @field_validator('a', mode='before')
        @classmethod
        def check_a1(cls, v: List[int]) -> List[int]:
            calls.append(f'check_a1 {v}')
            if 1 in v:
                raise ValueError('a1 broken')
            v[0] += 1
            return v

        @field_validator('a')
        @classmethod
        def check_a2(cls, v: List[int]) -> List[int]:
            calls.append(f'check_a2 {v}')
            if 10 in v:
                raise ValueError('a2 broken')
            return v

    assert Model(a=[3, 8]).a == [4, 8]
    assert calls == ['check_a1 [3, 8]', 'check_a2 [4, 8]']
    calls = []
    with pytest.raises(ValidationError) as exc_info:
        Model(a=[1, 3])
    assert exc_info.value.errors(include_url=False) == [{'ctx': {'error': HasRepr(repr(ValueError('a1 broken')))}, 'input': [1, 3], 'loc': ('a',), 'msg': 'Value error, a1 broken', 'type': 'value_error'}]
    assert calls == ['check_a1 [1, 3]']
    calls = []
    with pytest.raises(ValidationError) as exc_info:
        Model(a=[5, 10])
    assert exc_info.value.errors(include_url=False) == [{'ctx': {'error': HasRepr(repr(ValueError('a2 broken')))}, 'input': [6, 10], 'loc': ('a',), 'msg': 'Value error, a2 broken', 'type': 'value_error'}]
    assert calls == ['check_a1 [5, 10]', 'check_a2 [6, 10]']

@pytest.fixture(scope='session', name='ValidateAssignmentModel')
def validate_assignment_model_fixture() -> Type[BaseModel]:
    class ValidateAssignmentModel(BaseModel):
        a: int = 4
        b: str = ...
        c: int = 0

        @field_validator('b')
        @classmethod
        def b_length(cls, v: str, info: ValidationInfo) -> str:
            values = info.data
            if 'a' in values and len(v) < values['a']:
                raise ValueError('b too short')
            return v

        @field_validator('c')
        @classmethod
        def double_c(cls, v: int) -> int:
            return v * 2

        model_config = ConfigDict(validate_assignment=True, extra='allow')
    return ValidateAssignmentModel

def test_validating_assignment_ok(ValidateAssignmentModel: Type[BaseModel]) -> None:
    p: BaseModel = ValidateAssignmentModel(b='hello')
    assert p.b == 'hello'

def test_validating_assignment_fail(ValidateAssignmentModel: Type[BaseModel]) -> None:
    with pytest.raises(ValidationError):
        ValidateAssignmentModel(a=10, b='hello')
    p: BaseModel = ValidateAssignmentModel(b='hello')
    with pytest.raises(ValidationError):
        p.b = 'x'

def test_validating_assignment_value_change(ValidateAssignmentModel: Type[BaseModel]) -> None:
    p: BaseModel = ValidateAssignmentModel(b='hello', c=2)
    assert p.c == 4
    p = ValidateAssignmentModel(b='hello')
    assert p.c == 0
    p.c = 3
    assert p.c == 6
    assert p.model_dump()['c'] == 6

def test_validating_assignment_extra(ValidateAssignmentModel: Type[BaseModel]) -> None:
    p: BaseModel = ValidateAssignmentModel(b='hello', extra_field=1.23)
    assert p.extra_field == 1.23
    p = ValidateAssignmentModel(b='hello')
    p.extra_field = 1.23
    assert p.extra_field == 1.23
    p.extra_field = 'bye'
    assert p.extra_field == 'bye'
    assert p.model_dump()['extra_field'] == 'bye'

def test_validating_assignment_dict(ValidateAssignmentModel: Type[BaseModel]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ValidateAssignmentModel(a='x', b='xx')
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': ('a',), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'x'}]

def test_validating_assignment_values_dict() -> None:
    class ModelOne(BaseModel):
        a: int

    class ModelTwo(BaseModel):
        b: int
        m: ModelOne

        @field_validator('b')
        @classmethod
        def validate_b(cls, b: int, info: ValidationInfo) -> int:
            if 'm' in info.data:
                return b + info.data['m'].a
            else:
                return b

        model_config = ConfigDict(validate_assignment=True)

    model = ModelTwo(m=ModelOne(a=1), b=2)
    assert model.b == 3
    model.b = 3
    assert model.b == 4

def test_validate_multiple() -> None:
    class Model(BaseModel):
        a: str
        b: str

        @field_validator('a', 'b')
        @classmethod
        def check_a