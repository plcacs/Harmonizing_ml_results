#!/usr/bin/env python
from __future__ import annotations
import contextlib
import re
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from functools import partial, partialmethod
from itertools import product
from os.path import normcase
from typing import Any, Callable, Dict, Iterable, NamedTuple, Optional, Union, cast
from typing_extensions import Annotated, Literal, TypedDict
from unittest.mock import MagicMock

import pytest
from dirty_equals import HasRepr, IsInstance
from pydantic_core import core_schema
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
from pydantic.functional_validators import AfterValidator, BeforeValidator, PlainValidator, WrapValidator

V1_VALIDATOR_DEPRECATION_MATCH: str = 'Pydantic V1 style `@validator` validators are deprecated'

def test_annotated_validator_after() -> None:
    MyInt: Any = Annotated[int, AfterValidator(lambda x, _info: x if x != -1 else 0)]

    class Model(BaseModel):
        x: MyInt
    assert Model(x=0).x == 0
    assert Model(x=-1).x == 0
    assert Model(x=-2).x == -2
    assert Model(x=1).x == 1
    assert Model(x='-1').x == 0

def test_annotated_validator_before() -> None:
    FloatMaybeInf: Any = Annotated[float, BeforeValidator(lambda x, _info: x if x != 'zero' else 0.0)]

    class Model(BaseModel):
        x: FloatMaybeInf
    assert Model(x='zero').x == 0.0
    assert Model(x=1.0).x == 1.0
    assert Model(x='1.0').x == 1.0

def test_annotated_validator_builtin() -> None:
    """https://github.com/pydantic/pydantic/issues/6752"""
    TruncatedFloat: Any = Annotated[float, BeforeValidator(int)]
    DateTimeFromIsoFormat: Any = Annotated[datetime, BeforeValidator(datetime.fromisoformat)]

    class Model(BaseModel):
        x: TruncatedFloat
        y: DateTimeFromIsoFormat
    m = Model(x=1.234, y='2011-11-04T00:05:23')
    assert m.x == 1
    assert m.y == datetime(2011, 11, 4, 0, 5, 23)

def test_annotated_validator_plain() -> None:
    MyInt: Any = Annotated[int, PlainValidator(lambda x, _info: x if x != -1 else 0)]

    class Model(BaseModel):
        x: MyInt
    assert Model(x=0).x == 0
    assert Model(x=-1).x == 0
    assert Model(x=-2).x == -2

def test_annotated_validator_wrap() -> None:
    def sixties_validator(val: Any, handler: Callable[[Any], Any], info: Any) -> Any:
        if val == 'epoch':
            return date.fromtimestamp(0)
        newval = handler(val)
        if not date.fromisoformat('1960-01-01') <= newval < date.fromisoformat('1970-01-01'):
            raise ValueError(f'{val} is not in the sixties!')
        return newval
    SixtiesDateTime: Any = Annotated[date, WrapValidator(sixties_validator)]

    class Model(BaseModel):
        x: SixtiesDateTime
    assert Model(x='epoch').x == date.fromtimestamp(0)
    assert Model(x='1962-01-13').x == date(year=1962, month=1, day=13)
    assert Model(x=datetime(year=1962, month=1, day=13)).x == date(year=1962, month=1, day=13)
    with pytest.raises(ValidationError) as exc_info:
        Model(x=date(year=1970, month=4, day=17))
    expected = [{'ctx': {'error': HasRepr(repr(ValueError('1970-04-17 is not in the sixties!')))},
                 'input': date(1970, 4, 17),
                 'loc': ('x',),
                 'msg': 'Value error, 1970-04-17 is not in the sixties!',
                 'type': 'value_error'}]
    assert exc_info.value.errors(include_url=False) == expected

def test_annotated_validator_nested() -> None:
    MyInt: Any = Annotated[int, AfterValidator(lambda x: x if x != -1 else 0)]

    def non_decreasing_list(data: list[int]) -> list[int]:
        for prev, cur in zip(data, data[1:]):
            assert cur >= prev
        return data

    class Model(BaseModel):
        x: list[MyInt]
    assert Model(x=[0, -1, 2]).x == [0, 0, 2]
    with pytest.raises(ValidationError) as exc_info:
        Model(x=[0, -1, -2])
    expected = [{'ctx': {'error': HasRepr(repr(AssertionError('assert -2 >= 0')))},
                 'input': [0, -1, -2],
                 'loc': ('x',),
                 'msg': 'Assertion failed, assert -2 >= 0',
                 'type': 'assertion_error'}]
    assert exc_info.value.errors(include_url=False) == expected

def test_annotated_validator_runs_before_field_validators() -> None:
    MyInt: Any = Annotated[int, AfterValidator(lambda x: x if x != -1 else 0)]

    class Model(BaseModel):
        x: MyInt

        @field_validator('x')
        @classmethod
        def val_x(cls, v: Any) -> Any:
            assert v != -1
            return v
    assert Model(x=-1).x == 0

@pytest.mark.parametrize('validator, func', [
    (PlainValidator, lambda x, _info: x if x != -1 else 0),
    (WrapValidator, lambda x, nxt, _info: x if x != -1 else 0),
    (BeforeValidator, lambda x, _info: x if x != -1 else 0),
    (AfterValidator, lambda x, _info: x if x != -1 else 0)
])
def test_annotated_validator_typing_cache(validator: Callable[..., Any], func: Callable[..., Any]) -> None:
    FancyInt: Any = Annotated[int, validator(func)]

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
    expected = [{'ctx': {'error': HasRepr(repr(ValueError('"foobar" not found in a')))},
                 'input': 'snap',
                 'loc': ('a',),
                 'msg': 'Value error, "foobar" not found in a',
                 'type': 'value_error'}]
    assert exc_info.value.errors(include_url=False) == expected

def test_int_validation() -> None:
    class Model(BaseModel):
        a: int
    with pytest.raises(ValidationError) as exc_info:
        Model(a='snap')
    expected = [{'type': 'int_parsing',
                 'loc': ('a',),
                 'msg': 'Input should be a valid integer, unable to parse string as an integer',
                 'input': 'snap'}]
    assert exc_info.value.errors(include_url=False) == expected
    assert Model(a=3).a == 3
    assert Model(a=True).a == 1
    assert Model(a=False).a == 0
    with pytest.raises(ValidationError) as exc_info:
        Model(a=4.5)
    expected = [{'type': 'int_from_float',
                 'loc': ('a',),
                 'msg': 'Input should be a valid integer, got a number with a fractional part',
                 'input': 4.5}]
    assert exc_info.value.errors(include_url=False) == expected
    assert Model(a=2 ** 63 + 100).a == 2 ** 63 + 100

@pytest.mark.parametrize('value', [1e309, float('nan'), float('inf')])
def test_int_overflow_validation(value: float) -> None:
    class Model(BaseModel):
        a: int
    with pytest.raises(ValidationError) as exc_info:
        Model(a=value)
    expected = [{'type': 'finite_number',
                 'loc': ('a',),
                 'msg': 'Input should be a finite number',
                 'input': value}]
    assert exc_info.value.errors(include_url=False) == expected

def test_frozenset_validation() -> None:
    class Model(BaseModel):
        a: frozenset[int]
    with pytest.raises(ValidationError) as exc_info:
        Model(a='snap')
    expected = [{'type': 'frozen_set_type',
                 'loc': ('a',),
                 'msg': 'Input should be a valid frozenset',
                 'input': 'snap'}]
    assert exc_info.value.errors(include_url=False) == expected
    assert Model(a={1, 2, 3}).a == frozenset({1, 2, 3})
    assert Model(a=frozenset({1, 2, 3})).a == frozenset({1, 2, 3})
    assert Model(a=[4, 5]).a == frozenset({4, 5})
    assert Model(a=(6,)).a == frozenset({6})
    # converts string numbers to int
    assert Model(a={'1', '2', '3'}).a == frozenset({1, 2, 3})

def test_deque_validation() -> None:
    class Model(BaseModel):
        a: deque[int]
    with pytest.raises(ValidationError) as exc_info:
        Model(a='snap')
    expected = [{'type': 'list_type',
                 'loc': ('a',),
                 'msg': 'Input should be a valid list',
                 'input': 'snap'}]
    assert exc_info.value.errors(include_url=False) == expected
    with pytest.raises(ValidationError) as exc_info:
        Model(a=['a'])
    expected = [{'type': 'int_parsing',
                 'loc': ('a', 0),
                 'msg': 'Input should be a valid integer, unable to parse string as an integer',
                 'input': 'a'}]
    assert exc_info.value.errors(include_url=False) == expected
    with pytest.raises(ValidationError) as exc_info:
        Model(a=('a',))
    expected = [{'type': 'int_parsing',
                 'loc': ('a', 0),
                 'msg': 'Input should be a valid integer, unable to parse string as an integer',
                 'input': 'a'}]
    assert exc_info.value.errors(include_url=False) == expected
    assert Model(a={'1'}).a == deque([1])
    assert Model(a=[4, 5]).a == deque([4, 5])
    assert Model(a=(6,)).a == deque([6])

def test_validate_whole() -> None:
    class Model(BaseModel):
        a: list[Any]

        @field_validator('a', mode='before')
        @classmethod
        def check_a1(cls, v: list[Any]) -> list[Any]:
            v.append('123')
            return v

        @field_validator('a')
        @classmethod
        def check_a2(cls, v: list[Any]) -> list[Any]:
            v.append(456)
            return v
    assert Model(a=[1, 2]).a == [1, 2, 123, 456]

def test_validate_pre_error() -> None:
    calls: list[str] = []

    class Model(BaseModel):
        a: list[int]

        @field_validator('a', mode='before')
        @classmethod
        def check_a1(cls, v: list[int]) -> list[int]:
            calls.append(f'check_a1 {v}')
            if 1 in v:
                raise ValueError('a1 broken')
            v[0] += 1
            return v

        @field_validator('a')
        @classmethod
        def check_a2(cls, v: list[int]) -> list[int]:
            calls.append(f'check_a2 {v}')
            if 10 in v:
                raise ValueError('a2 broken')
            return v
    assert Model(a=[3, 8]).a == [4, 8]
    assert calls == ['check_a1 [3, 8]', 'check_a2 [4, 8]']
    calls.clear()
    with pytest.raises(ValidationError) as exc_info:
        Model(a=[1, 3])
    expected = [{'ctx': {'error': HasRepr(repr(ValueError('a1 broken')))},
                 'input': [1, 3],
                 'loc': ('a',),
                 'msg': 'Value error, a1 broken',
                 'type': 'value_error'}]
    assert exc_info.value.errors(include_url=False) == expected
    assert calls == ['check_a1 [1, 3]']
    calls.clear()
    with pytest.raises(ValidationError) as exc_info:
        Model(a=[5, 10])
    expected = [{'ctx': {'error': HasRepr(repr(ValueError('a2 broken')))},
                 'input': [6, 10],
                 'loc': ('a',),
                 'msg': 'Value error, a2 broken',
                 'type': 'value_error'}]
    assert exc_info.value.errors(include_url=False) == expected
    assert calls == ['check_a1 [5, 10]', 'check_a2 [6, 10]']

@pytest.fixture(scope='session', name='ValidateAssignmentModel')
def validate_assignment_model_fixture() -> type[BaseModel]:
    class ValidateAssignmentModel(BaseModel):
        a: int = 4
        b: str
        c: int = 0

        @field_validator('b')
        @classmethod
        def b_length(cls, v: str, info: ValidationInfo) -> str:
            values: Dict[str, Any] = info.data  # type: ignore
            if 'a' in values and len(v) < values['a']:
                raise ValueError('b too short')
            return v

        @field_validator('c')
        @classmethod
        def double_c(cls, v: int) -> int:
            return v * 2

        model_config = ConfigDict(validate_assignment=True, extra='allow')
    return ValidateAssignmentModel

def test_validating_assignment_ok(ValidateAssignmentModel: type[BaseModel]) -> None:
    p = ValidateAssignmentModel(b='hello')
    assert p.b == 'hello'

def test_validating_assignment_fail(ValidateAssignmentModel: type[BaseModel]) -> None:
    with pytest.raises(ValidationError):
        ValidateAssignmentModel(a=10, b='hello')
    p = ValidateAssignmentModel(b='hello')
    with pytest.raises(ValidationError):
        p.b = 'x'

def test_validating_assignment_value_change(ValidateAssignmentModel: type[BaseModel]) -> None:
    p = ValidateAssignmentModel(b='hello', c=2)
    assert p.c == 4
    p = ValidateAssignmentModel(b='hello')
    assert p.c == 0
    p.c = 3
    assert p.c == 6
    assert p.model_dump()['c'] == 6

def test_validating_assignment_extra(ValidateAssignmentModel: type[BaseModel]) -> None:
    p = ValidateAssignmentModel(b='hello', extra_field=1.23)
    assert p.extra_field == 1.23
    p = ValidateAssignmentModel(b='hello')
    p.extra_field = 1.23
    assert p.extra_field == 1.23
    p.extra_field = 'bye'
    assert p.extra_field == 'bye'
    assert p.model_dump()['extra_field'] == 'bye'

def test_validating_assignment_dict(ValidateAssignmentModel: type[BaseModel]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ValidateAssignmentModel(a='x', b='xx')
    expected = [{'type': 'int_parsing',
                 'loc': ('a',),
                 'msg': 'Input should be a valid integer, unable to parse string as an integer',
                 'input': 'x'}]
    assert exc_info.value.errors(include_url=False) == expected

def test_validating_assignment_values_dict() -> None:
    class ModelOne(BaseModel):
        a: int

    class ModelTwo(BaseModel):
        m: ModelOne
        b: int

        @field_validator('b')
        @classmethod
        def validate_b(cls, b: int, info: ValidationInfo) -> int:
            if 'm' in info.data:
                return b + info.data['m'].a  # type: ignore
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
        def check_a_and_b(cls, v: str, info: ValidationInfo) -> str:
            if len(v) < 4:
                field = cls.model_fields[info.field_name]
                raise AssertionError(f'{field.alias or info.field_name} is too short')
            return v + 'x'
    m = Model(a='1234', b='5678')
    assert m.model_dump() == {'a': '1234x', 'b': '5678x'}
    with pytest.raises(ValidationError) as exc_info:
        Model(a='x', b='x')
    expected = [
        {'ctx': {'error': HasRepr(repr(AssertionError('a is too short')))},
         'input': 'x',
         'loc': ('a',),
         'msg': 'Assertion failed, a is too short',
         'type': 'assertion_error'},
        {'ctx': {'error': HasRepr(repr(AssertionError('b is too short')))},
         'input': 'x',
         'loc': ('b',),
         'msg': 'Assertion failed, b is too short',
         'type': 'assertion_error'}
    ]
    assert exc_info.value.errors(include_url=False) == expected

def test_classmethod() -> None:
    class Model(BaseModel):
        a: str

        @field_validator('a')
        @classmethod
        def check_a(cls, v: str) -> str:
            assert cls is Model
            return v
    m = Model(a='this is foobar good')
    assert m.a == 'this is foobar good'
    m.check_a('x')

def test_use_bare() -> None:
    with pytest.raises(TypeError, match='`@validator` should be used with fields'):
        class Model(BaseModel):
            with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
                @validator
                def checker(cls, v: Any) -> Any:
                    return v

def test_use_bare_field_validator() -> None:
    with pytest.raises(TypeError, match='`@field_validator` should be used with fields'):
        class Model(BaseModel):
            @field_validator
            def checker(cls, v: Any) -> Any:
                return v

def test_use_no_fields() -> None:
    with pytest.raises(TypeError, match=re.escape("validator() missing 1 required positional argument: '__field'")):
        class Model(BaseModel):
            @validator()
            def checker(cls, v: Any) -> Any:
                return v

def test_use_no_fields_field_validator() -> None:
    with pytest.raises(TypeError, match=re.escape("field_validator() missing 1 required positional argument: 'field'")):
        class Model(BaseModel):
            @field_validator()
            def checker(cls, v: Any) -> Any:
                return v

def test_validator_bad_fields_throws_configerror() -> None:
    """
    Attempts to create a validator with fields set as a list of strings,
    rather than just multiple string args. Expects ConfigError to be raised.
    """
    with pytest.raises(TypeError, match='`@validator` fields should be passed as separate string args.'):
        class Model(BaseModel):
            with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
                @validator(['a', 'b'])
                def check_fields(cls, v: Any) -> Any:
                    return v

def test_field_validator_bad_fields_throws_configerror() -> None:
    """
    Attempts to create a validator with fields set as a list of strings,
    rather than just multiple string args. Expects ConfigError to be raised.
    """
    with pytest.raises(TypeError, match='`@field_validator` fields should be passed as separate string args.'):
        class Model(BaseModel):
            @field_validator(['a', 'b'])
            def check_fields(cls, v: Any) -> Any:
                return v

def test_validate_always() -> None:
    check_calls: int = 0

    class Model(BaseModel):
        a: Optional[str] = None
        with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
            @validator('a', pre=True, always=True)
            @classmethod
            def check_a(cls, v: Optional[str]) -> str:
                nonlocal check_calls
                check_calls += 1
                return v or 'xxx'
    m1 = Model()
    assert m1.a == 'xxx'
    assert check_calls == 1
    m2 = Model(a='y')
    assert m2.a == 'y'
    assert check_calls == 2

def test_field_validator_validate_default() -> None:
    check_calls: int = 0

    class Model(BaseModel):
        a: Optional[str] = Field(None, validate_default=True)

        @field_validator('a', mode='before')
        @classmethod
        def check_a(cls, v: Optional[str]) -> str:
            nonlocal check_calls
            check_calls += 1
            return v or 'xxx'
    m1 = Model()
    assert m1.a == 'xxx'
    assert check_calls == 1
    m2 = Model(a='y')
    assert m2.a == 'y'
    assert check_calls == 2

def test_validate_always_on_inheritance() -> None:
    check_calls: int = 0

    class ParentModel(BaseModel):
        a: Optional[str] = None

    class Model(ParentModel):
        with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
            @validator('a', pre=True, always=True)
            @classmethod
            def check_a(cls, v: Optional[str]) -> str:
                nonlocal check_calls
                check_calls += 1
                return v or 'xxx'
    m1 = Model()
    assert m1.a == 'xxx'
    assert check_calls == 1
    m2 = Model(a='y')
    assert m2.a == 'y'
    assert check_calls == 2

def test_field_validator_validate_default_on_inheritance() -> None:
    check_calls: int = 0

    class ParentModel(BaseModel):
        a: Optional[str] = Field(None, validate_default=True)

    class Model(ParentModel):
        @field_validator('a', mode='before')
        @classmethod
        def check_a(cls, v: Optional[str]) -> str:
            nonlocal check_calls
            check_calls += 1
            return v or 'xxx'
    m1 = Model()
    assert m1.a == 'xxx'
    assert check_calls == 1
    m2 = Model(a='y')
    assert m2.a == 'y'
    assert check_calls == 2

def test_validate_not_always() -> None:
    check_calls: int = 0

    class Model(BaseModel):
        a: Optional[str] = None

        @field_validator('a', mode='before')
        @classmethod
        def check_a(cls, v: Optional[str]) -> Optional[str]:
            nonlocal check_calls
            check_calls += 1
            return v or 'xxx'
    m1 = Model()
    assert m1.a is None
    assert check_calls == 0
    m2 = Model(a='y')
    assert m2.a == 'y'
    assert check_calls == 1

@pytest.mark.parametrize(
    'decorator, pytest_warns',
    [
        (validator, pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH)),
        (field_validator, contextlib.nullcontext())
    ]
)
def test_wildcard_validators(decorator: Callable[..., Any], pytest_warns: Any) -> None:
    calls: list[tuple[str, Any]] = []
    with pytest_warns:

        class Model(BaseModel):
            a: str
            b: int

            @decorator('a')
            def check_a(cls, v: Any) -> Any:
                calls.append(('check_a', v))
                return v

            @decorator('*')
            def check_all(cls, v: Any) -> Any:
                calls.append(('check_all', v))
                return v

            @decorator('*', 'a')
            def check_all_a(cls, v: Any) -> Any:
                calls.append(('check_all_a', v))
                return v
    m = Model(a='abc', b='123')
    # note: automatic conversion of '123' to int happens for b
    assert m.model_dump() == dict(a='abc', b=123)
    assert calls == [('check_a', 'abc'), ('check_all', 'abc'), ('check_all_a', 'abc'), ('check_all', 123), ('check_all_a', 123)]

@pytest.mark.parametrize(
    'decorator, pytest_warns',
    [
        (validator, pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH)),
        (field_validator, contextlib.nullcontext())
    ]
)
def test_wildcard_validator_error(decorator: Callable[..., Any], pytest_warns: Any) -> None:
    with pytest_warns:

        class Model(BaseModel):
            a: str

            @decorator('*')
            def check_all(cls, v: str) -> str:
                if 'foobar' not in v:
                    raise ValueError('"foobar" not found in a')
                return v
    # b is missing so it will cause missing validation error
    assert Model(a='foobar a').a == 'foobar a'
    with pytest.raises(ValidationError) as exc_info:
        Model(a='snap')
    expected = [{'ctx': {'error': HasRepr(repr(ValueError('"foobar" not found in a')))},
                 'input': 'snap',
                 'loc': ('a',),
                 'msg': 'Value error, "foobar" not found in a',
                 'type': 'value_error'},
                {'type': 'missing',
                 'loc': ('b',),
                 'msg': 'Field required',
                 'input': {'a': 'snap'}}]
    assert exc_info.value.errors(include_url=False) == expected

def test_invalid_field() -> None:
    msg: str = ("Decorators defined with incorrect fields: tests.test_validators.test_invalid_field.<locals>.Model:"
                "\\d+.check_b \\(use check_fields=False if you're inheriting from the model and intended this\\)")
    with pytest.raises(errors.PydanticUserError, match=msg):
        class Model(BaseModel):
            @field_validator('b')
            def check_b(cls, v: Any) -> Any:
                return v

def test_validate_child() -> None:
    class Parent(BaseModel):
        a: str

    class Child(Parent):
        @field_validator('a')
        @classmethod
        def check_a(cls, v: str) -> str:
            if 'foobar' not in v:
                raise ValueError('"foobar" not found in a')
            return v
    assert Parent(a='this is not a child').a == 'this is not a child'
    assert Child(a='this is foobar good').a == 'this is foobar good'
    with pytest.raises(ValidationError):
        Child(a='snap')

def test_validate_child_extra() -> None:
    class Parent(BaseModel):
        a: str

        @field_validator('a')
        @classmethod
        def check_a_one(cls, v: str) -> str:
            if 'foobar' not in v:
                raise ValueError('"foobar" not found in a')
            return v

    class Child(Parent):
        @field_validator('a')
        @classmethod
        def check_a_two(cls, v: str) -> str:
            return v.upper()
    assert Parent(a='this is foobar good').a == 'this is foobar good'
    assert Child(a='this is foobar good').a == 'THIS IS FOOBAR GOOD'
    with pytest.raises(ValidationError):
        Child(a='snap')

def test_validate_all() -> None:
    class MyModel(BaseModel):
        x: Any

        @field_validator('*')
        @classmethod
        def validate_all(cls, v: Any) -> Any:
            return v * 2
    assert MyModel(x=10).x == 20

def test_validate_child_all() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        class Parent(BaseModel):
            a: str

        class Child(Parent):
            @validator('*')
            @classmethod
            def check_a(cls, v: str) -> str:
                if 'foobar' not in v:
                    raise ValueError('"foobar" not found in a')
                return v
        assert Parent(a='this is not a child').a == 'this is not a child'
        assert Child(a='this is foobar good').a == 'this is foobar good'
        with pytest.raises(ValidationError):
            Child(a='snap')

    class Parent(BaseModel):
        a: str

    class Child(Parent):
        @field_validator('*')
        @classmethod
        def check_a(cls, v: str) -> str:
            if 'foobar' not in v:
                raise ValueError('"foobar" not found in a')
            return v
    assert Parent(a='this is not a child').a == 'this is not a child'
    assert Child(a='this is foobar good').a == 'this is foobar good'
    with pytest.raises(ValidationError):
        Child(a='snap')

def test_validate_parent() -> None:
    class Parent(BaseModel):
        a: str

        @field_validator('a')
        @classmethod
        def check_a(cls, v: str) -> str:
            if 'foobar' not in v:
                raise ValueError('"foobar" not found in a')
            return v

    class Child(Parent):
        pass
    assert Parent(a='this is foobar good').a == 'this is foobar good'
    assert Child(a='this is foobar good').a == 'this is foobar good'
    with pytest.raises(ValidationError):
        Parent(a='snap')
    with pytest.raises(ValidationError):
        Child(a='snap')

def test_validate_parent_all() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        class Parent(BaseModel):
            a: str

            @validator('*')
            @classmethod
            def check_a(cls, v: str) -> str:
                if 'foobar' not in v:
                    raise ValueError('"foobar" not found in a')
                return v

        class Child(Parent):
            pass
        assert Parent(a='this is foobar good').a == 'this is foobar good'
        assert Child(a='this is foobar good').a == 'this is foobar good'
        with pytest.raises(ValidationError):
            Parent(a='snap')
        with pytest.raises(ValidationError):
            Child(a='snap')

    class Parent(BaseModel):
        a: str

        @field_validator('*')
        @classmethod
        def check_a(cls, v: str) -> str:
            if 'foobar' not in v:
                raise ValueError('"foobar" not found in a')
            return v

    class Child(Parent):
        pass
    assert Parent(a='this is foobar good').a == 'this is foobar good'
    assert Child(a='this is foobar good').a == 'this is foobar good'
    with pytest.raises(ValidationError):
        Parent(a='snap')
    with pytest.raises(ValidationError):
        Child(a='snap')

def test_inheritance_keep() -> None:
    class Parent(BaseModel):
        x: int

        @field_validator('x')
        @classmethod
        def add_to_a(cls, v: int) -> int:
            return v + 1

    class Child(Parent):
        pass
    assert Child(x=0).x == 1

def test_inheritance_replace() -> None:
    """We promise that if you add a validator
    with the same _function_ name as an existing validator
    it replaces the existing validator and is run instead of it.
    """
    class Parent(BaseModel):
        x: list[str]

        @field_validator('x')
        @classmethod
        def parent_val_before(cls, v: list[str]) -> list[str]:
            v.append('parent before')
            return v

        @field_validator('x')
        @classmethod
        def val(cls, v: list[str]) -> list[str]:
            v.append('parent')
            return v

        @field_validator('x')
        @classmethod
        def parent_val_after(cls, v: list[str]) -> list[str]:
            v.append('parent after')
            return v

    class Child(Parent):
        @field_validator('x')
        @classmethod
        def child_val_before(cls, v: list[str]) -> list[str]:
            v.append('child before')
            return v

        @field_validator('x')
        @classmethod
        def val(cls, v: list[str]) -> list[str]:
            v.append('child')
            return v

        @field_validator('x')
        @classmethod
        def child_val_after(cls, v: list[str]) -> list[str]:
            v.append('child after')
            return v
    assert Parent(x=[]).x == ['parent before', 'parent', 'parent after']
    assert Child(x=[]).x == ['parent before', 'child', 'parent after', 'child before', 'child after']

def test_inheritance_replace_root_validator() -> None:
    """
    We promise that if you add a validator
    with the same _function_ name as an existing validator
    it replaces the existing validator and is run instead of it.
    """
    with pytest.warns(PydanticDeprecatedSince20):
        class Parent(BaseModel):
            x: list[str]

            @root_validator(skip_on_failure=True)
            def parent_val_before(cls, values: dict[str, Any]) -> dict[str, Any]:
                values['x'].append('parent before')
                return values

            @root_validator(skip_on_failure=True)
            def val(cls, values: dict[str, Any]) -> dict[str, Any]:
                values['x'].append('parent')
                return values

            @root_validator(skip_on_failure=True)
            def parent_val_after(cls, values: dict[str, Any]) -> dict[str, Any]:
                values['x'].append('parent after')
                return values

        class Child(Parent):
            @root_validator(skip_on_failure=True)
            def child_val_before(cls, values: dict[str, Any]) -> dict[str, Any]:
                values['x'].append('child before')
                return values

            @root_validator(skip_on_failure=True)
            def val(cls, values: dict[str, Any]) -> dict[str, Any]:
                values['x'].append('child')
                return values

            @root_validator(skip_on_failure=True)
            def child_val_after(cls, values: dict[str, Any]) -> dict[str, Any]:
                values['x'].append('child after')
                return values
    assert Parent(x=[]).x == ['parent before', 'parent', 'parent after']
    assert Child(x=[]).x == ['parent before', 'child', 'parent after', 'child before', 'child after']

def test_validation_each_item() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        class Model(BaseModel):
            foobar: Dict[Any, int]

            @validator('foobar', each_item=True)
            @classmethod
            def check_foobar(cls, v: int) -> int:
                return v + 1
    assert Model(foobar={1: 1}).foobar == {1: 2}

def test_validation_each_item_invalid_type() -> None:
    with pytest.raises(TypeError, match=re.escape('@validator(..., each_item=True)` cannot be applied to fields with a schema of int')):
        with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
            class Model(BaseModel):
                foobar: int

                @validator('foobar', each_item=True)
                @classmethod
                def check_foobar(cls, v: Any) -> Any:
                    ...

def test_validation_each_item_nullable() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        class Model(BaseModel):
            foobar: list[int]

            @validator('foobar', each_item=True)
            @classmethod
            def check_foobar(cls, v: int) -> int:
                return v + 1
    assert Model(foobar=[1]).foobar == [2]

def test_validation_each_item_one_sublevel() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        class Model(BaseModel):
            foobar: list[tuple[int, int]]

            @validator('foobar', each_item=True)
            @classmethod
            def check_foobar(cls, v: tuple[int, int]) -> tuple[int, int]:
                v1, v2 = v
                assert v1 == v2
                return v
    assert Model(foobar=[(1, 1), (2, 2)]).foobar == [(1, 1), (2, 2)]

def test_key_validation() -> None:
    class Model(BaseModel):
        foobar: Dict[int, int]

        @field_validator('foobar')
        @classmethod
        def check_foobar(cls, value: Dict[int, int]) -> Dict[int, int]:
            return {k + 1: v + 1 for k, v in value.items()}
    assert Model(foobar={1: 1}).foobar == {2: 2}

def test_validator_always_optional() -> None:
    check_calls: int = 0

    class Model(BaseModel):
        a: Optional[str] = None
        with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
            @validator('a', pre=True, always=True)
            @classmethod
            def check_a(cls, v: Optional[str]) -> str:
                nonlocal check_calls
                check_calls += 1
                return v or 'default value'
    m1 = Model(a='y')
    assert m1.a == 'y'
    m2 = Model()
    assert m2.a == 'default value'
    assert check_calls == 2

def test_field_validator_validate_default_optional() -> None:
    check_calls: int = 0

    class Model(BaseModel):
        a: Optional[str] = Field(None, validate_default=True)

        @field_validator('a', mode='before')
        @classmethod
        def check_a(cls, v: Optional[str]) -> str:
            nonlocal check_calls
            check_calls += 1
            return v or 'default value'
    m1 = Model(a='y')
    assert m1.a == 'y'
    m2 = Model()
    assert m2.a == 'default value'
    assert check_calls == 2

def test_validator_always_pre() -> None:
    check_calls: int = 0

    class Model(BaseModel):
        a: Optional[str] = None
        with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
            @validator('a', pre=True, always=True)
            @classmethod
            def check_a(cls, v: Optional[str]) -> str:
                nonlocal check_calls
                check_calls += 1
                return v or 'default value'
    _ = Model(a='y')
    _ = Model()
    assert check_calls == 2

def test_field_validator_validate_default_pre() -> None:
    check_calls: int = 0

    class Model(BaseModel]:
        a: Optional[str] = Field(None, validate_default=True)

        @field_validator('a', mode='before')
        @classmethod
        def check_a(cls, v: Optional[str]) -> str:
            nonlocal check_calls
            check_calls += 1
            return v or 'default value'
    _ = Model(a='y')
    _ = Model()
    assert check_calls == 2

def test_validator_always_post() -> None:
    class Model(BaseModel):
        a: str = ''
        with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
            @validator('a', always=True)
            @classmethod
            def check_a(cls, v: str) -> str:
                return v or 'default value'
    m1 = Model(a='y')
    assert m1.a == 'y'
    m2 = Model()
    assert m2.a == 'default value'

def test_field_validator_validate_default_post() -> None:
    class Model(BaseModel):
        a: str = Field('', validate_default=True)

        @field_validator('a')
        @classmethod
        def check_a(cls, v: str) -> str:
            return v or 'default value'
    m1 = Model(a='y')
    assert m1.a == 'y'
    m2 = Model()
    assert m2.a == 'default value'

def test_validator_always_post_optional() -> None:
    class Model(BaseModel):
        a: Optional[str] = None
        with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
            @validator('a', pre=True, always=True)
            @classmethod
            def check_a(cls, v: Optional[str]) -> str:
                return 'default value' if v is None else v
    m1 = Model(a='y')
    assert m1.a == 'y'
    m2 = Model()
    assert m2.a == 'default value'

def test_field_validator_validate_default_post_optional() -> None:
    class Model(BaseModel):
        a: Optional[str] = Field(None, validate_default=True)

        @field_validator('a', mode='before')
        @classmethod
        def check_a(cls, v: Optional[str]) -> str:
            return v or 'default value'
    m1 = Model(a='y')
    assert m1.a == 'y'
    m2 = Model()
    assert m2.a == 'default value'

def test_datetime_validator() -> None:
    check_calls: int = 0

    class Model(BaseModel):
        d: Optional[datetime] = None
        with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
            @validator('d', pre=True, always=True)
            @classmethod
            def check_d(cls, v: Optional[Any]) -> datetime:
                nonlocal check_calls
                check_calls += 1
                return v or datetime(2032, 1, 1)
    m1 = Model(d='2023-01-01T00:00:00')
    assert m1.d == datetime(2023, 1, 1)
    assert check_calls == 1
    m2 = Model()
    assert m2.d == datetime(2032, 1, 1)
    assert check_calls == 2
    m3 = Model(d=datetime(2023, 1, 1))
    assert m3.d == datetime(2023, 1, 1)
    assert check_calls == 3

def test_datetime_field_validator() -> None:
    check_calls: int = 0

    class Model(BaseModel):
        d: Optional[datetime] = Field(None, validate_default=True)

        @field_validator('d', mode='before')
        @classmethod
        def check_d(cls, v: Optional[Any]) -> datetime:
            nonlocal check_calls
            check_calls += 1
            return v or datetime(2032, 1, 1)
    m1 = Model(d='2023-01-01T00:00:00')
    assert m1.d == datetime(2023, 1, 1)
    assert check_calls == 1
    m2 = Model()
    assert m2.d == datetime(2032, 1, 1)
    assert check_calls == 2
    m3 = Model(d=datetime(2023, 1, 1))
    assert m3.d == datetime(2023, 1, 1)
    assert check_calls == 3

def test_pre_called_once() -> None:
    check_calls: int = 0

    class Model(BaseModel):
        a: tuple[int, ...]
        @field_validator('a', mode='before')
        @classmethod
        def check_a(cls, v: Any) -> tuple[int, ...]:
            nonlocal check_calls
            check_calls += 1
            # Assuming conversion happens only once
            return tuple(int(x) for x in v)
    m = Model(a=['1', '2', '3'])
    assert m.a == (1, 2, 3)
    assert check_calls == 1

def test_assert_raises_validation_error() -> None:
    class Model(BaseModel):
        a: str

        @field_validator('a')
        @classmethod
        def check_a(cls, v: str) -> str:
            if v != 'a':
                raise AssertionError('invalid a')
            return v
    Model(a='a')
    with pytest.raises(ValidationError) as exc_info:
        Model(a='snap')
    expected = [{'ctx': {'error': HasRepr(repr(AssertionError('invalid a')))},
                 'input': 'snap',
                 'loc': ('a',),
                 'msg': 'Assertion failed, invalid a',
                 'type': 'assertion_error'}]
    assert exc_info.value.errors(include_url=False) == expected

def test_root_validator() -> None:
    root_val_values: list[Dict[str, Any]] = []

    class Model(BaseModel):
        a: int = 1
        b: str
        c: str

        @field_validator('b')
        @classmethod
        def repeat_b(cls, v: str) -> str:
            return v * 2
        with pytest.warns(PydanticDeprecatedSince20):
            @root_validator(skip_on_failure=True)
            def example_root_validator(cls, values: dict[str, Any]) -> dict[str, Any]:
                root_val_values.append(values.copy())
                if 'snap' in values.get('b', ''):
                    raise ValueError('foobar')
                return dict(values, b='changed')

            @root_validator(skip_on_failure=True)
            def example_root_validator2(cls, values: dict[str, Any]) -> dict[str, Any]:
                root_val_values.append(values.copy())
                if 'snap' in values.get('c', ''):
                    raise ValueError('foobar2')
                return dict(values, c='changed')
    m = Model(a='123', b='bar', c='baz')
    assert m.model_dump() == {'a': 123, 'b': 'changed', 'c': 'changed'}
    with pytest.raises(ValidationError) as exc_info:
        Model(b='snap dragon', c='snap dragon2')
    expected = [{'ctx': {'error': HasRepr(repr(ValueError('foobar')))},
                 'input': {'b': 'snap dragon', 'c': 'snap dragon2'},
                 'loc': (),
                 'msg': 'Value error, foobar',
                 'type': 'value_error'}]
    assert exc_info.value.errors(include_url=False) == expected
    with pytest.raises(ValidationError) as exc_info:
        Model(a='broken', b='bar', c='baz')
    expected = [{'type': 'int_parsing',
                 'loc': ('a',),
                 'msg': 'Input should be a valid integer, unable to parse string as an integer',
                 'input': 'broken'}]
    assert exc_info.value.errors(include_url=False) == expected
    assert root_val_values == [
        {'a': 123, 'b': 'barbar', 'c': 'baz'},
        {'a': 123, 'b': 'changed', 'c': 'baz'},
        {'a': 1, 'b': 'snap dragonsnap dragon', 'c': 'snap dragon2'}
    ]

def test_root_validator_subclass() -> None:
    """
    https://github.com/pydantic/pydantic/issues/5388
    """
    class Parent(BaseModel):
        x: int
        expected: Any

        with pytest.warns(PydanticDeprecatedSince20):
            @root_validator(skip_on_failure=True)
            @classmethod
            def root_val(cls, values: dict[str, Any]) -> dict[str, Any]:
                assert cls is values['expected']
                return values

    class Child1(Parent):
        pass

    class Child2(Parent):
        with pytest.warns(PydanticDeprecatedSince20):
            @root_validator(skip_on_failure=True)
            @classmethod
            def root_val(cls, values: dict[str, Any]) -> dict[str, Any]:
                assert cls is Child2
                values['x'] = values['x'] * 2
                return values

    class Child3(Parent):
        @classmethod
        def root_val(cls, values: dict[str, Any]) -> dict[str, Any]:
            assert cls is Child3
            values['x'] = values['x'] * 3
            return values
    assert Parent(x=1, expected=Parent).x == 1
    assert Child1(x=1, expected=Child1).x == 1
    assert Child2(x=1, expected=Child2).x == 2
    assert Child3(x=1, expected=Child3).x == 3

def test_root_validator_pre() -> None:
    root_val_values: list[Dict[str, Any]] = []

    class Model(BaseModel):
        a: int = 1
        b: str

        @field_validator('b')
        @classmethod
        def repeat_b(cls, v: str) -> str:
            return v * 2
        with pytest.warns(PydanticDeprecatedSince20):
            @root_validator(pre=True)
            def root_validator(cls, values: dict[str, Any]) -> dict[str, Any]:
                root_val_values.append(values.copy())
                if 'snap' in values.get('b', ''):
                    raise ValueError('foobar')
                return {'a': 42, 'b': 'changed'}
    m = Model(a='123', b='bar')
    assert m.model_dump() == {'a': 42, 'b': 'changedchanged'}
    with pytest.raises(ValidationError) as exc_info:
        Model(b='snap dragon')
    assert root_val_values == [{'a': '123', 'b': 'bar'}, {'b': 'snap dragon'}]
    expected = [{'ctx': {'error': HasRepr(repr(ValueError('foobar')))},
                 'input': {'b': 'snap dragon'},
                 'loc': (),
                 'msg': 'Value error, foobar',
                 'type': 'value_error'}]
    assert exc_info.value.errors(include_url=False) == expected

def test_root_validator_types() -> None:
    root_val_values: Optional[tuple[Any, str]] = None

    class Model(BaseModel):
        a: int = 1
        with pytest.warns(PydanticDeprecatedSince20):
            @root_validator(skip_on_failure=True)
            def root_validator(cls, values: dict[str, Any]) -> dict[str, Any]:
                nonlocal root_val_values
                root_val_values = (cls, repr(values))
                return values
        model_config = ConfigDict(extra='allow')
    m = Model(b='bar', c='wobble')
    assert m.model_dump() == {'a': 1, 'b': 'bar', 'c': 'wobble'}
    assert root_val_values == (Model, "{'a': 1, 'b': 'bar', 'c': 'wobble'}")

def test_root_validator_returns_none_exception() -> None:
    class Model(BaseModel):
        a: int = 1
        with pytest.warns(PydanticDeprecatedSince20):
            @root_validator(skip_on_failure=True)
            def root_validator_repeated(cls, values: dict[str, Any]) -> None:
                return None
    with pytest.raises(TypeError, match="(:?__dict__ must be set to a dictionary, not a 'NoneType')|(:?setting dictionary to a non-dict)"):
        Model()

def test_model_validator_returns_ignore() -> None:
    class Model(BaseModel]:
        a: int = 1

        @model_validator(mode='after')
        def model_validator_return_none(self) -> Any:
            return None
    with pytest.warns(UserWarning, match='A custom validator is returning a value other than `self`'):
        m = Model(a=2)
    assert m.model_dump() == {'a': 2}

def reusable_validator(num: int) -> int:
    return num * 2

def test_reuse_global_validators() -> None:
    class Model(BaseModel):
        x: int
        y: int
        double_x = field_validator('x')(reusable_validator)
        double_y = field_validator('y')(reusable_validator)
    m = Model(x=1, y=1)
    assert dict(m) == {'x': 2, 'y': 2}

@pytest.mark.parametrize('validator_classmethod,root_validator_classmethod', product([True, False], [True, False]))
def test_root_validator_classmethod(validator_classmethod: bool, root_validator_classmethod: bool) -> None:
    root_val_values: list[Dict[str, Any]] = []

    class Model(BaseModel):
        a: int = 1

        def repeat_b(cls, v: str) -> str:
            return v * 2
        if validator_classmethod:
            repeat_b = classmethod(repeat_b)
        repeat_b = field_validator('b')(repeat_b)

        def example_root_validator(cls, values: dict[str, Any]) -> dict[str, Any]:
            root_val_values.append(values.copy())
            if 'snap' in values.get('b', ''):
                raise ValueError('foobar')
            return dict(values, b='changed')
        if root_validator_classmethod:
            example_root_validator = classmethod(example_root_validator)
        with pytest.warns(PydanticDeprecatedSince20):
            example_root_validator = root_validator(skip_on_failure=True)(example_root_validator)
    m1 = Model(a='123', b='bar')
    assert m1.model_dump() == {'a': 123, 'b': 'changed'}
    with pytest.raises(ValidationError) as exc_info:
        Model(b='snap dragon')
    expected = [{'ctx': {'error': HasRepr(repr(ValueError('foobar')))},
                 'input': {'b': 'snap dragon'},
                 'loc': (),
                 'msg': 'Value error, foobar',
                 'type': 'value_error'}]
    assert exc_info.value.errors(include_url=False) == expected
    with pytest.raises(ValidationError) as exc_info:
        Model(a='broken', b='bar')
    expected = [{'type': 'int_parsing',
                 'loc': ('a',),
                 'msg': 'Input should be a valid integer, unable to parse string as an integer',
                 'input': 'broken'}]
    assert exc_info.value.errors(include_url=False) == expected
    assert root_val_values == [{'a': 123, 'b': 'barbar'}, {'a': 1, 'b': 'snap dragonsnap dragon'}]

def test_assignment_validator_cls() -> None:
    validator_calls: int = 0

    class Model(BaseModel):
        name: str
        model_config = ConfigDict(validate_assignment=True)

        @field_validator('name')
        @classmethod
        def check_foo(cls, value: str) -> str:
            nonlocal validator_calls
            validator_calls += 1
            assert cls == Model
            return value
    m = Model(name='hello')
    m.name = 'goodbye'
    assert validator_calls == 2

def test_literal_validator() -> None:
    class Model(BaseModel):
        a: Literal['foo']
    Model(a='foo')
    with pytest.raises(ValidationError) as exc_info:
        Model(a='nope')
    expected = [{'type': 'literal_error',
                 'loc': ('a',),
                 'msg': "Input should be 'foo'",
                 'input': 'nope',
                 'ctx': {'expected': "'foo'"}}]
    assert exc_info.value.errors(include_url=False) == expected

def test_literal_validator_str_enum() -> None:
    class Bar(str, Enum):
        FIZ = 'fiz'
        FUZ = 'fuz'

    class Foo(BaseModel):
        bar: Bar
        barfiz: Bar
        fizfuz: Bar
    my_foo = Foo.model_validate({'bar': 'fiz', 'barfiz': 'fiz', 'fizfuz': 'fiz'})
    assert my_foo.bar is Bar.FIZ
    assert my_foo.barfiz is Bar.FIZ
    assert my_foo.fizfuz is Bar.FIZ
    my_foo = Foo.model_validate({'bar': 'fiz', 'barfiz': 'fiz', 'fizfuz': 'fuz'})
    assert my_foo.bar is Bar.FIZ
    assert my_foo.barfiz is Bar.FIZ
    assert my_foo.fizfuz is Bar.FUZ

def test_nested_literal_validator() -> None:
    L1: Any = Literal['foo']
    L2: Any = Literal['bar']

    class Model(BaseModel):
        a: Union[L1, L2]
    Model(a='foo')
    with pytest.raises(ValidationError) as exc_info:
        Model(a='nope')
    expected = [{'type': 'literal_error',
                 'loc': ('a',),
                 'msg': "Input should be 'foo' or 'bar'",
                 'input': 'nope',
                 'ctx': {'expected': "'foo' or 'bar'"}}]
    assert exc_info.value.errors(include_url=False) == expected

def test_union_literal_with_constraints() -> None:
    class Model(BaseModel, validate_assignment=True):
        x: int = Field(frozen=True)
    m = Model(x=42)
    with pytest.raises(ValidationError) as exc_info:
        m.x += 1
    expected = [{'input': 43, 'loc': ('x',), 'msg': 'Field is frozen', 'type': 'frozen_field'}]
    assert exc_info.value.errors(include_url=False) == expected

def test_field_that_is_being_validated_is_excluded_from_validator_values() -> None:
    check_values = MagicMock()

    class Model(BaseModel):
        bar: Any = Field(alias='pika')
        model_config = ConfigDict(validate_assignment=True)

        @field_validator('foo')
        @classmethod
        def validate_foo(cls, v: Any, info: ValidationInfo) -> Any:
            check_values({**info.data})
            return v

        @field_validator('bar')
        @classmethod
        def validate_bar(cls, v: Any, info: ValidationInfo) -> Any:
            check_values({**info.data})
            return v
    model = Model(foo='foo_value', pika='bar_value', baz='baz_value')
    check_values.reset_mock()
    assert list(dict(model).items()) == [('foo', 'foo_value'), ('bar', 'bar_value'), ('baz', 'baz_value')]
    model.foo = 'new_foo_value'
    check_values.assert_called_once_with({'bar': 'bar_value', 'baz': 'baz_value'})
    check_values.reset_mock()
    model.bar = 'new_bar_value'
    check_values.assert_called_once_with({'foo': 'new_foo_value', 'baz': 'baz_value'})
    assert list(dict(model).items()) == [('foo', 'new_foo_value'), ('bar', 'new_bar_value'), ('baz', 'baz_value')]

def test_exceptions_in_field_validators_restore_original_field_value() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(validate_assignment=True)

        @field_validator('foo')
        @classmethod
        def validate_foo(cls, v: str) -> str:
            if v == 'raise_exception':
                raise RuntimeError('test error')
            return v
    model = Model(foo='foo')
    with pytest.raises(RuntimeError, match='test error'):
        model.foo = 'raise_exception'
    assert model.foo == 'foo'

def test_overridden_root_validators() -> None:
    validate_stub: Any = MagicMock()

    class A(BaseModel):
        x: str

        @model_validator(mode='before')
        @classmethod
        def pre_root(cls, values: dict[str, Any]) -> dict[str, Any]:
            validate_stub('A', 'pre')
            return values

        @model_validator(mode='after')
        def post_root(self) -> BaseModel:
            validate_stub('A', 'post')
            return self

    class B(A):
        @model_validator(mode='before')
        @classmethod
        def pre_root(cls, values: dict[str, Any]) -> dict[str, Any]:
            validate_stub('B', 'pre')
            return values

        @model_validator(mode='after')
        def post_root(self) -> BaseModel:
            validate_stub('B', 'post')
            return self
    A(x='pika')
    assert validate_stub.call_args_list == [[('A', 'pre'), {}], [('A', 'post'), {}]]
    validate_stub.reset_mock()
    B(x='pika')
    assert validate_stub.call_args_list == [[('B', 'pre'), {}], [('B', 'post'), {}]]

def test_validating_assignment_pre_root_validator_fail() -> None:
    class Model(BaseModel):
        current_value: Any = Field(alias='current')
        model_config = ConfigDict(validate_assignment=True)
        with pytest.warns(PydanticDeprecatedSince20):
            @root_validator(pre=True)
            def values_are_not_string(cls, values: dict[str, Any]) -> dict[str, Any]:
                if any(isinstance(x, str) for x in values.values()):
                    raise ValueError('values cannot be a string')
                return values
    m = Model(current=100, max_value=200)
    with pytest.raises(ValidationError) as exc_info:
        m.current_value = '100'
    expected = [{'ctx': {'error': HasRepr(repr(ValueError('values cannot be a string')))},
                 'input': {'current_value': '100', 'max_value': 200.0},
                 'loc': (),
                 'msg': 'Value error, values cannot be a string',
                 'type': 'value_error'}]
    assert exc_info.value.errors(include_url=False) == expected

def test_validating_assignment_model_validator_before_fail() -> None:
    class Model(BaseModel):
        current_value: Any = Field(alias='current')
        model_config = ConfigDict(validate_assignment=True)

        @model_validator(mode='before')
        @classmethod
        def values_are_not_string(cls, values: dict[str, Any]) -> dict[str, Any]:
            assert isinstance(values, dict)
            if any(isinstance(x, str) for x in values.values()):
                raise ValueError('values cannot be a string')
            return values
    m = Model(current=100, max_value=200)
    with pytest.raises(ValidationError) as exc_info:
        m.current_value = '100'
    expected = [{'ctx': {'error': HasRepr(repr(ValueError('values cannot be a string')))},
                 'input': {'current_value': '100', 'max_value': 200.0},
                 'loc': (),
                 'msg': 'Value error, values cannot be a string',
                 'type': 'value_error'}]
    assert exc_info.value.errors(include_url=False) == expected

@pytest.mark.parametrize('kwargs', [{'skip_on_failure': False}, {'skip_on_failure': False, 'pre': False}, {'pre': False}])
def test_root_validator_skip_on_failure_invalid(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match='MUST specify `skip_on_failure=True`'):
        with pytest.warns(PydanticDeprecatedSince20, match='Pydantic V1 style `@root_validator` validators are deprecated.'):
            class Model(BaseModel):
                @root_validator(**kwargs)
                def root_val(cls, values: dict[str, Any]) -> dict[str, Any]:
                    return values

@pytest.mark.parametrize('kwargs', [{'skip_on_failure': True}, {'skip_on_failure': True, 'pre': False}, {'skip_on_failure': False, 'pre': True}, {'pre': True}])
def test_root_validator_skip_on_failure_valid(kwargs: dict[str, Any]) -> None:
    with pytest.warns(PydanticDeprecatedSince20, match='Pydantic V1 style `@root_validator` validators are deprecated.'):
        class Model(BaseModel):
            @root_validator(**kwargs)
            def root_val(cls, values: dict[str, Any]) -> dict[str, Any]:
                return values

def test_model_validator_many_values_change() -> None:
    """It should run root_validator on assignment and update ALL concerned fields"""

    class Rectangle(BaseModel):
        width: int
        height: int
        area: Optional[int] = None
        model_config = ConfigDict(validate_assignment=True)

        @model_validator(mode='after')
        def set_area(self) -> Rectangle:
            self.__dict__['area'] = self.width * self.height
            return self
    r = Rectangle(width=1, height=1)
    assert r.area == 1
    r.height = 5
    assert r.area == 5

def _get_source_line(filename: str, lineno: int) -> str:
    with open(filename) as f:
        for _ in range(lineno - 1):
            f.readline()
        return f.readline()

def test_v1_validator_deprecated() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH) as w:
        class Point(BaseModel):
            x: int
            y: int

            @validator('x')
            @classmethod
            def check_x(cls, x: int, values: dict[str, Any]) -> int:
                assert x * 2 == values['y']
                return x
    assert Point(x=1, y=2).model_dump() == {'x': 1, 'y': 2}
    warnings = w.list
    assert len(warnings) == 1
    w_item = warnings[0]
    assert normcase(w_item.filename) == normcase(__file__)
    source = _get_source_line(w_item.filename, w_item.lineno)
    assert "@validator('x')" in source

def test_info_field_name_data_before() -> None:
    """
    Test accessing info.field_name and info.data.
    We only test the `before` validator because they
    all share the same implementation.
    """
    class Model(BaseModel):
        a: str
        b: Any

        @field_validator('b', mode='before')
        @classmethod
        def check_a(cls, v: Any, info: ValidationInfo) -> Any:
            assert v == b'but my barbaz is better'
            assert info.field_name == 'b'
            assert info.data == {'a': 'your foobar is good'}
            return 'just kidding!'
    m = Model(a=b'your foobar is good', b=b'but my barbaz is better')
    assert m.b == 'just kidding!'

def test_decorator_proxy() -> None:
    def val(v: int) -> int:
        return v + 1

    class Model(BaseModel):
        x: int

        @field_validator('x')
        @staticmethod
        def val1(v: int) -> int:
            return v + 1

        @field_validator('x')
        @classmethod
        def val2(cls, v: int) -> int:
            return v + 1
        val3 = field_validator('x')(val)
    assert Model.val1(1) == 2
    assert Model.val2(1) == 2
    assert Model.val3(1) == 2

def test_root_validator_self() -> None:
    with pytest.raises(TypeError, match='`@root_validator` cannot be applied to instance methods'):
        with pytest.warns(PydanticDeprecatedSince20):
            class Model(BaseModel):
                a: int = 1

                @root_validator(skip_on_failure=True)
                def root_validator(self, values: dict[str, Any]) -> dict[str, Any]:
                    return values

def test_validator_self() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        with pytest.raises(TypeError, match='`@validator` cannot be applied to instance methods'):
            class Model(BaseModel):
                a: int = 1

                @validator('a')
                def check_a(self, values: Any) -> Any:
                    return values

def test_field_validator_self() -> None:
    with pytest.raises(TypeError, match='`@field_validator` cannot be applied to instance methods'):
        class Model(BaseModel):
            a: int = 1

            @field_validator('a')
            def check_a(self, values: Any) -> Any:
                return values

def test_v1_validator_signature_kwargs_not_allowed() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        with pytest.raises(TypeError, match='Unsupported signature for V1 style validator'):
            class Model(BaseModel):
                a: int

                @validator('a')
                def check_a(cls, value: int, foo: Any) -> int:
                    ...

def test_v1_validator_signature_kwargs1() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        class Model(BaseModel):
            a: int
            b: int

            @validator('b')
            def check_b(cls, value: int, **kwargs: Any) -> int:
                assert kwargs == {'values': {'a': 1}}
                assert value == 2
                return value + 1
    assert Model(a=1, b=2).model_dump() == {'a': 1, 'b': 3}

def test_v1_validator_signature_kwargs2() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        class Model(BaseModel):
            a: int
            b: int

            @validator('b')
            def check_b(cls, value: int, values: dict[str, Any], **kwargs: Any) -> int:
                assert kwargs == {}
                assert values == {'a': 1}
                assert value == 2
                return value + 1
    assert Model(a=1, b=2).model_dump() == {'a': 1, 'b': 3}

def test_v1_validator_signature_with_values() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        class Model(BaseModel):
            a: int
            b: int

            @validator('b')
            def check_b(cls, value: int, values: dict[str, Any]) -> int:
                assert values == {'a': 1}
                assert value == 2
                return value + 1
    assert Model(a=1, b=2).model_dump() == {'a': 1, 'b': 3}

def test_v1_validator_signature_with_values_kw_only() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        class Model(BaseModel):
            a: int
            b: int

            @validator('b')
            def check_b(cls, value: int, *, values: dict[str, Any]) -> int:
                assert values == {'a': 1}
                assert value == 2
                return value + 1
    assert Model(a=1, b=2).model_dump() == {'a': 1, 'b': 3}

def test_v1_validator_signature_with_field() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        with pytest.raises(TypeError, match='The `field` and `config` parameters are not available in Pydantic V2'):
            class Model(BaseModel):
                a: int
                b: int

                @validator('b')
                def check_b(cls, value: int, field: Any) -> int:
                    ...

def test_v1_validator_signature_with_config() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        with pytest.raises(TypeError, match='The `field` and `config` parameters are not available in Pydantic V2'):
            class Model(BaseModel):
                a: int
                b: int

                @validator('b')
                def check_b(cls, value: int, config: Any) -> int:
                    ...

def test_model_config_validate_default() -> None:
    class Model(BaseModel):
        x: int = -1

        @field_validator('x')
        @classmethod
        def force_x_positive(cls, v: int) -> int:
            assert v > 0
            return v
    assert Model().x == -1

    class ValidatingModel(Model):
        model_config = ConfigDict(validate_default=True)
    with pytest.raises(ValidationError) as exc_info:
        ValidatingModel()
    expected = [{'ctx': {'error': HasRepr(repr(AssertionError('assert -1 > 0')))},
                 'input': -1,
                 'loc': ('x',),
                 'msg': 'Assertion failed, assert -1 > 0',
                 'type': 'assertion_error'}]
    assert exc_info.value.errors(include_url=False) == expected

def partial_val_func1(value: Any, allowed: int) -> Any:
    assert value == allowed
    return value

def partial_val_func2(value: Any, *, allowed: int) -> Any:
    assert value == allowed
    return value

def partial_values_val_func1(value: Any, values: dict[str, Any], *, allowed: int) -> Any:
    assert isinstance(values, dict)
    assert value == allowed
    return value

def partial_values_val_func2(value: Any, *, values: dict[str, Any], allowed: int) -> Any:
    assert isinstance(values, dict)
    assert value == allowed
    return value

def partial_info_val_func(value: Any, info: ValidationInfo, *, allowed: int) -> Any:
    assert isinstance(info.data, dict)
    assert value == allowed
    return value

def partial_cls_val_func1(cls: Any, value: Any, allowed: int, expected_cls: str) -> Any:
    assert cls.__name__ == expected_cls
    assert value == allowed
    return value

def partial_cls_val_func2(cls: Any, value: Any, *, allowed: int, expected_cls: str) -> Any:
    assert cls.__name__ == expected_cls
    assert value == allowed
    return value

def partial_cls_values_val_func1(cls: Any, value: Any, values: dict[str, Any], *, allowed: int, expected_cls: str) -> Any:
    assert cls.__name__ == expected_cls
    assert isinstance(values, dict)
    assert value == allowed
    return value

def partial_cls_values_val_func2(cls: Any, value: Any, *, values: dict[str, Any], allowed: int, expected_cls: str) -> Any:
    assert cls.__name__ == expected_cls
    assert isinstance(values, dict)
    assert value == allowed
    return value

def partial_cls_info_val_func(cls: Any, value: Any, info: ValidationInfo, *, allowed: int, expected_cls: str) -> Any:
    assert cls.__name__ == expected_cls
    assert isinstance(info.data, dict)
    assert value == allowed
    return value

@pytest.mark.parametrize('func', [partial_val_func1, partial_val_func2, partial_info_val_func])
def test_functools_partial_validator_v2(func: Callable[..., Any]) -> None:
    class Model(BaseModel):
        x: int
        val = field_validator('x')(partial(func, allowed=42))
    Model(x=42)
    with pytest.raises(ValidationError):
        Model(x=123)

@pytest.mark.parametrize('func', [partial_val_func1, partial_val_func2, partial_info_val_func])
def test_functools_partialmethod_validator_v2(func: Callable[..., Any]) -> None:
    class Model(BaseModel]:
        x: int
        val = field_validator('x')(partialmethod(func, allowed=42))
    Model(x=42)
    with pytest.raises(ValidationError):
        Model(x=123)

@pytest.mark.parametrize('func', [partial_cls_val_func1, partial_cls_val_func2, partial_cls_info_val_func])
def test_functools_partialmethod_validator_v2_cls_method(func: Callable[..., Any]) -> None:
    class Model(BaseModel):
        x: int
        val = field_validator('x')(partialmethod(classmethod(func), allowed=42, expected_cls='Model'))
    Model(x=42)
    with pytest.raises(ValidationError):
        Model(x=123)

@pytest.mark.parametrize('func', [partial_val_func1, partial_val_func2, partial_values_val_func1, partial_values_val_func2])
def test_functools_partial_validator_v1(func: Callable[..., Any]) -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        class Model(BaseModel):
            x: int
            val = validator('x')(partial(func, allowed=42))
    Model(x=42)
    with pytest.raises(ValidationError):
        Model(x=123)

@pytest.mark.parametrize('func', [partial_val_func1, partial_val_func2, partial_values_val_func1, partial_values_val_func2])
def test_functools_partialmethod_validator_v1(func: Callable[..., Any]) -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        class Model(BaseModel):
            x: int
            val = validator('x')(partialmethod(func, allowed=42))
        Model(x=42)
        with pytest.raises(ValidationError):
            Model(x=123)

@pytest.mark.parametrize('func', [partial_cls_val_func1, partial_cls_val_func2, partial_cls_values_val_func1, partial_cls_values_val_func2])
def test_functools_partialmethod_validator_v1_cls_method(func: Callable[..., Any]) -> None:
    with pytest.warns(PydanticDeprecatedSince20, match=V1_VALIDATOR_DEPRECATION_MATCH):
        class Model(BaseModel):
            x: int
            val = validator('x')(partialmethod(classmethod(func), allowed=42, expected_cls='Model'))
    Model(x=42)
    with pytest.raises(ValidationError):
        Model(x=123)

def test_validator_allow_reuse_inheritance() -> None:
    class Parent(BaseModel):
        x: int

        @field_validator('x')
        def val(cls, v: int) -> int:
            return v + 1

    class Child(Parent):
        @field_validator('x')
        def val(cls, v: int) -> int:
            assert v == 1
            v = super().val(v)
            assert v == 2
            return 4
    assert Parent(x=1).model_dump() == {'x': 2}
    assert Child(x=1).model_dump() == {'x': 4}

def test_validator_allow_reuse_same_field() -> None:
    with pytest.warns(UserWarning, match='`val_x` overrides an existing Pydantic `@field_validator` decorator'):
        class Model(BaseModel):
            x: int

            @field_validator('x')
            def val_x(cls, v: int) -> int:
                return v + 1

            @field_validator('x')
            def val_x(cls, v: int) -> int:
                return v + 2
        assert Model(x=1).model_dump() == {'x': 3}

def test_validator_allow_reuse_different_field_1() -> None:
    with pytest.warns(UserWarning, match='`val` overrides an existing Pydantic `@field_validator` decorator'):
        class Model(BaseModel):
            x: int
            y: int

            @field_validator('x')
            def val(cls, v: int) -> int:
                return v + 1

            @field_validator('y')
            def val(cls, v: int) -> int:
                return v + 2
    assert Model(x=1, y=2).model_dump() == {'x': 1, 'y': 4}

def test_validator_allow_reuse_different_field_2() -> None:
    with pytest.warns(UserWarning, match='`val_x` overrides an existing Pydantic `@field_validator` decorator'):
        def val(cls: Any, v: int) -> int:
            return v + 2

        class Model(BaseModel):
            x: int
            y: int

            @field_validator('x')
            def val_x(cls, v: int) -> int:
                return v + 1
            val_x = field_validator('y')(val)
    assert Model(x=1, y=2).model_dump() == {'x': 1, 'y': 4}

def test_validator_allow_reuse_different_field_3() -> None:
    with pytest.warns(UserWarning, match='`val_x` overrides an existing Pydantic `@field_validator` decorator'):
        def val1(v: int) -> int:
            return v + 1

        def val2(v: int) -> int:
            return v + 2

        class Model(BaseModel):
            x: int
            y: int
            val_x = field_validator('x')(val1)
            val_x = field_validator('y')(val2)
    assert Model(x=1, y=2).model_dump() == {'x': 1, 'y': 4}

def test_validator_allow_reuse_different_field_4() -> None:
    def val(v: int) -> int:
        return v + 1

    class Model(BaseModel):
        x: int
        y: int
        val_x = field_validator('x')(val)
        not_val_x = field_validator('y')(val)
    assert Model(x=1, y=2).model_dump() == {'x': 2, 'y': 3}

@pytest.mark.filterwarnings('ignore:Pydantic V1 style `@root_validator` validators are deprecated.*:pydantic.warnings.PydanticDeprecatedSince20')
def test_root_validator_allow_reuse_same_field() -> None:
    with pytest.warns(UserWarning, match='`root_val` overrides an existing Pydantic `@root_validator` decorator'):
        class Model(BaseModel]:
            x: int

            @root_validator(skip_on_failure=True)
            def root_val(cls, v: dict[str, Any]) -> dict[str, Any]:
                v['x'] += 1
                return v

            @root_validator(skip_on_failure=True)
            def root_val(cls, v: dict[str, Any]) -> dict[str, Any]:
                v['x'] += 2
                return v
        assert Model(x=1).model_dump() == {'x': 3}

def test_root_validator_allow_reuse_inheritance() -> None:
    with pytest.warns(PydanticDeprecatedSince20):
        class Parent(BaseModel):
            x: int

            @root_validator(skip_on_failure=True)
            def root_val(cls, v: dict[str, Any]) -> dict[str, Any]:
                v['x'] += 1
                return v
    with pytest.warns(PydanticDeprecatedSince20):
        class Child(Parent):
            @root_validator(skip_on_failure=True)
            def root_val(cls, v: dict[str, Any]) -> dict[str, Any]:
                assert v == {'x': 1}
                v = super().root_val(v)
                assert v == {'x': 2}
                return {'x': 4}
    assert Parent(x=1).model_dump() == {'x': 2}
    assert Child(x=1).model_dump() == {'x': 4}

def test_bare_root_validator() -> None:
    with pytest.raises(PydanticUserError, match=re.escape('If you use `@root_validator` with pre=False (the default) you MUST specify `skip_on_failure=True`. Note that `@root_validator` is deprecated and should be replaced with `@model_validator`.')):
        with pytest.warns(PydanticDeprecatedSince20, match='Pydantic V1 style `@root_validator` validators are deprecated.'):
            class Model(BaseModel):
                @root_validator
                @classmethod
                def validate_values(cls, values: dict[str, Any]) -> dict[str, Any]:
                    return values

def test_validator_with_underscore_name() -> None:
    """
    https://github.com/pydantic/pydantic/issues/5252
    """
    def f(name: str) -> str:
        return name.lower()

    class Model(BaseModel):
        name: str
        _normalize_name = field_validator('name')(f)
    assert Model(name='Adrian').name == 'adrian'

@pytest.mark.parametrize(
    'mode,config,input_str',
    (
        ('before', {}, "type=value_error, input_value='123', input_type=str"),
        ('before', {'hide_input_in_errors': False}, "type=value_error, input_value='123', input_type=str"),
        ('before', {'hide_input_in_errors': True}, 'type=value_error'),
        ('after', {}, "type=value_error, input_value='123', input_type=str"),
        ('after', {'hide_input_in_errors': False}, "type=value_error, input_value='123', input_type=str"),
        ('after', {'hide_input_in_errors': True}, 'type=value_error'),
        ('plain', {}, "type=value_error, input_value='123', input_type=str"),
        ('plain', {'hide_input_in_errors': False}, "type=value_error, input_value='123', input_type=str"),
        ('plain', {'hide_input_in_errors': True}, 'type=value_error'),
    )
)
def test_validator_function_error_hide_input(mode: str, config: dict[str, Any], input_str: str) -> None:
    class Model(BaseModel):
        x: Any
        model_config = ConfigDict(**config)

        @field_validator('x', mode=mode)
        @classmethod
        def check_a1(cls, v: Any) -> Any:
            raise ValueError('foo')
    with pytest.raises(ValidationError, match=re.escape(f'Value error, foo [{input_str}]')):
        Model(x='123')

def foobar_validate(value: Any, info: ValidationInfo) -> dict[str, Any]:
    data: Any = info.data
    if isinstance(data, dict):
        data = data.copy()
    return {'value': value, 'field_name': info.field_name, 'data': data}

class Foobar:
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> Any:
        return core_schema.with_info_plain_validator_function(foobar_validate, field_name=handler.field_name)

def test_custom_type_field_name_model() -> None:
    class MyModel(BaseModel):
        foobar: Any
        tuple_nesting: tuple[Any, ...]

    m = MyModel(foobar=1, tuple_nesting=(1, 2))
    assert m.foobar == {'value': 1, 'field_name': 'foobar', 'data': {}}

def test_custom_type_field_name_model_nested() -> None:
    class MyModel(BaseModel):
        x: str
        tuple_nested: tuple[Any, ...]

    m = MyModel(x='123', tuple_nested=(1, 2))
    assert m.tuple_nested[1] == {'value': 2, 'field_name': 'tuple_nested', 'data': {'x': 123}}

def test_custom_type_field_name_typed_dict() -> None:
    class MyDict(TypedDict):
        ...

    ta = TypeAdapter(MyDict)
    m = ta.validate_python({'x': '123', 'foobar': 1})
    assert m['foobar'] == {'value': 1, 'field_name': 'foobar', 'data': {'x': 123}}

def test_custom_type_field_name_dataclass() -> None:
    @pydantic_dataclass
    class MyDc:
        ...

    ta = TypeAdapter(MyDc)
    m = ta.validate_python({'x': '123', 'foobar': 1})
    assert m.foobar == {'value': 1, 'field_name': 'foobar', 'data': {'x': 123}}

def test_custom_type_field_name_named_tuple() -> None:
    class MyNamedTuple(NamedTuple):
        ...

    ta = TypeAdapter(MyNamedTuple)
    m = ta.validate_python({'x': '123', 'foobar': 1})
    assert m.foobar == {'value': 1, 'field_name': 'foobar', 'data': None}

def test_custom_type_field_name_validate_call() -> None:
    @validate_call
    def foobar(x: Any, y: Any) -> tuple[Any, Any]:
        return (x, y)
    result = foobar(1, 2)
    assert result == (1, {'value': 2, 'field_name': 'y', 'data': None})

def test_after_validator_field_name() -> None:
    class MyModel(BaseModel):
        foobar: Any
        x: str
    m = MyModel(x='123', foobar='1')
    assert m.foobar == {'value': 1, 'field_name': 'foobar', 'data': {'x': 123}}

def test_before_validator_field_name() -> None:
    class MyModel(BaseModel):
        foobar: Any
        x: str
    m = MyModel(x='123', foobar='1')
    assert m.foobar == {'value': '1', 'field_name': 'foobar', 'data': {'x': 123}}

def test_plain_validator_field_name() -> None:
    class MyModel(BaseModel):
        foobar: Any
        x: str
    m = MyModel(x='123', foobar='1')
    assert m.foobar == {'value': '1', 'field_name': 'foobar', 'data': {'x': 123}}

def validate_wrap(value: Any, handler: Callable[[Any], Any], info: ValidationInfo) -> dict[str, Any]:
    data: Any = info.data
    if isinstance(data, dict):
        data = data.copy()
    return {'value': handler(value), 'field_name': info.field_name, 'data': data}

def test_wrap_validator_field_name() -> None:
    class MyModel(BaseModel):
        foobar: Any
        x: str
    m = MyModel(x='123', foobar='1')
    # Here the wrap validator should convert '1' to integer using the handler
    assert m.foobar == {'value': 1, 'field_name': 'foobar', 'data': {'x': 123}}

def test_validate_default_raises_for_basemodel() -> None:
    class Model(BaseModel):
        value_a: Any
        value_b: Any
        with pytest.warns(PydanticDeprecatedSince20):
            @field_validator('value_a', mode='after')
            def value_a_validator(cls, value: Any) -> Any:
                raise AssertionError

            @field_validator('value_b', mode='after')
            def value_b_validator(cls, value: Any) -> Any:
                raise AssertionError
    with pytest.raises(ValidationError) as exc_info:
        Model()
    expected = [
        {'type': 'missing', 'loc': ('value_0',), 'msg': 'Field required', 'input': {}},
        {'type': 'assertion_error', 'loc': ('value_a',), 'msg': 'Assertion failed, ', 'input': None, 'ctx': {'error': IsInstance(AssertionError)}},
        {'type': 'assertion_error', 'loc': ('value_b',), 'msg': 'Assertion failed, ', 'input': None, 'ctx': {'error': IsInstance(AssertionError)}}
    ]
    assert exc_info.value.errors(include_url=False) == expected

def test_validate_default_raises_for_dataclasses() -> None:
    @pydantic_dataclass
    class Model:
        # Fields will be auto-detected; validators are applied
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        @field_validator('value_a', mode='after')
        def value_a_validator(cls, value: Any) -> Any:
            raise AssertionError

        @field_validator('value_b', mode='after')
        def value_b_validator(cls, value: Any) -> Any:
            raise AssertionError
    with pytest.raises(ValidationError) as exc_info:
        Model()
    expected = [
        {'type': 'missing', 'loc': ('value_0',), 'msg': 'Field required', 'input': HasRepr('ArgsKwargs(())')},
        {'type': 'assertion_error', 'loc': ('value_a',), 'msg': 'Assertion failed, ', 'input': None, 'ctx': {'error': IsInstance(AssertionError)}},
        {'type': 'assertion_error', 'loc': ('value_b',), 'msg': 'Assertion failed, ', 'input': None, 'ctx': {'error': IsInstance(AssertionError)}}
    ]
    assert exc_info.value.errors(include_url=False) == expected

def test_plain_validator_plain_serializer() -> None:
    """https://github.com/pydantic/pydantic/issues/8512"""
    ser_type = str
    serializer = PlainSerializer(lambda x: ser_type(int(x)), return_type=ser_type)
    validator_obj = PlainValidator(lambda x: bool(int(x)))
    class Blah(BaseModel):
        foo: Any
        bar: Any
    blah = Blah(foo='0', bar='1')
    data = blah.model_dump()
    assert isinstance(data['foo'], ser_type)
    assert isinstance(data['bar'], ser_type)

def test_plain_validator_plain_serializer_single_ser_call() -> None:
    """https://github.com/pydantic/pydantic/issues/10385"""
    ser_count = 0
    def ser(v: Any) -> Any:
        nonlocal ser_count
        ser_count += 1
        return v
    class Model(BaseModel):
        foo: Any
    model = Model(foo=True)
    data = model.model_dump()
    assert data == {'foo': True}
    assert ser_count == 1

@pytest.mark.xfail(reason='https://github.com/pydantic/pydantic/issues/10428')
def test_plain_validator_with_filter_dict_schema() -> None:
    class MyDict:
        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> Any:
            return core_schema.dict_schema(
                keys_schema=handler.generate_schema(str),
                values_schema=handler.generate_schema(int),
                serialization=core_schema.filter_dict_schema(include={'a'})
            )
    class Model(BaseModel):
        f: MyDict
    assert Model(f={'a': 1, 'b': 1}).model_dump() == {'f': {'a': 1}}

def test_plain_validator_with_unsupported_type() -> None:
    class UnsupportedClass:
        pass
    PreviouslySupportedType: Any = Annotated[UnsupportedClass, PlainValidator(lambda _: UnsupportedClass())]
    type_adapter = TypeAdapter(PreviouslySupportedType)
    model = type_adapter.validate_python('abcdefg')
    assert isinstance(model, UnsupportedClass)
    assert isinstance(type_adapter.dump_python(model), UnsupportedClass)

def test_validator_with_default_values() -> None:
    def validate_x(v: int, unrelated_arg: int = 1, other_unrelated_arg: int = 2) -> int:
        assert v != -1
        return v
    class Model(BaseModel):
        x: int
        val_x = field_validator('x')(validate_x)
    with pytest.raises(ValidationError):
        Model(x=-1)

def test_field_validator_input_type_invalid_mode() -> None:
    with pytest.raises(PydanticUserError, match=re.escape("`json_schema_input_type` can't be used when mode is set to 'after'")):
        class Model(BaseModel):
            a: int
            @field_validator('a', mode='after', json_schema_input_type=Union[int, str])
            @classmethod
            def validate_a(cls, value: int) -> int:
                ...

def test_non_self_return_val_warns() -> None:
    class Child(BaseModel):
        name: str

        @model_validator(mode='after')
        def validate_model(self) -> BaseModel:
            return Child.model_construct(name='different')
    with pytest.warns(UserWarning, match='A custom validator is returning a value other than `self`'):
        c = Child(name='name')
        assert c.name == 'name'
