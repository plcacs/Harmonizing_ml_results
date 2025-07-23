import json
import platform
import re
import sys
import warnings
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from functools import cache, cached_property, partial
from typing import Annotated, Any, Callable, ClassVar, Final, Generic, Literal, Optional, Type, TypeVar, Union, get_type_hints

from uuid import UUID, uuid4

import pytest
from pydantic_core import CoreSchema, core_schema
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    PrivateAttr,
    PydanticDeprecatedSince211,
    PydanticUndefinedAnnotation,
    PydanticUserError,
    SecretStr,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    constr,
    field_validator,
)
from pydantic._internal._generate_schema import GenerateSchema
from pydantic._internal._mock_val_ser import MockCoreSchema
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.v1 import BaseModel as BaseModelV1

KT = TypeVar('KT')
VT = TypeVar('VT')


class ArbitraryType:
    pass


class OtherClass:
    pass


def test_success() -> None:
    class Model(BaseModel):
        a: float
        b: int = 10

    m: Model = Model(a=10.2)
    assert m.a == 10.2
    assert m.b == 10


@pytest.fixture(name='UltraSimpleModel', scope='session')
def ultra_simple_model_fixture() -> Type[BaseModel]:
    class UltraSimpleModel(BaseModel):
        a: float
        b: int = 10

    return UltraSimpleModel


def test_ultra_simple_missing(UltraSimpleModel: Type[BaseModel]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        UltraSimpleModel()
    assert exc_info.value.errors(include_url=False) == [
        {'loc': ('a',), 'msg': 'Field required', 'type': 'missing', 'input': {}}
    ]
    assert str(exc_info.value) == '1 validation error for UltraSimpleModel\na\n  Field required [type=missing, input_value={}, input_type=dict]'


def test_ultra_simple_failed(UltraSimpleModel: Type[BaseModel]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        UltraSimpleModel(a='x', b='x')
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'float_parsing', 'loc': ('a',), 'msg': 'Input should be a valid number, unable to parse string as a number', 'input': 'x'},
        {'type': 'int_parsing', 'loc': ('b',), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'x'}
    ]


def test_ultra_simple_repr(UltraSimpleModel: Type[BaseModel]) -> None:
    m: UltraSimpleModel = UltraSimpleModel(a=10.2)
    assert str(m) == 'a=10.2 b=10'
    assert repr(m) == 'UltraSimpleModel(a=10.2, b=10)'
    assert repr(UltraSimpleModel.model_fields['a']) == 'FieldInfo(annotation=float, required=True)'
    assert repr(UltraSimpleModel.model_fields['b']) == 'FieldInfo(annotation=int, required=False, default=10)'
    assert dict(m) == {'a': 10.2, 'b': 10}
    assert m.model_dump() == {'a': 10.2, 'b': 10}
    assert m.model_dump_json() == '{"a":10.2,"b":10}'
    assert str(m) == 'a=10.2 b=10'


def test_recursive_repr() -> None:
    class A(BaseModel):
        a: Optional['A'] = None

    class B(BaseModel):
        a: Optional[A] = None

    a: A = A()
    a.a = a
    b: B = B(a=a)
    assert re.match("B\\(a=A\\(a='<Recursion on A with id=\\d+>'\\)\\)", repr(b)) is not None


def test_default_factory_field() -> None:
    def myfunc() -> int:
        return 1

    class Model(BaseModel):
        a: int = Field(default_factory=myfunc)

    m: Model = Model()
    assert str(m) == 'a=1'
    assert repr(Model.model_fields['a']) == 'FieldInfo(annotation=int, required=False, default_factory=myfunc)'
    assert dict(m) == {'a': 1}
    assert m.model_dump_json() == '{"a":1}'


def test_comparing(UltraSimpleModel: Type[BaseModel]) -> None:
    m: UltraSimpleModel = UltraSimpleModel(a=10.2, b='100')
    assert m.model_dump() == {'a': 10.2, 'b': 100}
    assert m != {'a': 10.2, 'b': 100}  # type: ignore
    assert m == UltraSimpleModel(a=10.2, b=100)


@pytest.fixture(scope='session', name='NoneCheckModel')
def none_check_model_fix() -> Type[BaseModel]:
    class NoneCheckModel(BaseModel):
        existing_str_value: str = 'foo'
        required_str_value: str
        required_str_none_value: Optional[str]
        existing_bytes_value: bytes = b'foo'
        required_bytes_value: bytes
        required_bytes_none_value: Optional[bytes]

    return NoneCheckModel


def test_nullable_strings_success(NoneCheckModel: Type[BaseModel]) -> None:
    m = NoneCheckModel(required_str_value='v1', required_str_none_value=None, required_bytes_value='v2', required_bytes_none_value=None)
    assert m.required_str_value == 'v1'
    assert m.required_str_none_value is None
    assert m.required_bytes_value == b'v2'
    assert m.required_bytes_none_value is None


def test_nullable_strings_fails(NoneCheckModel: Type[BaseModel]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        NoneCheckModel(required_str_value=None, required_str_none_value=None, required_bytes_value=None, required_bytes_none_value=None)
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'string_type', 'loc': ('required_str_value',), 'msg': 'Input should be a valid string', 'input': None},
        {'type': 'bytes_type', 'loc': ('required_bytes_value',), 'msg': 'Input should be a valid bytes', 'input': None}
    ]


@pytest.fixture(name='ParentModel', scope='session')
def parent_sub_model_fixture() -> Type[BaseModel]:
    class UltraSimpleModel(BaseModel):
        a: float
        b: int = 10

    class ParentModel(BaseModel):
        grape: bool
        banana: UltraSimpleModel

    return ParentModel


def test_parent_sub_model(ParentModel: Type[BaseModel]) -> None:
    m: ParentModel = ParentModel(grape=1, banana={'a': 1})
    assert m.grape is True
    assert m.banana.a == 1.0
    assert m.banana.b == 10
    assert repr(m) == 'ParentModel(grape=True, banana=UltraSimpleModel(a=1.0, b=10))'


def test_parent_sub_model_fails(ParentModel: Type[BaseModel]) -> None:
    with pytest.raises(ValidationError):
        ParentModel(grape=1, banana=123)


def test_not_required() -> None:
    class Model(BaseModel):
        a: Optional[float] = None

    assert Model(a=12.2).a == 12.2
    assert Model().a is None
    with pytest.raises(ValidationError) as exc_info:
        Model(a=None)
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'float_type', 'loc': ('a',), 'msg': 'Input should be a valid number', 'input': None}
    ]


def test_allow_extra() -> None:
    class Model(BaseModel):
        a: float
        model_config = ConfigDict(extra='allow')

    m: Model = Model(a='10.2', b=12)
    assert m.__dict__ == {'a': 10.2}
    assert m.__pydantic_extra__ == {'b': 12}
    assert m.a == 10.2
    assert m.b == 12
    assert m.model_extra == {'b': 12}
    m.c = 42
    assert 'c' not in m.__dict__
    assert m.__pydantic_extra__ == {'b': 12, 'c': 42}
    assert m.model_dump() == {'a': 10.2, 'b': 12, 'c': 42}


@pytest.mark.parametrize('extra', ['ignore', 'forbid', 'allow'])
def test_allow_extra_from_attributes(extra: Literal['ignore', 'forbid', 'allow']) -> None:
    class Model(BaseModel):
        model_config = ConfigDict(extra=extra, from_attributes=True)

    class TestClass:
        a: float = 1.0
        b: int = 12

    m: Model = Model.model_validate(TestClass())
    assert m.a == 1.0
    assert not hasattr(m, 'b')


def test_allow_extra_repr() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(extra='allow')
        a: float

    m: Model = Model(a='10.2', b=12)
    assert str(m) == 'a=10.2 b=12'


def test_forbidden_extra_success() -> None:
    class ForbiddenExtra(BaseModel):
        foo: str = 'whatever'
        model_config = ConfigDict(extra='forbid')

    m: ForbiddenExtra = ForbiddenExtra()
    assert m.foo == 'whatever'


def test_forbidden_extra_fails() -> None:
    class ForbiddenExtra(BaseModel):
        foo: str = 'whatever'
        model_config = ConfigDict(extra='forbid')

    with pytest.raises(ValidationError) as exc_info:
        ForbiddenExtra(foo='ok', bar='wrong', spam='xx')
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'extra_forbidden', 'loc': ('bar',), 'msg': 'Extra inputs are not permitted', 'input': 'wrong'},
        {'type': 'extra_forbidden', 'loc': ('spam',), 'msg': 'Extra inputs are not permitted', 'input': 'xx'}
    ]


def test_assign_extra_no_validate() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(validate_assignment=True)

    model: Model = Model(a=0.2)
    with pytest.raises(ValidationError, match="b\\s+Object has no attribute 'b'"):
        model.b = 2


def test_assign_extra_validate() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(validate_assignment=True)

    model: Model = Model(a=0.2)
    with pytest.raises(ValidationError, match="b\\s+Object has no attribute 'b'"):
        model.b = 2


def test_model_property_attribute_error() -> None:
    class Model(BaseModel):
        @property
        def a_property(self) -> Any:
            raise AttributeError('Internal Error')

    with pytest.raises(AttributeError, match='Internal Error'):
        Model().a_property


def test_extra_allowed() -> None:
    class Model(BaseModel):
        a: float
        model_config = ConfigDict(extra='allow')

    model: Model = Model(a=0.2, b=0.1)
    assert model.b == 0.1
    assert not hasattr(model, 'c')
    model.c = 1
    assert hasattr(model, 'c')
    assert model.c == 1


def test_reassign_instance_method_with_extra_allow() -> None:
    class Model(BaseModel):
        name: str
        model_config = ConfigDict(extra='allow')

        def not_extra_func(self) -> str:
            return f'hello {self.name}'

    def not_extra_func_replacement(self_sub: Model) -> str:
        return f'hi {self_sub.name}'

    m: Model = Model(name='james')
    assert m.not_extra_func() == 'hello james'
    m.not_extra_func = partial(not_extra_func_replacement, m)
    assert m.not_extra_func() == 'hi james'
    assert 'not_extra_func' in m.__dict__


def test_extra_ignored() -> None:
    class Model(BaseModel):
        a: float
        model_config = ConfigDict(extra='ignore')

    model: Model = Model(a=0.2, b=0.1)
    assert not hasattr(model, 'b')
    with pytest.raises(ValueError, match='"Model" object has no field "b"'):
        model.b = 1
    assert model.model_extra is None


def test_field_order_is_preserved_with_extra() -> None:
    """This test covers https://github.com/pydantic/pydantic/issues/1234."""
    class Model(BaseModel):
        model_config = ConfigDict(extra='allow')
        a: float
        b: str
        c: float
        d: int

    model: Model = Model(a=1, b='2', c=3.0, d=4)
    assert repr(model) == "Model(a=1, b='2', c=3.0, d=4)"
    assert str(model.model_dump()) == "{'a': 1, 'b': '2', 'c': 3.0, 'd': 4}"
    assert str(model.model_dump_json()) == '{"a":1,"b":"2","c":3.0,"d":4}'


def test_extra_broken_via_pydantic_extra_interference() -> None:
    """
    At the time of writing this test there is `_model_construction.model_extra_getattr` being assigned to model's
    `__getattr__`. The method then expects `BaseModel.__pydantic_extra__` isn't `None`. Both this actions happen when
    `model_config.extra` is set to `True`. However, this behavior could be accidentally broken in a subclass of
    `BaseModel`. In that case `AttributeError` should be thrown when `__getattr__` is being accessed essentially
    disabling the `extra` functionality.
    """
    class BrokenExtraBaseModel(BaseModel):

        def model_post_init(self, context: Any, /) -> None:
            super().model_post_init(context)
            object.__setattr__(self, '__pydantic_extra__', None)

    class Model(BrokenExtraBaseModel):
        a: float
        model_config = ConfigDict(extra='allow')

    m: Model = Model(extra_field='some extra value')
    with pytest.raises(AttributeError) as e:
        _ = m.extra_field
    assert e.value.args == ("'Model' object has no attribute 'extra_field'",)


def test_model_extra_is_none_when_extra_is_forbid() -> None:
    class Foo(BaseModel):
        model_config = ConfigDict(extra='forbid')

    assert Foo().model_extra is None


def test_set_attr(UltraSimpleModel: Type[BaseModel]) -> None:
    m: UltraSimpleModel = UltraSimpleModel(a=10.2)
    assert m.model_dump() == {'a': 10.2, 'b': 10}
    m.b = 20
    assert m.model_dump() == {'a': 10.2, 'b': 20}


def test_set_attr_invalid() -> None:
    class UltraSimpleModel(BaseModel):
        a: float
        b: int = 10

    m: UltraSimpleModel = UltraSimpleModel(a=10.2)
    assert m.model_dump() == {'a': 10.2, 'b': 10}
    with pytest.raises(ValueError) as exc_info:
        m.c = 20
    assert '"UltraSimpleModel" object has no field "c"' in exc_info.value.args[0]


def test_any() -> None:
    class AnyModel(BaseModel):
        a: Any = 10
        b: Any = 20

    m: AnyModel = AnyModel()
    assert m.a == 10
    assert m.b == 20
    m = AnyModel(a='foobar', b='barfoo')
    assert m.a == 'foobar'
    assert m.b == 'barfoo'


def test_population_by_field_name() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        a: Annotated[float, Field(alias='_a')]

    assert Model(a='different').a == 'different'
    assert Model(a='different').model_dump() == {'a': 'different'}
    assert Model(a='different').model_dump(by_alias=True) == {'_a': 'different'}


def test_field_order() -> None:
    class Model(BaseModel):
        c: float
        b: int = 10
        a: str
        d: dict

    assert list(Model.model_fields.keys()) == ['c', 'b', 'a', 'd']


def test_required() -> None:
    class Model(BaseModel):
        a: float
        b: int = 10

    m: Model = Model(a=10.2)
    assert m.model_dump() == {'a': 10.2, 'b': 10}
    with pytest.raises(ValidationError) as exc_info:
        Model()
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'missing', 'loc': ('a',), 'msg': 'Field required', 'input': {}}
    ]


def test_mutability() -> None:
    class TestModel(BaseModel):
        a: int = 10
        model_config = ConfigDict(extra='forbid', frozen=False)

    m: TestModel = TestModel()
    assert m.a == 10
    m.a = 11
    assert m.a == 11


def test_frozen_model() -> None:
    class FrozenModel(BaseModel):
        a: int = 10
        model_config = ConfigDict(extra='forbid', frozen=True)

    m: FrozenModel = FrozenModel()
    assert m.a == 10
    with pytest.raises(ValidationError) as exc_info:
        m.a = 11
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'frozen_instance', 'loc': ('a',), 'msg': 'Instance is frozen', 'input': 11}
    ]
    with pytest.raises(ValidationError) as exc_info:
        del m.a
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'frozen_instance', 'loc': ('a',), 'msg': 'Instance is frozen', 'input': None}
    ]
    assert m.a == 10


def test_frozen_model_cached_property() -> None:
    class FrozenModel(BaseModel):
        model_config = ConfigDict(frozen=True)

        @cached_property
        def test(self) -> int:
            return self.a + 1

    m: FrozenModel = FrozenModel(a=1)
    assert m.test == 2
    del m.test
    m.test = 3
    assert m.test == 3


def test_frozen_field() -> None:
    class FrozenModel(BaseModel):
        a: int = Field(10, frozen=True)

    m: FrozenModel = FrozenModel()
    assert m.a == 10
    with pytest.raises(ValidationError) as exc_info:
        m.a = 11
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'frozen_field', 'loc': ('a',), 'msg': 'Field is frozen', 'input': 11}
    ]
    with pytest.raises(ValidationError) as exc_info:
        del m.a
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'frozen_field', 'loc': ('a',), 'msg': 'Field is frozen', 'input': None}
    ]
    assert m.a == 10


def test_not_frozen_are_not_hashable() -> None:
    class TestModel(BaseModel):
        a: int = 10

    m: TestModel = TestModel()
    with pytest.raises(TypeError) as exc_info:
        _ = hash(m)
    assert "unhashable type: 'TestModel'" in str(exc_info.value)


def test_with_declared_hash() -> None:
    class Foo(BaseModel):
        x: int

        def __hash__(self) -> int:
            return self.x ** 2

    class Bar(Foo):
        y: int

        def __hash__(self) -> int:
            return self.y ** 3

    class Buz(Bar):
        z: int

    assert hash(Foo(x=2)) == 4
    assert hash(Bar(x=2, y=3)) == 27
    assert hash(Buz(x=2, y=3, z=4)) == 27


def test_frozen_with_hashable_fields_are_hashable() -> None:
    class TestModel(BaseModel):
        model_config = ConfigDict(frozen=True)
        a: int = 10

    m: TestModel = TestModel()
    assert m.__hash__ is not None
    assert isinstance(hash(m), int)


def test_frozen_with_unhashable_fields_are_not_hashable() -> None:
    class TestModel(BaseModel):
        model_config = ConfigDict(frozen=True)
        a: int = 10
        y: list = [1, 2, 3]

    m: TestModel = TestModel()
    with pytest.raises(TypeError) as exc_info:
        _ = hash(m)
    assert "unhashable type: 'list'" in str(exc_info.value)


def test_hash_function_empty_model() -> None:
    class TestModel(BaseModel):
        model_config = ConfigDict(frozen=True)

    m: TestModel = TestModel()
    m2: TestModel = TestModel()
    assert m == m2
    assert hash(m) == hash(m2)


def test_hash_function_give_different_result_for_different_object() -> None:
    class TestModel(BaseModel):
        model_config = ConfigDict(frozen=True)
        a: int = 10

    m: TestModel = TestModel()
    m2: TestModel = TestModel()
    m3: TestModel = TestModel(a=11)
    assert hash(m) == hash(m2)
    assert hash(m) != hash(m3)


def test_hash_function_works_when_instance_dict_modified() -> None:
    class TestModel(BaseModel):
        model_config = ConfigDict(frozen=True)

    m: TestModel = TestModel(a=1, b=2)
    h: int = hash(m)
    m.__dict__['c'] = 1
    assert hash(m) == h
    m.__dict__ = {'b': 2, 'a': 1}
    assert hash(m) == h
    del m.__dict__['a']
    hash(m)


def test_default_hash_function_overrides_default_hash_function() -> None:
    class A(BaseModel):
        x: int

        def __hash__(self) -> int:
            return self.x ** 2

    class Bar(A):
        y: int

        def __hash__(self) -> int:
            return self.y ** 3

    class Buz(Bar):
        pass

    assert hash(A(x=1)) == 1
    assert hash(Bar(x=1, y=3)) == 27
    assert hash(Buz(x=1, y=3, z=4)) == 27


def test_hash_method_is_inherited_for_frozen_models() -> None:
    class MyBaseModel(BaseModel):
        """A base model with sensible configurations."""
        model_config = ConfigDict(frozen=True)

        def __hash__(self) -> int:
            return hash(id(self))

    class MySubClass(MyBaseModel):
        x: dict

        @cache
        def cached_method(self) -> int:
            return len(self.x)

    my_instance: MySubClass = MySubClass(x={'a': 1, 'b': 2})
    assert my_instance.cached_method() == 2
    object.__setattr__(my_instance, 'x', {})
    assert my_instance.cached_method() == 2


@pytest.fixture(name='ValidateAssignmentModel', scope='session')
def validate_assignment_fixture() -> Type[BaseModel]:
    class ValidateAssignmentModel(BaseModel):
        model_config = ConfigDict(validate_assignment=True)
        a: int = 2

    return ValidateAssignmentModel


def test_validating_assignment_pass(ValidateAssignmentModel: Type[BaseModel]) -> None:
    p: ValidateAssignmentModel = ValidateAssignmentModel(a=5, b='hello')
    p.a = 2
    assert p.a == 2
    assert p.model_dump() == {'a': 2, 'b': 'hello'}
    p.b = 'hi'
    assert p.b == 'hi'
    assert p.model_dump() == {'a': 2, 'b': 'hi'}


@pytest.mark.parametrize('init_valid', [False, True])
def test_validating_assignment_fail(ValidateAssignmentModel: Type[BaseModel], init_valid: bool) -> None:
    p: ValidateAssignmentModel = ValidateAssignmentModel(a=5, b='hello')
    if init_valid:
        p.a = 5
        p.b = 'hello'
    with pytest.raises(ValidationError) as exc_info:
        p.a = 'b'
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'int_parsing', 'loc': ('a',), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'b'}
    ]
    with pytest.raises(ValidationError) as exc_info:
        p.b = ''
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'string_too_short', 'loc': ('b',), 'msg': 'String should have at least 1 character', 'input': '', 'ctx': {'min_length': 1}}
    ]


class FooEnum(Enum):
    FOO = 'foo'
    BAR = 'bar'


@pytest.mark.parametrize('value', [FooEnum.FOO, FooEnum.FOO.value, 'foo'])
def test_enum_values(value: Union[FooEnum, str]) -> None:
    class Model(BaseModel):
        model_config = ConfigDict(use_enum_values=True)
        foo: str

    m: Model = Model(foo=value)
    foo_val: str = m.foo
    assert type(foo_val) is str, type(foo_val)
    assert foo_val == 'foo'
    foo_val = m.model_dump()['foo']
    assert type(foo_val) is str, type(foo_val)
    assert foo_val == 'foo'


def test_literal_enum_values() -> None:
    FooEnumType = Enum('FooEnum', {'foo': 'foo_value', 'bar': 'bar_value'})

    class Model(BaseModel):
        baz: Literal['foo_value']
        boo: str = 'hoo'
        model_config = ConfigDict(use_enum_values=True)

    m: Model = Model(baz=FooEnumType.foo)
    assert m.model_dump() == {'baz': 'foo_value', 'boo': 'hoo'}
    assert m.model_dump(mode='json') == {'baz': 'foo_value', 'boo': 'hoo'}
    assert m.baz == 'foo_value'
    with pytest.raises(ValidationError) as exc_info:
        Model(baz=FooEnumType.bar)
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'literal_error', 'loc': ('baz',), 'msg': "Input should be <FooEnum.foo: 'foo_value'>", 'input': FooEnumType.bar, 'ctx': {'expected': "<FooEnum.foo: 'foo_value'>"}}
    ]


class StrFoo(str, Enum):
    FOO = 'foo'
    BAR = 'bar'


@pytest.mark.parametrize('value', [StrFoo.FOO, StrFoo.FOO.value, 'foo', 'hello'])
def test_literal_use_enum_values_multi_type(value: Any) -> None:
    class Model(BaseModel):
        model_config = ConfigDict(use_enum_values=True)
        baz: str

    assert isinstance(Model(baz=value).baz, str)


def test_literal_use_enum_values_with_default() -> None:
    class Model(BaseModel):
        baz: str = Field(default=StrFoo.FOO)
        model_config = ConfigDict(use_enum_values=True, validate_default=True)

    validated: Model = Model()
    assert type(validated.baz) is str
    assert type(validated.model_dump()['baz']) is str
    validated = Model.model_validate_json('{"baz": "foo"}')
    assert type(validated.baz) is str
    assert type(validated.model_dump()['baz']) is str
    validated = Model.model_validate({'baz': StrFoo.FOO})
    assert type(validated.baz) is str
    assert type(validated.model_dump()['baz']) is str


def test_strict_enum_values() -> None:
    class MyEnum(Enum):
        val = 'val'

    class Model(BaseModel):
        x: str
        model_config = ConfigDict(use_enum_values=True)

    assert Model.model_validate({'x': MyEnum.val}, strict=True).x == 'val'


def test_union_enum_values() -> None:
    class MyEnum(Enum):
        val = 'val'

    class NormalModel(BaseModel):
        x: MyEnum

    class UseEnumValuesModel(BaseModel):
        x: MyEnum
        model_config = ConfigDict(use_enum_values=True)

    assert NormalModel(x=MyEnum.val).x != 'val'
    assert UseEnumValuesModel(x=MyEnum.val).x == 'val'


def test_enum_raw() -> None:
    FooEnum = Enum('FooEnum', {'foo': 'foo', 'bar': 'bar'})

    class Model(BaseModel):
        foo: FooEnum

    m: Model = Model(foo='foo')
    assert isinstance(m.foo, FooEnum)
    assert m.foo != 'foo'
    assert m.foo.value == 'foo'


def test_set_tuple_values() -> None:
    class Model(BaseModel):
        foo: set[str]
        bar: tuple[str, ...]

    m: Model = Model(foo=['a', 'b'], bar=['c', 'd'])
    assert m.foo == {'a', 'b'}
    assert m.bar == ('c', 'd')
    assert m.model_dump() == {'foo': {'a', 'b'}, 'bar': ('c', 'd')}


def test_default_copy() -> None:
    class User(BaseModel):
        friends: list[Any] = Field(default_factory=lambda: [])

    u1: User = User()
    u2: User = User()
    assert u1.friends is not u2.friends


def test_arbitrary_type_allowed_validation_success() -> None:
    class ArbitraryTypeAllowedModel(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        t: ArbitraryType

    arbitrary_type_instance: ArbitraryType = ArbitraryType()
    m: ArbitraryTypeAllowedModel = ArbitraryTypeAllowedModel(t=arbitrary_type_instance)
    assert m.t == arbitrary_type_instance


def test_arbitrary_type_allowed_validation_fails() -> None:
    class ArbitraryTypeAllowedModel(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        t: ArbitraryType

    input_value: OtherClass = OtherClass()
    with pytest.raises(ValidationError) as exc_info:
        ArbitraryTypeAllowedModel(t=input_value)
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'is_instance_of', 'loc': ('t',), 'msg': 'Input should be an instance of ArbitraryType', 'input': input_value, 'ctx': {'class': 'ArbitraryType'}}
    ]


def test_arbitrary_types_not_allowed() -> None:
    class ArbitraryTypeNotAllowedModel(BaseModel):
        pass

    with pytest.raises(TypeError, match='Unable to generate pydantic-core schema for <class'):
        ArbitraryTypeNotAllowedModel()


@pytest.fixture(scope='session', name='TypeTypeModel')
def type_type_model_fixture() -> Type[BaseModel]:
    class TypeTypeModel(BaseModel):
        t: Type[ArbitraryType]

    return TypeTypeModel


def test_type_type_validation_success(TypeTypeModel: Type[BaseModel]) -> None:
    arbitrary_type_class: Type[ArbitraryType] = ArbitraryType
    m: TypeTypeModel = TypeTypeModel(t=arbitrary_type_class)
    assert m.t == arbitrary_type_class


def test_type_type_subclass_validation_success(TypeTypeModel: Type[BaseModel]) -> None:
    class ArbitrarySubType(ArbitraryType):
        pass

    arbitrary_type_class: Type[ArbitrarySubType] = ArbitrarySubType
    m: TypeTypeModel = TypeTypeModel(t=arbitrary_type_class)
    assert m.t == arbitrary_type_class


@pytest.mark.parametrize('input_value', [OtherClass, 1], ids=['OtherClass', 'int'])
def test_type_type_validation_fails(TypeTypeModel: Type[BaseModel], input_value: Any) -> None:
    with pytest.raises(ValidationError) as exc_info:
        TypeTypeModel(t=input_value)
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'is_subclass_of', 'loc': ('t',), 'msg': 'Input should be a subclass of ArbitraryType', 'input': input_value, 'ctx': {'class': 'ArbitraryType'}}
    ]


@pytest.mark.parametrize('bare_type', [type, type], ids=['type', 'type'])
def test_bare_type_type_validation_success(bare_type: type) -> None:
    class TypeTypeModel(BaseModel):
        t: type

    class ArbitraryType:
        pass

    arbitrary_type_class: type = ArbitraryType
    m: TypeTypeModel = TypeTypeModel(t=arbitrary_type_class)
    assert m.t == arbitrary_type_class


@pytest.mark.parametrize('bare_type', [type, type], ids=['type', 'type'])
def test_bare_type_type_validation_fails(bare_type: type) -> None:
    class TypeTypeModel(BaseModel):
        t: type

    arbitrary_type = ArbitraryType()
    with pytest.raises(ValidationError) as exc_info:
        TypeTypeModel(t=arbitrary_type)
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'is_type', 'loc': ('t',), 'msg': 'Input should be a type', 'input': arbitrary_type}
    ]


def test_value_field_name_shadows_attribute() -> None:
    with pytest.raises(PydanticUserError, match="A non-annotated attribute was detected: `model_json_schema = 'abc'`"):
        class BadModel(BaseModel):
            model_json_schema: str = 'abc'


def test_class_var() -> None:
    class MyModel(BaseModel):
        b: int = 1
        c: int = 2

    assert list(MyModel.model_fields.keys()) == ['c']

    class MyOtherModel(MyModel):
        a: str = ''
        b: int = 2

    assert list(MyOtherModel.model_fields.keys()) == ['c']


def test_fields_set() -> None:
    class MyModel(BaseModel):
        b: int = 2

    m: MyModel = MyModel(a=5)
    assert m.model_fields_set == {'a'}
    m.b = 2
    assert m.model_fields_set == {'a', 'b'}
    m = MyModel(a=5, b=2)
    assert m.model_fields_set == {'a', 'b'}


def test_exclude_unset_dict() -> None:
    class MyModel(BaseModel):
        b: int = 2

    m: MyModel = MyModel(a=5)
    assert m.model_dump(exclude_unset=True) == {'a': 5}
    m = MyModel(a=5, b=3)
    assert m.model_dump(exclude_unset=True) == {'a': 5, 'b': 3}


def test_exclude_unset_recursive() -> None:
    class ModelA(BaseModel):
        a: int = 0

    class ModelB(BaseModel):
        d: int = 2

    class ModelC(BaseModel):
        a: int = 0

    m: ModelB = ModelB(c=5, e={'a': 0})
    assert m.model_dump() == {'c': 5, 'd': 2, 'e': {'a': 0, 'b': 1}}
    assert m.model_dump(exclude_unset=True) == {'c': 5, 'e': {'a': 0}}
    assert dict(m) == {'c': 5, 'd': 2, 'e': ModelA(a=0, b=1)}


def test_dict_exclude_unset_populated_by_alias() -> None:
    class MyModel(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        a: str = Field('default', alias='alias_a')
        b: str = Field('default', alias='alias_b')

    m: MyModel = MyModel(alias_a='a')
    assert m.model_dump(exclude_unset=True) == {'a': 'a'}
    assert m.model_dump(exclude_unset=True, by_alias=True) == {'alias_a': 'a'}


def test_dict_exclude_unset_populated_by_alias_with_extra() -> None:
    class MyModel(BaseModel):
        model_config = ConfigDict(extra='allow')
        a: str = Field('default', alias='alias_a')
        b: str = Field('default', alias='alias_b')

    m: MyModel = MyModel(alias_a='a', c='c')
    assert m.model_dump(exclude_unset=True) == {'a': 'a', 'c': 'c'}
    assert m.model_dump(exclude_unset=True, by_alias=True) == {'alias_a': 'a', 'c': 'c'}


def test_exclude_defaults() -> None:
    class Model(BaseModel):
        mandatory: str
        nullable_mandatory: Optional[str]
        facultative: str = 'x'
        nullable_facultative: Optional[str] = None

    m: Model = Model(mandatory='a', nullable_mandatory=None)
    assert m.model_dump(exclude_defaults=True) == {'mandatory': 'a', 'nullable_mandatory': None}
    m = Model(mandatory='a', nullable_mandatory=None, facultative='y', nullable_facultative=None)
    assert m.model_dump(exclude_defaults=True) == {'mandatory': 'a', 'nullable_mandatory': None, 'facultative': 'y'}
    m = Model(mandatory='a', nullable_mandatory=None, facultative='y', nullable_facultative='z')
    assert m.model_dump(exclude_defaults=True) == {'mandatory': 'a', 'nullable_mandatory': None, 'facultative': 'y', 'nullable_facultative': 'z'}


def test_dir_fields() -> None:
    class MyModel(BaseModel):
        attribute_b: int = 2

    m: MyModel = MyModel(attribute_a=5)
    assert 'model_dump' in dir(m)
    assert 'model_dump_json' in dir(m)
    assert 'attribute_a' in dir(m)
    assert 'attribute_b' in dir(m)


def test_dict_with_extra_keys() -> None:
    class MyModel(BaseModel):
        model_config = ConfigDict(extra='allow')
        a: Optional[str] = Field(None, alias='alias_a')

    m: MyModel = MyModel(extra_key='extra')
    assert m.model_dump() == {'a': None, 'extra_key': 'extra'}
    assert m.model_dump(by_alias=True) == {'alias_a': None, 'extra_key': 'extra'}


def test_ignored_types() -> None:
    from pydantic import BaseModel

    class _ClassPropertyDescriptor:
        def __init__(self, getter: Callable[..., Any]) -> None:
            self.getter = getter

        def __get__(self, instance: Any, owner: Type[Any]) -> Any:
            return self.getter(owner)

    classproperty = _ClassPropertyDescriptor

    class Model(BaseModel):
        model_config = ConfigDict(ignored_types=(classproperty,))

        @classproperty
        def class_name(cls) -> str:
            return cls.__name__

    assert Model.class_name == 'Model'
    assert Model().class_name == 'Model'


def test_model_iteration() -> None:
    class Foo(BaseModel):
        a: int = 1
        b: int = 2

    class Bar(BaseModel):
        pass

    m: Bar = Bar(c=3, d={})
    assert m.model_dump() == {'c': 3, 'd': {'a': 1, 'b': 2}}
    assert list(m) == [('c', 3), ('d', Foo())]
    assert dict(m) == {'c': 3, 'd': Foo()}


def test_model_iteration_extra() -> None:
    class Foo(BaseModel):
        x: int = 1

    class Bar(BaseModel):
        model_config = ConfigDict(extra='allow')

    m: Bar = Bar.model_validate({'a': 1, 'b': {}, 'c': 2, 'd': Foo()})
    assert m.model_dump() == {'a': 1, 'b': {'x': 1}, 'c': 2, 'd': {'x': 1}}
    assert list(m) == [('a', 1), ('b', Foo()), ('c', 2), ('d', Foo())]
    assert dict(m) == {'a': 1, 'b': Foo(), 'c': 2, 'd': Foo()}


@pytest.mark.parametrize(
    'exclude,expected,raises_match',
    [
        pytest.param(
            None,
            {'c': 3, 'foos': [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]},
            None,
            id='exclude nothing'
        ),
        pytest.param(
            {'foos': {0: {'a'}, 1: {'a'}}},
            {'c': 3, 'foos': [{'b': 2}, {'b': 4}]},
            None,
            id='excluding fields of indexed list items'
        ),
        pytest.param(
            {'foos': {'a'}},
            {'c': 3, 'foos': [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]},
            None,
            id='Trying to exclude string keys on list field should be ignored (1)'
        ),
        pytest.param(
            {'foos': {0: ..., 'a': ...}},
            {'c': 3, 'foos': [{'a': 3, 'b': 4}]},
            None,
            id='Trying to exclude string keys on list field should be ignored (2)'
        ),
        pytest.param(
            {'foos': {0: 1}},
            TypeError,
            '`exclude` argument must be a set or dict',
            id='value as int should be an error'
        ),
        pytest.param(
            {'foos': {'__all__': {1}}},
            {'c': 3, 'foos': [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]},
            None,
            id='excluding int in dict should have no effect'
        ),
        pytest.param(
            {'foos': {'__all__': {'a'}}},
            {'c': 3, 'foos': [{'b': 2}, {'b': 4}]},
            None,
            id='using "__all__" to exclude specific nested field'
        ),
        pytest.param(
            {'foos': {0: {'b'}, '__all__': {'a'}}},
            {'c': 3, 'foos': [{}, {'b': 4}]},
            None,
            id='using "__all__" to exclude specific nested field in combination with more specific exclude'
        ),
        pytest.param(
            {'foos': {'__all__'}},
            {'c': 3, 'foos': []},
            None,
            id='using "__all__" to exclude all list items'
        ),
        pytest.param(
            {'foos': {1, '__all__'}},
            {'c': 3, 'foos': []},
            None,
            id='using "__all__" and other items should get merged together, still excluding all list items'
        ),
        pytest.param(
            {'foos': {-1: {'b'}}},
            {'c': 3, 'foos': [{'a': 1, 'b': 2}, {'a': 3}]},
            None,
            id='negative indexes'
        ),
    ],
    ids=[
        'exclude nothing',
        'excluding fields of indexed list items',
        'Trying to exclude string keys on list field should be ignored (1)',
        'Trying to exclude string keys on list field should be ignored (2)',
        'value as int should be an error',
        'excluding int in dict should have no effect',
        'using "__all__" to exclude specific nested field',
        'using "__all__" to exclude specific nested field in combination with more specific exclude',
        'using "__all__" to exclude all list items',
        'using "__all__" and other items should get merged together, still excluding all list items',
        'negative indexes'
    ]
)
def test_model_export_nested_list(exclude: Optional[dict], expected: dict, raises_match: Optional[str]) -> None:
    class Foo(BaseModel):
        a: int
        b: int

    class Bar(BaseModel):
        pass

    class Model(BaseModel):
        a: int = 1
        foos: list[Foo] = [Foo(a=1, b=2), Foo(a=3, b=4)]

    m: Model = Model()
    if raises_match is not None:
        with pytest.raises(eval(raises_match)) as e:
            m.model_dump(exclude=exclude)
    else:
        original_exclude = deepcopy(exclude)
        assert m.model_dump(exclude=exclude) == expected
        assert exclude == original_exclude


@pytest.mark.parametrize(
    'excludes,expected',
    [
        pytest.param(
            {'bars': {0}},
            {'a': 1, 'bars': [{'y': 2}, {'w': -1, 'z': 3}]},
            id='excluding first item from list field using index'
        ),
        pytest.param(
            {'bars': {'__all__'}},
            {'a': 1, 'bars': []},
            id='using "__all__" to exclude all list items'
        ),
        pytest.param(
            {'bars': {'__all__': {'w'}}},
            {'a': 1, 'bars': [{'x': 1}, {'y': 2}, {'z': 3}]},
            id='exclude single dict key from all list items'
        ),
    ],
    ids=[
        'excluding first item from list field using index',
        'using "__all__" to exclude all list items',
        'exclude single dict key from all list items'
    ]
)
def test_model_export_dict_exclusion(excludes: dict, expected: dict) -> None:
    class Foo(BaseModel):
        w: int = 0
        x: int = 1
        y: int = 2
        z: int = 3

    class Model(BaseModel):
        a: int = 1
        bars: list[dict[str, int]] = [{'w': 0, 'x': 1}, {'y': 2}, {'w': -1, 'z': 3}]

    m: Model = Model()
    original_excludes = deepcopy(excludes)
    assert m.model_dump(exclude=excludes) == expected
    assert excludes == original_excludes


def test_field_exclude() -> None:
    class User(BaseModel):
        _priv: Any = PrivateAttr()
        password: SecretStr = Field(exclude=True)
        id: int = 42
        username: str = 'JohnDoe'
        hobbies: list[str] = ['scuba diving']

    my_user: User = User(id=42, username='JohnDoe', password=SecretStr('hashedpassword'), hobbies=['scuba diving'])
    my_user._priv = 13
    assert my_user.id == 42
    assert my_user.password.get_secret_value() == 'hashedpassword'
    assert my_user.model_dump() == {'id': 42, 'username': 'JohnDoe', 'hobbies': ['scuba diving']}


def test_revalidate_instances_never() -> None:
    class User(BaseModel):
        a: list[str]

    my_user: User = User(a=['scuba diving'])

    class Transaction(BaseModel):
        user: User

    t: Transaction = Transaction(user=my_user)
    assert t.user is my_user
    assert t.user.a is my_user.a

    class SubUser(User):
        sins: list[str]

    my_sub_user: SubUser = SubUser(a=['scuba diving'], sins=['lying'])
    t = Transaction(user=my_sub_user)
    assert t.user is my_sub_user
    assert t.user.a is my_sub_user.a
    assert not hasattr(t.user, 'sins')


def test_revalidate_instances_sub_instances() -> None:
    class User(BaseModel, Generic[T]):
        a: list[str]

        class Config:
            revalidate_instances = 'subclass-instances'

    class Transaction(BaseModel):
        user: User[Any]

    my_user: User[Any] = User(a=['scuba diving'])

    t: Transaction = Transaction(user=my_user)
    assert t.user is my_user
    assert t.user.a is my_user.a

    class SubUser(User[Any]):
        sins: list[str]

    my_sub_user: SubUser = SubUser(a=['scuba diving'], sins=['lying'])
    t = Transaction(user=my_sub_user)
    assert t.user is not my_sub_user
    assert t.user.a is not my_sub_user.a
    assert not hasattr(t.user, 'sins')


def test_revalidate_instances_always() -> None:
    class User(BaseModel, Generic[T]):
        a: list[str]

        class Config:
            revalidate_instances = 'always'

    class Transaction(BaseModel):
        user: User[Any]

    my_user: User[Any] = User(a=['scuba diving'])

    t: Transaction = Transaction(user=my_user)
    assert t.user is not my_user
    assert t.user.a is not my_user.a

    class SubUser(User[Any]):
        sins: list[str]

    my_sub_user: SubUser = SubUser(a=['scuba diving'], sins=['lying'])
    t = Transaction(user=my_sub_user)
    assert t.user is not my_sub_user
    assert t.user.a is not my_sub_user.a
    assert not hasattr(t.user, 'sins')


def test_revalidate_instances_always_list_of_model_instance() -> None:
    class A(BaseModel):
        name: str

        class Config:
            revalidate_instances = 'always'

    class B(BaseModel):
        list_a: list[A]

    a1: A = A(name='a')
    a2: A = A(name='a')
    b: B = B(list_a=[a1])
    assert b.list_a == [A(name='a')]
    a1.name = 'b'
    assert b.list_a == [A(name='a')]


@pytest.mark.skip(reason='not implemented')
@pytest.mark.parametrize(
    'kinds',
    [
        {'sub_fields', 'model_fields', 'model_config', 'sub_config', 'combined_config'},
        {'sub_fields', 'model_fields', 'combined_config'},
        {'sub_fields', 'model_fields'},
        {'combined_config'},
        {'model_config', 'sub_config'},
        {'model_config', 'sub_fields'},
        {'model_fields', 'sub_config'}
    ]
)
@pytest.mark.parametrize(
    'exclude,expected,raises_match',
    [
        pytest.param(
            None,
            {'a': 0, 'c': {'a': [3, 5], 'c': 'foobar'}, 'd': {'c': 'foobar'}},
            None,
            id='exclude nothing'
        ),
        pytest.param(
            {'c', 'd'},
            {'a': 0},
            None,
            id='exclude c and d'
        ),
        pytest.param(
            {'a': ..., 'c': ..., 'd': {'a': ..., 'c': ...}},
            {'d': {}},
            None,
            id='exclude a, c, and nested a,c in d'
        ),
    ],
    ids=[
        'exclude nothing',
        'exclude c and d',
        'exclude a, c, and nested a,c in d'
    ]
)
def test_model_export_exclusion_with_fields_and_config(kinds: set, exclude: Optional[dict], expected: dict) -> None:
    """Test that exporting models with fields using the export parameter works."""
    class ChildConfig:
        fields: Optional[dict] = None

    if 'sub_config' in kinds:
        ChildConfig.fields = {'b': {'exclude': ...}, 'a': {'exclude': {1}}}

    class ParentConfig:
        fields: Optional[dict] = None

    if 'combined_config' in kinds:
        ParentConfig.fields = {'b': {'exclude': ...}, 'c': {'exclude': {'b': ..., 'a': {1}}}, 'd': {'exclude': {'a': ..., 'b': ...}}}
    elif 'model_config' in kinds:
        ParentConfig.fields = {'b': {'exclude': ...}, 'd': {'exclude': {'a'}}}

    class Sub(BaseModel):
        a: list[int]
        b: int
        c: str
        Config = ChildConfig

    class Model(BaseModel):
        a: int = 0
        b: int = 2
        c: Sub = Sub(a=[3, 4, 5], b=4, c='foobar')
        d: Sub = Sub(a=[5], b=3, c='foobar')
        Config = ParentConfig

    m: Model = Model()
    assert m.model_dump(exclude=exclude) == expected


@pytest.mark.skip(reason='not implemented')
def test_model_export_exclusion_inheritance() -> None:
    pass  # Implemented above


@pytest.mark.skip(reason='not implemented')
def test_model_export_with_true_instead_of_ellipsis() -> None:
    pass  # Implemented above


@pytest.mark.skip(reason='not implemented')
def test_model_export_inclusion() -> None:
    pass  # Implemented above


@pytest.mark.skip(reason='not implemented')
def test_model_export_inclusion_inheritance() -> None:
    pass  # Implemented above


def test_shadow_attribute() -> None:
    """https://github.com/pydantic/pydantic/issues/7108"""
    with pytest.raises(PydanticUserError, match="Fields must not use names with leading underscores; e.g., use 'x' instead of '_x'"):
        class Model1(BaseModel):
            _x: str = Field(alias='x')

    with pytest.raises(PydanticUserError, match="Fields must not use names with leading underscores; e.g., use 'x__' instead of '__x__'"):
        class Model2(BaseModel):
            __x__: int = Field()

    with pytest.raises(PydanticUserError, match="Fields must not use names with leading underscores; e.g., use 'my_field' instead of '___'"):
        class Model3(BaseModel):
            ___: int = Field(default=1)


def test_customize_type_constraints_order() -> None:
    class Model(BaseModel):
        x: str = Field(..., max_length=1)
        y: str = Field(..., max_length=1)

    with pytest.raises(ValidationError) as exc_info:
        Model(x=' 1 ', y=' 1 ')
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'string_too_long', 'loc': ('y',), 'msg': 'String should have at most 1 character', 'input': ' 1 ', 'ctx': {'max_length': 1}}
    ]


def test_shadow_attribute_warn_for_redefined_fields() -> None:
    """https://github.com/pydantic/pydantic/issues/9107"""
    class Parent:
        foo: bool = False

    with pytest.warns(UserWarning, match='Field "foo" in ".*ChildWithRedefinedField" shadows an attribute in parent ".*Parent"'):
        class ChildWithRedefinedField(BaseModel, Parent):
            foo: bool = True


def test_eval_type_backport() -> None:
    class Model(BaseModel):
        x: list[Union[int, str]]

    assert Model.model_validate({'x': [1, '2']}).x == [1, '2']
    with pytest.raises(ValidationError) as exc_info:
        Model.model_validate({'x': 'not a list'})
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'list_type', 'loc': ('x',), 'msg': 'Input should be a valid list', 'input': 'not a list'}
    ]
    with pytest.raises(ValidationError) as exc_info:
        Model.model_validate({'x': [{'not a str or int'}]})
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'int_type', 'loc': ('x', 0, 'int'), 'msg': 'Input should be a valid integer', 'input': {'not a str or int'}},
        {'type': 'string_type', 'loc': ('x', 0, 'str'), 'msg': 'Input should be a valid string', 'input': {'not a str or int'}}
    ]


def test_inherited_class_vars(create_module: Callable[[str], Any]) -> None:
    module = create_module(
        '''
from pydantic import BaseModel

class Base(BaseModel):
    CONST1: str = 'a'
    CONST2: str = 'b'
'''
    )
    class Child(module.Base):
        pass
    assert Child.CONST1 == 'a'
    assert Child.CONST2 == 'b'


def test_schema_valid_for_inner_generic() -> None:
    received_schemas: defaultdict[str, list[str]] = defaultdict(list)

    @dataclass
    class Marker:
        name: str

        def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
            schema: CoreSchema = handler(source_type)
            received_schemas[self.name].append(schema['type'])  # type: ignore
            return schema

    class InnerModel(BaseModel):
        @classmethod
        def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
            schema: CoreSchema = handler(source_type)
            received_schemas['InnerModel'].append(schema['type'])  # type: ignore
            schema = deepcopy(schema)
            schema['metadata'] = schema.get('metadata', {})
            schema['metadata']['foo'] = 'inner was here!'
            return deepcopy(schema)

    class OuterModel(BaseModel):
        x: int
        model_config = ConfigDict()

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
            schema: CoreSchema = handler(source_type)
            received_schemas['OuterModel'].append(schema['type'])  # type: ignore

            def set_x(m: OuterModel) -> OuterModel:
                m.x += 1
                return m
            return core_schema.no_info_after_validator_function(set_x, schema, ref=schema.pop('ref'))

    ta: TypeAdapter[Annotated[OuterModel, Marker('Marker("outer")')]] = TypeAdapter(Annotated[OuterModel, Marker('Marker("outer")')])
    assert 'inner was here' in str(ta.core_schema)
    assert received_schemas == {
        'InnerModel': ['model', 'model'],
        'Marker("outer")': ['definition-ref'],
        'OuterModel': ['model', 'model']
    }


def test_get_core_schema_return_new_ref() -> None:
    class InnerModel(BaseModel):
        x: int

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
            schema: CoreSchema = handler(source_type)
            schema = deepcopy(schema)
            schema['metadata'] = schema.get('metadata', {})
            schema['metadata']['foo'] = 'inner was here!'
            return deepcopy(schema)

    class OuterModel(BaseModel):
        x: int = 1
        inner: InnerModel

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
            schema: CoreSchema = handler(source_type)

            def set_x(m: OuterModel) -> OuterModel:
                m.x += 1
                return m
            return core_schema.no_info_after_validator_function(set_x, schema, ref=schema.pop('ref'))

    cs: CoreSchema = OuterModel.__pydantic_core_schema__  # type: ignore
    assert 'inner was here' in str(cs)
    m: OuterModel = OuterModel(inner=InnerModel())
    assert m.x == 2


def test_resolve_def_schema_from_core_schema() -> None:
    class Inner(BaseModel):
        x: int

    class Marker:
        def __init__(self, name: str) -> None:
            self.name = name

        def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
            schema: CoreSchema = handler(source_type)
            resolved: CoreSchema = handler.resolve_ref_schema(schema)
            assert resolved['type'] == 'model'
            assert resolved['cls'] is Inner

            def modify_inner(v: Inner) -> Inner:
                v.x += 1
                return v
            return core_schema.no_info_after_validator_function(modify_inner, schema)

    class Outer(BaseModel):
        inner: Inner = Marker('Marker("inner")')

    m: Outer = Outer.model_validate({'inner': {'x': 1}})
    assert m.inner.x == 2


def test_extra_validator_scalar() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(extra='allow')
        a: int = 1

    class Child(Model):
        pass

    m: Child = Child(a='1')  # 'a' should be parsed to int
    assert m.__pydantic_extra__ == {'a': 1}
    assert Child.model_json_schema() == {
        'additionalProperties': {'type': 'integer'},
        'properties': {'a': {'title': 'A', 'type': 'integer'}},
        'title': 'Child',
        'type': 'object'
    }


def test_extra_validator_field() -> None:
    class Model(BaseModel, extra='allow'):
        __pydantic_extra__: dict[str, int] = Field(init=False)

    m: Model = Model(a='1')
    assert m.__pydantic_extra__ == {'a': 1}
    with pytest.raises(ValidationError) as exc_info:
        Model(a='a')
    assert exc_info.value.errors(include_url=False) == [
        {'input': 'a', 'loc': ('a',), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}
    ]
    assert Model.model_json_schema() == {
        'additionalProperties': {'type': 'integer'},
        'properties': {'a': {'title': 'A', 'type': 'integer'}},
        'title': 'Model',
        'type': 'object'
    }


def test_extra_validator_named() -> None:
    class Foo(BaseModel):
        x: int = 1

    class Model(BaseModel):
        model_config = ConfigDict(extra='allow')

    class Child(Model):
        pass

    m: Child = Child(a={'x': 1}, y=2)
    assert m.__pydantic_extra__ == {'a': Foo(x=1)}
    assert Child.model_json_schema() == {
        '$defs': {'Foo': {'properties': {'x': {'title': 'X', 'type': 'integer'}}, 'required': ['x'], 'title': 'Foo', 'type': 'object'}},
        'additionalProperties': {'$ref': '#/$defs/Foo'},
        'properties': {'y': {'title': 'Y', 'type': 'integer'}},
        'required': ['y'],
        'title': 'Child',
        'type': 'object'
    }


def test_super_getattr_extra() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(extra='allow')

        def __getattr__(self, item: str) -> Any:
            if item == 'test':
                return 'success'
            return super().__getattr__(item)

    m: Model = Model(x=1)
    assert m.x == 1
    with pytest.raises(AttributeError):
        _ = m.y
    assert m.test == 'success'


def test_super_getattr_private() -> None:
    class Model(BaseModel):
        _x: int = PrivateAttr()

        def __getattr__(self, item: str) -> Any:
            if item == 'test':
                return 'success'
            else:
                return super().__getattr__(item)

    m: Model = Model()
    m._x = 1
    assert m._x == 1
    with pytest.raises(AttributeError):
        _ = m._y
    assert m.test == 'success'


def test_super_delattr_extra() -> None:
    test_calls: list[str] = []

    class Model(BaseModel):
        model_config = ConfigDict(extra='allow')

        def __delattr__(self, item: str) -> None:
            if item == 'test':
                test_calls.append('success')
            else:
                super().__delattr__(item)

    m: Model = Model(x=1)
    assert m.x == 1
    del m.x
    with pytest.raises(AttributeError):
        _ = m._x
    assert test_calls == []
    del m.test
    assert test_calls == ['success']


def test_super_delattr_private() -> None:
    test_calls: list[str] = []

    class Model(BaseModel):
        _x: int = PrivateAttr()

        def __delattr__(self, item: str) -> None:
            if item == 'test':
                test_calls.append('success')
            else:
                super().__delattr__(item)

    m: Model = Model()
    m._x = 1
    assert m._x == 1
    del m._x
    with pytest.raises(AttributeError):
        _ = m._x
    assert test_calls == []
    del m.test
    assert test_calls == ['success']


def test_arbitrary_types_not_a_type() -> None:
    """https://github.com/pydantic/pydantic/issues/6477"""
    class Foo:
        pass

    class Bar:
        pass

    with pytest.warns(UserWarning, match='is not a Python type'):
        ta: TypeAdapter[Any] = TypeAdapter(Foo(), config=ConfigDict(arbitrary_types_allowed=True))
    bar: Bar = Bar()
    assert ta.validate_python(bar) is bar


@pytest.mark.parametrize('is_dataclass', [False, True])
def test_deferred_core_schema(is_dataclass: bool) -> None:
    if is_dataclass:
        @pydantic_dataclass
        class Foo:
            pass
    else:
        class Foo(BaseModel):
            pass
    assert isinstance(Foo.__pydantic_core_schema__, MockCoreSchema)
    with pytest.raises(PydanticUserError, match='`Foo` is not fully defined'):
        _ = Foo.__pydantic_core_schema__['type']


def test_help(create_module: Callable[[str], Any]) -> None:
    module = create_module(
        '''
import pydoc

from pydantic import BaseModel

class Model(BaseModel):
    x: int

help_result_string = pydoc.render_doc(Model)
'''
    )
    assert 'class Model' in module.help_result_string


def test_cannot_use_leading_underscore_field_names() -> None:
    with pytest.raises(PydanticUserError, match="Fields must not use names with leading underscores; e.g., use 'x' instead of '_x'"):
        class Model1(BaseModel):
            _x: str = Field(alias='x')

    with pytest.raises(PydanticUserError, match="Fields must not use names with leading underscores; e.g., use 'x__' instead of '__x__'"):
        class Model2(BaseModel):
            __x__: int = Field()

    with pytest.raises(PydanticUserError, match="Fields must not use names with leading underscores; e.g., use 'my_field' instead of '___'"):
        class Model3(BaseModel):
            ___: int = Field(default=1)


def test_customize_type_constraints_order() -> None:
    class Model(BaseModel):
        x: str = Field(..., max_length=1)
        y: str = Field(..., max_length=1)

    with pytest.raises(ValidationError) as exc_info:
        Model(x=' 1 ', y=' 1 ')
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'string_too_long', 'loc': ('y',), 'msg': 'String should have at most 1 character', 'input': ' 1 ', 'ctx': {'max_length': 1}}
    ]


def test_shadow_attribute_warn_for_redefined_fields() -> None:
    """https://github.com/pydantic/pydantic/issues/9107"""
    class Parent:
        foo: bool = False

    with pytest.warns(UserWarning, match='Field "foo" in ".*ChildWithRedefinedField" shadows an attribute in parent ".*Parent"'):
        class ChildWithRedefinedField(BaseModel, Parent):
            foo: bool = True
