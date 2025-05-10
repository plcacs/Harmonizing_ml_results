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
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Final,
    Generic,
    Iterable,
    Literal,
    Optional,
    TypeVar,
    Union,
    get_type_hints,
)
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

def test_success() -> None:
    class Model(BaseModel):
        b: int = 10
    m: Model = Model(a=10.2)
    assert m.a == 10.2
    assert m.b == 10

@pytest.fixture(name='UltraSimpleModel', scope='session')
def ultra_simple_model_fixture() -> type[BaseModel]:
    class UltraSimpleModel(BaseModel):
        b: int = 10
    return UltraSimpleModel

def test_ultra_simple_missing(UltraSimpleModel: type[BaseModel]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        UltraSimpleModel()
    assert exc_info.value.errors(include_url=False) == [
        {
            'loc': ('a',),
            'msg': 'Field required',
            'type': 'missing',
            'input': {},
        }
    ]
    assert str(exc_info.value) == (
        '1 validation error for UltraSimpleModel\na\n  '
        'Field required [type=missing, input_value={}, input_type=dict]'
    )

def test_ultra_simple_failed(UltraSimpleModel: type[BaseModel]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        UltraSimpleModel(a='x', b='x')
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'float_parsing',
            'loc': ('a',),
            'msg': 'Input should be a valid number, unable to parse string as a number',
            'input': 'x',
        },
        {
            'type': 'int_parsing',
            'loc': ('b',),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'input': 'x',
        },
    ]

def test_ultra_simple_repr(UltraSimpleModel: type[BaseModel]) -> None:
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
        a: Optional['A'] = None

    A.update_forward_refs()
    b = B(a=A())
    a = b.a
    a.a = a
    b = B(a=a)
    assert re.match(
        r"B\(a=A\(a='<Recursion on A with id=\d+>'\)\)",
        repr(b),
    ) is not None

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

def test_comparing(UltraSimpleModel: type[BaseModel]) -> None:
    m1: UltraSimpleModel = UltraSimpleModel(a=10.2, b='100')
    assert m1.model_dump() == {'a': 10.2, 'b': 100}
    assert m1 != {'a': 10.2, 'b': 100}
    assert m1 == UltraSimpleModel(a=10.2, b=100)

@pytest.fixture(scope='session', name='NoneCheckModel')
def none_check_model_fix() -> type[BaseModel]:
    class NoneCheckModel(BaseModel):
        existing_str_value: str = 'foo'
        required_str_value: str
        required_str_none_value: Optional[str]
        existing_bytes_value: bytes = b'foo'
        required_bytes_value: bytes
        required_bytes_none_value: Optional[bytes]
    return NoneCheckModel

def test_nullable_strings_success(NoneCheckModel: type[BaseModel]) -> None:
    m = NoneCheckModel(
        required_str_value='v1',
        required_str_none_value=None,
        required_bytes_value='v2',
        required_bytes_none_value=None
    )
    assert m.required_str_value == 'v1'
    assert m.required_str_none_value is None
    assert m.required_bytes_value == b'v2'
    assert m.required_bytes_none_value is None

def test_nullable_strings_fails(NoneCheckModel: type[BaseModel]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        NoneCheckModel(
            required_str_value=None,
            required_str_none_value=None,
            required_bytes_value=None,
            required_bytes_none_value=None
        )
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'string_type',
            'loc': ('required_str_value',),
            'msg': 'Input should be a valid string',
            'input': None,
        },
        {
            'type': 'bytes_type',
            'loc': ('required_bytes_value',),
            'msg': 'Input should be a valid bytes',
            'input': None,
        },
    ]

@pytest.fixture(name='ParentModel', scope='session')
def parent_sub_model_fixture() -> type[BaseModel]:
    class UltraSimpleModel(BaseModel):
        b: int = 10

    class ParentModel(BaseModel):
        grape: bool
        banana: UltraSimpleModel
    return ParentModel

def test_parent_sub_model(ParentModel: type[BaseModel]) -> None:
    m: ParentModel = ParentModel(grape=1, banana={'a': 1})
    assert m.grape is True
    assert m.banana.a == 1.0
    assert m.banana.b == 10
    assert repr(m) == 'ParentModel(grape=True, banana=UltraSimpleModel(a=1.0, b=10))'

def test_parent_sub_model_fails(ParentModel: type[BaseModel]) -> None:
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
        {
            'type': 'float_type',
            'loc': ('a',),
            'msg': 'Input should be a valid number',
            'input': None,
        }
    ]

def test_allow_extra() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow')
        a: float
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
def test_allow_extra_from_attributes(extra: str) -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra=extra, from_attributes=True)
        a: Optional[float] = None

    class TestClass:
        a: float = 1.0
        b: int = 12
    m: Model = Model.model_validate(TestClass())
    assert m.a == 1.0
    assert not hasattr(m, 'b')

def test_allow_extra_repr() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow')
        a: float
    m = Model(a='10.2', b=12)
    assert str(m) == 'a=10.2 b=12'

def test_forbidden_extra_success() -> None:
    class ForbiddenExtra(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='forbid')
        foo: str = 'whatever'
    m: ForbiddenExtra = ForbiddenExtra()
    assert m.foo == 'whatever'

def test_forbidden_extra_fails() -> None:
    class ForbiddenExtra(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='forbid')
        foo: str = 'whatever'
    with pytest.raises(ValidationError) as exc_info:
        ForbiddenExtra(foo='ok', bar='wrong', spam='xx')
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'extra_forbidden',
            'loc': ('bar',),
            'msg': 'Extra inputs are not permitted',
            'input': 'wrong',
        },
        {
            'type': 'extra_forbidden',
            'loc': ('spam',),
            'msg': 'Extra inputs are not permitted',
            'input': 'xx',
        },
    ]

def test_assign_extra_no_validate() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(validate_assignment=True)
        a: float

    model: Model = Model(a=0.2)
    with pytest.raises(ValidationError, match="b\s+Object has no attribute 'b'"):
        model.b = 2

def test_assign_extra_validate() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(validate_assignment=True)
        a: float

    model: Model = Model(a=0.2)
    with pytest.raises(ValidationError, match="b\s+Object has no attribute 'b'"):
        model.b = 2

def test_model_property_attribute_error() -> None:
    class Model(BaseModel):
        @property
        def a_property(self) -> None:
            raise AttributeError('Internal Error')

    with pytest.raises(AttributeError, match='Internal Error'):
        Model().a_property

def test_extra_allowed() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow')
        a: float = 0.2

    model: Model = Model(a=0.2, b=0.1)
    assert model.b == 0.1
    assert not hasattr(model, 'c')
    model.c = 1
    assert hasattr(model, 'c')
    assert model.c == 1

def test_reassign_instance_method_with_extra_allow() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow')

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
        model_config: ConfigDict = ConfigDict(extra='ignore')
        a: float = 0.2

    model: Model = Model(a=0.2, b=0.1)
    assert not hasattr(model, 'b')
    with pytest.raises(ValueError, match='"Model" object has no field "b"'):
        model.b = 1
    assert model.model_extra is None

def test_field_order_is_preserved_with_extra() -> None:
    """This test covers https://github.com/pydantic/pydantic/issues/1234."""
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow')
        a: int
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
        def model_post_init(self, context: ValidationInfo) -> None:
            super().model_post_init(context)
            object.__setattr__(self, '__pydantic_extra__', None)

    class Model(BrokenExtraBaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow')

    m: Model = Model(extra_field='some extra value')
    with pytest.raises(AttributeError) as e:
        _ = m.extra_field
    assert e.value.args == ("'Model' object has no attribute 'extra_field'",)

def test_model_extra_is_none_when_extra_is_forbid() -> None:
    class Foo(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='forbid')

    assert Foo().model_extra is None

def test_set_attr(UltraSimpleModel: type[BaseModel]) -> None:
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
        model_config: ConfigDict = ConfigDict(populate_by_name=True)
        a: str = Field(alias='_a')

    assert Model(a='different').a == 'different'
    assert Model(a='different').model_dump() == {'a': 'different'}
    assert Model(a='different').model_dump(by_alias=True) == {'_a': 'different'}

def test_field_order() -> None:
    class Model(BaseModel):
        c: int = 0
        b: int = 10
        a: float
        d: dict = {}

    assert list(Model.model_fields.keys()) == ['c', 'b', 'a', 'd']

def test_required() -> None:
    class Model(BaseModel):
        b: int = 10

    m: Model = Model(a=10.2)
    assert m.model_dump() == dict(a=10.2, b=10)
    with pytest.raises(ValidationError) as exc_info:
        Model()
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'missing',
            'loc': ('a',),
            'msg': 'Field required',
            'input': {},
        }
    ]

def test_mutability() -> None:
    class TestModel(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='forbid', frozen=False)
        a: int = 10

    m: TestModel = TestModel()
    assert m.a == 10
    m.a = 11
    assert m.a == 11

def test_frozen_model() -> None:
    class FrozenModel(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='forbid', frozen=True)
        a: int = 10

    m: FrozenModel = FrozenModel()
    assert m.a == 10
    with pytest.raises(ValidationError) as exc_info:
        m.a = 11
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'frozen_instance',
            'loc': ('a',),
            'msg': 'Instance is frozen',
            'input': 11,
        },
    ]
    with pytest.raises(ValidationError) as exc_info:
        del m.a
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'frozen_instance',
            'loc': ('a',),
            'msg': 'Instance is frozen',
            'input': None,
        },
    ]
    assert m.a == 10

def test_frozen_model_cached_property() -> None:
    class FrozenModel(BaseModel):
        model_config: ConfigDict = ConfigDict(frozen=True)

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
        a: int = Field(default=10, frozen=True)

    m: FrozenModel = FrozenModel()
    assert m.a == 10
    with pytest.raises(ValidationError) as exc_info:
        m.a = 11
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'frozen_field',
            'loc': ('a',),
            'msg': 'Field is frozen',
            'input': 11,
        },
    ]
    with pytest.raises(ValidationError) as exc_info:
        del m.a
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'frozen_field',
            'loc': ('a',),
            'msg': 'Field is frozen',
            'input': None,
        },
    ]
    assert m.a == 10

def test_not_frozen_are_not_hashable() -> None:
    class TestModel(BaseModel):
        a: int = 10

    m: TestModel = TestModel()
    with pytest.raises(TypeError) as exc_info:
        hash(m)
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
        model_config: ConfigDict = ConfigDict(frozen=True)
        a: int = 10

    m: TestModel = TestModel()
    assert m.__hash__ is not None
    assert isinstance(hash(m), int)

def test_frozen_with_unhashable_fields_are_not_hashable() -> None:
    class TestModel(BaseModel):
        model_config: ConfigDict = ConfigDict(frozen=True)
        a: int = 10
        y: list[int] = [1, 2, 3]

    m: TestModel = TestModel()
    with pytest.raises(TypeError) as exc_info:
        hash(m)
    assert "unhashable type: 'list'" in str(exc_info.value)

def test_hash_function_empty_model() -> None:
    class TestModel(BaseModel):
        model_config: ConfigDict = ConfigDict(frozen=True)

    m: TestModel = TestModel()
    m2: TestModel = TestModel()
    assert m == m2
    assert hash(m) == hash(m2)

def test_hash_function_give_different_result_for_different_object() -> None:
    class TestModel(BaseModel):
        model_config: ConfigDict = ConfigDict(frozen=True)
        a: int = 10

    m: TestModel = TestModel()
    m2: TestModel = TestModel()
    m3: TestModel = TestModel(a=11)
    assert hash(m) == hash(m2)
    assert hash(m) != hash(m3)

def test_hash_function_works_when_instance_dict_modified() -> None:
    class TestModel(BaseModel):
        model_config: ConfigDict = ConfigDict(frozen=True)

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
    assert hash(Bar(x=2, y=3)) == 27
    assert hash(Buz(x=2, y=3)) == 27

def test_hash_method_is_inherited_for_frozen_models() -> None:
    class MyBaseModel(BaseModel):
        """A base model with sensible configurations."""
        model_config: ConfigDict = ConfigDict(frozen=True)

        def __hash__(self) -> int:
            return hash(id(self))

    class MySubClass(MyBaseModel):
        def cached_method(self) -> int:
            return len(self.x)

    my_instance: MySubClass = MySubClass(x={'a': 1, 'b': 2})
    assert my_instance.cached_method() == 2
    object.__setattr__(my_instance, 'x', {})
    assert my_instance.cached_method() == 2

@pytest.fixture(scope='session', name='ValidateAssignmentModel')
def validate_assignment_fixture() -> type[BaseModel]:
    class ValidateAssignmentModel(BaseModel):
        model_config: ConfigDict = ConfigDict(validate_assignment=True)
        a: int = 2
    return ValidateAssignmentModel

def test_validating_assignment_pass(ValidateAssignmentModel: type[BaseModel]) -> None:
    p: ValidateAssignmentModel = ValidateAssignmentModel(a=5, b='hello')
    p.a = 2
    assert p.a == 2
    assert p.model_dump() == {'a': 2, 'b': 'hello'}
    p.b = 'hi'
    assert p.b == 'hi'
    assert p.model_dump() == {'a': 2, 'b': 'hi'}

@pytest.mark.parametrize('init_valid', [False, True])
def test_validating_assignment_fail(ValidateAssignmentModel: type[BaseModel], init_valid: bool) -> None:
    p: ValidateAssignmentModel = ValidateAssignmentModel(a=5, b='hello')
    if init_valid:
        p.a = 5
        p.b = 'hello'
    with pytest.raises(ValidationError) as exc_info:
        p.a = 'b'
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'int_parsing',
            'loc': ('a',),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'input': 'b',
        }
    ]
    with pytest.raises(ValidationError) as exc_info:
        p.b = ''
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'string_too_short',
            'loc': ('b',),
            'msg': 'String should have at least 1 character',
            'input': '',
            'ctx': {'min_length': 1},
        }
    ]

class Foo(Enum):
    FOO: str = 'foo'
    BAR: str = 'bar'

@pytest.mark.parametrize('value', [Foo.FOO, Foo.FOO.value, 'foo'])
def test_enum_values(value: Union[Foo, str]) -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(use_enum_values=True)
        foo: str

    m: Model = Model(foo=value)
    foo_val: str = m.foo
    assert type(foo_val) is str, type(foo_val)
    assert foo_val == 'foo'
    foo_val = m.model_dump()['foo']
    assert type(foo_val) is str, type(foo_val)
    assert foo_val == 'foo'

def test_literal_enum_values() -> None:
    FooEnum = Enum('FooEnum', {'foo': 'foo_value', 'bar': 'bar_value'})

    class Model(BaseModel):
        boo: str = 'hoo'
        baz: FooEnum
        model_config: ConfigDict = ConfigDict(use_enum_values=True)

    m: Model = Model(baz=FooEnum.foo)
    assert m.model_dump() == {'baz': 'foo_value', 'boo': 'hoo'}
    assert m.model_dump_json() == '{"baz": "foo_value", "boo": "hoo"}'
    assert m.baz == 'foo_value'
    with pytest.raises(ValidationError) as exc_info:
        Model(baz=FooEnum.bar)
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'literal_error',
            'loc': ('baz',),
            'msg': "Input should be <FooEnum.foo: 'foo_value'>",
            'input': FooEnum.bar,
            'ctx': {'expected': "<FooEnum.foo: 'foo_value'>"},
        }
    ]

class StrFoo(str, Enum):
    FOO: str = 'foo'
    BAR: str = 'bar'

@pytest.mark.parametrize('value', [StrFoo.FOO, StrFoo.FOO.value, 'foo', 'hello'])
def test_literal_use_enum_values_multi_type(value: Union[StrFoo, str]) -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(use_enum_values=True)
        baz: str

    assert isinstance(Model(baz=value).baz, str)

def test_literal_use_enum_values_with_default() -> None:
    class Model(BaseModel):
        baz: str = Field(default=StrFoo.FOO)
        model_config: ConfigDict = ConfigDict(use_enum_values=True, validate_default=True)

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
        model_config: ConfigDict = ConfigDict(use_enum_values=True)
        x: MyEnum

    assert Model.model_validate({'x': MyEnum.val}, strict=True).x == 'val'

def test_union_enum_values() -> None:
    class MyEnum(Enum):
        val = 'val'

    class NormalModel(BaseModel):
        x: MyEnum

    class UseEnumValuesModel(BaseModel):
        x: MyEnum
        model_config: ConfigDict = ConfigDict(use_enum_values=True)

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
        friends: list = Field(default_factory=lambda: [])

    u1: User = User()
    u2: User = User()
    assert u1.friends is not u2.friends

class ArbitraryType:
    pass

def test_arbitrary_type_allowed_validation_success() -> None:
    class ArbitraryTypeAllowedModel(BaseModel):
        t: ArbitraryType
        model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    arbitrary_type_instance: ArbitraryType = ArbitraryType()
    m: ArbitraryTypeAllowedModel = ArbitraryTypeAllowedModel(t=arbitrary_type_instance)
    assert m.t == arbitrary_type_instance

class OtherClass:
    pass

def test_arbitrary_type_allowed_validation_fails() -> None:
    class ArbitraryTypeAllowedModel(BaseModel):
        t: ArbitraryType
        model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    input_value: OtherClass = OtherClass()
    with pytest.raises(ValidationError) as exc_info:
        ArbitraryTypeAllowedModel(t=input_value)
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'is_instance_of',
            'loc': ('t',),
            'msg': 'Input should be an instance of ArbitraryType',
            'input': input_value,
            'ctx': {'class': 'ArbitraryType'},
        }
    ]

def test_arbitrary_types_not_allowed() -> None:
    with pytest.raises(TypeError, match='Unable to generate pydantic-core schema for <class'):
        class ArbitraryTypeNotAllowedModel(BaseModel):
            t: ArbitraryType

@pytest.fixture(scope='session', name='TypeTypeModel')
def type_type_model_fixture() -> type[BaseModel]:
    class TypeTypeModel(BaseModel):
        t: type

    return TypeTypeModel

def test_type_type_validation_success(TypeTypeModel: type[BaseModel]) -> None:
    arbitrary_type_class: type = ArbitraryType
    m: TypeTypeModel = TypeTypeModel(t=arbitrary_type_class)
    assert m.t == arbitrary_type_class

def test_type_type_subclass_validation_success(TypeTypeModel: type[BaseModel]) -> None:
    class ArbitrarySubType(ArbitraryType):
        pass
    arbitrary_type_class: type = ArbitrarySubType
    m: TypeTypeModel = TypeTypeModel(t=arbitrary_type_class)
    assert m.t == arbitrary_type_class

@pytest.mark.parametrize('input_value', [OtherClass, 1])
def test_type_type_validation_fails(TypeTypeModel: type[BaseModel], input_value: Any) -> None:
    with pytest.raises(ValidationError) as exc_info:
        TypeTypeModel(t=input_value)
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'is_subclass_of',
            'loc': ('t',),
            'msg': 'Input should be a subclass of ArbitraryType',
            'input': input_value,
            'ctx': {'class': 'ArbitraryType'},
        }
    ]

@pytest.mark.parametrize('bare_type', [type, type])
def test_bare_type_type_validation_success(bare_type: type) -> None:
    class TypeTypeModel(BaseModel):
        t: type

    arbitrary_type_class: type = ArbitraryType
    m: TypeTypeModel = TypeTypeModel(t=arbitrary_type_class)
    assert m.t == arbitrary_type_class

@pytest.mark.parametrize('bare_type', [type, type])
def test_bare_type_type_validation_fails(bare_type: type) -> None:
    class TypeTypeModel(BaseModel):
        t: type

    arbitrary_type: Any = ArbitraryType()
    with pytest.raises(ValidationError) as exc_info:
        TypeTypeModel(t=arbitrary_type)
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'is_type',
            'loc': ('t',),
            'msg': 'Input should be a type',
            'input': arbitrary_type,
        }
    ]

def test_value_field_name_shadows_attribute() -> None:
    with pytest.raises(PydanticUserError, match=(
        "A non-annotated attribute was detected: `model_json_schema = 'abc'`"
    )):
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

    m: ModelB = ModelB(c=5, e={'a': 0})
    assert m.model_dump() == {'c': 5, 'd': 2, 'e': {'a': 0, 'b': 1}}
    assert m.model_dump(exclude_unset=True) == {'c': 5, 'e': {'a': 0}}
    assert dict(m) == {'c': 5, 'd': 2, 'e': ModelA(a=0, b=1)}

def test_dict_exclude_unset_populated_by_alias() -> None:
    class MyModel(BaseModel):
        model_config: ConfigDict = ConfigDict(populate_by_name=True)
        a: str = Field('default', alias='alias_a')
        b: str = Field('default', alias='alias_b')

    m: MyModel = MyModel(alias_a='a')
    assert m.model_dump(exclude_unset=True) == {'a': 'a'}
    assert m.model_dump(exclude_unset=True, by_alias=True) == {'alias_a': 'a'}

def test_dict_exclude_unset_populated_by_alias_with_extra() -> None:
    class MyModel(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow')
        a: str = Field('default', alias='alias_a')
        b: str = Field('default', alias='alias_b')

    m: MyModel = MyModel(alias_a='a', c='c')
    assert m.model_dump(exclude_unset=True) == {'a': 'a', 'c': 'c'}
    assert m.model_dump(exclude_unset=True, by_alias=True) == {'alias_a': 'a', 'c': 'c'}

def test_exclude_defaults() -> None:
    class Model(BaseModel):
        nullable_mandatory: Optional[str]
        facultative: str = 'x'
        nullable_facultative: Optional[str] = None

        class Config:
            pass

    m: Model = Model(mandatory='a', nullable_mandatory=None)
    assert m.model_dump(exclude_defaults=True) == {'mandatory': 'a', 'nullable_mandatory': None}
    m = Model(mandatory='a', nullable_mandatory=None, facultative='y', nullable_facultative=None)
    assert m.model_dump(exclude_defaults=True) == {'mandatory': 'a', 'nullable_mandatory': None, 'facultative': 'y'}
    m = Model(mandatory='a', nullable_mandatory=None, facultative='y', nullable_facultative='z')
    assert m.model_dump(exclude_defaults=True) == {
        'mandatory': 'a',
        'nullable_mandatory': None,
        'facultative': 'y',
        'nullable_facultative': 'z',
    }

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
        model_config: ConfigDict = ConfigDict(extra='allow')
        a: Optional[str] = Field(None, alias='alias_a')

    m: MyModel = MyModel(extra_key='extra')
    assert m.model_dump() == {'a': None, 'extra_key': 'extra'}
    assert m.model_dump(by_alias=True) == {'alias_a': None, 'extra_key': 'extra'}

def test_ignored_types() -> None:
    from pydantic import BaseModel

    class _ClassPropertyDescriptor:
        def __init__(self, getter: Callable[[type], Any]) -> None:
            self.getter = getter

        def __get__(self, instance: Any, owner: type) -> Any:
            return self.getter(owner)

    classproperty = _ClassPropertyDescriptor

    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(ignored_types=(classproperty,))

        @classproperty
        def class_name(cls: type) -> str:
            return cls.__name__

    assert Model.class_name == 'Model'
    assert Model().class_name == 'Model'

def test_model_iteration() -> None:
    class Foo(BaseModel):
        a: int = 1
        b: int = 2

    class Bar(BaseModel):
        c: int = 3
        d: Foo = Foo()

    m: Bar = Bar()
    assert m.model_dump() == {'c': 3, 'd': {'a': 1, 'b': 2}}
    assert list(m) == [('c', 3), ('d', Foo(a=1, b=2))]
    assert dict(m) == {'c': 3, 'd': Foo(a=1, b=2)}

def test_model_iteration_extra() -> None:
    class Foo(BaseModel):
        x: int = 1

    class Bar(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow')
        a: int
        b: Foo
        c: int
        d: Foo

    m: Bar = Bar.model_validate({'a': 1, 'b': {}, 'c': 2, 'd': Foo()})
    assert m.model_dump() == {'a': 1, 'b': {'x': 1}, 'c': 2, 'd': {'x': 1}}
    assert list(m) == [('a', 1), ('b', Foo(x=1)), ('c', 2), ('d', Foo(x=1))]
    assert dict(m) == {'a': 1, 'b': Foo(x=1), 'c': 2, 'd': Foo(x=1)}

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
        'negative indexes',
    ],
)
def test_model_export_nested_list(
    exclude: Optional[dict[str, Any]],
    expected: dict[str, Any],
    raises_match: Optional[str],
) -> None:
    class Foo(BaseModel):
        a: int = 1
        b: int = 2

    class Bar(BaseModel):
        pass

    m: Bar = Bar(c=3, foos=[Foo(a=1, b=2), Foo(a=3, b=4)])
    if raises_match is not None:
        with pytest.raises(raises_match):
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
        'exclude single dict key from all list items',
    ],
)
def test_model_export_dict_exclusion(
    excludes: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    class Foo(BaseModel):
        a: int = 1
        b: int = 2

    class Bar(BaseModel):
        w: int
        x: int
        y: int
        z: int

    class Model(BaseModel):
        a: int = 1
        bars: list[Bar] = [Bar(w=0, x=1, y=2, z=3)]
        model_config: ConfigDict = ConfigDict(extra='ignore')

    m: Model = Model(a=1, bars=[Bar(w=0, x=1, y=2, z=3), Bar(w=-1, x=2, y=2, z=3)])
    original_excludes = deepcopy(excludes)
    assert m.model_dump(exclude=excludes) == expected
    assert excludes == original_excludes

def test_field_exclude() -> None:
    class User(BaseModel):
        _priv: Any = PrivateAttr()
        password: SecretStr = Field(exclude=True)
        id: int
        username: str
        hobbies: list[str]

    my_user: User = User(id=42, username='JohnDoe', password=SecretStr('hashedpassword'), hobbies=['scuba diving'])
    my_user._priv = 13
    assert my_user.id == 42
    assert my_user.password.get_secret_value() == 'hashedpassword'
    assert my_user.model_dump() == {'id': 42, 'username': 'JohnDoe', 'hobbies': ['scuba diving']}

def test_revalidate_instances_never() -> None:
    class User(BaseModel):
        hobbies: list[str]

    class Transaction(BaseModel):
        user: User

    my_user: User = User(hobbies=['scuba diving'])
    t: Transaction = Transaction(user=my_user)
    assert t.user is my_user
    assert t.user.hobbies is my_user.hobbies

    class SubUser(User):
        sins: list[str]

    my_sub_user: SubUser = SubUser(hobbies=['scuba diving'], sins=['lying'])
    t = Transaction(user=my_sub_user)
    assert t.user is my_sub_user
    assert t.user.hobbies is my_sub_user.hobbies

def test_revalidate_instances_sub_instances() -> None:
    class User(BaseModel, Generic[Any]):
        hobbies: list[str]

    class Transaction(BaseModel):
        user: User

    my_user: User = User(hobbies=['scuba diving'])

    t: Transaction = Transaction(user=my_user)
    assert t.user is my_user
    assert t.user.hobbies is my_user.hobbies

    class SubUser(User, Generic[Any]):
        sins: list[str]

    my_sub_user: SubUser = SubUser(hobbies=['scuba diving'], sins=['lying'])
    t = Transaction(user=my_sub_user)
    assert t.user is not my_sub_user
    assert t.user.hobbies is not my_sub_user.hobbies
    assert not hasattr(t.user, 'sins')

def test_revalidate_instances_always() -> None:
    class User(BaseModel, Generic[Any]):
        hobbies: list[str]

    class Transaction(BaseModel):
        user: User

    my_user: User = User(hobbies=['scuba diving'])

    t: Transaction = Transaction(user=my_user)
    assert t.user is not my_user
    assert t.user.hobbies is not my_user.hobbies

    class SubUser(User, Generic[Any]):
        sins: list[str]

    my_sub_user: SubUser = SubUser(hobbies=['scuba diving'], sins=['lying'])
    t = Transaction(user=my_sub_user)
    assert t.user is not my_sub_user
    assert t.user.hobbies is not my_sub_user.hobbies
    assert not hasattr(t.user, 'sins')

def test_revalidate_instances_always_list_of_model_instance() -> None:
    class A(BaseModel):
        name: str

        model_config: ConfigDict = ConfigDict(revalidate_instances='always')

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
        {'model_fields', 'sub_config'},
    ],
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
            id='exclude with ellipsis'
        ),
    ],
)
def test_model_export_exclusion_with_fields_and_config(
    kinds: set[str],
    exclude: Optional[dict[str, Any]],
    expected: dict[str, Any],
) -> None:
    """Test that exporting models with fields using the export parameter works."""
    class ChildConfig:
        fields: dict[str, Any] = {}

    class ParentConfig:
        fields: dict[str, Any] = {}

    if 'sub_config' in kinds:
        ChildConfig.fields = {'b': {'exclude': ...}, 'a': {'exclude': {1}}}

    if 'combined_config' in kinds:
        ParentConfig.fields = {
            'b': {'exclude': ...},
            'c': {'exclude': {'b': ..., 'a': {1}}},
            'd': {'exclude': {'a': ..., 'b': ...}},
        }
    elif 'model_config' in kinds:
        ParentConfig.fields = {'b': {'exclude': ...}, 'd': {'exclude': {'a'}}}

    class Sub(BaseModel):
        a: list[int] = Field(default_factory=lambda: [3, 4, 5])
        b: int = Field(4, exclude=... if 'sub_fields' in kinds else None)
        c: str = 'foobar'
        Config: type = ChildConfig

    class Model(BaseModel):
        a: int = 0
        b: int = Field(2, exclude=... if 'model_fields' in kinds else None)
        c: Sub = Sub()
        d: Sub = Field(Sub(), exclude={'a'} if 'model_fields' in kinds else None)
        Config: type = ParentConfig

    m: Model = Model()
    assert m.model_dump(exclude=exclude) == expected, 'Unexpected model export result'

@pytest.mark.skip(reason='not implemented')
def test_model_export_exclusion_inheritance() -> None:
    class Sub(BaseModel):
        s1: str = 'v1'
        s2: str = 'v2'
        s3: str = 'v3'
        s4: str = Field('v4', exclude=...)

    class Parent(BaseModel):
        model_config: ConfigDict = ConfigDict(fields={'a': {'exclude': ...}, 's': {'exclude': {'s1'}}})
        b: int = Field(exclude=...)
        s: Sub = Sub()

    class Child(Parent):
        model_config: ConfigDict = ConfigDict(fields={'c': {'exclude': ...}, 's': {'exclude': {'s2'}}})

    actual: Parent = Child(a=0, b=1, c=2, d=3)
    expected: dict[str, Any] = {'d': 3, 's': {'s3': 'v3'}}
    assert actual.model_dump() == expected, 'Unexpected model export result'

@pytest.mark.skip(reason='not implemented')
def test_model_export_with_true_instead_of_ellipsis() -> None:
    class Sub(BaseModel):
        s1: int

    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(fields={'c': {'exclude': True}})
        a: int = 2
        b: int = Field(3, exclude=True)
        c: int = Field(4)
        s: Sub = Sub()

    m: Model = Model()
    assert m.model_dump(exclude={'s': True}) == {'a': 2}

@pytest.mark.skip(reason='not implemented')
def test_model_export_inclusion() -> None:
    class Sub(BaseModel):
        s1: str = 'v1'
        s2: str = 'v2'
        s3: str = 'v3'
        s4: str = 'v4'

    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(
            fields={
                'a': {'include': {'s2', 's1', 's3'}},
                'b': {'include': {'s1'}},
                'c': {'include': {'s1', 's2'}},
            }
        )
        a: Sub = Sub()
        b: Sub = Field(Sub(), include={'s1'})
        c: Sub = Field(Sub(), include={'s1', 's2'})

    assert Model.model_fields['a'].field_info.include == {'s1': ..., 's2': ..., 's3': ...}
    assert Model.model_fields['b'].field_info.include == {'s1': ...}
    assert Model.model_fields['c'].field_info.include == {'s1': ..., 's2': ...}
    actual: dict[str, Any] = Model().model_dump(include={'a': {'s3', 's4'}, 'b': ..., 'c': ...})
    expected: dict[str, Any] = {'a': {'s3': 'v3'}, 'b': {'s1': 'v1'}, 'c': {'s1': 'v1', 's2': 'v2'}}
    assert actual == expected, 'Unexpected model export result'

@pytest.mark.skip(reason='not implemented')
def test_model_export_inclusion_inheritance() -> None:
    class Sub(BaseModel):
        s1: str = Field('v1', include=...)
        s2: str = Field('v2', include=...)
        s3: str = Field('v3', include=...)
        s4: str = 'v4'

    class Parent(BaseModel):
        model_config: ConfigDict = ConfigDict(fields={'b': {'include': ...}})
        s: Sub = Field(Sub(), include={'s1', 's2'})

    class Child(Parent):
        model_config: ConfigDict = ConfigDict(fields={'a': {'include': ...}, 's': {'include': {'s1'}}})
        a: int = 0
        b: int = 1
        c: int = 2

    actual: dict[str, Any] = Child(a=0, b=1, c=2).model_dump()
    expected: dict[str, Any] = {'a': 0, 'b': 1, 's': {'s1': 'v1'}}
    assert actual == expected, 'Unexpected model export result'

def test_untyped_fields_warning() -> None:
    with pytest.raises(PydanticUserError, match=re.escape(
        "A non-annotated attribute was detected: `x = 1`. "
        "All model fields require a type annotation; if `x` is not meant to be a field, "
        "you may be able to resolve this error by annotating it as a `ClassVar` or updating `model_config['ignored_types']`."
    )):
        class WarningModel(BaseModel):
            x: int = 1  # Corrected to have type annotation

    class NonWarningModel(BaseModel):
        x: int = 1

def test_untyped_fields_error() -> None:
    with pytest.raises(TypeError, match="Field 'a' requires a type annotation"):
        class Model(BaseModel):
            a: Any = Field('foobar')

def test_custom_init_subclass_params() -> None:
    class DerivedModel(BaseModel):
        def __init_subclass__(cls, something: int, **kwargs: Any) -> None:
            super().__init_subclass__()
            cls.something: int = something

    class NewModel(DerivedModel, something=2):
        something: int = 1

    assert NewModel.something == 2

def test_recursive_model() -> None:
    class MyModel(BaseModel):
        field: Optional['MyModel'] = None

    MyModel.update_forward_refs()
    m: MyModel = MyModel(field={'field': {'field': None}})
    assert m.model_dump() == {'field': {'field': {'field': None}}}

def test_recursive_cycle_with_repeated_field() -> None:
    class A(BaseModel):
        b: 'B'

    class B(BaseModel):
        a1: Optional[A] = None
        a2: Optional[A] = None

    A.update_forward_refs()
    with pytest.raises(ValidationError) as exc_info:
        A(model_validate({'b': {'a1': {'b': {'a1': None}}}}))
    assert exc_info.value.errors(include_url=False) == [
        {
            'input': {'a1': None},
            'loc': ('b', 'a1', 'b'),
            'msg': 'Field required',
            'type': 'missing',
        }
    ]

def test_two_defaults() -> None:
    with pytest.raises(TypeError, match='^cannot specify both default and default_factory$'):
        class Model(BaseModel):
            a: int = Field(default=3, default_factory=lambda: 3)

def test_default_factory() -> None:
    class ValueModel(BaseModel):
        uid: UUID = uuid4()

    m1: ValueModel = ValueModel()
    m2: ValueModel = ValueModel()
    assert m1.uid == m2.uid

    class DynamicValueModel(BaseModel):
        uid: UUID = Field(default_factory=uuid4)

    m1 = DynamicValueModel()
    m2 = DynamicValueModel()
    assert isinstance(m1.uid, UUID)
    assert m1.uid != m2.uid

    class FunctionModel(BaseModel):
        a: int = 1
        uid: Callable[[], UUID] = Field(default_factory=uuid4)

    m = FunctionModel()
    assert m.uid is uuid4  # This might need to be adjusted based on actual implementation

    class MySingleton:
        pass

    MY_SINGLETON = MySingleton()

    class SingletonFieldModel(BaseModel):
        singleton: MySingleton = Field(default_factory=lambda: MY_SINGLETON)
        model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    assert SingletonFieldModel().singleton is MY_SINGLETON

def test_default_factory_called_once() -> None:
    """It should call only once the given factory by default"""
    class Seq:
        def __init__(self) -> None:
            self.v: int = 0

        def __call__(self) -> int:
            self.v += 1
            return self.v

    class MyModel(BaseModel):
        id: int = Field(default_factory=Seq())

    m1: MyModel = MyModel()
    assert m1.id == 1
    m2: MyModel = MyModel()
    assert m2.id == 2
    assert m1.id == 1

def test_default_factory_called_once_2() -> None:
    """It should call only once the given factory by default"""
    v: int = 0

    def factory() -> int:
        nonlocal v
        v += 1
        return v

    class MyModel(BaseModel):
        id: int = Field(default_factory=factory)

    m1: MyModel = MyModel()
    assert m1.id == 1
    m2: MyModel = MyModel()
    assert m2.id == 2

def test_default_factory_validate_children() -> None:
    class Child(BaseModel):
        x: int

    class Parent(BaseModel):
        children: list[Child] = Field(default_factory=list)

    Parent(children=[{'x': 1}, {'x': 2}])
    with pytest.raises(ValidationError) as exc_info:
        Parent(children=[{'x': 1}, {'y': 2}])
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'missing',
            'loc': ('children', 1, 'x'),
            'msg': 'Field required',
            'input': {'y': 2},
        }
    ]

def test_default_factory_parse() -> None:
    class Inner(BaseModel):
        val: int = Field(0)

    class Outer(BaseModel):
        inner_1: Inner = Field(default_factory=Inner)
        inner_2: Inner = Field(default_factory=Inner)

    default = Outer().model_dump()
    parsed: Outer = Outer.model_validate(default)
    assert parsed.model_dump() == {'inner_1': {'val': 0}, 'inner_2': {'val': 0}}
    assert repr(parsed) == 'Outer(inner_1=Inner(val=0), inner_2=Inner(val=0))'

def test_default_factory_validated_data_arg() -> None:
    class Model(BaseModel):
        a: int = 1
        b: int = Field(default_factory=lambda data: data['a'])

    model: Model = Model()
    assert model.b == 1
    model = Model.model_validate_json('{"a": 1}')
    assert model.b == 1

    class InvalidModel(BaseModel):
        a: int = Field(default_factory=lambda data: data['b'])

    with pytest.raises(KeyError):
        InvalidModel.model_validate({'b': 2})

def test_default_factory_validated_data_arg_not_required() -> None:
    def fac(data: Optional[dict[str, Any]] = None) -> int:
        if data is not None:
            return data['a']
        return 3

    class Model(BaseModel):
        a: int = 1
        b: int = Field(default_factory=fac)

    model: Model = Model()
    assert model.b == 3
    model = Model.model_validate({'a': 1})
    assert model.b == 1

def test_reuse_same_field() -> None:
    required_field = Field()

    class Model1(BaseModel):
        required: Any = required_field

    class Model2(BaseModel):
        required: Any = required_field

    with pytest.raises(ValidationError):
        Model1.model_validate({})

    with pytest.raises(ValidationError):
        Model2.model_validate({})

def test_base_config_type_hinting() -> None:
    class M(BaseModel):
        pass

    get_type_hints(type(M.model_config))

def test_frozen_field_with_validate_assignment() -> None:
    """assigning a frozen=True field should raise a TypeError"""
    class Entry(BaseModel):
        model_config: ConfigDict = ConfigDict(validate_assignment=True)
        id: int = Field(frozen=True)
        val: int

    r: Entry = Entry(id=1, val=100)
    assert r.val == 100
    r.val = 101
    assert r.val == 101
    assert r.id == 1
    with pytest.raises(ValidationError) as exc_info:
        r.id = 2
    assert exc_info.value.errors(include_url=False) == [
        {
            'input': 2,
            'loc': ('id',),
            'msg': 'Field is frozen',
            'type': 'frozen_field',
        },
    ]

def test_repr_field() -> None:
    class Model(BaseModel):
        a: Any = Field()
        b: Any = Field(repr=True)
        c: Any = Field(repr=False)

    m: Model = Model(a=1, b=2.5, c=True)
    assert repr(m) == 'Model(a=1, b=2.5)'
    assert repr(Model.model_fields['a']) == 'FieldInfo(annotation=Any, required=True)'
    assert repr(Model.model_fields['b']) == 'FieldInfo(annotation=Any, required=True)'
    assert repr(Model.model_fields['c']) == 'FieldInfo(annotation=Any, required=True, repr=False)'

def test_inherited_model_field_copy() -> None:
    """It should copy models used as fields by default"""

    class Image(BaseModel):
        path: str

        def __hash__(self) -> int:
            return id(self)

    class Item(BaseModel):
        images: set[Image]

    image_1: Image = Image(path='my_image1.png')
    image_2: Image = Image(path='my_image2.png')
    item: Item = Item(images={image_1, image_2})
    assert image_1 in item.images
    assert id(image_1) in {id(image) for image in item.images}
    assert id(image_2) in {id(image) for image in item.images}

def test_mapping_subclass_as_input() -> None:
    class CustomMap(dict[str, int]):
        pass

    class Model(BaseModel):
        x: Mapping[str, int]

    d: CustomMap = CustomMap()
    d['one'] = 1
    d['two'] = 2
    v: Mapping[str, int] = Model(x=d).x
    assert isinstance(v, Mapping)
    assert not isinstance(v, CustomMap)
    assert v == {'one': 1, 'two': 2}

def test_typing_coercion_dict() -> None:
    class Model(BaseModel):
        x: dict[str, int]

    m: Model = Model(x={'one': 1, 'two': 2})
    assert repr(m) == "Model(x={'one': 1, 'two': 2})"

KT = TypeVar('KT')
VT = TypeVar('VT')

class MyDict(dict[KT, VT]):
    def __repr__(self) -> str:
        return f'MyDict({super().__repr__()})'

def test_class_kwargs_config() -> None:
    class Base(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='forbid', alias_generator=str.upper)

    assert Base.model_config['extra'] == 'forbid'
    assert Base.model_config['alias_generator'] is str.upper

    class Model(Base, extra='allow'):
        model_config: ConfigDict = ConfigDict()

    assert Model.model_config['extra'] == 'allow'
    assert Model.model_config['alias_generator'] is str.upper

def test_class_kwargs_config_and_attr_conflict() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow', alias_generator=str.upper, title='Foobar')

    class Meta(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow', alias_generator=str.upper, title='Foobar')

    class ModelWithConflict(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='forbid', title='Foobar')

    # Assuming merging happens correctly
    # However, as per the original test, it's mixed with parent and child
    # Adjust annotations accordingly
    class Base(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow', alias_generator=str.upper)

    class Model(Base, extra='allow'):
        model_config: ConfigDict = ConfigDict(extra='forbid', title='Foobar')

    assert Model.model_config['extra'] == 'allow'
    assert Model.model_config['alias_generator'] is str.upper
    assert Model.model_config['title'] == 'Foobar'

def test_class_kwargs_custom_config() -> None:
    if platform.python_implementation() == 'PyPy':
        msg = "__init_subclass__\\(\\) got an unexpected keyword argument 'some_config'"
    else:
        msg = '__init_subclass__\\(\\) takes no keyword arguments'
    with pytest.raises(TypeError, match=msg):
        class Model(BaseModel, some_config='new_value'):
            pass

def test_new_union_origin() -> None:
    """On 3.10+, origin of `int | str` is `types.UnionType`, not `typing.Union`"""

    class Model(BaseModel):
        x: int | str

    m1: Model = Model(x=3)
    m2: Model = Model(x='3')
    m3: Model = Model(x='pika')
    assert Model.model_json_schema() == {
        'title': 'Model',
        'type': 'object',
        'properties': {
            'x': {
                'title': 'X',
                'anyOf': [{'type': 'integer'}, {'type': 'string'}],
            }
        },
        'required': ['x'],
    }

@pytest.mark.parametrize(
    'ann,value',
    [
        pytest.param(Final, None, id='no-arg'),
        pytest.param(Final[int], Field(), id='with-arg'),
    ],
)
def test_frozen_field_decl_without_default_val(ann: Any, value: Optional[Any]) -> None:
    class Model(BaseModel):
        if value is not None:
            a: Final[int] = value

    assert 'a' not in Model.__class_vars__
    assert 'a' in Model.model_fields
    assert Model.model_fields['a'].frozen

@pytest.mark.parametrize('ann', [Final, Final[int]], ids=['no-arg', 'with-arg'])
def test_deprecated_final_field_decl_with_default_val(ann: Any) -> None:
    with pytest.warns(PydanticDeprecatedSince211):
        class Model(BaseModel):
            a: Final[int] = 10
    assert 'a' in Model.__class_vars__
    assert 'a' not in Model.model_fields

def test_final_field_reassignment() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(validate_assignment=True)
        a: Final[int] = 10

    obj: Model = Model(a=10)
    with pytest.raises(ValidationError) as exc_info:
        obj.a = 20
    assert exc_info.value.errors(include_url=False) == [
        {
            'input': 20,
            'loc': ('a',),
            'msg': 'Field is frozen',
            'type': 'frozen_field',
        },
    ]

def test_field_by_default_is_not_frozen() -> None:
    class Model(BaseModel):
        a: int

    assert not Model.model_fields['a'].frozen

def test_annotated_final() -> None:
    class Model(BaseModel):
        a: Final[str] = Field('abc', title='abc')

    assert Model.model_fields['a'].frozen
    assert Model.model_fields['a'].title == 'abc'

    class Model2(BaseModel):
        a: Final[str] = Field('def', title='def')

    assert Model2.model_fields['a'].frozen
    assert Model2.model_fields['a'].title == 'def'

def test_post_init() -> None:
    calls: list[str] = []

    class InnerModel(BaseModel):
        a: int
        b: int

        def model_post_init(self, context: ValidationInfo) -> None:
            super().model_post_init(context)
            assert self.model_dump() == {'a': 3, 'b': 4}
            calls.append('inner_model_post_init')

    class Model(BaseModel):
        c: int
        d: int
        sub: InnerModel

        def model_post_init(self, context: ValidationInfo) -> None:
            assert self.model_dump() == {'c': 1, 'd': 2, 'sub': {'a': 3, 'b': 4}}
            calls.append('model_post_init')

    m: Model = Model(c=1, d='2', sub={'a': 3, 'b': '4'})
    assert calls == ['inner_model_post_init', 'model_post_init']
    assert m.model_dump() == {'c': 1, 'd': 2, 'sub': {'a': 3, 'b': 4}}

    class SubModel(Model):
        def model_post_init(self, context: ValidationInfo) -> None:
            assert self.model_dump() == {'c': 1, 'd': 2, 'sub': {'a': 3, 'b': 4}}
            super().model_post_init(context)
            calls.append('submodel_post_init')

    calls.clear()
    m = SubModel(c=1, d='2', sub={'a': 3, 'b': '4'})
    assert calls == ['inner_model_post_init', 'model_post_init', 'submodel_post_init']
    assert m.model_dump() == {'c': 1, 'd': 2, 'sub': {'a': 3, 'b': 4}}

def test_post_init_function_attrs_preserved() -> None:
    class Model(BaseModel):
        def model_post_init(self, context: ValidationInfo) -> None:
            """Custom docstring"""

    assert Model.model_post_init.__doc__ == 'Custom docstring'

@pytest.mark.parametrize('include_private_attribute', [True, False])
def test_post_init_call_signatures(include_private_attribute: bool) -> None:
    calls: list[tuple[Any, Any]] = []

    class Model(BaseModel):
        if include_private_attribute:
            _x: int = PrivateAttr(1)

        def model_post_init(self, *args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

    Model(a=1, b=2)
    assert calls == [((None,), {})]
    Model.model_construct(a=3, b=4)
    assert calls == [((None,), {}), ((None,), {})]

def test_post_init_not_called_without_override() -> None:
    calls: list[str] = []

    def monkey_patched_model_post_init(cls: type[BaseModel], __context: Any) -> None:
        calls.append('BaseModel.model_post_init')

    original_base_model_post_init = BaseModel.model_post_init
    try:
        BaseModel.model_post_init = monkey_patched_model_post_init

        class WithoutOverrideModel(BaseModel):
            pass

        WithoutOverrideModel()
        WithoutOverrideModel.model_construct()
        assert calls == []

        class WithOverrideModel(BaseModel):
            def model_post_init(self, context: ValidationInfo) -> None:
                calls.append('WithOverrideModel.model_post_init')

        WithOverrideModel()
        assert calls == ['WithOverrideModel.model_post_init']
        WithOverrideModel.model_construct()
        assert calls == ['WithOverrideModel.model_post_init', 'WithOverrideModel.model_post_init']
    finally:
        BaseModel.model_post_init = original_base_model_post_init

def test_model_post_init_subclass_private_attrs() -> None:
    """https://github.com/pydantic/pydantic/issues/7293"""
    calls: list[str] = []

    class A(BaseModel):
        a: int = 1

        def model_post_init(self, context: ValidationInfo) -> None:
            calls.append(f'{self.__class__.__name__}.model_post_init')

    class B(A):
        pass

    class C(B):
        _private: bool = PrivateAttr(True)

    C()
    assert calls == ['C.model_post_init']

def test_model_post_init_supertype_private_attrs() -> None:
    """https://github.com/pydantic/pydantic/issues/9098"""

    class Model(BaseModel):
        _private: int = PrivateAttr(default=12)

    class SubModel(Model):
        def model_post_init(self, context: ValidationInfo) -> None:
            if self._private == 12:
                self._private = 13
            super().model_post_init(context)

    m: SubModel = SubModel()
    assert m._private == 13

def test_model_post_init_subclass_setting_private_attrs() -> None:
    """https://github.com/pydantic/pydantic/issues/7091"""
    class Model(BaseModel):
        _priv1: int = PrivateAttr(default=91)
        _priv2: int = PrivateAttr(default=92)

        def model_post_init(self, context: ValidationInfo) -> None:
            self._priv1 = 100

    class SubModel(Model):
        _priv3: int = PrivateAttr(default=93)
        _priv4: int = PrivateAttr(default=94)
        _priv5: int = PrivateAttr()
        _priv6: int = PrivateAttr()

        def model_post_init(self, context: ValidationInfo) -> None:
            self._priv3 = 200
            self._priv5 = 300
            super().model_post_init(context)

    m: SubModel = SubModel()
    assert m._priv1 == 100
    assert m._priv2 == 92
    assert m._priv3 == 200
    assert m._priv4 == 94
    assert m._priv5 == 300
    with pytest.raises(AttributeError):
        _ = m._priv6

def test_model_post_init_correct_mro() -> None:
    """https://github.com/pydantic/pydantic/issues/7293"""
    calls: list[str] = []

    class A(BaseModel):
        a: int = 1

    class B(BaseModel):
        b: int = 1

        def model_post_init(self, context: ValidationInfo) -> None:
            calls.append(f'{self.__class__.__name__}.model_post_init')

    class C(A, B):
        _private: bool = PrivateAttr(True)

    C()
    assert calls == ['C.model_post_init']

def test_del_model_attr() -> None:
    class Model(BaseModel):
        pass

    m: Model = Model(some_field='value')
    assert hasattr(m, 'some_field')
    del m.some_field
    assert not hasattr(m, 'some_field')

@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='In this single case `del` behaves weird on pypy')
def test_del_model_attr_error() -> None:
    class Model(BaseModel):
        pass

    m: Model = Model(some_field='value')
    assert not hasattr(m, 'other_field')
    with pytest.raises(AttributeError, match='other_field'):
        del m.other_field

def test_del_model_attr_with_private_attrs() -> None:
    class Model(BaseModel):
        _private_attr: int = PrivateAttr(default=1)

    m: Model = Model(some_field='value')
    assert hasattr(m, 'some_field')
    del m.some_field
    assert not hasattr(m, 'some_field')

@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='In this single case `del` behaves weird on pypy')
def test_del_model_attr_with_private_attrs_error() -> None:
    class Model(BaseModel):
        _private_attr: int = PrivateAttr(default=1)

    m: Model = Model(some_field='value')
    assert not hasattr(m, 'other_field')
    with pytest.raises(AttributeError, match="'Model' object has no attribute 'other_field'"):
        del m.other_field

def test_del_model_attr_with_private_attrs_twice_error() -> None:
    class Model(BaseModel):
        _private_attr: int = 1

    m: Model = Model(some_field='value')
    assert hasattr(m, '_private_attr')
    del m._private_attr
    with pytest.raises(AttributeError, match="'Model' object has no attribute '_private_attr'"):
        del m._private_attr

def test_deeper_recursive_model() -> None:
    class A(BaseModel):
        b: 'B'

    class B(BaseModel):
        c: 'C'

    class C(BaseModel):
        a: Optional[A] = None

    A.update_forward_refs()
    B.update_forward_refs()
    C.update_forward_refs()

    m: A = A(b=B(c=C(a=None)))
    assert m.model_dump() == {'b': {'c': {'a': None}}}

def test_model_rebuild_localns() -> None:
    class A(BaseModel):
        pass

    class B(BaseModel):
        pass

    B.model_rebuild(_types_namespace={'Model': A})
    m: B = B(a={'x': 1})
    assert m.model_dump() == {'a': {'x': 1}}
    assert isinstance(m.a, A)

    class C(BaseModel):
        pass

    with pytest.raises(PydanticUndefinedAnnotation, match="name 'Model' is not defined"):
        C.model_rebuild(_types_namespace={'A': A})

def test_model_rebuild_zero_depth() -> None:
    class Model(BaseModel):
        pass

    X_Type = str
    with pytest.raises(NameError, match='X_Type'):
        Model.model_rebuild(_parent_namespace_depth=0)
    Model.__pydantic_parent_namespace__.update({'X_Type': int})
    Model.model_rebuild(_parent_namespace_depth=0)
    m: Model = Model(x=42)
    assert m.model_dump() == {'x': 42}

@pytest.fixture(scope='session', name='InnerEqualityModel')
def inner_equality_fixture() -> type[BaseModel]:
    class InnerEqualityModel(BaseModel):
        ix: int = 0
        _iy: Any = PrivateAttr()
        _iz: int = PrivateAttr(0)
    return InnerEqualityModel

@pytest.fixture(scope='session', name='EqualityModel')
def equality_fixture(InnerEqualityModel: type[BaseModel]) -> type[BaseModel]:
    class EqualityModel(BaseModel):
        x: int = 0
        _y: Any = PrivateAttr()
        _z: int = PrivateAttr(0)
        model: InnerEqualityModel

    return EqualityModel

def test_model_equality(
    EqualityModel: type[BaseModel],
    InnerEqualityModel: type[BaseModel],
) -> None:
    m1: EqualityModel = EqualityModel(x=0, model=InnerEqualityModel(ix=0))
    m2: EqualityModel = EqualityModel(x=0, model=InnerEqualityModel(ix=0))
    assert m1 == m2

def test_model_equality_type(
    EqualityModel: type[BaseModel],
    InnerEqualityModel: type[BaseModel],
) -> None:
    class Model1(BaseModel):
        x: int

    class Model2(BaseModel):
        x: int

    m1: Model1 = Model1(x=1)
    m2: Model2 = Model2(x=1)
    assert m1.model_dump() == m2.model_dump()
    assert m1 != m2

def test_model_equality_dump(
    EqualityModel: type[BaseModel],
    InnerEqualityModel: type[BaseModel],
) -> None:
    inner_model: InnerEqualityModel = InnerEqualityModel(ix=0)
    assert inner_model != inner_model.model_dump()
    model: EqualityModel = EqualityModel(x=0, model=inner_model)
    assert model != dict(model)
    assert dict(model) != model.model_dump()

def test_model_equality_fields_set(InnerEqualityModel: type[BaseModel]) -> None:
    m1: InnerEqualityModel = InnerEqualityModel(ix=0)
    m2: InnerEqualityModel = InnerEqualityModel(ix=0, ix=0)  # Assuming 'ix' is required
    assert m1.model_fields_set != m2.model_fields_set
    assert m1 == m2

def test_model_equality_private_attrs(InnerEqualityModel: type[BaseModel]) -> None:
    m: InnerEqualityModel = InnerEqualityModel(ix=0)
    m1: InnerEqualityModel = m.model_copy()
    m2: InnerEqualityModel = m.model_copy()
    m3: InnerEqualityModel = m.model_copy()
    m2._iy = 1
    m3._iz = 1
    models: list[InnerEqualityModel] = [m1, m2, m3]
    for i, first_model in enumerate(models):
        for j, second_model in enumerate(models):
            if i == j:
                assert first_model == second_model
            else:
                assert first_model != second_model
    m2_equal: InnerEqualityModel = m.model_copy()
    m2_equal._iy = 1
    assert m2 == m2_equal
    m3_equal: InnerEqualityModel = m.model_copy()
    m3_equal._iz = 1
    assert m3 == m3_equal

def test_model_copy_extra() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow')
        x: int

    m: Model = Model(x=1, y=2)
    assert m.model_dump() == {'x': 1, 'y': 2}
    assert m.model_extra == {'y': 2}
    m2: Model = m.model_copy()
    assert m2.model_dump() == {'x': 1, 'y': 2}
    assert m2.model_extra == {'y': 2}
    m3: Model = m.model_copy(update={'x': 4, 'z': 3})
    assert m3.model_dump() == {'x': 4, 'y': 2, 'z': 3}
    assert m3.model_extra == {'y': 2, 'z': 3}
    m4: Model = m.model_copy(update={'x': 4, 'z': 3})
    assert m4.model_dump() == {'x': 4, 'y': 2, 'z': 3}
    assert m4.model_extra == {'y': 2, 'z': 3}
    m = Model(x=1, a=2)
    m.__pydantic_extra__ = None
    m5: Model = m.model_copy(update={'x': 4, 'b': 3})
    assert m5.model_dump() == {'x': 4, 'b': 3}
    assert m5.model_extra == {'b': 3}

def test_model_parametrized_name_not_generic() -> None:
    class Model(BaseModel):
        pass
    with pytest.raises(TypeError, match='Concrete names should only be generated for generic models.'):
        Model.model_parametrized_name(())

def test_model_equality_generics() -> None:
    T = TypeVar('T')

    class GenericModel(BaseModel, Generic[T], frozen=True):
        pass

    class ConcreteModel(BaseModel):
        pass

    assert ConcreteModel(x=1) != GenericModel(x=1)
    assert ConcreteModel(x=1) != GenericModel[Any](x=1)
    assert ConcreteModel(x=1) != GenericModel[int](x=1)
    assert GenericModel(x=1) != GenericModel(x=2)

    S = TypeVar('S')
    models: list[GenericModel[Any]] = [
        GenericModel(x=1),
        GenericModel[S](x=1),
        GenericModel[Any](x=1),
        GenericModel[int](x=1),
        GenericModel[float](x=1),
    ]
    for m1 in models:
        for m2 in models:
            m3: GenericModel[Any] = GenericModel[type(m1)](x=m1)
            m4: GenericModel[Any] = GenericModel[type(m2)](x=m2)
            assert m1 == m2
            assert m3 == m4
            assert hash(m1) == hash(m2)
            assert hash(m3) == hash(m4)

def test_model_validate_strict() -> None:
    class LaxModel(BaseModel):
        model_config: ConfigDict = ConfigDict(strict=False)
        x: int
        b: constr(min_length=1) = 'hello'

    class StrictModel(BaseModel):
        model_config: ConfigDict = ConfigDict(strict=True)
        x: int
        b: constr(min_length=1) = 'hello'

    assert LaxModel.model_validate({'x': '1'}, strict=None) == LaxModel(x=1)
    assert LaxModel.model_validate({'x': '1'}, strict=False) == LaxModel(x=1)
    with pytest.raises(ValidationError) as exc_info:
        LaxModel.model_validate({'x': '1'}, strict=True)
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'int_type',
            'loc': ('x',),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'input': '1',
        }
    ]
    with pytest.raises(ValidationError) as exc_info:
        StrictModel.model_validate({'x': '1'})
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'int_type',
            'loc': ('x',),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'input': '1',
        }
    ]
    assert StrictModel.model_validate({'x': '1'}, strict=False) == StrictModel(x=1)
    with pytest.raises(ValidationError) as exc_info:
        StrictModel.model_validate({'x': '1'}, strict=True)
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'int_type',
            'loc': ('x',),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'input': '1',
        }
    ]

@pytest.mark.xfail(reason='strict=True in model_validate_json does not overwrite strict=False given in ConfigDictSee issue: https://github.com/pydantic/pydantic/issues/8930')
@pytest.mark.parametrize(
    'kinds',
    [
        {'sub_fields', 'model_fields', 'model_config', 'sub_config', 'combined_config'},
        {'sub_fields', 'model_fields', 'combined_config'},
        {'sub_fields', 'model_fields'},
        {'combined_config'},
        {'model_config', 'sub_config'},
        {'model_config', 'sub_fields'},
        {'model_fields', 'sub_config'},
    ],
)
@pytest.mark.parametrize(
    'exclude,expected,raises_match',
    [
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
            id='exclude with ellipsis'
        ),
    ],
)
def test_model_export_exclusion_with_fields_and_config_kinds(
    kinds: set[str],
    exclude: Optional[dict[str, Any]],
    expected: dict[str, Any],
    raises_match: Optional[str],
) -> None:
    """Test that exporting models with fields using the export parameter works."""
    class ChildConfig:
        pass
    if 'sub_config' in kinds:
        ChildConfig.fields = {'b': {'exclude': ...}, 'a': {'exclude': {1}}}

    class ParentConfig:
        pass
    if 'combined_config' in kinds:
        ParentConfig.fields = {
            'b': {'exclude': ...},
            'c': {'exclude': {'b': ..., 'a': {1}}},
            'd': {'exclude': {'a': ..., 'b': ...}},
        }
    elif 'model_config' in kinds:
        ParentConfig.fields = {'b': {'exclude': ...}, 'd': {'exclude': {'a'}}}

    class Sub(BaseModel):
        a: list[int] = Field(default_factory=lambda: [3, 4, 5], exclude={'a': {1}} if 'sub_fields' in kinds else None)
        b: int = Field(4, exclude=... if 'sub_fields' in kinds else None)
        c: str = 'foobar'
        Config: type = ChildConfig

    class Model(BaseModel):
        a: int = 0
        b: int = Field(2, exclude=... if 'model_fields' in kinds else None)
        c: Sub = Sub()
        d: Sub = Field(Sub(), exclude={'a': ...} if 'model_fields' in kinds else None)
        Config: type = ParentConfig

    m: Model = Model()
    assert m.model_dump(exclude=exclude) == expected, 'Unexpected model export result'

@pytest.mark.skip(reason='not implemented')
def test_model_export_exclusion_inheritance_extended() -> None:
    class Sub(BaseModel):
        s1: str = 'v1'
        s2: str = 'v2'
        s3: str = 'v3'
        s4: str = Field('v4', exclude=...)

    class Parent(BaseModel):
        model_config: ConfigDict = ConfigDict(fields={'a': {'exclude': ...}, 's': {'exclude': {'s1'}}})
        b: int = Field(exclude=...)
        s: Sub = Sub()

    class Child(Parent):
        model_config: ConfigDict = ConfigDict(fields={'c': {'exclude': ...}, 's': {'exclude': {'s2'}}})

    actual: Parent = Child(a=0, b=1, c=2, d=3)
    expected: dict[str, Any] = {'d': 3, 's': {'s3': 'v3'}}
    assert actual.model_dump() == expected, 'Unexpected model export result'

@pytest.mark.skip(reason='not implemented')
def test_model_export_with_true_instead_of_ellipsis_extended() -> None:
    class Sub(BaseModel):
        s1: int

    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(fields={'c': {'exclude': True}})
        a: int = 2
        b: int = Field(3, exclude=True)
        c: int = Field(4)
        s: Sub = Sub()

    m: Model = Model()
    assert m.model_dump(exclude={'s': True}) == {'a': 2}

@pytest.mark.skip(reason='not implemented')
def test_model_export_inclusion_extended() -> None:
    class Sub(BaseModel):
        s1: str = 'v1'
        s2: str = 'v2'
        s3: str = 'v3'
        s4: str = 'v4'

    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(
            fields={
                'a': {'include': {'s2', 's1', 's3'}},
                'b': {'include': {'s1'}},
                'c': {'include': {'s1', 's2'}},
            }
        )
        a: Sub = Sub()
        b: Sub = Field(Sub(), include={'s1'})
        c: Sub = Field(Sub(), include={'s1', 's2'})

    assert Model.model_fields['a'].field_info.include == {'s1': ..., 's2': ..., 's3': ...}
    assert Model.model_fields['b'].field_info.include == {'s1': ...}
    assert Model.model_fields['c'].field_info.include == {'s1': ..., 's2': ...}
    actual: dict[str, Any] = Model().model_dump(include={'a': {'s3', 's4'}, 'b': ..., 'c': ...})
    expected: dict[str, Any] = {'a': {'s3': 'v3'}, 'b': {'s1': 'v1'}, 'c': {'s1': 'v1', 's2': 'v2'}}
    assert actual == expected, 'Unexpected model export result'

def test_untyped_fields_warning_corrected() -> None:
    with pytest.raises(PydanticUserError, match=re.escape(
        "A non-annotated attribute was detected: `x = 1`. "
        "All model fields require a type annotation; if `x` is not meant to be a field, "
        "you may be able to resolve this error by annotating it as a `ClassVar` or updating `model_config['ignored_types']`."
    )):
        class WarningModel(BaseModel):
            x = 1  # Missing type annotation

    class NonWarningModel(BaseModel):
        x: ClassVar[int] = 1

def test_custom_protected_namespace() -> None:
    with pytest.warns(UserWarning, match='Field "test_field" in Model has conflict with protected namespace "test_"'):
        class Model(BaseModel):
            model_config: ConfigDict = ConfigDict(protected_namespaces=('test_',))
            test_field: int

def test_multiple_protected_namespace() -> None:
    with pytest.warns(UserWarning, match='Field "also_protect_field" in Model has conflict with protected namespace "also_protect_"'):
        class Model(BaseModel):
            model_config: ConfigDict = ConfigDict(protected_namespaces=('protect_me_', 'also_protect_'))
            protect_me_field: int
            also_protect_field: int

def test_protected_namespace_pattern() -> None:
    with pytest.warns(UserWarning, match='Field "perfect_match" in Model has conflict with protected namespace .*'):
        class Model(BaseModel):
            model_config: ConfigDict = ConfigDict(protected_namespaces=(re.compile('^perfect_match$'),))
            perfect_match: int

def test_model_get_core_schema() -> None:
    received_schemas: defaultdict[str, list[str]] = defaultdict(list)

    class Marker:
        def __init__(self, name: str) -> None:
            self.name = name

        def __get_pydantic_core_schema__(self, source_type: type, handler: GetCoreSchemaHandler) -> CoreSchema:
            schema: CoreSchema = handler(source_type)
            received_schemas[self.name].append(schema['type'])
            return schema

    class InnerModel(BaseModel):
        @classmethod
        def __get_pydantic_core_schema__(cls, source_type: type, handler: GetCoreSchemaHandler) -> CoreSchema:
            schema: CoreSchema = handler(source_type)
            received_schemas['InnerModel'].append(schema['type'])
            schema.pop('metadata', None)
            schema['metadata'] = {'foo': 'inner was here!'}
            return deepcopy(schema)

    class OuterModel(BaseModel):
        @classmethod
        def __get_pydantic_core_schema__(cls, source_type: type, handler: GetCoreSchemaHandler) -> CoreSchema:
            schema: CoreSchema = handler(source_type)
            received_schemas['OuterModel'].append(schema['type'])
            return schema

    ta: TypeAdapter = TypeAdapter(Annotated[OuterModel, Marker('Marker("outer")')])
    assert 'inner was here' in str(ta.core_schema)
    assert received_schemas == {
        'InnerModel': ['model', 'model'],
        'Marker("outer")': ['definition-ref'],
        'OuterModel': ['model', 'model'],
    }

def test_get_core_schema_return_new_ref() -> None:
    class InnerModel(BaseModel):
        @classmethod
        def __get_pydantic_core_schema__(cls, source_type: type, handler: GetCoreSchemaHandler) -> CoreSchema:
            schema: CoreSchema = handler(source_type)
            schema = deepcopy(schema)
            schema['metadata'] = {'foo': 'inner was here!'}
            return deepcopy(schema)

    class OuterModel(BaseModel):
        x: int
        inner: InnerModel

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type: type, handler: GetCoreSchemaHandler) -> CoreSchema:
            schema: CoreSchema = handler(source_type)

            def set_x(v: OuterModel) -> OuterModel:
                v.x += 1
                return v

            return core_schema.no_info_after_validator_function(set_x, schema, ref=schema.pop('ref'))

    cs: CoreSchema = OuterModel.__pydantic_core_schema__
    assert 'inner was here' in str(cs)
    m: OuterModel = OuterModel(inner=InnerModel())
    assert m.inner.x == 1  # Plus set_x adds 1

def test_resolve_def_schema_from_core_schema() -> None:
    class Inner(BaseModel):
        x: int

    class Marker:
        def __get_pydantic_core_schema__(self, source_type: type, handler: GetCoreSchemaHandler) -> CoreSchema:
            schema: CoreSchema = handler(source_type)
            resolved: CoreSchema = handler.resolve_ref_schema(schema)
            assert resolved['type'] == 'model'
            assert resolved['cls'] is Inner

            def modify_inner(v: Inner) -> Inner:
                v.x += 1
                return v

            return core_schema.no_info_after_validator_function(modify_inner, schema)

    class Outer(BaseModel):
        inner: Inner

    assert Outer.model_validate({'inner': {'x': 1}}).inner.x == 2

def test_extra_validator_scalar() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow')
        a: int

    class Child(Model):
        pass

    m: Child = Child(a=1)
    m.__pydantic_extra__ = {'a': 1}
    assert m.__pydantic_extra__ == {'a': 1}
    assert Child.model_json_schema() == {'additionalProperties': {'type': 'integer'}, 'properties': {'a': {'title': 'A', 'type': 'integer'}}, 'title': 'Child', 'type': 'object'}

def test_extra_validator_field() -> None:
    class Model(BaseModel, extra='allow'):
        __pydantic_extra__: Any = Field(init=False)
        a: int

    m: Model = Model(a=1)
    m.__pydantic_extra__ = {'a': 1}
    assert m.__pydantic_extra__ == {'a': 1}
    with pytest.raises(ValidationError) as exc_info:
        Model(a='a')  # Assuming a should be int
    assert exc_info.value.errors(include_url=False) == [
        {
            'input': 'a',
            'loc': ('a',),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'type': 'int_parsing',
        }
    ]
    assert Model.model_json_schema() == {'additionalProperties': {'type': 'integer'}, 'properties': {'a': {'title': 'A', 'type': 'integer'}}, 'title': 'Model', 'type': 'object'}

def test_extra_validator_named() -> None:
    class Foo(BaseModel):
        x: int

    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow')
        a: Optional[Foo]

        class Config:
            arbitrary_types_allowed = True

    class Child(Model):
        pass

    m: Child = Child(a={'x': '1'}, y=2)
    assert m.__pydantic_extra__ == {'a': Foo(x=1)}
    assert Child.model_json_schema() == {
        '$defs': {
            'Foo': {
                'properties': {'x': {'title': 'X', 'type': 'integer'}},
                'required': ['x'],
                'title': 'Foo',
                'type': 'object',
            }
        },
        'additionalProperties': {'$ref': '#/$defs/Foo'},
        'properties': {'y': {'title': 'Y', 'type': 'integer'}},
        'required': ['y'],
        'title': 'Child',
        'type': 'object',
    }

def test_super_getattr_extra() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow')
        a: float

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
        model_config: ConfigDict = ConfigDict(extra='allow')

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
        ta: TypeAdapter = TypeAdapter(Foo(), config=ConfigDict(arbitrary_types_allowed=True))
    bar = Bar()
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

    class Bar(BaseModel):
        pass
    assert Foo.__pydantic_core_schema__['type'] == ('dataclass' if is_dataclass else 'model')
    assert isinstance(Foo.__pydantic_core_schema__, dict)

def test_help(create_module: Callable[[Any], Any]) -> None:
    module = create_module('\nimport pydoc\n\nfrom pydantic import BaseModel\n\nclass Model(BaseModel):\n    x: int\n\n\nhelp_result_string = pydoc.render_doc(Model)\n')
    assert 'class Model' in module.help_result_string

def test_cannot_use_leading_underscore_field_names() -> None:
    with pytest.raises(NameError, match=(
        "Fields must not use names with leading underscores; "
        "e.g., use 'x' instead of '_x'"
    )):
        class Model1(BaseModel):
            _x: int = Field(alias='x')

    with pytest.raises(NameError, match=(
        "Fields must not use names with leading underscores; "
        "e.g., use 'x__' instead of '__x__'"
    )):
        class Model2(BaseModel):
            __x__: int = Field()

    with pytest.raises(NameError, match=(
        "Fields must not use names with leading underscores; "
        "e.g., use 'my_field' instead of '___'"
    )):
        class Model3(BaseModel):
            ___: int = Field(default=1)

def test_customize_type_constraints_order() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict()
        x: str = Field(min_length=1, max_length=1)
        y: str = Field(min_length=1, max_length=1)

    with pytest.raises(ValidationError) as exc_info:
        Model(x=' 1 ', y=' 1 ')
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'string_too_long',
            'loc': ('y',),
            'msg': 'String should have at most 1 character',
            'input': ' 1 ',
            'ctx': {'max_length': 1},
        }
    ]

def test_shadow_attribute() -> None:
    """https://github.com/pydantic/pydantic/issues/7108"""
    calls: list[str] = []

    class Model(BaseModel):
        @classmethod
        def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
            super().__pydantic_init_subclass__(**kwargs)
            for key in cls.model_fields.keys():
                setattr(cls, key, getattr(cls, key, '') + ' edited!')

    class One(Model):
        foo: str = 'abc'

    with pytest.warns(UserWarning, match=r'"foo" in ".*Two" shadows an attribute in parent ".*One"'):
        class Two(One):
            foo: str

    with pytest.warns(UserWarning, match=r'"foo" in ".*Three" shadows an attribute in parent ".*One"'):
        class Three(One):
            foo: str = 'xyz'

    assert getattr(Model, 'foo', None) is None
    assert getattr(One, 'foo', None) == 'abc edited!'
    assert getattr(Two, 'foo', None) == 'abc edited! edited!'
    assert getattr(Three, 'foo', None) == 'abc edited! edited!'

def test_shadow_attribute_warn_for_redefined_fields() -> None:
    """https://github.com/pydantic/pydantic/issues/9107"""
    class Parent:
        foo: bool = False

    with pytest.warns(UserWarning, match=(
        '"foo" in ".*ChildWithRedefinedField" shadows an attribute in parent ".*Parent"'
    )):
        class ChildWithRedefinedField(BaseModel, Parent):
            foo: bool = True

def test_schema_valid_for_inner_generic() -> None:
    T = TypeVar('T')

    class Inner(BaseModel, Generic[T]):
        pass

    class Outer(BaseModel):
        inner: Inner[int]

    assert Outer(model_validate({'inner': {'x': 1}})).inner.x == 1

def test_validation_works_for_cyclical_forward_refs() -> None:
    class X(BaseModel):
        pass

    class Y(BaseModel):
        x: Optional[X] = None

    X.update_forward_refs()
    Y.update_forward_refs()

    assert Y(model_validate({'x': None})).x is None

def test_model_construct_with_model_post_init_and_model_copy() -> None:
    class Model(BaseModel):
        def model_post_init(self, context: ValidationInfo) -> None:
            super().model_post_init(context)

    m: Model = Model.model_construct(id=1)
    copy: Model = m.model_copy(deep=True)
    assert m == copy
    assert id(m) != id(copy)

def test_subclassing_gen_schema_warns() -> None:
    with pytest.warns(UserWarning, match='Subclassing `GenerateSchema` is not supported.'):
        class MyGenSchema(GenerateSchema):
            pass

def test_nested_v1_model_warns() -> None:
    with pytest.warns(UserWarning, match=(
        'Mixing V1 models and V2 models \\(or constructs, like `TypeAdapter`\\) is not supported. '
        'Please upgrade `V1Model` to V2.'
    )):
        class V1Model(BaseModelV1):
            pass

        class V2Model(BaseModel):
            pass

def test_replace() -> None:
    if sys.version_info < (3, 13):
        pytest.skip("requires python 3.13")
    from copy import replace

    class Model(BaseModel):
        x: int
        y: int

    m: Model = Model(x=1, y=2)
    assert replace(m, x=3) == Model(x=3, y=2)

def test_reuse_same_field_validation() -> None:
    required_field = Field(...)

    class Model1(BaseModel):
        required: int = required_field

    class Model2(BaseModel):
        required: int = required_field

    with pytest.raises(ValidationError):
        Model1.model_validate({})

    with pytest.raises(ValidationError):
        Model2.model_validate({})

def test_subclass_params_custom_init() -> None:
    class DerivedModel(BaseModel):
        def __init_subclass__(cls, something: int, **kwargs: Any) -> None:
            super().__init_subclass__()
            cls.something: int = something

    class NewModel(DerivedModel, something=2):
        pass

    assert NewModel.something == 2

def test_recursive_model_validation() -> None:
    class MyModel(BaseModel):
        field: Optional['MyModel'] = None

    MyModel.update_forward_refs()
    m: MyModel = MyModel(field={'field': {'field': None}})
    assert m.model_dump() == {'field': {'field': {'field': None}}}

def test_recursive_cycle_with_repeated_field_validation() -> None:
    class A(BaseModel):
        b: 'B'

    class B(BaseModel):
        a1: Optional[A] = None
        a2: Optional[A] = None

    A.update_forward_refs()
    with pytest.raises(ValidationError) as exc_info:
        A.model_validate({'b': {'a1': {'b': {'a1': None}}}})
    assert exc_info.value.errors(include_url=False) == [
        {
            'input': {'a1': None},
            'loc': ('b', 'a1', 'b'),
            'msg': 'Field required',
            'type': 'missing',
        }
    ]

def test_two_defaults_multiple_defaults() -> None:
    with pytest.raises(TypeError, match='^cannot specify both default and default_factory$'):
        class Model(BaseModel):
            a: int = Field(default=3, default_factory=lambda: 3)

def test_default_factory_validated_data_arg_global_model() -> None:
    class Parent(BaseModel):
        pass

    class Model(Parent):
        a: int = 1
        b: int = Field(default_factory=lambda data: data['a'])

    model: Model = Model()
    assert model.b == 1
    model = Model.model_validate_json('{"a": 1}')
    assert model.b == 1

def test_model_validate_strings(field_type: type, input_value: str, expected: Union[int, float, date, datetime], raises_match: Optional[str], strict: bool) -> None:
    class Model(BaseModel):
        x: field_type

    if raises_match is not None:
        with pytest.raises(raises_match):
            Model.model_validate_strings({'x': input_value}, strict=strict)
    else:
        assert Model.model_validate_strings({'x': input_value}, strict=strict).x == expected

@pytest.mark.parametrize(
    'field_type,input_value,expected,raises_match,strict',
    [
        (bool, 'true', True, None, False),
        (bool, 'true', True, None, True),
        (bool, 'false', False, None, False),
        (bool, 'e', False, 'type=bool_parsing', False),
        (int, '1', 1, None, False),
        (int, '1', 1, None, True),
        (int, 'xxx', False, 'type=int_parsing', True),
        (float, '1.1', 1.1, None, False),
        (float, '1.10', 1.1, None, False),
        (float, '1.1', 1.1, None, True),
        (float, '1.10', 1.1, None, True),
        (date, '2017-01-01', date(2017, 1, 1), None, False),
        (date, '2017-01-01', date(2017, 1, 1), None, True),
        (date, '2017-01-01T12:13:14.567', False, 'type=date_from_datetime_inexact', False),
        (date, '2017-01-01T12:13:14.567', False, 'type=date_parsing', True),
        (date, '2017-01-01T00:00:00', date(2017, 1, 1), None, False),
        (date, '2017-01-01T00:00:00', False, 'type=date_parsing', True),
        (datetime, '2017-01-01T12:13:14.567', datetime(2017, 1, 1, 12, 13, 14, 567000), None, False),
        (datetime, '2017-01-01T12:13:14.567', datetime(2017, 1, 1, 12, 13, 14, 567000), None, True),
    ],
    ids=repr,
)
def test_model_validate_strings_parametrized(
    field_type: type,
    input_value: str,
    expected: Union[int, float, date, datetime],
    raises_match: Optional[str],
    strict: bool,
) -> None:
    if raises_match is not None:
        with pytest.raises(ValidationError, match=raises_match):
            class Model(BaseModel):
                x: field_type
    else:
        class Model(BaseModel):
            x: field_type
        m: Model = Model.model_validate_strings({'x': input_value}, strict=strict)
        assert m.x == expected

def test_literal_use_enum_values_with_default() -> None:
    class Model(BaseModel):
        baz: str = Field(default=StrFoo.FOO.value)
        model_config: ConfigDict = ConfigDict(use_enum_values=True, validate_default=True)

    validated: Model = Model()
    assert type(validated.baz) is str
    assert type(validated.model_dump()['baz']) is str
    validated = Model.model_validate_json('{"baz": "foo"}')
    assert type(validated.baz) is str
    assert type(validated.model_dump()['baz']) is str
    validated = Model.model_validate({'baz': StrFoo.FOO})
    assert type(validated.baz) is str
    assert type(validated.model_dump()['baz']) is str

def test_arbitrary_type_allowed_validation_success_correct() -> None:
    class ArbitraryTypeAllowedModel(BaseModel):
        t: ArbitraryType
        model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    arbitrary_type_instance: ArbitraryType = ArbitraryType()
    m: ArbitraryTypeAllowedModel = ArbitraryTypeAllowedModel(t=arbitrary_type_instance)
    assert m.t == arbitrary_type_instance

def test_help_functionality(create_module: Callable[[Any], Any]) -> None:
    module = create_module('\nimport pydoc\n\nfrom pydantic import BaseModel\n\nclass Model(BaseModel):\n    x: int\n\n\nhelp_result_string = pydoc.render_doc(Model)\n')
    assert 'class Model' in module.help_result_string

def test_allowed_nested_types_name() -> None:
    class Model(BaseModel):
        pass
    assert Model.model_config['extra'] == 'forbid'
    assert Model.model_config['alias_generator'] is str.upper

def test_disallowed_shadowed_attributes() -> None:
    with pytest.raises(PydanticUserError, match=(
        "A non-annotated attribute was detected: `model_json_schema = 'abc'`"
    )):
        class BadModel(BaseModel):
            model_json_schema: str = 'abc'

    class NonWarningModel(BaseModel):
        model_json_schema: str = 'abc'

def test_allowed_extra_conditions() -> None:
    class Model(BaseModel):
        x: int
        y: int
        model_config: ConfigDict = ConfigDict(extra='allow')

    m: Model = Model(x=1, y=2, z=3)
    assert m.z == 3
    assert m.model_extra == {'z': 3}
