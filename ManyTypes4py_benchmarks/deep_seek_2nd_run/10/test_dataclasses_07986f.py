import dataclasses
import inspect
import pickle
import re
import sys
import traceback
from collections.abc import Hashable
from datetime import date, datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)
from typing_extensions import Annotated

import pytest
from dirty_equals import HasRepr
from pydantic_core import ArgsKwargs, SchemaValidator
import pydantic
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FieldInfo,
    PydanticDeprecatedSince20,
    PydanticUndefinedAnnotation,
    PydanticUserError,
    RootModel,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
    with_config,
)
from pydantic._internal._mock_val_ser import MockValSer
from pydantic.dataclasses import (
    is_pydantic_dataclass,
    rebuild_dataclass,
    dataclass as pydantic_dataclass,
)
from pydantic.fields import Field as PydanticField
from pydantic.json_schema import model_json_schema
from pydantic.typing import InitVar


def test_cannot_create_dataclass_from_basemodel_subclass() -> None:
    msg = 'Cannot create a Pydantic dataclass from SubModel as it is already a Pydantic model'
    with pytest.raises(PydanticUserError, match=msg):

        @pydantic.dataclasses.dataclass
        class SubModel(BaseModel):
            pass


def test_simple() -> None:

    @pydantic.dataclasses.dataclass
    class MyDataclass:
        a: int
        b: float

    d = MyDataclass('1', '2.5')
    assert d.a == 1
    assert d.b == 2.5
    d = MyDataclass(b=10, a=20)
    assert d.a == 20
    assert d.b == 10


def test_model_name() -> None:

    @pydantic.dataclasses.dataclass
    class MyDataClass:
        model_name: str

    d = MyDataClass('foo')
    assert d.model_name == 'foo'
    d = MyDataClass(model_name='foo')
    assert d.model_name == 'foo'


def test_value_error() -> None:

    @pydantic.dataclasses.dataclass
    class MyDataclass:
        a: int
        b: int

    with pytest.raises(ValidationError) as exc_info:
        MyDataclass(1, 'wrong')
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'int_parsing',
            'loc': ('b',),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'input': 'wrong'
        }
    ]


def test_frozen() -> None:

    @pydantic.dataclasses.dataclass(frozen=True)
    class MyDataclass:
        a: int

    d = MyDataclass(1)
    assert d.a == 1
    with pytest.raises(AttributeError):
        d.a = 7


def test_validate_assignment() -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(validate_assignment=True))
    class MyDataclass:
        a: int

    d = MyDataclass(1)
    assert d.a == 1
    d.a = '7'
    assert d.a == 7


def test_validate_assignment_error() -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(validate_assignment=True))
    class MyDataclass:
        a: int

    d = MyDataclass(1)
    with pytest.raises(ValidationError) as exc_info:
        d.a = 'xxx'
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'int_parsing',
            'loc': ('a',),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'input': 'xxx'
        }
    ]


def test_not_validate_assignment() -> None:

    @pydantic.dataclasses.dataclass
    class MyDataclass:
        a: int

    d = MyDataclass(1)
    assert d.a == 1
    d.a = '7'
    assert d.a == '7'


def test_validate_assignment_value_change() -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(validate_assignment=True), frozen=False)
    class MyDataclass:
        a: int

        @field_validator('a')
        @classmethod
        def double_a(cls, v: int) -> int:
            return v * 2

    d = MyDataclass(2)
    assert d.a == 4
    d.a = 3
    assert d.a == 6


@pytest.mark.parametrize('config', [
    ConfigDict(validate_assignment=False),
    ConfigDict(extra=None),
    ConfigDict(extra='forbid'),
    ConfigDict(extra='ignore'),
    ConfigDict(validate_assignment=False, extra=None),
    ConfigDict(validate_assignment=False, extra='forbid'),
    ConfigDict(validate_assignment=False, extra='ignore'),
    ConfigDict(validate_assignment=False, extra='allow'),
    ConfigDict(validate_assignment=True, extra='allow')
])
def test_validate_assignment_extra_unknown_field_assigned_allowed(config: ConfigDict) -> None:

    @pydantic.dataclasses.dataclass(config=config)
    class MyDataclass:
        a: int

    d = MyDataclass(1)
    assert d.a == 1
    d.extra_field = 123  # type: ignore
    assert d.extra_field == 123


@pytest.mark.parametrize('config', [
    ConfigDict(validate_assignment=True),
    ConfigDict(validate_assignment=True, extra=None),
    ConfigDict(validate_assignment=True, extra='forbid'),
    ConfigDict(validate_assignment=True, extra='ignore')
])
def test_validate_assignment_extra_unknown_field_assigned_errors(config: ConfigDict) -> None:

    @pydantic.dataclasses.dataclass(config=config)
    class MyDataclass:
        a: int

    d = MyDataclass(1)
    assert d.a == 1
    with pytest.raises(ValidationError) as exc_info:
        d.extra_field = 1.23  # type: ignore
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'no_such_attribute',
            'loc': ('extra_field',),
            'msg': "Object has no attribute 'extra_field'",
            'input': 1.23,
            'ctx': {'attribute': 'extra_field'}
        }
    ]


def test_post_init() -> None:
    post_init_called = False

    @pydantic.dataclasses.dataclass
    class MyDataclass:
        a: int

        def __post_init__(self) -> None:
            nonlocal post_init_called
            post_init_called = True

    d = MyDataclass('1')
    assert d.a == 1
    assert post_init_called


def test_post_init_validation() -> None:

    @dataclasses.dataclass
    class DC:
        a: int

        def __post_init__(self) -> None:
            self.a *= 2

    assert DC(a='2').a == '22'
    PydanticDC = pydantic.dataclasses.dataclass(DC)
    assert DC(a='2').a == '22'
    assert PydanticDC(a='2').a == 4


def test_convert_vanilla_dc() -> None:

    @dataclasses.dataclass
    class DC:
        a: int
        b: str = dataclasses.field(init=False)

        def __post_init__(self) -> None:
            self.a *= 2
            self.b = 'hello'

    dc1 = DC(a='2')
    assert dc1.a == '22'
    assert dc1.b == 'hello'
    PydanticDC = pydantic.dataclasses.dataclass(DC)
    dc2 = DC(a='2')
    assert dc2.a == '22'
    assert dc2.b == 'hello'
    py_dc = PydanticDC(a='2')
    assert py_dc.a == 4
    assert py_dc.b == 'hello'


def test_std_dataclass_with_parent() -> None:

    @dataclasses.dataclass
    class DCParent:
        a: int
        b: int

    @dataclasses.dataclass
    class DC(DCParent):
        def __post_init__(self) -> None:
            self.a *= 2

    assert dataclasses.asdict(DC(a='2', b='1')) == {'a': '22', 'b': '1'}
    PydanticDC = pydantic.dataclasses.dataclass(DC)
    assert dataclasses.asdict(DC(a='2', b='1')) == {'a': '22', 'b': '1'}
    assert dataclasses.asdict(PydanticDC(a='2', b='1')) == {'a': 4, 'b': 1}


def test_post_init_inheritance_chain() -> None:
    parent_post_init_called = False
    post_init_called = False

    @pydantic.dataclasses.dataclass
    class ParentDataclass:
        a: int
        b: int

        def __post_init__(self) -> None:
            nonlocal parent_post_init_called
            parent_post_init_called = True

    @pydantic.dataclasses.dataclass
    class MyDataclass(ParentDataclass):
        def __post_init__(self) -> None:
            super().__post_init__()
            nonlocal post_init_called
            post_init_called = True

    d = MyDataclass(a=1, b=2)
    assert d.a == 1
    assert d.b == 2
    assert parent_post_init_called
    assert post_init_called


def test_post_init_post_parse() -> None:
    with pytest.warns(PydanticDeprecatedSince20, match='Support for `__post_init_post_parse__` has been dropped'):

        @pydantic.dataclasses.dataclass
        class MyDataclass:
            def __post_init_post_parse__(self) -> None:
                pass


def test_post_init_assignment() -> None:

    @pydantic.dataclasses.dataclass
    class C:
        a: float
        b: float
        c: float = dataclasses.field(init=False)

        def __post_init__(self) -> None:
            self.c = self.a + self.b

    c = C(0.1, 0.2)
    assert c.a == 0.1
    assert c.b == 0.2
    assert c.c == 0.30000000000000004


def test_inheritance() -> None:

    @pydantic.dataclasses.dataclass
    class A:
        a: bytes

    a_ = A(a=b'a')
    assert a_.a == 'a'

    @pydantic.dataclasses.dataclass
    class B(A):
        b: int

    b = B(a='a', b=12)
    assert b.a == 'a'
    assert b.b == 12
    with pytest.raises(ValidationError):
        B(a='a', b='b')
    a_ = A(a=b'a')
    assert a_.a == 'a'


def test_validate_long_string_error() -> None:

    @pydantic.dataclasses.dataclass(config=dict(str_max_length=3))
    class MyDataclass:
        a: str

    with pytest.raises(ValidationError) as exc_info:
        MyDataclass('xxxx')
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'string_too_long',
            'loc': ('a',),
            'msg': 'String should have at most 3 characters',
            'input': 'xxxx',
            'ctx': {'max_length': 3}
        }
    ]


def test_validate_assignment_long_string_error() -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(str_max_length=3, validate_assignment=True))
    class MyDataclass:
        a: str

    d = MyDataclass('xxx')
    with pytest.raises(ValidationError) as exc_info:
        d.a = 'xxxx'
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'string_too_long',
            'loc': ('a',),
            'msg': 'String should have at most 3 characters',
            'input': 'xxxx',
            'ctx': {'max_length': 3}
        }
    ]


def test_no_validate_assignment_long_string_error() -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(str_max_length=3, validate_assignment=False))
    class MyDataclass:
        a: str

    d = MyDataclass('xxx')
    d.a = 'xxxx'
    assert d.a == 'xxxx'


def test_nested_dataclass() -> None:

    @pydantic.dataclasses.dataclass
    class Nested:
        number: int

    @pydantic.dataclasses.dataclass
    class Outer:
        n: Nested

    navbar = Outer(n=Nested(number='1'))
    assert isinstance(navbar.n, Nested)
    assert navbar.n.number == 1
    navbar = Outer(n={'number': '3'})
    assert isinstance(navbar.n, Nested)
    assert navbar.n.number == 3
    with pytest.raises(ValidationError) as exc_info:
        Outer(n='not nested')
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'dataclass_type',
            'loc': ('n',),
            'msg': 'Input should be a dictionary or an instance of Nested',
            'input': 'not nested',
            'ctx': {'class_name': 'Nested'}
        }
    ]
    with pytest.raises(ValidationError) as exc_info:
        Outer(n={'number': 'x'})
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'int_parsing',
            'loc': ('n', 'number'),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'input': 'x'
        }
    ]


def test_arbitrary_types_allowed() -> None:

    class Button:
        def __init__(self, href: str) -> None:
            self.href = href

    @pydantic.dataclasses.dataclass(config=dict(arbitrary_types_allowed=True))
    class Navbar:
        button: Button

    btn = Button(href='a')
    navbar = Navbar(button=btn)
    assert navbar.button.href == 'a'
    with pytest.raises(ValidationError) as exc_info:
        Navbar(button=('b',))
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'is_instance_of',
            'loc': ('button',),
            'msg': 'Input should be an instance of test_arbitrary_types_allowed.<locals>.Button',
            'input': ('b',),
            'ctx': {'class': 'test_arbitrary_types_allowed.<locals>.Button'}
        }
    ]


def test_nested_dataclass_model() -> None:

    @pydantic.dataclasses.dataclass
    class Nested:
        number: int

    class Outer(BaseModel):
        n: Nested

    navbar = Outer(n=Nested(number='1'))
    assert navbar.n.number == 1


def test_fields() -> None:

    @pydantic.dataclasses.dataclass
    class User:
        id: int
        name: str = 'John Doe'
        signup_ts: Optional[datetime] = None

    user = User(id=123)
    fields = user.__pydantic_fields__
    assert fields['id'].is_required() is True
    assert fields['name'].is_required() is False
    assert fields['name'].default == 'John Doe'
    assert fields['signup_ts'].is_required() is False
    assert fields['signup_ts'].default is None


@pytest.mark.parametrize('field_constructor', [dataclasses.field, pydantic.dataclasses.Field])
def test_default_factory_field(field_constructor: Callable) -> None:

    @pydantic.dataclasses.dataclass
    class User:
        id: int
        other: Dict[str, str] = field_constructor(default_factory=lambda: {'John': 'Joey'})

    user = User(id=123)
    assert user.id == 123
    assert user.other == {'John': 'Joey'}
    fields = user.__pydantic_fields__
    assert fields['id'].is_required() is True
    assert repr(fields['id'].default) == 'PydanticUndefined'
    assert fields['other'].is_required() is False
    assert fields['other'].default_factory() == {'John': 'Joey'}


def test_default_factory_singleton_field() -> None:

    class MySingleton:
        pass

    MY_SINGLETON = MySingleton()

    @pydantic.dataclasses.dataclass(config=dict(arbitrary_types_allowed=True))
    class Foo:
        singleton: MySingleton = dataclasses.field(default_factory=lambda: MY_SINGLETON)

    assert Foo().singleton is Foo().singleton


def test_schema() -> None:

    @pydantic.dataclasses.dataclass
    class User:
        id: int
        name: str = 'John Doe'
        ali