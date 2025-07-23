import dataclasses
import inspect
import pickle
import re
import sys
import traceback
from collections.abc import Hashable
from dataclasses import InitVar
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Callable, ClassVar, Generic, Literal, Optional, TypeVar, Union
import pytest
from dirty_equals import HasRepr
from pydantic_core import ArgsKwargs, SchemaValidator
import pydantic
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
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
from pydantic.dataclasses import is_pydantic_dataclass, rebuild_dataclass
from pydantic.fields import Field, FieldInfo
from pydantic.json_schema import model_json_schema


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
            'loc': (1,),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'input': 'wrong',
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
            'input': 'xxx',
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

    @pydantic.dataclasses.dataclass(
        config=ConfigDict(validate_assignment=True), frozen=False
    )
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


@pytest.mark.parametrize(
    'config',
    [
        ConfigDict(validate_assignment=False),
        ConfigDict(extra=None),
        ConfigDict(extra='forbid'),
        ConfigDict(extra='ignore'),
        ConfigDict(validate_assignment=False, extra=None),
        ConfigDict(validate_assignment=False, extra='forbid'),
        ConfigDict(validate_assignment=False, extra='ignore'),
        ConfigDict(validate_assignment=False, extra='allow'),
        ConfigDict(validate_assignment=True, extra='allow'),
    ],
)
def test_validate_assignment_extra_unknown_field_assigned_allowed(
    config: ConfigDict,
) -> None:

    @pydantic.dataclasses.dataclass(config=config)
    class MyDataclass:
        a: int = 1

    d = MyDataclass(1)
    assert d.a == 1
    d.extra_field = 123
    assert d.extra_field == 123


@pytest.mark.parametrize(
    'config',
    [
        ConfigDict(validate_assignment=True),
        ConfigDict(validate_assignment=True, extra=None),
        ConfigDict(validate_assignment=True, extra='forbid'),
        ConfigDict(validate_assignment=True, extra='ignore'),
    ],
)
def test_validate_assignment_extra_unknown_field_assigned_errors(
    config: ConfigDict,
) -> None:

    @pydantic.dataclasses.dataclass(config=config)
    class MyDataclass:
        a: int = 1

    d = MyDataclass(1)
    assert d.a == 1
    with pytest.raises(ValidationError) as exc_info:
        d.extra_field = 1.23
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'no_such_attribute',
            'loc': ('extra_field',),
            'msg': "Object has no attribute 'extra_field'",
            'input': 1.23,
            'ctx': {'attribute': 'extra_field'},
        }
    ]


def test_post_init() -> None:
    post_init_called: bool = False

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
        a: str

        def __post_init__(self) -> None:
            self.a *= 2

    assert DC(a='2').a == '22'
    PydanticDC = pydantic.dataclasses.dataclass(DC)
    assert DC(a='2').a == '22'
    assert PydanticDC(a='2').a == 4


def test_convert_vanilla_dc() -> None:

    @dataclasses.dataclass
    class DC:
        a: str
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
        a: str

    @dataclasses.dataclass
    class DC(DCParent):
        b: str

        def __post_init__(self) -> None:
            self.a *= 2

    assert dataclasses.asdict(DC(a='2', b='1')) == {'a': '22', 'b': '1'}
    PydanticDC = pydantic.dataclasses.dataclass(DC)
    assert dataclasses.asdict(DC(a='2', b='1')) == {'a': '22', 'b': '1'}
    assert dataclasses.asdict(PydanticDC(a='2', b='1')) == {'a': 4, 'b': 1}


def test_post_init_inheritance_chain() -> None:
    parent_post_init_called: bool = False
    post_init_called: bool = False

    @pydantic.dataclasses.dataclass
    class ParentDataclass:
        a: int
        b: int

        def __post_init__(self) -> None:
            nonlocal parent_post_init_called
            parent_post_init_called = True

    @pydantic.dataclasses.dataclass
    class MyDataclass(ParentDataclass):
        c: int

        def __post_init__(self) -> None:
            super().__post_init__()
            nonlocal post_init_called
            post_init_called = True

    d = MyDataclass(a=1, b=2, c=3)
    assert d.a == 1
    assert d.b == 2
    assert parent_post_init_called
    assert post_init_called


def test_post_init_post_parse() -> None:
    with pytest.warns(
        PydanticDeprecatedSince20,
        match='Support for `__post_init_post_parse__` has been dropped',
    ):

        @pydantic.dataclasses.dataclass
        class MyDataclass:
            def __post_init_post_parse__(self) -> None:
                pass


def test_post_init_assignment() -> None:
    from dataclasses import field

    @pydantic.dataclasses.dataclass
    class C:
        a: float
        b: float
        c: float = field(init=False)

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
            'loc': (0,),
            'msg': 'String should have at most 3 characters',
            'input': 'xxxx',
            'ctx': {'max_length': 3},
        }
    ]


def test_validate_assignment_long_string_error() -> None:

    @pydantic.dataclasses.dataclass(
        config=ConfigDict(str_max_length=3, validate_assignment=True)
    )
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
            'ctx': {'max_length': 3},
        }
    ]


def test_no_validate_assignment_long_string_error() -> None:

    @pydantic.dataclasses.dataclass(
        config=ConfigDict(str_max_length=3, validate_assignment=False)
    )
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
            'ctx': {'class_name': 'Nested'},
        }
    ]
    with pytest.raises(ValidationError) as exc_info:
        Outer(n={'number': 'x'})
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'int_parsing',
            'loc': ('n', 'number'),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'input': 'x',
        }
    ]


def test_arbitrary_types_allowed() -> None:

    class Button:
        href: str

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
            'ctx': {'class': 'test_arbitrary_types_allowed.<locals>.Button'},
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
        name: str = 'John Doe'
        signup_ts: Optional[str] = None

    user = User(id=123)
    fields: dict[str, FieldInfo] = user.__pydantic_fields__
    assert fields['id'].is_required() is True
    assert fields['name'].is_required() is False
    assert fields['name'].default == 'John Doe'
    assert fields['signup_ts'].is_required() is False
    assert fields['signup_ts'].default is None


@pytest.mark.parametrize('field_constructor', [dataclasses.field, pydantic.dataclasses.Field])
@pytest.mark.parametrize(
    'extra', ['ignore', 'forbid']
)
def test_default_factory_field(field_constructor: Callable[..., Any]) -> None:

    @pydantic.dataclasses.dataclass
    class User:
        other: dict[str, str] = field_constructor(default_factory=lambda: {'John': 'Joey'})

    user = User(id=123)
    assert user.id == 123
    assert user.other == {'John': 'Joey'}
    fields: dict[str, FieldInfo] = user.__pydantic_fields__
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
        name: str = 'John Doe'
        aliases: dict[str, str] = dataclasses.field(default_factory=lambda: {'John': 'Joey'})
        signup_ts: Optional[str] = None
        age: Optional[int] = dataclasses.field(
            default=None, metadata=dict(title='The age of the user', description='do not lie!')
        )
        height: Optional[int] = pydantic.Field(
            None, title='The height in cm', ge=50, le=300
        )

    User(id=123)
    assert model_json_schema(User) == {
        'properties': {
            'age': {
                'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                'default': None,
                'title': 'The age of the user',
                'description': 'do not lie!',
            },
            'aliases': {
                'additionalProperties': {'type': 'string'},
                'title': 'Aliases',
                'type': 'object',
            },
            'height': {
                'anyOf': [{'maximum': 300, 'minimum': 50, 'type': 'integer'}, {'type': 'null'}],
                'default': None,
                'title': 'The height in cm',
            },
            'id': {'title': 'Id', 'type': 'integer'},
            'name': {'default': 'John Doe', 'title': 'Name', 'type': 'string'},
            'signup_ts': {
                'default': None,
                'format': 'date-time',
                'title': 'Signup Ts',
                'type': 'string',
            },
        },
        'required': ['id'],
        'title': 'User',
        'type': 'object',
    }


def test_nested_schema() -> None:

    @pydantic.dataclasses.dataclass
    class Nested:
        number: int

    @pydantic.dataclasses.dataclass
    class Outer:
        n: Nested

    assert model_json_schema(Outer) == {
        '$defs': {
            'Nested': {
                'properties': {'number': {'title': 'Number', 'type': 'integer'}},
                'required': ['number'],
                'title': 'Nested',
                'type': 'object',
            }
        },
        'properties': {'n': {'$ref': '#/$defs/Nested'}},
        'required': ['n'],
        'title': 'Outer',
        'type': 'object',
    }


def test_initvar() -> None:

    @pydantic.dataclasses.dataclass
    class TestInitVar:
        x: int
        y: InitVar[int]

        def __post_init__(self, y: int) -> None:
            self.x = y

    tiv = TestInitVar(1, 2)
    assert tiv.x == 1
    with pytest.raises(AttributeError):
        tiv.y


def test_derived_field_from_initvar() -> None:

    @pydantic.dataclasses.dataclass
    class DerivedWithInitVar:
        a: int
        plusone: int = dataclasses.field(init=False)

        def __post_init__(self, number: int) -> None:
            self.plusone = number + 1

    derived = DerivedWithInitVar('1')
    assert derived.plusone == 2
    with pytest.raises(ValidationError, match='Input should be a valid integer, unable to parse string as an integer'):
        DerivedWithInitVar('Not A Number')


def test_initvars_post_init() -> None:

    @pydantic.dataclasses.dataclass
    class PathDataPostInit:
        path: Path
        base_path: Optional[Path] = None

        def __post_init__(self, base_path: Optional[str]) -> None:
            if base_path is not None:
                self.path = Path(base_path) / self.path

    path_data = PathDataPostInit('world')
    assert 'path' in path_data.__dict__
    assert 'base_path' not in path_data.__dict__
    assert path_data.path == Path('world')
    p = PathDataPostInit('world', base_path='/hello')
    assert p.path == Path('/hello/world')


def test_classvar() -> None:

    @pydantic.dataclasses.dataclass
    class TestClassVar:
        a: int
        klassvar: ClassVar[str] = "I'm a Class variable"

    tcv = TestClassVar(2)
    assert tcv.klassvar == "I'm a Class variable"


def test_frozenset_field() -> None:

    @pydantic.dataclasses.dataclass
    class TestFrozenSet:
        set: frozenset[int]

    test_set = frozenset({1, 2, 3})
    object_under_test = TestFrozenSet(set=test_set)
    assert object_under_test.set == test_set


def test_inheritance_post_init() -> None:
    post_init_called: bool = False

    @pydantic.dataclasses.dataclass
    class Base:
        a: int
        b: int

        def __post_init__(self) -> None:
            nonlocal post_init_called
            post_init_called = True

    @pydantic.dataclasses.dataclass
    class Child(Base):
        pass

    Child(a=1, b=2)
    assert post_init_called


def test_hashable_required() -> None:

    @pydantic.dataclasses.dataclass
    class MyDataclass:
        v: Hashable

    MyDataclass(v=None)
    with pytest.raises(ValidationError) as exc_info:
        MyDataclass(v=[])
    assert exc_info.value.errors(include_url=False) == [
        {
            'input': [],
            'loc': ('v',),
            'msg': 'Input should be hashable',
            'type': 'is_hashable',
        }
    ]
    with pytest.raises(ValidationError) as exc_info:
        MyDataclass()
    assert exc_info.value.errors(include_url=False) == [
        {
            'input': HasRepr('ArgsKwargs(())'),
            'loc': ('v',),
            'msg': 'Field required',
            'type': 'missing',
        }
    ]


@pytest.mark.parametrize('default', [1, None])
def test_default_value(default: int | None) -> None:

    @pydantic.dataclasses.dataclass
    class MyDataclass:
        v: Optional[int] = default

    assert dataclasses.asdict(MyDataclass()) == {'v': default}
    assert dataclasses.asdict(MyDataclass(v=42)) == {'v': 42}


def test_default_value_ellipsis() -> None:
    """
    https://github.com/pydantic/pydantic/issues/5488
    """

    @pydantic.dataclasses.dataclass
    class MyDataclass:
        v: int = ...

    assert dataclasses.asdict(MyDataclass(v=42)) == {'v': 42}
    with pytest.raises(ValidationError, match='type=missing'):
        MyDataclass()


def test_override_builtin_dataclass() -> None:

    @dataclasses.dataclass
    class File:
        hash: str
        name: bytes
        size: str

    ValidFile = pydantic.dataclasses.dataclass(File)

    file = File(hash='xxx', name=b'whatever.txt', size='456')
    valid_file = ValidFile(hash='xxx', name=b'whatever.txt', size='456')
    assert file.name == b'whatever.txt'
    assert file.size == '456'
    assert valid_file.name == 'whatever.txt'
    assert valid_file.size == 456
    assert isinstance(valid_file, File)
    assert isinstance(valid_file, ValidFile)
    with pytest.raises(ValidationError) as e:
        ValidFile(hash=[1], name='name', size=3)
    assert e.value.errors(include_url=False) == [
        {
            'type': 'string_type',
            'loc': ('hash',),
            'msg': 'Input should be a valid string',
            'input': [1],
        }
    ]


def test_override_builtin_dataclass_2() -> None:

    @dataclasses.dataclass
    class Meta:
        modified_date: Optional[datetime] = None
        seen_count: int = 0

    Meta(modified_date='not-validated', seen_count=0)

    @pydantic.dataclasses.dataclass
    @dataclasses.dataclass
    class File(Meta):
        filename: bytes

    Meta(modified_date='still-not-validated', seen_count=0)
    f = File(filename=b'thefilename', modified_date='2020-01-01T00:00', seen_count='7')
    assert f.filename == 'thefilename'
    assert f.modified_date == datetime(2020, 1, 1, 0, 0)
    assert f.seen_count == 7

    class Foo(BaseModel):
        file: File

    foo = Foo.model_validate({'file': {'filename': b'thefilename', 'meta': {'modified_date': '2020-01-01T00:00', 'seen_count': '7'}}})
    assert foo.file.filename == 'thefilename'
    assert foo.file.modified_date == datetime(2020, 1, 1, 0, 0)
    assert foo.file.seen_count == 7


def test_override_builtin_dataclass_nested_schema() -> None:

    @dataclasses.dataclass
    class Meta:
        modified_date: Optional[datetime] = None
        seen_count: int

    @dataclasses.dataclass
    class File:
        filename: bytes
        meta: Meta

    FileChecked = pydantic.dataclasses.dataclass(File)
    assert model_json_schema(FileChecked) == {
        '$defs': {
            'Meta': {
                'properties': {
                    'modified_date': {'anyOf': [{'format': 'date-time', 'type': 'string'}, {'type': 'null'}], 'title': 'Modified Date'},
                    'seen_count': {'title': 'Seen Count', 'type': 'integer'},
                },
                'required': ['modified_date', 'seen_count'],
                'title': 'Meta',
                'type': 'object',
            }
        },
        'properties': {
            'filename': {'title': 'Filename', 'type': 'string'},
            'meta': {'$ref': '#/$defs/Meta'},
        },
        'required': ['filename', 'meta'],
        'title': 'File',
        'type': 'object',
    }


def test_inherit_builtin_dataclass() -> None:

    @dataclasses.dataclass
    class Z:
        z: int

    @dataclasses.dataclass
    class Y(Z):
        y: int

    @pydantic.dataclasses.dataclass
    class X(Y):
        x: int

    pika = X(x='2', y='4', z='3')
    assert pika.x == 2
    assert pika.y == 4
    assert pika.z == 3


def test_forward_stdlib_dataclass_params() -> None:

    @dataclasses.dataclass(frozen=True)
    class Item:
        name: bytes

    class Example(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

    e = Example(item=Item(name='pika'))
    e.other = 'bulbi2'
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.item.name = 'pika2'


def test_pydantic_callable_field() -> None:
    """pydantic callable fields behaviour should be the same as stdlib dataclass"""

    def foo(arg1: Any, arg2: Any) -> tuple[Any, Any]:
        return (arg1, arg2)

    def bar(x: int, y: int, z: str) -> bool:
        return str(x + y) == z

    class PydanticModel(BaseModel):
        default_callable: Callable[[Any, Any], tuple[Any, Any]] = foo
        default_callable_2: Callable[[int, int, str], bool] = bar

    @pydantic.dataclasses.dataclass
    class PydanticDataclass:
        default_callable: Callable[[Any, Any], tuple[Any, Any]] = foo
        default_callable_2: Callable[[int, int, str], bool] = bar

    @dataclasses.dataclass
    class StdlibDataclass:
        default_callable: Callable[[Any, Any], tuple[Any, Any]] = foo
        default_callable_2: Callable[[int, int, str], bool] = bar

    pyd_m = PydanticModel(required_callable=foo, required_callable_2=bar)
    pyd_dc = PydanticDataclass(required_callable=foo, required_callable_2=bar)
    std_dc = StdlibDataclass(required_callable=foo, required_callable_2=bar)
    assert (
        pyd_m.required_callable is pyd_m.default_callable is pyd_dc.required_callable is pyd_dc.default_callable is std_dc.required_callable is std_dc.default_callable
    )
    assert (
        pyd_m.required_callable_2 is pyd_m.default_callable_2 is pyd_dc.required_callable_2 is pyd_dc.default_callable_2 is std_dc.required_callable_2 is std_dc.default_callable_2
    )


def test_pickle_overridden_builtin_dataclass(create_module: Callable[..., Any]) -> None:
    module = create_module(
        'import dataclasses\nimport pydantic\n\n\n@pydantic.dataclasses.dataclass(config=pydantic.config.ConfigDict(validate_assignment=True))\nclass BuiltInDataclassForPickle:\n    value: int\n        '
    )
    obj = module.BuiltInDataclassForPickle(value=5)
    pickled_obj = pickle.dumps(obj)
    restored_obj = pickle.loads(pickled_obj)
    assert restored_obj.value == 5
    assert restored_obj == obj
    with pytest.raises(ValidationError):
        restored_obj.value = 'value of a wrong type'


def lazy_cases_for_dataclass_equality_checks():
    """
    The reason for the convoluted structure of this function is to avoid
    creating the classes while collecting tests, which may trigger breakpoints
    etc. while working on one specific test.
    """
    cases: list[tuple[Any, Any]] = []

    def get_cases() -> list[tuple[Any, Any]]:
        if cases:
            return cases

        @dataclasses.dataclass(frozen=True)
        class StdLibFoo:
            a: str
            b: int

        @pydantic.dataclasses.dataclass(frozen=True)
        class PydanticFoo:
            a: str
            b: int

        @dataclasses.dataclass(frozen=True)
        class StdLibBar:
            c: Any

        @pydantic.dataclasses.dataclass(frozen=True)
        class PydanticBar:
            c: Any

        @dataclasses.dataclass(frozen=True)
        class StdLibBaz:
            c: Any

        @pydantic.dataclasses.dataclass(frozen=True)
        class PydanticBaz:
            c: Any

        foo = StdLibFoo(a='Foo', b=1)
        cases.append((foo, StdLibBar(c=foo)))

        foo = PydanticFoo(a='Foo', b=1)
        cases.append((foo, PydanticBar(c=foo)))

        foo = PydanticFoo(a='Foo', b=1)
        cases.append((foo, StdLibBaz(c=foo)))

        foo = StdLibFoo(a='Foo', b=1)
        cases.append((foo, PydanticBaz(c=foo)))

        return cases

    case_ids = ['stdlib_stdlib', 'pydantic_pydantic', 'pydantic_stdlib', 'stdlib_pydantic']

    def case(i: int) -> Callable[[], tuple[Any, Any]]:

        def get_foo_bar() -> tuple[Any, Any]:
            return get_cases()[i]

        get_foo_bar.__name__ = case_ids[i]
        return get_foo_bar

    return [case(i) for i in range(4)]


@pytest.mark.parametrize(
    'foo_bar_getter',
    lazy_cases_for_dataclass_equality_checks(),
)
def test_dataclass_equality_for_field_values(
    foo_bar_getter: Callable[[], tuple[Any, Any]]
) -> None:
    foo, bar = foo_bar_getter()
    assert dataclasses.asdict(foo) == dataclasses.asdict(bar.c)
    assert dataclasses.astuple(foo) == dataclasses.astuple(bar.c)
    assert foo == bar.c


def test_issue_2383() -> None:

    @dataclasses.dataclass
    class A:
        a: str

        def __hash__(self) -> int:
            return 123

    class B(pydantic.BaseModel):
        a: A

    a = A('')

    b = B(a=a)
    assert hash(a) == 123
    assert hash(b.a) == 123


def test_issue_2398() -> None:

    @dataclasses.dataclass(order=True)
    class DC:
        num: int = 42

    class Model(pydantic.BaseModel):
        dc: DC

    real_dc = DC()
    model = Model(dc=real_dc)
    assert real_dc <= real_dc
    assert model.dc <= model.dc
    assert real_dc <= model.dc


def test_issue_2424() -> None:

    @dataclasses.dataclass
    class Base:
        pass

    @dataclasses.dataclass
    class Thing(Base):
        y: str = dataclasses.field(default_factory=str)

    assert Thing(x='hi').y == ''

    @pydantic.dataclasses.dataclass
    class ValidatedThing(Base):
        y: str = dataclasses.field(default_factory=str)

    assert Thing(x='hi').y == ''
    assert ValidatedThing(x='hi').y == ''


def test_issue_2541() -> None:

    @dataclasses.dataclass(frozen=True)
    class Infos:
        id: int

    @dataclasses.dataclass(frozen=True)
    class Item:
        name: str
        infos: Infos

    class Example(BaseModel):
        item: Item

    e = Example.model_validate({'item': {'name': '123', 'infos': {'id': '1'}}})
    assert e.item.name == '123'
    assert e.item.infos.id == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.item.infos.id = 2


def test_complex_nested_vanilla_dataclass() -> None:

    @dataclasses.dataclass
    class Span:
        first: int
        last: int
        label: str

    @dataclasses.dataclass
    class LabeledSpan(Span):
        pass

    @dataclasses.dataclass
    class BinaryRelation:
        subject: LabeledSpan
        object: LabeledSpan
        label: str

    @dataclasses.dataclass
    class Sentence:
        relations: list[BinaryRelation]

    class M(pydantic.BaseModel):
        s: Sentence

    assert M.model_json_schema() == {
        '$defs': {
            'BinaryRelation': {
                'properties': {
                    'label': {'title': 'Label', 'type': 'string'},
                    'object': {'$ref': '#/$defs/LabeledSpan'},
                    'subject': {'$ref': '#/$defs/LabeledSpan'},
                },
                'required': ['subject', 'object', 'label'],
                'title': 'BinaryRelation',
                'type': 'object',
            },
            'LabeledSpan': {
                'properties': {
                    'first': {'title': 'First', 'type': 'integer'},
                    'last': {'title': 'Last', 'type': 'integer'},
                    'label': {'title': 'Label', 'type': 'string'},
                },
                'required': ['first', 'last', 'label'],
                'title': 'LabeledSpan',
                'type': 'object',
            },
            'Sentence': {
                'properties': {'relations': {'type': 'array', 'items': {'$ref': '#/$defs/BinaryRelation'}}},
                'required': ['relations'],
                'title': 'Sentence',
                'type': 'object',
            },
        },
        'properties': {'s': {'$ref': '#/$defs/Sentence'}},
        'required': ['s'],
        'title': 'M',
        'type': 'object',
    }


def test_json_schema_with_computed_field() -> None:

    @dataclasses.dataclass
    class MyDataclass:
        x: int

        @computed_field
        @property
        def double_x(self) -> int:
            return 2 * self.x

    class Model(BaseModel):
        dc: MyDataclass

    assert Model.model_json_schema(mode='validation') == {
        '$defs': {
            'MyDataclass': {
                'properties': {'x': {'title': 'X', 'type': 'integer'}},
                'required': ['x'],
                'title': 'MyDataclass',
                'type': 'object',
            }
        },
        'properties': {'dc': {'$ref': '#/$defs/MyDataclass'}},
        'required': ['dc'],
        'title': 'Model',
        'type': 'object',
    }
    assert Model.model_json_schema(mode='serialization') == {
        '$defs': {
            'MyDataclass': {
                'properties': {
                    'double_x': {'readOnly': True, 'title': 'Double X', 'type': 'integer'},
                    'x': {'title': 'X', 'type': 'integer'},
                },
                'required': ['x', 'double_x'],
                'title': 'MyDataclass',
                'type': 'object',
            }
        },
        'properties': {'dc': {'$ref': '#/$defs/MyDataclass'}},
        'required': ['dc'],
        'title': 'Model',
        'type': 'object',
    }


def test_issue_2594() -> None:

    @dataclasses.dataclass
    class Empty:
        pass

    @pydantic.dataclasses.dataclass
    class M:
        e: Empty

    assert isinstance(M(e=Empty()).e, Empty)


def test_schema_description_unset() -> None:

    @pydantic.dataclasses.dataclass
    class A:
        pass

    assert 'description' not in model_json_schema(A)

    @pydantic.dataclasses.dataclass
    @dataclasses.dataclass
    class B:
        pass

    assert 'description' not in model_json_schema(B)


def test_schema_description_set() -> None:

    @pydantic.dataclasses.dataclass
    class A:
        """my description"""

    assert model_json_schema(A)['description'] == 'my description'

    @pydantic.dataclasses.dataclass
    @dataclasses.dataclass
    class B:
        """my description"""

    assert model_json_schema(A)['description'] == 'my description'


def test_issue_3011() -> None:
    """Validation of a subclass of a dataclass"""

    @dataclasses.dataclass
    class A:
        thing_a: str

    class B(pydantic.BaseModel):
        pass

    @pydantic.dataclasses.dataclass
    class C:
        thing: B

    b = B()
    c = C(thing=b)
    assert c.thing.thing_a == 'Thing A'


def test_issue_3162() -> None:

    @dataclasses.dataclass
    class User:
        id: int
        name: str

    class Users(BaseModel):
        user: User
        other_user: User

    assert Users.model_json_schema() == {
        '$defs': {
            'User': {
                'properties': {
                    'id': {'title': 'Id', 'type': 'integer'},
                    'name': {'title': 'Name', 'type': 'string'},
                },
                'required': ['id', 'name'],
                'title': 'User',
                'type': 'object',
            }
        },
        'properties': {
            'other_user': {'$ref': '#/$defs/User'},
            'user': {'$ref': '#/$defs/User'},
        },
        'required': ['user', 'other_user'],
        'title': 'Users',
        'type': 'object',
    }


def test_discriminated_union_basemodel_instance_value() -> None:

    @pydantic.dataclasses.dataclass
    class A:
        l: Literal['a']

    @pydantic.dataclasses.dataclass
    class B:
        l: Literal['b']

    @pydantic.dataclasses.dataclass
    class Top:
        sub: Union[A, B]

    t = Top(sub=A(l='a'))
    assert isinstance(t, Top)
    assert model_json_schema(Top) == {
        'title': 'Top',
        'type': 'object',
        'properties': {
            'sub': {
                'title': 'Sub',
                'discriminator': {
                    'mapping': {'a': '#/$defs/A', 'b': '#/$defs/B'},
                    'propertyName': 'l',
                },
                'oneOf': [{'$ref': '#/$defs/A'}, {'$ref': '#/$defs/B'}],
            }
        },
        'required': ['sub'],
        '$defs': {
            'A': {
                'properties': {'l': {'const': 'a', 'title': 'L', 'type': 'string'}},
                'required': ['l'],
                'title': 'A',
                'type': 'object',
            },
            'B': {
                'properties': {'l': {'const': 'b', 'title': 'L', 'type': 'string'}},
                'required': ['l'],
                'title': 'B',
                'type': 'object',
            },
        },
    }


def test_post_init_after_validation() -> None:

    @dataclasses.dataclass
    class SetWrapper:
        set: set[int]

        def __post_init__(self) -> None:
            assert isinstance(
                self.set, set
            ), f"self.set should be a set but it's {self.set!r} of type {type(self.set).__name__}"

    class Model(pydantic.BaseModel):
        set_wrapper: SetWrapper

    model = Model(set_wrapper=SetWrapper({1, 2, 3}))
    json_text = model.model_dump_json()
    assert Model.model_validate_json(json_text).model_dump() == model.model_dump()


def test_new_not_called() -> None:
    """
    pydantic dataclasses do not preserve sunder attributes set in __new__
    """

    class StandardClass:
        """Class which modifies instance creation."""

        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            instance = super().__new__(cls)
            instance._special_property = 1
            return instance

    @dataclasses.dataclass
    class StandardLibDataclass(StandardClass):
        a: str

    PydanticDataclass = pydantic.dataclasses.dataclass(StandardClass)

    test_string = 'string'
    std_instance = StandardLibDataclass(a=test_string)
    assert std_instance._special_property == 1
    assert std_instance.a == test_string

    @pydantic.dataclasses.dataclass
    class PydanticDataclassModel(StandardClass):
        a: str

    pyd_instance = PydanticDataclassModel(a=test_string)
    assert not hasattr(pyd_instance, '_special_property')
    assert pyd_instance.a == test_string


def test_ignore_extra() -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra='ignore'))
    class Foo:
        a: int = 1

    foo = Foo(**{'x': '1', 'y': '2'})
    assert foo.__dict__ == {'a': 1, 'x': 1}


def test_ignore_extra_subclass() -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra='ignore'))
    class Foo:
        a: int = 1

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra='ignore'))
    class Bar(Foo):
        b: int = 2

    bar = Bar(**{'x': '1', 'y': '2', 'z': '3'})
    assert bar.__dict__ == {'a': 1, 'b': 2, 'x': 1, 'y': 2}


def test_allow_extra() -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra='allow'))
    class Foo:
        a: int = 1

    foo = Foo(**{'x': '1', 'y': '2'})
    assert foo.__dict__ == {'a': 1, 'x': '1', 'y': '2'}


def test_allow_extra_subclass() -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra='allow'))
    class Foo:
        a: int = 1

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra='allow'))
    class Bar(Foo):
        b: int = 2

    bar = Bar(**{'x': '1', 'y': '2', 'z': '3'})
    assert bar.__dict__ == {'a': 1, 'b': 2, 'x': '1', 'y': '2', 'z': '3'}


def test_forbid_extra() -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra='forbid'))
    class Foo:
        a: int = 1

    msg = re.escape(
        "Unexpected keyword argument [type=unexpected_keyword_argument, input_value='2', input_type=str]"
    )
    with pytest.raises(ValidationError, match=msg):
        Foo(**{'x': '1', 'y': '2'})


def test_self_reference_dataclass() -> None:

    @pydantic.dataclasses.dataclass
    class MyDataclass:
        self_reference: Optional['MyDataclass'] = None

    assert MyDataclass.__pydantic_fields__['self_reference'].annotation == Optional[
        MyDataclass
    ]
    instance = MyDataclass(self_reference=MyDataclass(self_reference=MyDataclass()))
    assert TypeAdapter(MyDataclass).dump_python(instance) == {
        'self_reference': {'self_reference': {'self_reference': None}}
    }
    with pytest.raises(ValidationError) as exc_info:
        MyDataclass(self_reference=1)
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'dataclass_type',
            'loc': ('self_reference',),
            'msg': 'Input should be a dictionary or an instance of MyDataclass',
            'input': 1,
            'ctx': {'class_name': 'MyDataclass'},
        }
    ]


def test_cyclic_reference_dataclass(create_module: Callable[..., Any]) -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra='forbid'))
    class D1:
        d2: Optional['D2'] = None

    @create_module
    def module():
        from typing import Optional
        import pydantic

        @pydantic.dataclasses.dataclass(config=pydantic.ConfigDict(extra='forbid'))
        class D2:
            d1: Optional[D1] = None

    D2 = module.D2

    assert isinstance(D1.__pydantic_validator__, MockValSer)
    assert isinstance(D2.__pydantic_validator__, MockValSer)

    instance = D1(d2=D2(d1=D1(d2=D2(d1=D1()))))
    assert isinstance(D1.__pydantic_validator__, SchemaValidator)
    assert isinstance(D2.__pydantic_validator__, SchemaValidator)
    assert TypeAdapter(D1).dump_python(instance) == {'d2': {'d1': {'d2': {'d1': {'d2': None}}}}}
    with pytest.raises(ValidationError) as exc_info:
        D2(d1=D2())
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'dataclass_type',
            'loc': ('d1',),
            'msg': 'Input should be a dictionary or an instance of D1',
            'input': D2(d1=None),
            'ctx': {'class_name': 'D1'},
        }
    ]
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(D1).validate_python(
            {'d2': {'d1': {'d2': {'d2': {'d1': None}}}}}
        )
    assert exc_info.value.errors(include_url=False) == [
        {
            'input': {},
            'loc': ('d2', 'd1', 'd2', 'd2'),
            'msg': 'Unexpected keyword argument',
            'type': 'unexpected_keyword_argument',
        }
    ]


def test_cross_module_cyclic_reference_dataclass(create_module: Callable[..., Any]) -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra='forbid'))
    class D1:
        d2: Optional['D2'] = None

    @create_module
    def module():
        from typing import Optional
        import pydantic

        @pydantic.dataclasses.dataclass(config=pydantic.ConfigDict(extra='forbid'))
        class D2:
            d1: Optional[D1] = None

    with pytest.raises(
        PydanticUserError,
        match=re.escape(
            '`D1` is not fully defined; you should define `D2`, then call `pydantic.dataclasses.rebuild_dataclass(D1)`.'
        ),
    ):
        D1()

    rebuild_dataclass(D1, _types_namespace={'D2': module.D2, 'D1': D1})
    assert isinstance(module.D2.__pydantic_validator__, SchemaValidator)
    instance = D1(d2=module.D2(d1=D1(d2=module.D2(d1=D1()))))
    assert isinstance(module.D2.__pydantic_validator__, SchemaValidator)
    assert TypeAdapter(D1).dump_python(instance) == {'d2': {'d1': {'d2': {'d1': {'d2': None}}}}}
    with pytest.raises(ValidationError) as exc_info:
        module.D2(d1=module.D2())
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'dataclass_type',
            'loc': ('d1',),
            'msg': 'Input should be a dictionary or an instance of D1',
            'input': module.D2(d1=None),
            'ctx': {'class_name': 'D1'},
        }
    ]
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(D1).validate_python(
            {'d2': {'d1': {'d2': {'d2': {'d1': None}}}}}
        )
    assert exc_info.value.errors(include_url=False) == [
        {
            'input': {},
            'loc': ('d2', 'd1', 'd2', 'd2'),
            'msg': 'Unexpected keyword argument',
            'type': 'unexpected_keyword_argument',
        }
    ]


@pytest.mark.parametrize(
    'dataclass_decorator',
    [pydantic.dataclasses.dataclass, dataclasses.dataclass],
    ids=['pydantic', 'stdlib'],
)
def test_base_dataclasses_annotations_resolving(
    create_module: Callable[..., Any],
    dataclass_decorator: Callable[..., Any],
) -> None:

    @create_module
    def module():
        import dataclasses
        from typing import NewType

        OddInt = NewType('OddInt', int)

        @dataclasses.dataclass
        class D1:
            __pydantic_config__: ConfigDict = ConfigDict(str_to_lower=True)
            d1: int
            s: str

    @dataclass_decorator
    class D2(module.D1):
        d2: int
        s: str

    assert TypeAdapter(D2).validate_python({'d1': 1, 'd2': 2, 's': 'ABC'}) == D2(d1=1, d2=2, s='abc')


@pytest.mark.parametrize(
    'dataclass_decorator',
    [pydantic.dataclasses.dataclass, dataclasses.dataclass],
    ids=['pydantic', 'stdlib'],
)
def test_base_dataclasses_annotations_resolving_with_override(
    create_module: Callable[..., Any],
    dataclass_decorator: Callable[..., Any],
) -> None:

    @create_module
    def module1():
        import dataclasses
        from typing import NewType

        IDType = NewType('IDType', int)

        @dataclasses.dataclass
        class D1:
            __pydantic_config__: ConfigDict = ConfigDict(str_to_lower=True)
            db_id: IDType
            s: str

    @create_module
    def module2():
        import dataclasses
        from typing import NewType

        IDType = NewType('IDType', str)

        @dataclasses.dataclass
        class D2:
            __pydantic_config__: ConfigDict = ConfigDict(str_to_lower=False)
            db_id: IDType
            s: str

    @dataclass_decorator
    class D3(module1.D1, module2.D2):
        db_id: int
        s: str

    assert TypeAdapter(D3).validate_python({'db_id': 42, 's': 'ABC'}) == D3(db_id=42, s='abc')


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason='kw_only is not available in python < 3.10',
)
def test_kw_only() -> None:

    @pydantic.dataclasses.dataclass(kw_only=True)
    class A:
        a: int
        y: str = 'default'

    with pytest.raises(ValidationError):
        A(1, '')

    a = A(b='hi')
    assert a.b == 'hi'


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason='kw_only is not available in python < 3.10',
)
def test_kw_only_subclass() -> None:

    @pydantic.dataclasses.dataclass
    class A:
        x: int
        y: int = pydantic.Field(default=0, kw_only=True)

    @pydantic.dataclasses.dataclass
    class B(A):
        z: int

    assert B(1, 2, z=3) == B(x=1, y=2, z=3)
    assert B(1, z=2) == B(x=1, y=0, z=2)


@pytest.mark.parametrize('field_constructor', [pydantic.dataclasses.Field, dataclasses.field])
def test_repr_false(field_constructor: Callable[..., Any]) -> None:

    @pydantic.dataclasses.dataclass
    class A:
        visible_field: str
        hidden_field: str = field_constructor(repr=False)

    instance = A(visible_field='this_should_be_included', hidden_field='this_should_not_be_included')
    assert "visible_field='this_should_be_included'" in repr(instance)
    assert "hidden_field='this_should_not_be_included'" not in repr(instance)


def dataclass_decorators(
    include_identity: bool = False,
    exclude_combined: bool = False,
) -> dict[str, Any]:
    decorators: list[Callable[..., Any]] = [pydantic.dataclasses.dataclass, dataclasses.dataclass]
    ids: list[str] = ['pydantic', 'stdlib']
    if not exclude_combined:

        def combined_decorator(cls: Any) -> Any:
            """
            Should be equivalent to:
            @pydantic.dataclasses.dataclass
            @dataclasses.dataclass
            """
            return pydantic.dataclasses.dataclass(dataclasses.dataclass(cls))

        decorators.append(combined_decorator)
        ids.append('combined')
    if include_identity:

        def identity_decorator(cls: Any) -> Any:
            return cls

        decorators.append(identity_decorator)
        ids.append('identity')
    return {'argvalues': decorators, 'ids': ids}


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason='kw_only is not available in python < 3.10',
)
@pytest.mark.parametrize(
    'decorator1',
    **dataclass_decorators(exclude_combined=True),
)
@pytest.mark.parametrize(
    'decorator2',
    **dataclass_decorators(exclude_combined=True),
)
def test_kw_only_inheritance(decorator1: Callable[..., Any], decorator2: Callable[..., Any]) -> None:

    @decorator1(kw_only=True)
    class Parent:
        a: int
        y: int

    @decorator2
    class Child(Parent):
        b: int
        z: int

    child = Child(1, x=2)
    assert child.x == 2
    assert child.y == 1


@pytest.mark.parametrize('field_constructor', [dataclasses.field, pydantic.dataclasses.Field])
@pytest.mark.parametrize('extra', ['ignore', 'forbid'])
def test_init_false_not_in_signature(extra: str, field_constructor: Callable[..., Any]) -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra=extra))
    class MyDataclass:
        a: int
        b: int = field_constructor(init=False, default=-1)

    signature = inspect.signature(MyDataclass)
    assert 'a' in signature.parameters
    assert 'b' not in signature.parameters


init_test_cases = [
    ({'a': 2, 'b': -1}, 'ignore', {'a': 2, 'b': 1}),
    ({'a': 2}, 'ignore', {'a': 2, 'b': 1}),
    (
        {'a': 2, 'b': -1},
        'forbid',
        [
            {
                'type': 'unexpected_keyword_argument',
                'loc': ('b',),
                'msg': 'Unexpected keyword argument',
                'input': -1,
            }
        ],
    ),
    ({'a': 2}, 'forbid', {'a': 2, 'b': 1}),
]


@pytest.mark.parametrize(
    'field_constructor',
    [dataclasses.field, pydantic.dataclasses.Field],
)
@pytest.mark.parametrize(
    'input_data,extra,expected',
    init_test_cases,
)
def test_init_false_with_post_init(
    input_data: dict[str, Any],
    extra: str,
    expected: Union[dict[str, Any], list[dict[str, Any]]],
    field_constructor: Callable[..., Any],
) -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra=extra))
    class MyDataclass:
        a: int
        b: int = field_constructor(init=False)

        def __post_init__(self) -> None:
            self.b = 1

    if isinstance(expected, list):
        with pytest.raises(ValidationError) as exc_info:
            MyDataclass(**input_data)
        assert exc_info.value.errors(include_url=False) == expected
    else:
        assert dataclasses.asdict(MyDataclass(**input_data)) == expected


@pytest.mark.parametrize(
    'field_constructor',
    [dataclasses.field, pydantic.dataclasses.Field],
)
@pytest.mark.parametrize(
    'input_data,extra,expected',
    init_test_cases,
)
def test_init_false_with_default(
    input_data: dict[str, Any],
    extra: str,
    expected: Union[dict[str, Any], list[dict[str, Any]]],
    field_constructor: Callable[..., Any],
) -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra=extra))
    class MyDataclass:
        a: int
        b: int = field_constructor(init=False, default=1)

    if isinstance(expected, list):
        with pytest.raises(ValidationError) as exc_info:
            MyDataclass(**input_data)
        assert exc_info.value.errors(include_url=False) == expected
    else:
        assert dataclasses.asdict(MyDataclass(**input_data)) == expected


def test_disallow_extra_allow_and_init_false() -> None:
    with pytest.raises(
        PydanticUserError, match='This combination is not allowed.'
    ):

        @pydantic.dataclasses.dataclass(config=ConfigDict(extra='allow'))
        class A:
            a: int = Field(init=False, default=1)


def test_disallow_init_false_and_init_var_true() -> None:
    with pytest.raises(
        PydanticUserError, match='mutually exclusive.'
    ):

        @pydantic.dataclasses.dataclass
        class Foo:
            bar: int = Field(init=False, init_var=True)


def test_annotations_valid_for_field_inheritance() -> None:

    @pydantic.dataclasses.dataclass()
    class A:
        a: int = pydantic.dataclasses.Field()
        # Assuming default type as int

    @pydantic.dataclasses.dataclass()
    class B(A):
        pass

    assert B.__pydantic_fields__['a'].annotation is int
    assert B(a=1).a == 1


def test_annotations_valid_for_field_inheritance_with_existing_field() -> None:

    @pydantic.dataclasses.dataclass()
    class A:
        a: int = pydantic.dataclasses.Field()

    @pydantic.dataclasses.dataclass()
    class B(A):
        b: str = pydantic.dataclasses.Field()

    assert B.__pydantic_fields__['a'].annotation is int
    assert B.__pydantic_fields__['b'].annotation is str
    b_instance = B(a=1, b='b')
    assert b_instance.a == 1
    assert b_instance.b == 'b'


def test_annotation_with_double_override() -> None:

    @pydantic.dataclasses.dataclass()
    class A:
        c: str = pydantic.dataclasses.Field()
        d: str = pydantic.dataclasses.Field()

    @pydantic.dataclasses.dataclass()
    class B(A):
        b: str = pydantic.dataclasses.Field()
        d: str = pydantic.dataclasses.Field()

    @pydantic.dataclasses.dataclass()
    class C(B):
        a: str

    for cls in [B, C]:
        instance = cls(a='a', b='b', c='c', d='d')
        for field_name in ['a', 'b', 'c', 'd']:
            assert cls.__pydantic_fields__[field_name].annotation is str
            assert getattr(instance, field_name) == field_name


def test_schema_valid_for_inner_generic() -> None:
    T = TypeVar('T')

    @pydantic.dataclasses.dataclass()
    class Inner(Generic[T]):
        x: T

    @pydantic.dataclasses.dataclass()
    class Outer:
        inner: Inner[int]

    assert Outer(inner={'x': None}).inner.x is None
    assert Outer(inner={'x': 1}).inner.x == 1
    with pytest.raises(ValidationError) as exc_info:
        Outer(inner={'y': None})
    assert exc_info.value.errors(include_url=False) == [
        {'input': {'y': None}, 'loc': ('inner', 'x'), 'msg': 'Field required', 'type': 'missing'}
    ]


def test_validation_works_for_cyclical_forward_refs() -> None:

    @pydantic.dataclasses.dataclass()
    class X:
        y: Optional['Y'] = None

    @pydantic.dataclasses.dataclass()
    class Y:
        x: Optional[X] = None

    assert Y(x={'y': None}).x.y is None


def test_annotated_with_field_default_factory():
    """
    https://github.com/pydantic/pydantic/issues/9947
    """
    field_constructor = dataclasses.field

    @pydantic.dataclasses.dataclass()
    class A:
        b: int = Field(default=1)
        c: int = Field(default=1)
        d: int = Field(default_factory=lambda: 2)
        e: int = Field(default_factory=lambda: 2)
        f: int = Field(default_factory=lambda: 2)

    @pydantic.dataclasses.dataclass()
    class B:
        b: int = field_constructor(default=1)
        c: int = field_constructor(default=1)
        d: int = Field(default_factory=lambda: 2)
        e: int = field_constructor(default_factory=lambda: 2)
        f: int = field_constructor(default_factory=lambda: 2)

    for cls in (A, B):
        instance = cls()
        field_names = ('a', 'b', 'c', 'd', 'e', 'f')
        results = (1, 1, 1, 2, 2, 2)
        for field_name, result in zip(field_names, results):
            assert getattr(instance, field_name) == result


def test_frozen_with_validate_assignment() -> None:
    """Test for https://github.com/pydantic/pydantic/issues/10041."""

    @pydantic.dataclasses.dataclass(
        frozen=True, config=ConfigDict(validate_assignment=True)
    )
    class MyDataclass:
        a: int

    inst = MyDataclass(1)
    try:
        inst.a = 'other'
    except Exception as e:
        assert "cannot assign to field 'a'" in repr(e)

    @pydantic.dataclasses.dataclass(
        config=ConfigDict(frozen=True, validate_assignment=True)
    )
    class MyDataclass2:
        a: int

    inst2 = MyDataclass2(1)
    try:
        inst2.a = 'other'
    except ValidationError as e:
        assert 'Instance is frozen' in repr(e)


def test_warns_on_double_frozen() -> None:
    with pytest.warns(
        UserWarning,
        match='`frozen` is set via both the `dataclass` decorator and `config`',
    ):

        @pydantic.dataclasses.dataclass(frozen=True, config=ConfigDict(frozen=True))
        class DC:
            a: int


def test_warns_on_double_config() -> None:
    with pytest.warns(
        UserWarning,
        match='`config` is set via both the `dataclass` decorator and `__pydantic_config__`',
    ):

        @pydantic.dataclasses.dataclass(config=ConfigDict(title='from decorator'))
        class Foo:
            __pydantic_config__: ConfigDict = ConfigDict(title='from __pydantic_config__')


def test_dataclasses_with_config_decorator() -> None:

    @dataclasses.dataclass
    @with_config(ConfigDict(str_to_lower=True))
    class Model1:
        x: str

    ta1 = TypeAdapter(Model1)
    assert ta1.validate_python({'x': 'ABC'}).x == 'abc'

    @with_config(ConfigDict(str_to_lower=True))
    @dataclasses.dataclass
    class Model2:
        x: str

    ta2 = TypeAdapter(Model2)
    assert ta2.validate_python({'x': 'ABC'}).x == 'abc'


def test_pydantic_field_annotation() -> None:

    @pydantic.dataclasses.dataclass
    class Model:
        x: int = Field(gt=0)

    with pytest.raises(ValidationError) as exc_info:
        Model(x=-1)
    assert exc_info.value.errors(include_url=False) == [
        {
            'ctx': {'gt': 0},
            'input': -1,
            'loc': ('x',),
            'msg': 'Input should be greater than 0',
            'type': 'greater_than',
        }
    ]


def test_combined_field_annotations() -> None:
    """
    This test is included to document the fact that `Field` and `field` can be used together.
    That said, if you mix them like this, there is a good chance you'll run into surprising behavior/bugs.

    (E.g., `x: Annotated[int, Field(gt=1, validate_default=True)] = field(default=0)` doesn't cause an error)

    I would generally advise against doing this, and if we do change the behavior in the future to somehow merge
    pydantic.FieldInfo and dataclasses.Field in a way that changes runtime behavior for existing code, I would probably
    consider it a bugfix rather than a breaking change.
    """

    @pydantic.dataclasses.dataclass
    class Model:
        x: int = dataclasses.field(default=1, metadata={'gt': 1})

    assert Model().x == 1
    with pytest.raises(ValidationError) as exc_info:
        Model(x=0)
    assert exc_info.value.errors(include_url=False) == [
        {
            'ctx': {'gt': 1},
            'input': 0,
            'loc': ('x',),
            'msg': 'Input should be greater than 1',
            'type': 'greater_than',
        }
    ]


def test_dataclass_field_default_factory_with_init() -> None:

    @pydantic.dataclasses.dataclass
    class Model:
        x: int = dataclasses.field(default_factory=lambda: 3, init=False)

    m = Model()
    assert 'x' in Model.__pydantic_fields__
    assert m.x == 3
    assert RootModel[Model](m).model_dump() == {'x': 3}


def test_dataclass_field_default_with_init() -> None:

    @pydantic.dataclasses.dataclass
    class Model:
        x: int = dataclasses.field(default=3, init=False)

    m = Model()
    assert 'x' in Model.__pydantic_fields__
    assert m.x == 3
    assert RootModel[Model](m).model_dump() == {'x': 3}


def test_metadata() -> None:

    @dataclasses.dataclass
    class Test:
        value: int = dataclasses.field(metadata={'info': 'Some int value', 'json_schema_extra': {'a': 'b'}})

    PydanticTest = pydantic.dataclasses.dataclass(Test)
    assert TypeAdapter(PydanticTest).json_schema() == {
        'properties': {
            'value': {
                'a': 'b',
                'title': 'Value',
                'type': 'integer',
            }
        },
        'required': ['value'],
        'title': 'Test',
        'type': 'object',
    }


def test_signature() -> None:

    @pydantic.dataclasses.dataclass
    class Model:
        y: str = 'y'
        z: float = dataclasses.field(default=1.0)
        a: float = dataclasses.field(default_factory=float)
        b: float = Field(default=1.0)
        c: float = Field(default_factory=float)
        d: int = dataclasses.field(metadata={'alias': 'dd'}, default=1)

    assert str(inspect.signature(Model)) == "(x: int, y: str = 'y', z: float = 1.0, a: float = <factory>, b: float = 1.0, c: float = <factory>, dd: int = 1) -> None"


def test_inherited_dataclass_signature() -> None:

    @pydantic.dataclasses.dataclass
    class A:
        a: int

    @pydantic.dataclasses.dataclass
    class B(A):
        b: int

    assert str(inspect.signature(A)) == '(a: int) -> None'
    assert str(inspect.signature(B)) == '(a: int, b: int) -> None'


def test_dataclasses_with_slots_and_default() -> None:

    @pydantic.dataclasses.dataclass(slots=True)
    class A:
        a: int = 0

    assert A().a == 0

    @pydantic.dataclasses.dataclass(slots=True)
    class B:
        b: float = pydantic.dataclasses.Field(1)

    assert B().b == 1


@pytest.mark.parametrize(
    'decorator1',
    **dataclass_decorators(),
)
def test_annotated_before_validator_called_once(
    decorator1: Callable[..., Any],
) -> None:
    count: int = 0

    def convert(value: Any) -> str:
        nonlocal count
        count += 1
        return str(value)

    IntToStr = Annotated[str, BeforeValidator(convert)]

    @decorator1
    class A:
        x: IntToStr

    assert count == 0
    TypeAdapter(A).validate_python({'x': 123})
    assert count == 1


def test_is_pydantic_dataclass() -> None:

    @pydantic.dataclasses.dataclass
    class PydanticDataclass:
        a: int

    @dataclasses.dataclass
    class StdLibDataclass:
        a: int

    assert is_pydantic_dataclass(PydanticDataclass) is True
    assert is_pydantic_dataclass(StdLibDataclass) is False


def test_can_inherit_stdlib_dataclasses_with_defaults() -> None:

    @dataclasses.dataclass
    class Base:
        a: Optional[int] = None

    class Model(BaseModel, Base):
        pass

    assert Model().a is None


def test_can_inherit_stdlib_dataclasses_default_factories_and_use_them() -> None:
    """This test documents that default factories are not supported"""

    @dataclasses.dataclass
    class Base:
        a: str = dataclasses.field(default_factory=lambda: 'TEST')

    class Model(BaseModel, Base):
        pass

    with pytest.raises(ValidationError):
        assert Model().a == 'TEST'


def test_can_inherit_stdlib_dataclasses_default_factories_and_provide_a_value() -> None:

    @dataclasses.dataclass
    class Base:
        a: str = dataclasses.field(default_factory=lambda: 'TEST')

    class Model(BaseModel, Base):
        pass

    assert Model(a='NOT_THE_SAME').a == 'NOT_THE_SAME'


def test_can_inherit_stdlib_dataclasses_with_dataclass_fields() -> None:

    @dataclasses.dataclass
    class Base:
        a: int = dataclasses.field(default=5)

    class Model(BaseModel, Base):
        pass

    assert Model().a == 5


def test_alias_with_dashes() -> None:
    """Test for fix issue #7226."""

    @pydantic.dataclasses.dataclass
    class Foo:
        some_var: str = Field(alias='some-var')

    obj = Foo(**{'some-var': 'some_value'})
    assert obj.some_var == 'some_value'
    with pytest.raises(ValidationError) as exc_info:
        Foo(name='test name')
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'missing',
            'loc': ('some-var',),
            'msg': 'Field required',
            'input': {'name': 'test name'},
        }
    ]


def test_validate_strings() -> None:

    @pydantic.dataclasses.dataclass
    class Nested:
        d: date

    class Model(BaseModel):
        n: Nested

    assert Model.model_validate_strings({'n': {'d': '2017-01-01'}}).n.d == date(2017, 1, 1)


@pytest.mark.parametrize(
    'field_constructor',
    [dataclasses.field, pydantic.dataclasses.Field],
)
@pytest.mark.parametrize(
    'extra',
    ['ignore', 'forbid'],
)
def test_init_false_not_in_signature(
    field_constructor: Callable[..., Any],
    extra: str,
) -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra=extra))
    class MyDataclass:
        a: int
        b: int = field_constructor(init=False, default=-1)
        c: int = 1  # Assuming an additional field for testing

    signature = inspect.signature(MyDataclass)
    assert 'a' in signature.parameters
    assert 'b' not in signature.parameters


def test_disallow_extra_allow_and_init_false() -> None:
    with pytest.raises(
        PydanticUserError, match='This combination is not allowed.'
    ):

        @pydantic.dataclasses.dataclass(config=ConfigDict(extra='allow'))
        class A:
            a: int = Field(init=False, default=1)


def test_disallow_init_false_and_init_var_true() -> None:
    with pytest.raises(
        PydanticUserError, match='mutually exclusive.'
    ):

        @pydantic.dataclasses.dataclass
        class Foo:
            bar: int = Field(init=False, init_var=True)


def test_annotations_valid_for_field_inheritance() -> None:

    @pydantic.dataclasses.dataclass()
    class A:
        a: int = pydantic.dataclasses.Field()

    @pydantic.dataclasses.dataclass()
    class B(A):
        pass

    assert B.__pydantic_fields__['a'].annotation is int
    assert B(a=1).a == 1


def test_annotations_valid_for_field_inheritance_with_existing_field() -> None:

    @pydantic.dataclasses.dataclass()
    class A:
        a: int = pydantic.dataclasses.Field()

    @pydantic.dataclasses.dataclass()
    class B(A):
        b: str = pydantic.dataclasses.Field()

    assert B.__pydantic_fields__['a'].annotation is int
    assert B.__pydantic_fields__['b'].annotation is str
    b_instance = B(a=1, b='b')
    assert b_instance.a == 1
    assert b_instance.b == 'b'


def test_annotation_with_double_override() -> None:

    @pydantic.dataclasses.dataclass()
    class A:
        c: str = pydantic.dataclasses.Field()
        d: str = pydantic.dataclasses.Field()

    @pydantic.dataclasses.dataclass()
    class B(A):
        b: str = pydantic.dataclasses.Field()
        d: str = pydantic.dataclasses.Field()

    @pydantic.dataclasses.dataclass()
    class C(B):
        a: str

    for cls in [B, C]:
        instance = cls(a='a', b='b', c='c', d='d')
        for field_name in ['a', 'b', 'c', 'd']:
            assert cls.__pydantic_fields__[field_name].annotation is str
            assert getattr(instance, field_name) == field_name


def test_schema_valid_for_inner_generic() -> None:
    T = TypeVar('T')

    @pydantic.dataclasses.dataclass()
    class Inner(Generic[T]):
        x: T

    @pydantic.dataclasses.dataclass()
    class Outer:
        inner: Inner[int]

    assert Outer(inner={'x': None}).inner.x is None
    assert Outer(inner={'x': 1}).inner.x == 1
    with pytest.raises(ValidationError) as exc_info:
        Outer(inner={'y': None})
    assert exc_info.value.errors(include_url=False) == [
        {
            'input': {'y': None},
            'loc': ('inner', 'x'),
            'msg': 'Field required',
            'type': 'missing',
        }
    ]


def test_validation_works_for_cyclical_forward_refs() -> None:

    @pydantic.dataclasses.dataclass()
    class X:
        y: Optional['Y'] = None

    @pydantic.dataclasses.dataclass()
    class Y:
        x: Optional[X] = None

    assert Y(x={'y': None}).x.y is None


def test_annotated_with_field_default_factory():
    """
    https://github.com/pydantic/pydantic/issues/9947
    """
    field_constructor = dataclasses.field

    @pydantic.dataclasses.dataclass()
    class A:
        b: int = Field()
        c: int = Field()
        d: int = Field(default_factory=lambda: 2)
        e: int = Field(default_factory=lambda: 2)
        f: int = Field(default_factory=lambda: 2)

    @pydantic.dataclasses.dataclass()
    class B:
        b: int = field_constructor()
        c: int = field_constructor()
        d: int = Field(default_factory=lambda: 2)
        e: int = field_constructor(default_factory=lambda: 2)
        f: int = field_constructor(default_factory=lambda: 2)

    for cls in (A, B):
        instance = cls()
        field_names = ('b', 'c', 'd', 'e', 'f')
        results = (1, 1, 2, 2, 2)
        for field_name, result in zip(field_names, results):
            assert getattr(instance, field_name) == result


def test_frozen_with_validate_assignment() -> None:
    """Test for https://github.com/pydantic/pydantic/issues/10041."""

    @pydantic.dataclasses.dataclass(
        frozen=True, config=ConfigDict(validate_assignment=True)
    )
    class MyDataclass:
        a: int

    inst = MyDataclass(1)
    try:
        inst.a = 'other'
    except Exception as e:
        assert "cannot assign to field 'a'" in repr(e)

    @pydantic.dataclasses.dataclass(
        config=ConfigDict(frozen=True, validate_assignment=True)
    )
    class MyDataclass2:
        a: int

    inst2 = MyDataclass2(1)
    try:
        inst2.a = 'other'
    except ValidationError as e:
        assert 'Instance is frozen' in repr(e)


def test_warns_on_double_frozen() -> None:
    with pytest.warns(
        UserWarning,
        match='`frozen` is set via both the `dataclass` decorator and `config`',
    ):

        @pydantic.dataclasses.dataclass(frozen=True, config=ConfigDict(frozen=True))
        class DC:
            a: int


def test_warns_on_double_config() -> None:
    with pytest.warns(
        UserWarning,
        match='`config` is set via both the `dataclass` decorator and `__pydantic_config__`',
    ):

        @pydantic.dataclasses.dataclass(config=ConfigDict(title='from decorator'))
        class Foo:
            __pydantic_config__: ConfigDict = ConfigDict(title='from __pydantic_config__')


def test_dataclasses_with_config_decorator() -> None:

    @dataclasses.dataclass
    @with_config(ConfigDict(str_to_lower=True))
    class Model1:
        x: str

    ta1 = TypeAdapter(Model1)
    assert ta1.validate_python({'x': 'ABC'}).x == 'abc'

    @with_config(ConfigDict(str_to_lower=True))
    @dataclasses.dataclass
    class Model2:
        x: str

    ta2 = TypeAdapter(Model2)
    assert ta2.validate_python({'x': 'ABC'}).x == 'abc'


def test_pydantic_field_annotation() -> None:

    @pydantic.dataclasses.dataclass
    class Model:
        x: int = Field(gt=0)

    with pytest.raises(ValidationError) as exc_info:
        Model(x=-1)
    assert exc_info.value.errors(include_url=False) == [
        {
            'ctx': {'gt': 0},
            'input': -1,
            'loc': ('x',),
            'msg': 'Input should be greater than 0',
            'type': 'greater_than',
        }
    ]


def test_combined_field_annotations() -> None:
    """
    This test is included to document the fact that `Field` and `field` can be used together.
    That said, if you mix them like this, there is a good chance you'll run into surprising behavior/bugs.

    (E.g., `x: Annotated[int, Field(gt=1, validate_default=True)] = field(default=0)` doesn't cause an error)

    I would generally advise against doing this, and if we do change the behavior in the future to somehow merge
    pydantic.FieldInfo and dataclasses.Field in a way that changes runtime behavior for existing code, I would probably
    consider it a bugfix rather than a breaking change.
    """
    @pydantic.dataclasses.dataclass
    class Model:
        x: int = dataclasses.field(default=1, metadata={'gt': 1})

    assert Model().x == 1
    with pytest.raises(ValidationError) as exc_info:
        Model(x=0)
    assert exc_info.value.errors(include_url=False) == [
        {
            'ctx': {'gt': 1},
            'input': 0,
            'loc': ('x',),
            'msg': 'Input should be greater than 1',
            'type': 'greater_than',
        }
    ]


def test_dataclass_field_default_factory_with_init() -> None:

    @pydantic.dataclasses.dataclass
    class Model:
        x: int = dataclasses.field(default_factory=lambda: 3, init=False)

    m = Model()
    assert 'x' in Model.__pydantic_fields__
    assert m.x == 3
    assert RootModel[Model](m).model_dump() == {'x': 3}


def test_dataclass_field_default_with_init() -> None:

    @pydantic.dataclasses.dataclass
    class Model:
        x: int = dataclasses.field(default=3, init=False)

    m = Model()
    assert 'x' in Model.__pydantic_fields__
    assert m.x == 3
    assert RootModel[Model](m).model_dump() == {'x': 3}


def test_metadata() -> None:

    @dataclasses.dataclass
    class Test:
        value: int = dataclasses.field(metadata={'info': 'Some int value', 'json_schema_extra': {'a': 'b'}})

    PydanticTest = pydantic.dataclasses.dataclass(Test)
    assert TypeAdapter(PydanticTest).json_schema() == {
        'properties': {
            'value': {
                'a': 'b',
                'title': 'Value',
                'type': 'integer',
            }
        },
        'required': ['value'],
        'title': 'Test',
        'type': 'object',
    }


def test_signature() -> None:

    @pydantic.dataclasses.dataclass
    class Model:
        y: str = 'y'
        z: float = dataclasses.field(default=1.0)
        a: float = dataclasses.field(default_factory=float)
        b: float = Field(default=1.0)
        c: float = Field(default_factory=float)
        d: int = dataclasses.field(metadata={'alias': 'dd'}, default=1)

    assert str(inspect.signature(Model)) == "(y: str = 'y', z: float = 1.0, a: float = <factory>, b: float = 1.0, c: float = <factory>, dd: int = 1) -> None"


def test_inherited_dataclass_signature() -> None:

    @pydantic.dataclasses.dataclass
    class A:
        a: int

    @pydantic.dataclasses.dataclass
    class B(A):
        b: int

    assert str(inspect.signature(A)) == '(a: int) -> None'
    assert str(inspect.signature(B)) == '(a: int, b: int) -> None'


def test_dataclasses_with_slots_and_default() -> None:

    @pydantic.dataclasses.dataclass(slots=True)
    class A:
        a: int = 0

    assert A().a == 0

    @pydantic.dataclasses.dataclass(slots=True)
    class B:
        b: float = pydantic.dataclasses.Field(1)

    assert B().b == 1


@pytest.mark.parametrize(
    'decorator1',
    dataclass_decorators(),
)
def test_annotated_before_validator_called_once(decorator1: Callable[..., Any]) -> None:
    count: int = 0

    def convert(value: Any) -> str:
        nonlocal count
        count += 1
        return str(value)

    IntToStr = Annotated[str, BeforeValidator(convert)]

    @decorator1
    class A:
        x: IntToStr

    assert count == 0
    TypeAdapter(A).validate_python({'x': 123})
    assert count == 1


def test_is_pydantic_dataclass() -> None:

    @pydantic.dataclasses.dataclass
    class PydanticDataclass:
        a: int

    @dataclasses.dataclass
    class StdLibDataclass:
        a: int

    assert is_pydantic_dataclass(PydanticDataclass) is True
    assert is_pydantic_dataclass(StdLibDataclass) is False


def test_can_inherit_stdlib_dataclasses_with_defaults() -> None:

    @dataclasses.dataclass
    class Base:
        a: Optional[int] = None

    class Model(BaseModel, Base):
        pass

    assert Model().a is None


def test_can_inherit_stdlib_dataclasses_default_factories_and_use_them() -> None:
    """This test documents that default factories are not supported"""

    @dataclasses.dataclass
    class Base:
        a: str = dataclasses.field(default_factory=lambda: 'TEST')

    class Model(BaseModel, Base):
        pass

    with pytest.raises(ValidationError):
        assert Model().a == 'TEST'


def test_can_inherit_stdlib_dataclasses_default_factories_and_provide_a_value() -> None:

    @dataclasses.dataclass
    class Base:
        a: str = dataclasses.field(default_factory=lambda: 'TEST')

    class Model(BaseModel, Base):
        pass

    assert Model(a='NOT_THE_SAME').a == 'NOT_THE_SAME'


def test_can_inherit_stdlib_dataclasses_with_dataclass_fields() -> None:

    @dataclasses.dataclass
    class Base:
        a: int = dataclasses.field(default=5)

    class Model(BaseModel, Base):
        pass

    assert Model().a == 5


def test_alias_with_dashes() -> None:
    """Test for fix issue #7226."""

    @pydantic.dataclasses.dataclass
    class Foo:
        some_var: str = Field(alias='some-var')

    obj = Foo(**{'some-var': 'some_value'})
    assert obj.some_var == 'some_value'
    with pytest.raises(ValidationError) as exc_info:
        Foo(name='test name')
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'missing',
            'loc': ('some-var',),
            'msg': 'Field required',
            'input': {'name': 'test name'},
        }
    ]


def test_validate_strings() -> None:

    @pydantic.dataclasses.dataclass
    class Nested:
        d: date

    class Model(BaseModel):
        n: Nested

    assert Model.model_validate_strings({'n': {'d': '2017-01-01'}}).n.d == date(2017, 1, 1)


@pytest.mark.parametrize(
    'field_constructor',
    [dataclasses.field, pydantic.dataclasses.Field],
)
@pytest.mark.parametrize(
    'extra',
    ['ignore', 'forbid'],
)
@pytest.mark.parametrize(
    'input_data,extra,expected',
    init_test_cases,
)
def test_init_false_with_post_init(
    input_data: dict[str, Any],
    extra: str,
    expected: Union[dict[str, Any], list[dict[str, Any]]],
    field_constructor: Callable[..., Any],
) -> None:

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra=extra))
    class MyDataclass:
        a: int
        b: int = field_constructor(init=False)

        def __post_init__(self) -> None:
            self.b = 1

    if isinstance(expected, list):
        with pytest.raises(ValidationError) as exc_info:
            MyDataclass(**input_data)
        assert exc_info.value.errors(include_url=False) == expected
    else:
        assert dataclasses.asdict(MyDataclass(**input_data)) == expected


def test_dataclass_config_validate_default() -> None:

    @pydantic.dataclasses.dataclass
    class Model:
        x: int = -1

        @field_validator('x')
        @classmethod
        def force_x_positive(cls, v: int) -> int:
            assert v > 0
            return v

    assert Model().x == -1

    @pydantic.dataclasses.dataclass(config=ConfigDict(validate_default=True))
    class ValidatingModel(Model):
        pass

    with pytest.raises(ValidationError) as exc_info:
        ValidatingModel()
    assert exc_info.value.errors(include_url=False) == [
        {
            'ctx': {'error': HasRepr(repr(AssertionError('assert -1 > 0')))},
            'input': -1,
            'loc': ('x',),
            'msg': 'Assertion failed, assert -1 > 0',
            'type': 'assertion_error',
        }
    ]


@pytest.mark.parametrize(
    'dataclass_decorator',
    dataclass_decorators(include_identity=True),
)
def test_pydantic_dataclass_preserves_metadata(
    dataclass_decorator: Callable[..., Any],
) -> None:

    @dataclass_decorator
    class FooStd:
        """Docstring"""

    FooPydantic = pydantic.dataclasses.dataclass(FooStd)
    assert FooPydantic.__module__ == FooStd.__module__
    assert FooPydantic.__name__ == FooStd.__name__
    assert FooPydantic.__qualname__ == FooStd.__qualname__


def test_recursive_dataclasses_gh_4509(create_module: Callable[..., Any]) -> None:

    @create_module
    def module():
        import dataclasses
        import pydantic

        @dataclasses.dataclass
        class Recipe:
            author: Any

        @dataclasses.dataclass
        class Cook:
            recipes: list[Recipe]

        @pydantic.dataclasses.dataclass
        class Foo(Cook):
            recipes: list[Recipe]

    gordon = module.Cook(recipes=[])
    burger = module.Recipe(author=gordon)
    me = module.Foo(recipes=[burger])
    assert me.recipes == [burger]


def test_dataclass_alias_generator() -> None:

    def alias_generator(name: str) -> str:
        return 'alias_' + name

    @pydantic.dataclasses.dataclass(config=ConfigDict(alias_generator=alias_generator))
    class User:
        name: str = Field(alias='alias_name')
        score: int = Field(alias='my_score')

    user = User(**{'alias_name': 'test name', 'my_score': 2})
    assert user.name == 'test name'
    assert user.score == 2
    with pytest.raises(ValidationError) as exc_info:
        User(name='test name', score=2)
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'missing',
            'loc': ('alias_name',),
            'msg': 'Field required',
            'input': {'name': 'test name', 'score': 2},
        },
        {
            'type': 'missing',
            'loc': ('my_score',),
            'msg': 'Field required',
            'input': {'name': 'test name', 'score': 2},
        },
    ]


def test_init_vars_inheritance() -> None:
    init_vars: list[int] = []

    @pydantic.dataclasses.dataclass
    class Foo:
        init: int

    @pydantic.dataclasses.dataclass
    class Bar(Foo):
        a: int

        def __post_init__(self) -> None:
            init_vars.append(self.init)

    bar = Bar(init=1, a=2)
    assert TypeAdapter(Bar).dump_python(bar) == {'a': 2}
    assert init_vars == [1]
    with pytest.raises(ValidationError) as exc_info:
        Bar(init='a', a=2)
    assert exc_info.value.errors(include_url=False) == [
        {
            'input': 'a',
            'loc': ('init',),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'type': 'int_parsing',
        }
    ]


@pytest.mark.skipif(
    not hasattr(pydantic.dataclasses, '_call_initvar'),
    reason='InitVar was not modified',
)
@pytest.mark.parametrize('remove_monkeypatch', [True, False])
def test_init_vars_call_monkeypatch(remove_monkeypatch: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if remove_monkeypatch:
        monkeypatch.delattr(InitVar, '__call__')
    InitVar[int]()
    with pytest.raises(TypeError, match="'InitVar' object is not callable") as exc:
        InitVar[int]()
    stack_depth = len(traceback.extract_tb(exc.value.__traceback__))
    assert stack_depth == (1 if remove_monkeypatch else 2)


@pytest.mark.parametrize(
    'decorator1,expected_parent,expected_child',
    [
        (
            pydantic.dataclasses.dataclass,
            ['parent before', 'parent', 'parent after'],
            ['parent before', 'parent', 'parent after', 'child before', 'child after'],
        ),
        (
            dataclasses.dataclass,
            [],
            ['parent before', 'child', 'parent after', 'child before', 'child after'],
        ),
    ],
    ids=['pydantic', 'stdlib'],
)
def test_inheritance_replace(
    decorator1: Callable[..., Any],
    expected_parent: list[str],
    expected_child: list[str],
) -> None:
    """We promise that if you add a validator
    with the same _function_ name as an existing validator
    it replaces the existing validator and is run instead of it.
    """

    @decorator1
    class Parent:
        a: list[str]

        @field_validator('a', mode='before')
        @classmethod
        def parent_val_before(cls, v: list[str]) -> list[str]:
            v.append('parent before')
            return v

        @field_validator('a', mode='before')
        @classmethod
        def val(cls, v: list[str]) -> list[str]:
            v.append('parent')
            return v

        @field_validator('a', mode='before')
        @classmethod
        def parent_val_after(cls, v: list[str]) -> list[str]:
            v.append('parent after')
            return v

    @pydantic.dataclasses.dataclass
    class Child(Parent):
        a: list[str]
        b: int

        @field_validator('a', mode='before')
        @classmethod
        def child_val_before(cls, v: list[str]) -> list[str]:
            v.append('child before')
            return v

        @field_validator('a', mode='before')
        @classmethod
        def val(cls, v: list[str]) -> list[str]:
            v.append('child')
            return v

        @field_validator('a', mode='before')
        @classmethod
        def child_val_after(cls, v: list[str]) -> list[str]:
            v.append('child after')
            return v

    parent_instance = Parent(a=[])
    assert parent_instance.a == expected_parent

    child_instance = Child(a=[], b=1)
    assert child_instance.a == expected_child


@pytest.mark.parametrize(
    'decorator1',
    dataclass_decorators(),
    ids=['pydantic', 'stdlib', 'combined', 'identity'],
)
@pytest.mark.parametrize(
    'default',
    [1, dataclasses.field(default=1), Field(default=1)],
    ids=['1', 'dataclasses.field(default=1)', 'pydantic.Field(default=1)'],
)
def test_dataclasses_inheritance_default_value_is_not_deleted(
    decorator1: Callable[..., Any],
    default: Any,
) -> None:
    if decorator1 is dataclasses.dataclass and isinstance(default, FieldInfo):
        pytest.skip(reason="stdlib dataclasses don't support Pydantic fields")

    @decorator1
    class Parent:
        a: int = default

    assert Parent.a == 1
    assert Parent().a == 1

    @pydantic.dataclasses.dataclass
    class Child(Parent):
        pass

    assert Child.a == 1
    assert Child().a == 1


def test_dataclass_config_validate_default() -> None:

    @pydantic.dataclasses.dataclass
    class Model:
        x: int = -1

        @field_validator('x')
        @classmethod
        def force_x_positive(cls, v: int) -> int:
            assert v > 0
            return v

    assert Model().x == -1

    @pydantic.dataclasses.dataclass(config=ConfigDict(validate_default=True))
    class ValidatingModel(Model):
        pass

    with pytest.raises(ValidationError) as exc_info:
        ValidatingModel()
    assert exc_info.value.errors(include_url=False) == [
        {
            'ctx': {'error': HasRepr(repr(AssertionError('assert -1 > 0')))},
            'input': -1,
            'loc': ('x',),
            'msg': 'Assertion failed, assert -1 > 0',
            'type': 'assertion_error',
        }
    ]


@pytest.mark.parametrize(
    'dataclass_decorator', dataclass_decorators(include_identity=True),
)
@pytest.mark.parametrize(
    'default',
    [1, dataclasses.field(default=1), Field(default=1)],
    ids=['1', 'dataclasses.field(default=1)', 'pydantic.Field(default=1)'],
)
def test_dataclasses_inheritance_default_value_is_not_deleted(
    dataclass_decorator: Callable[..., Any],
    default: Any,
) -> None:
    if dataclass_decorator is dataclasses.dataclass and isinstance(default, FieldInfo):
        pytest.skip(reason="stdlib dataclasses don't support Pydantic fields")

    @dataclass_decorator
    class Parent:
        a: int = default

    assert Parent.a == 1
    assert Parent().a == 1

    @pydantic.dataclasses.dataclass
    class Child(Parent):
        pass

    assert Child.a == 1
    assert Child().a == 1


def test_dataclass_config_override_in_decorator() -> None:
    with pytest.warns(
        UserWarning,
        match='`config` is set via both the `dataclass` decorator and `__pydantic_config__`',
    ):

        @pydantic.dataclasses.dataclass(
            config=ConfigDict(str_to_lower=False, str_strip_whitespace=True)
        )
        class Model:
            __pydantic_config__: ConfigDict = ConfigDict(str_to_lower=True)


def test_model_config_override_in_decorator_empty_config() -> None:
    with pytest.warns(
        UserWarning,
        match='`config` is set via both the `dataclass` decorator and `__pydantic_config__`',
    ):

        @pydantic.dataclasses.dataclass(config=ConfigDict())
        class Model:
            __pydantic_config__: ConfigDict = ConfigDict(str_to_lower=True)


def test_override_builtin_dataclass_nested() -> None:

    @dataclasses.dataclass
    class Meta:
        modified_date: Optional[datetime]
        seen_count: int

    @dataclasses.dataclass
    class File:
        filename: bytes
        meta: Meta

    FileChecked = pydantic.dataclasses.dataclass(File)
    f = FileChecked(filename=b'thefilename', meta=Meta(modified_date='2020-01-01T00:00', seen_count='7'))
    assert f.filename == 'thefilename'
    assert f.meta.modified_date == datetime(2020, 1, 1, 0, 0)
    assert f.meta.seen_count == 7
    with pytest.raises(ValidationError) as e:
        FileChecked(filename=b'thefilename', meta=Meta(modified_date='2020-01-01T00:00', seen_count=['7']))
    assert e.value.errors(include_url=False) == [
        {
            'type': 'int_type',
            'loc': ('meta', 'seen_count'),
            'msg': 'Input should be a valid integer',
            'input': ['7'],
        }
    ]

    class Foo(BaseModel):
        file: File

    foo = Foo.model_validate({'file': {'filename': b'thefilename', 'meta': {'modified_date': '2020-01-01T00:00', 'seen_count': '7'}}})
    assert foo.file.filename == 'thefilename'
    assert foo.file.meta.modified_date == datetime(2020, 1, 1, 0, 0)
    assert foo.file.meta.seen_count == 7


def test_override_builtin_dataclass_nested_schema() -> None:

    @dataclasses.dataclass
    class Meta:
        pass

    @dataclasses.dataclass
    class File:
        filename: bytes
        meta: Meta

    FileChecked = pydantic.dataclasses.dataclass(File)
    assert model_json_schema(FileChecked) == {
        '$defs': {
            'Meta': {
                'properties': {
                    'modified_date': {
                        'anyOf': [
                            {'format': 'date-time', 'type': 'string'},
                            {'type': 'null'},
                        ],
                        'title': 'Modified Date',
                    },
                    'seen_count': {'title': 'Seen Count', 'type': 'integer'},
                },
                'required': ['modified_date', 'seen_count'],
                'title': 'Meta',
                'type': 'object',
            }
        },
        'properties': {
            'filename': {'title': 'Filename', 'type': 'string'},
            'meta': {'$ref': '#/$defs/Meta'},
        },
        'required': ['filename', 'meta'],
        'title': 'File',
        'type': 'object',
    }


def test_inherit_builtin_dataclass() -> None:

    @dataclasses.dataclass
    class Z:
        z: int

    @dataclasses.dataclass
    class Y(Z):
        y: int

    @pydantic.dataclasses.dataclass
    class X(Y):
        x: int

    pika = X(x='2', y='4', z='3')
    assert pika.x == 2
    assert pika.y == 4
    assert pika.z == 3


def test_copy_dataclass() -> None:
    @pydantic.dataclasses.dataclass
    class Original:
        a: int
        b: str

    original = Original(a=1, b='test')
    copy_instance = dataclasses.replace(original, a=2)
    assert copy_instance.a == 2
    assert copy_instance.b == 'test'


def test_model_config(dataclass_decorator: Callable[..., Any]) -> None:
    @dataclass_decorator
    class Model:
        __pydantic_config__: ConfigDict = ConfigDict(str_to_lower=True)
        x: str

    ta = TypeAdapter(Model)
    assert ta.validate_python({'x': 'ABC'}).x == 'abc'
