import functools
import importlib.util
import re
import sys
import typing
from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Annotated,
    Any,
    Callable,
    ForwardRef,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
)
import pytest
from dirty_equals import HasRepr, IsStr
from pydantic_core import (
    ErrorDetails,
    InitErrorDetails,
    PydanticSerializationError,
    PydanticUndefined,
    core_schema,
)
from typing_extensions import TypeAliasType, TypedDict, get_args
from pydantic import (
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    PrivateAttr,
    PydanticDeprecatedSince20,
    PydanticSchemaGenerationError,
    PydanticUserError,
    RootModel,
    TypeAdapter,
    ValidationError,
    constr,
    errors,
    field_validator,
    model_validator,
    root_validator,
    validator,
)
from pydantic._internal._model_construction import ModelMetaclass
from pydantic.fields import Field, computed_field
from pydantic.functional_serializers import field_serializer, model_serializer


def test_str_bytes() -> None:
    class Model(BaseModel):
        v: Union[str, bytes]

    m = Model(v="s")
    assert m.v == "s"
    assert (
        repr(Model.model_fields["v"])
        == "FieldInfo(annotation=Union[str, bytes], required=True)"
    )
    m = Model(v=b"b")
    assert m.v == b"b"
    with pytest.raises(ValidationError) as exc_info:
        Model(v=None)
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "string_type",
            "loc": ("v", "str"),
            "msg": "Input should be a valid string",
            "input": None,
        },
        {
            "type": "bytes_type",
            "loc": ("v", "bytes"),
            "msg": "Input should be a valid bytes",
            "input": None,
        },
    ]


def test_str_bytes_none() -> None:
    class Model(BaseModel):
        v: Optional[Union[str, bytes]]

    m = Model(v="s")
    assert m.v == "s"
    m = Model(v=b"b")
    assert m.v == b"b"
    m = Model(v=None)
    assert m.v is None


def test_union_int_str() -> None:
    class Model(BaseModel):
        v: Union[int, str]

    m = Model(v=123)
    assert m.v == 123
    m = Model(v="123")
    assert m.v == "123"
    m = Model(v=b"foobar")
    assert m.v == "foobar"
    m = Model(v=12.0)
    assert m.v == 12
    with pytest.raises(ValidationError) as exc_info:
        Model(v=None)
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "int_type",
            "loc": ("v", "int"),
            "msg": "Input should be a valid integer",
            "input": None,
        },
        {
            "type": "string_type",
            "loc": ("v", "str"),
            "msg": "Input should be a valid string",
            "input": None,
        },
    ]


def test_union_int_any() -> None:
    class Model(BaseModel):
        v: Any

    m = Model(v=123)
    assert m.v == 123
    m = Model(v="123")
    assert m.v == "123"
    m = Model(v="foobar")
    assert m.v == "foobar"
    m = Model(v=None)
    assert m.v is None


def test_typed_list() -> None:
    class Model(BaseModel):
        v: list[int]

    m = Model(v=[1, 2, "3"])
    assert m.v == [1, 2, 3]
    with pytest.raises(ValidationError) as exc_info:
        Model(v=[1, "x", "y"])
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "int_parsing",
            "loc": ("v", 1),
            "msg": "Input should be a valid integer, unable to parse string as an integer",
            "input": "x",
        },
        {
            "type": "int_parsing",
            "loc": ("v", 2),
            "msg": "Input should be a valid integer, unable to parse string as an integer",
            "input": "y",
        },
    ]
    with pytest.raises(ValidationError) as exc_info:
        Model(v=1)
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "list_type",
            "loc": ("v",),
            "msg": "Input should be a valid list",
            "input": 1,
        }
    ]


def test_typed_set() -> None:
    class Model(BaseModel):
        v: set[int]

    assert Model(v={1, 2, "3"}).v == {1, 2, 3}
    assert Model(v=[1, 2, "3"]).v == {1, 2, 3}
    with pytest.raises(ValidationError) as exc_info:
        Model(v=[1, "x"])
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "int_parsing",
            "loc": ("v", 1),
            "msg": "Input should be a valid integer, unable to parse string as an integer",
            "input": "x",
        }
    ]


def test_dict_dict() -> None:
    class Model(BaseModel):
        v: dict[str, int]

    assert Model(v={"foo": 1}).model_dump() == {"v": {"foo": 1}}


def test_none_list() -> None:
    class Model(BaseModel):
        v: list[None] = [None]

    assert Model.model_json_schema() == {
        "title": "Model",
        "type": "object",
        "properties": {
            "v": {
                "title": "V",
                "default": [None],
                "type": "array",
                "items": {"type": "null"},
            }
        },
    }


@pytest.mark.parametrize(
    "value,result",
    [
        ({"a": 2, "b": 4}, {"a": 2, "b": 4}),
        ({b"a": "2", "b": 4}, {"a": 2, "b": 4}),
    ],
)
def test_typed_dict(value: dict, result: dict) -> None:
    class Model(BaseModel):
        v: dict[str, int]

    assert Model(v=value).v == result


@pytest.mark.parametrize(
    "value,errors",
    [
        (
            1,
            [
                {
                    "type": "dict_type",
                    "loc": ("v",),
                    "msg": "Input should be a valid dictionary",
                    "input": 1,
                }
            ],
        ),
        (
            {"a": "b"},
            [
                {
                    "type": "int_parsing",
                    "loc": ("v", "a"),
                    "msg": "Input should be a valid integer, unable to parse string as an integer",
                    "input": "b",
                }
            ],
        ),
        (
            [1, 2, 3],
            [
                {
                    "type": "dict_type",
                    "loc": ("v",),
                    "msg": "Input should be a valid dictionary",
                    "input": [1, 2, 3],
                }
            ],
        ),
    ],
)
def test_typed_dict_error(value: Any, errors: list[dict]) -> None:
    class Model(BaseModel):
        v: dict[str, int]

    with pytest.raises(ValidationError) as exc_info:
        Model(v=value)
    assert exc_info.value.errors(include_url=False) == errors


def test_dict_key_error() -> None:
    class Model(BaseModel):
        v: dict[int, int]

    assert Model(v={1: 2, "3": "4"}).v == {1: 2, 3: 4}
    with pytest.raises(ValidationError) as exc_info:
        Model(v={"foo": 2, "3": "4"})
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "int_parsing",
            "loc": ("v", "foo", "[key]"),
            "msg": "Input should be a valid integer, unable to parse string as an integer",
            "input": "foo",
        }
    ]


def test_tuple() -> None:
    class Model(BaseModel):
        v: tuple[int, float, bool]

    m = Model(v=["1.0", "2.2", "true"])
    assert m.v == (1, 2.2, True)


def test_tuple_more() -> None:
    class Model(BaseModel):
        simple_tuple: tuple[int, ...]
        tuple_of_different_types: tuple[int, float, str, bool]
        tuple_of_single_tuples: tuple[tuple[int], ...]

    m = Model(
        empty_tuple=[],
        simple_tuple=[1, 2, 3, 4],
        tuple_of_different_types=[4, 3.1, "str", 1],
        tuple_of_single_tuples=(("1",), (2,)),
    )
    assert m.model_dump() == {
        "empty_tuple": (),
        "simple_tuple": (1, 2, 3, 4),
        "tuple_of_different_types": (4, 3.1, "str", True),
        "tuple_of_single_tuples": ((1,), (2,)),
    }


@pytest.mark.parametrize(
    "dict_cls,frozenset_cls,list_cls,set_cls,tuple_cls,type_cls",
    [
        (typing.Dict, typing.FrozenSet, typing.List, typing.Set, typing.Tuple, typing.Type),
        (dict, frozenset, list, set, tuple, type),
    ],
)
def test_pep585_generic_types(
    dict_cls: type,
    frozenset_cls: type,
    list_cls: type,
    set_cls: type,
    tuple_cls: type,
    type_cls: type,
) -> None:
    class Type1:
        pass

    class Type2:
        pass

    class Model(BaseModel, arbitrary_types_allowed=True):
        a: dict_cls[str, int]
        a1: dict_cls[str, int]
        b: frozenset_cls[int]
        b1: frozenset_cls[int]
        c: list_cls[int]
        c1: list_cls[int]
        d: set_cls[int]
        d1: set_cls[int]
        e: tuple_cls[int, ...]
        e1: tuple_cls[int, ...]
        e2: tuple_cls[int, ...]
        e3: tuple_cls[()]
        f: type_cls[Type1]
        f1: type_cls[Type1]

    default_model_kwargs = dict(
        a={},
        a1={"a": "1"},
        b=[],
        b1=("1",),
        c=[],
        c1=("1",),
        d=[],
        d1=["1"],
        e=[],
        e1=["1"],
        e2=["1", "2"],
        e3=[],
        f=Type1,
        f1=Type1,
    )
    m = Model(**default_model_kwargs)
    assert m.a == {}
    assert m.a1 == {"a": 1}
    assert m.b == frozenset()
    assert m.b1 == frozenset({1})
    assert m.c == []
    assert m.c1 == [1]
    assert m.d == set()
    assert m.d1 == {1}
    assert m.e == ()
    assert m.e1 == (1,)
    assert m.e2 == (1, 2)
    assert m.e3 == ()
    assert m.f == Type1
    assert m.f1 == Type1
    with pytest.raises(ValidationError) as exc_info:
        Model(**{**default_model_kwargs, "e3": (1,)})
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "too_long",
            "loc": ("e3",),
            "msg": "Tuple should have at most 0 items after validation, not 1",
            "input": (1,),
            "ctx": {"field_type": "Tuple", "max_length": 0, "actual_length": 1},
        }
    ]
    Model(**{**default_model_kwargs, "f": Type2})
    with pytest.raises(ValidationError) as exc_info:
        Model(**{**default_model_kwargs, "f1": Type2})
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "is_subclass_of",
            "loc": ("f1",),
            "msg": "Input should be a subclass of test_pep585_generic_types.<locals>.Type1",
            "input": HasRepr(IsStr(regex=".+\\.Type2'>")),
            "ctx": {"class": "test_pep585_generic_types.<locals>.Type1"},
        }
    ]


def test_tuple_length_error() -> None:
    class Model(BaseModel):
        v: tuple[int, int, int]
        w: tuple[()]

    with pytest.raises(ValidationError) as exc_info:
        Model(v=[1, 2], w=[1])
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "missing",
            "loc": ("v", 2),
            "msg": "Field required",
            "input": [1, 2],
        },
        {
            "type": "too_long",
            "loc": ("w",),
            "msg": "Tuple should have at most 0 items after validation, not 1",
            "input": [1],
            "ctx": {"field_type": "Tuple", "max_length": 0, "actual_length": 1},
        },
    ]


def test_tuple_invalid() -> None:
    class Model(BaseModel):
        v: tuple[int, float, Decimal]

    with pytest.raises(ValidationError) as exc_info:
        Model(v="xxx")
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "tuple_type",
            "loc": ("v",),
            "msg": "Input should be a valid tuple",
            "input": "xxx",
        }
    ]


def test_tuple_value_error() -> None:
    class Model(BaseModel):
        v: tuple[int, float, Decimal]

    with pytest.raises(ValidationError) as exc_info:
        Model(v=["x", "y", "x"])
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "int_parsing",
            "loc": ("v", 0),
            "msg": "Input should be a valid integer, unable to parse string as an integer",
            "input": "x",
        },
        {
            "type": "float_parsing",
            "loc": ("v", 1),
            "msg": "Input should be a valid number, unable to parse string as a number",
            "input": "y",
        },
        {
            "type": "decimal_parsing",
            "loc": ("v", 2),
            "msg": "Input should be a valid decimal",
            "input": "x",
        },
    ]


def test_recursive_list() -> None:
    class SubModel(BaseModel):
        name: str
        count: Optional[int] = None

    class Model(BaseModel):
        v: list[SubModel]

    m = Model(v=[])
    assert m.v == []
    m = Model(v=[{"name": "testing", "count": 4}])
    assert repr(m) == "Model(v=[SubModel(name='testing', count=4)])"
    assert m.v[0].name == "testing"
    assert m.v[0].count == 4
    assert m.model_dump() == {"v": [{"count": 4, "name": "testing"}]}
    with pytest.raises(ValidationError) as exc_info:
        Model(v=["x"])
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "model_type",
            "loc": ("v", 0),
            "msg": "Input should be a valid dictionary or instance of SubModel",
            "input": "x",
            "ctx": {"class_name": "SubModel"},
        }
    ]


def test_recursive_list_error() -> None:
    class SubModel(BaseModel):
        name: str
        count: Optional[int] = None

    class Model(BaseModel):
        v: list[SubModel]

    with pytest.raises(ValidationError) as exc_info:
        Model(v=[{}])
    assert exc_info.value.errors(include_url=False) == [
        {"input": {}, "loc": ("v", 0, "name"), "msg": "Field required", "type": "missing"}
    ]


def test_list_unions() -> None:
    class Model(BaseModel):
        v: list[Union[int, str]]

    assert Model(v=[123, "456", "foobar"]).v == [123, "456", "foobar"]
    with pytest.raises(ValidationError) as exc_info:
        Model(v=[1, 2, None])
    errors = exc_info.value.errors(include_url=False)
    expected_errors = [
        {
            "input": None,
            "loc": ("v", 2, "int"),
            "msg": "Input should be a valid integer",
            "type": "int_type",
        },
        {
            "input": None,
            "loc": ("v", 2, "str"),
            "msg": "Input should be a valid string",
            "type": "string_type",
        },
    ]
    assert sorted(errors, key=str) == sorted(expected