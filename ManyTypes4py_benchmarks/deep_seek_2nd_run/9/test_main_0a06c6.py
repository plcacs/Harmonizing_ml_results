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

    m = Model(a=10.2)
    assert m.a == 10.2
    assert m.b == 10


@pytest.fixture(name="UltraSimpleModel", scope="session")
def ultra_simple_model_fixture() -> type[BaseModel]:
    class UltraSimpleModel(BaseModel):
        b: int = 10

    return UltraSimpleModel


def test_ultra_simple_missing(UltraSimpleModel: type[BaseModel]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        UltraSimpleModel()
    assert exc_info.value.errors(include_url=False) == [
        {
            "loc": ("a",),
            "msg": "Field required",
            "type": "missing",
            "input": {},
        }
    ]
    assert (
        str(exc_info.value)
        == "1 validation error for UltraSimpleModel\na\n  Field required [type=missing, input_value={}, input_type=dict]"
    )


def test_ultra_simple_failed(UltraSimpleModel: type[BaseModel]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        UltraSimpleModel(a="x", b="x")
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "float_parsing",
            "loc": ("a",),
            "msg": "Input should be a valid number, unable to parse string as a number",
            "input": "x",
        },
        {
            "type": "int_parsing",
            "loc": ("b",),
            "msg": "Input should be a valid integer, unable to parse string as an integer",
            "input": "x",
        },
    ]


def test_ultra_simple_repr(UltraSimpleModel: type[BaseModel]) -> None:
    m = UltraSimpleModel(a=10.2)
    assert str(m) == "a=10.2 b=10"
    assert repr(m) == "UltraSimpleModel(a=10.2, b=10)"
    assert (
        repr(UltraSimpleModel.model_fields["a"])
        == "FieldInfo(annotation=float, required=True)"
    )
    assert (
        repr(UltraSimpleModel.model_fields["b"])
        == "FieldInfo(annotation=int, required=False, default=10)"
    )
    assert dict(m) == {"a": 10.2, "b": 10}
    assert m.model_dump() == {"a": 10.2, "b": 10}
    assert m.model_dump_json() == '{"a":10.2,"b":10}'
    assert str(m) == "a=10.2 b=10"


def test_recursive_repr() -> None:
    class A(BaseModel):
        a: Optional["A"] = None

    class B(BaseModel):
        a: Optional[A] = None

    a = A()
    a.a = a
    b = B(a=a)
    assert re.match(r"B\(a=A\(a='<Recursion on A with id=\d+>'\)\)", repr(b)) is not None


def test_default_factory_field() -> None:
    def myfunc() -> int:
        return 1

    class Model(BaseModel):
        a: int = Field(default_factory=myfunc)

    m = Model()
    assert str(m) == "a=1"
    assert (
        repr(Model.model_fields["a"])
        == "FieldInfo(annotation=int, required=False, default_factory=myfunc)"
    )
    assert dict(m) == {"a": 1}
    assert m.model_dump_json() == '{"a":1}'


def test_comparing(UltraSimpleModel: type[BaseModel]) -> None:
    m = UltraSimpleModel(a=10.2, b="100")
    assert m.model_dump() == {"a": 10.2, "b": 100}
    assert m != {"a": 10.2, "b": 100}
    assert m == UltraSimpleModel(a=10.2, b=100)


@pytest.fixture(scope="session", name="NoneCheckModel")
def none_check_model_fix() -> type[BaseModel]:
    class NoneCheckModel(BaseModel):
        existing_str_value: str = "foo"
        required_str_value: str
        required_str_none_value: Optional[str]
        existing_bytes_value: bytes = b"foo"
        required_bytes_value: bytes
        required_bytes_none_value: Optional[bytes]

    return NoneCheckModel


def test_nullable_strings_success(NoneCheckModel: type[BaseModel]) -> None:
    m = NoneCheckModel(
        required_str_value="v1",
        required_str_none_value=None,
        required_bytes_value="v2",
        required_bytes_none_value=None,
    )
    assert m.required_str_value == "v1"
    assert m.required_str_none_value is None
    assert m.required_bytes_value == b"v2"
    assert m.required_bytes_none_value is None


def test_nullable_strings_fails(NoneCheckModel: type[BaseModel]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        NoneCheckModel(
            required_str_value=None,
            required_str_none_value=None,
            required_bytes_value=None,
            required_bytes_none_value=None,
        )
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "string_type",
            "loc": ("required_str_value",),
            "msg": "Input should be a valid string",
            "input": None,
        },
        {
            "type": "bytes_type",
            "loc": ("required_bytes_value",),
            "msg": "Input should be a valid bytes",
            "input": None,
        },
    ]


@pytest.fixture(name="ParentModel", scope="session")
def parent_sub_model_fixture() -> type[BaseModel]:
    class UltraSimpleModel(BaseModel):
        b: int = 10

    class ParentModel(BaseModel):
        pass

    return ParentModel


def test_parent_sub_model(ParentModel: type[BaseModel]) -> None:
    m = ParentModel(grape=1, banana={"a": 1})
    assert m.grape is True
    assert m.banana.a == 1.0
    assert m.banana.b == 10
    assert repr(m) == "ParentModel(grape=True, banana=UltraSimpleModel(a=1.0, b=10))"


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
            "type": "float_type",
            "loc": ("a",),
            "msg": "Input should be a valid number",
            "input": None,
        }
    ]


def test_allow_extra() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(extra="allow")

    m = Model(a="10.2", b=12)
    assert m.__dict__ == {"a": 10.2}
    assert m.__pydantic_extra__ == {"b": 12}
    assert m.a == 10.2
    assert m.b == 12
    assert m.model_extra == {"b": 12}
    m.c = 42
    assert "c" not in m.__dict__
    assert m.__pydantic_extra__ == {"b": 12, "c": 42}
    assert m.model_dump() == {"a": 10.2, "b": 12, "c": 42}


@pytest.mark.parametrize("extra", ["ignore", "forbid", "allow"])
def test_allow_extra_from_attributes(extra: str) -> None:
    class Model(BaseModel):
        model_config = ConfigDict(extra=extra, from_attributes=True)

    class TestClass:
        a = 1.0
        b = 12

    m = Model.model_validate(TestClass())
    assert m.a == 1.0
    assert not hasattr(m, "b")


def test_allow_extra_repr() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(extra="allow")
        a: float

    assert str(Model(a="10.2", b=12)) == "a=10.2 b=12"


def test_forbidden_extra_success() -> None:
    class ForbiddenExtra(BaseModel):
        model_config = ConfigDict(extra="forbid")
        foo: str = "whatever"

    m = ForbiddenExtra()
    assert m.foo == "whatever"


def test_forbidden_extra_fails() -> None:
    class ForbiddenExtra(BaseModel):
        model_config = ConfigDict(extra="forbid")
        foo: str = "whatever"

    with pytest.raises(ValidationError) as exc_info:
        ForbiddenExtra(foo="ok", bar="wrong", spam="xx")
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "extra_forbidden",
            "loc": ("bar",),
            "msg": "Extra inputs are not permitted",
            "input": "wrong",
        },
        {
            "type": "extra_forbidden",
            "loc": ("spam",),
            "msg": "Extra inputs are not permitted",
            "input": "xx",
        },
    ]


def test_assign_extra_no_validate() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(validate_assignment=True)

    model = Model(a=0.2)
    with pytest.raises(ValidationError, match="b\\s+Object has no attribute 'b'"):
        model.b = 2


def test_assign_extra_validate() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(validate_assignment=True)

    model = Model(a=0.2)
    with pytest.raises(ValidationError, match="b\\s+Object has no attribute 'b'"):
        model.b = 2


def test_model_property_attribute_error() -> None:
    class Model(BaseModel):
        @property
        def a_property(self) -> None:
            raise AttributeError("Internal Error")

    with pytest.raises(AttributeError, match="Internal Error"):
        Model().a_property


def test_extra_allowed() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(extra="allow")

    model = Model(a=0.2, b=0.1)
    assert model.b == 0.1
    assert not hasattr(model, "c")
    model.c = 1
    assert hasattr(model, "c")
    assert model.c == 1


def test_reassign_instance_method_with_extra_allow() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(extra="allow")

        def not_extra_func(self) -> str:
            return f"hello {self.name}"

    def not_extra_func_replacement(self_sub: Any) -> str:
        return f"hi {self_sub.name}"

    m = Model(name="james")
    assert m.not_extra_func() == "hello james"
    m.not_extra_func = partial(not_extra_func_replacement, m)
    assert m.not_extra_func() == "hi james"
    assert "not_extra_func" in m.__dict__


def test_extra_ignored() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(extra="ignore")

    model = Model(a=0.2, b=0.1)
    assert not hasattr(model, "b")
    with pytest.raises(ValueError, match='"Model" object has no field "b"'):
        model.b = 1
    assert model.model_extra is None


def test_field_order_is_preserved_with_extra() -> None:
    """This test covers https://github.com/pydantic/pydantic/issues/1234."""

    class Model(BaseModel):
        model_config = ConfigDict(extra="allow")

    model = Model(a=1, b="2", c=3.0, d=4)
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
            object.__setattr__(self, "__pydantic_extra__", None)

    class Model(BrokenExtraBaseModel):
        model_config = ConfigDict(extra="allow")

    m = Model(extra_field="some extra value")
    with pytest.raises(AttributeError) as e:
        m.extra_field
    assert e.value.args == ("'Model' object has no attribute 'extra_field'",)


def test_model_extra_is_none_when_extra_is_forbid() -> None:
    class Foo(BaseModel):
        model_config = ConfigDict(extra="forbid")

    assert Foo().model_extra is None


def test_set_attr(UltraSimpleModel: type[BaseModel]) -> None:
    m = UltraSimpleModel(a=10.2)
    assert m.model_dump() == {"a": 10.2, "b": 10}
    m.b = 20
    assert m.model_dump() == {"a": 10.2, "b": 20}


def test_set_attr_invalid() -> None:
    class UltraSimpleModel(BaseModel):
        a: float
        b: int = 10

    m = UltraSimpleModel(a=10.2)
    assert m.model_dump() == {"a": 10.2, "b": 10}
    with pytest.raises(ValueError) as exc_info:
        m.c = 20
    assert '"UltraSimpleModel" object has no field "c"' in exc_info.value.args[0]


def test_any() -> None:
    class AnyModel(BaseModel):
        a: Any = 10
        b: Any = 20

    m = AnyModel()
    assert m.a == 10
    assert m.b == 20
    m = AnyModel(a="foobar", b="barfoo")
    assert m.a == "foobar"
    assert m.b == "barfoo"


def test_population_by_field_name() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        a: str = Field(alias="_a")

    assert Model(a="different").a == "different"
    assert Model(a="different").model_dump() == {"a": "different"}
    assert Model(a="different").model_dump(by_alias=True) == {"_a": "different"}


def test_field_order() -> None:
    class Model(BaseModel):
        b: int = 10
        d: dict = {}

    assert list(Model.model_fields.keys()) == ["c", "b", "a", "d"]


def test_required() -> None:
    class Model(BaseModel):
        b: int = 10

    m = Model(a=10.2)
    assert m.model_dump() == dict(a=10.2, b=10)
    with pytest.raises(ValidationError) as exc_info:
        Model()
    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "missing",
            "loc": ("a",),
            "msg": "Field required",
            "input": {},
        }
    ]


def test_mutability() -> None:
    class TestModel(BaseModel):
        a: int = 10
        model_config = ConfigDict(extra="forbid", frozen=False)

    m = TestModel()
    assert m.a == 10
    m.a = 11
    assert m.a == 11


