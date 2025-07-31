#!/usr/bin/env python3
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import pickle
import pytest
from pydantic_core import CoreSchema
from pydantic_core.core_schema import SerializerFunctionWrapHandler
from pydantic import (
    Base64Str,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    PydanticDeprecatedSince20,
    PydanticUserError,
    RootModel,
    ValidationError,
    field_serializer,
    model_validator,
)


def parametrize_root_model() -> Any:
    class InnerModel(BaseModel):
        pass

    return pytest.mark.parametrize(
        ("root_type", "root_value", "dump_value"),
        [
            pytest.param(int, 42, 42, id="int"),
            pytest.param(str, "forty two", "forty two", id="str"),
            pytest.param(dict[int, bool], {1: True, 2: False}, {1: True, 2: False}, id="dict[int, bool]"),
            pytest.param(list[int], [4, 2, -1], [4, 2, -1], id="list[int]"),
            pytest.param(InnerModel, InnerModel(int_field=42, str_field="forty two"), {"int_field": 42, "str_field": "forty two"}, id="InnerModel"),
        ],
    )


def check_schema(schema: Dict[str, Any]) -> None:
    assert schema["type"] == "model"
    assert schema["root_model"] is True
    assert schema["custom_init"] is False


@parametrize_root_model()
def test_root_model_specialized(root_type: Type[Any], root_value: Any, dump_value: Any) -> None:
    Model: Type[RootModel[Any]] = RootModel[root_type]
    check_schema(Model.__pydantic_core_schema__)
    m: RootModel[Any] = Model(root_value)
    assert m.model_dump() == dump_value
    assert dict(m) == {"root": m.root}
    assert m.__pydantic_fields_set__ == {"root"}


@parametrize_root_model()
def test_root_model_inherited(root_type: Type[Any], root_value: Any, dump_value: Any) -> None:
    class Model(RootModel[root_type]):  # type: ignore[type-var]
        pass

    check_schema(Model.__pydantic_core_schema__)
    m: RootModel[Any] = Model(root_value)
    assert m.model_dump() == dump_value
    assert dict(m) == {"root": m.root}
    assert m.__pydantic_fields_set__ == {"root"}


def test_root_model_validation_error() -> None:
    Model: Type[RootModel[int]] = RootModel[int]
    with pytest.raises(ValidationError) as e:
        Model("forty two")
    assert e.value.errors(include_url=False) == [
        {
            "input": "forty two",
            "loc": (),
            "msg": "Input should be a valid integer, unable to parse string as an integer",
            "type": "int_parsing",
        }
    ]


def test_root_model_repr() -> None:
    SpecializedRootModel: Type[RootModel[int]] = RootModel[int]

    class SubRootModel(RootModel):
        pass

    class SpecializedSubRootModel(RootModel[int]):
        pass

    assert repr(SpecializedRootModel(1)) == "RootModel[int](root=1)"
    assert repr(SubRootModel(1)) == "SubRootModel(root=1)"
    assert repr(SpecializedSubRootModel(1)) == "SpecializedSubRootModel(root=1)"


def test_root_model_recursive() -> None:
    class A(RootModel[List["B"]]):
        def my_a_method(self) -> None:
            pass

    class B(RootModel[Dict[str, Optional[A]]]):
        def my_b_method(self) -> None:
            pass

    assert repr(A.model_validate([{}])) == "A(root=[B(root={})])"


def test_root_model_nested() -> None:
    calls: List[Any] = []

    class B(RootModel[int]):
        def my_b_method(self) -> None:
            calls.append(("my_b_method", self.root))

    class A(RootModel[B]):
        def my_a_method(self) -> None:
            calls.append(("my_a_method", self.root.root))

    m1: RootModel[Any] = A.model_validate(1)
    m1.my_a_method()
    m1.root.my_b_method()
    assert calls == [("my_a_method", 1), ("my_b_method", 1)]
    calls.clear()
    m2: RootModel[Any] = A.model_validate_json("2")
    m2.my_a_method()
    m2.root.my_b_method()
    assert calls == [("my_a_method", 2), ("my_b_method", 2)]


def test_root_model_as_field() -> None:
    class MyRootModel(RootModel[int]):
        pass

    class MyModel(BaseModel):
        root_model: Any  # type: ignore

    m: MyModel = MyModel.model_validate({"root_model": 1})
    assert isinstance(m.root_model, MyRootModel)


def test_v1_compatibility_serializer() -> None:
    class MyInnerModel(BaseModel):
        pass

    class MyRootModel(RootModel[MyInnerModel]):
        @field_serializer("root", mode="wrap")
        def embed_in_dict(self, v: Any, handler: SerializerFunctionWrapHandler) -> Dict[str, Any]:
            return {"__root__": handler(v)}

    class MyOuterModel(BaseModel):
        my_root: Any

    m: MyOuterModel = MyOuterModel.model_validate({"my_root": {"x": 1}})
    assert m.model_dump() == {"my_root": {"__root__": {"x": 1}}}
    with pytest.warns(PydanticDeprecatedSince20):
        assert m.dict() == {"my_root": {"__root__": {"x": 1}}}


def test_construct() -> None:
    class Base64Root(RootModel[Base64Str]):
        pass

    v: RootModel[Base64Str] = Base64Root.model_construct("test")
    assert v.model_dump() == "dGVzdA=="


def test_construct_nested() -> None:
    class Base64RootProperty(BaseModel):
        data: Any

    v: Base64RootProperty = Base64RootProperty.model_construct(data=RootModel[Base64Str].model_construct("test"))
    assert v.model_dump() == {"data": "dGVzdA=="}
    v = Base64RootProperty.model_construct(data="test")
    assert isinstance(v.data, str)
    with pytest.raises(AttributeError, match="'str' object has no attribute 'root'"):
        v.model_dump()


def test_assignment() -> None:
    Model: Type[RootModel[int]] = RootModel[int]
    m: RootModel[int] = Model(1)
    assert m.model_fields_set == {"root"}
    assert m.root == 1
    m.root = 2
    assert m.root == 2


def test_validate_assignment_false() -> None:
    Model: Type[RootModel[int]] = RootModel[int]
    m: RootModel[int] = Model(42)
    m.root = "abc"  # type: ignore
    assert m.root == "abc"


def test_validate_assignment_true() -> None:
    class Model(RootModel[int]):
        model_config = ConfigDict(validate_assignment=True)

    m: RootModel[int] = Model(42)
    with pytest.raises(ValidationError) as e:
        m.root = "abc"  # type: ignore
    assert e.value.errors(include_url=False) == [
        {
            "input": "abc",
            "loc": (),
            "msg": "Input should be a valid integer, unable to parse string as an integer",
            "type": "int_parsing",
        }
    ]


def test_root_model_literal() -> None:
    assert RootModel[int](42).root == 42


def test_root_model_equality() -> None:
    assert RootModel[int](42) == RootModel[int](42)
    assert RootModel[int](42) != RootModel[int](7)
    assert RootModel[int](42) != RootModel[float](42)
    assert RootModel[int](42) == RootModel[int].model_construct(42)


def test_root_model_with_private_attrs_equality() -> None:
    class Model(RootModel[int]):
        _private_attr: Any = PrivateAttr(default="abc")

    m: RootModel[int] = Model(42)
    assert m == Model(42)
    m._private_attr = "xyz"
    assert m != Model(42)


def test_root_model_nested_equality() -> None:
    class Model(BaseModel):
        value: Any

    assert Model(value=42).value == RootModel[int](42)


def test_root_model_base_model_equality() -> None:
    class R(RootModel[int]):
        pass

    class B(BaseModel):
        root: Any

    assert R(42) != B(root=42)
    assert B(root=42) != R(42)


@pytest.mark.parametrize("extra_value", ["ignore", "allow", "forbid"])
def test_extra_error(extra_value: str) -> None:
    with pytest.raises(PydanticUserError, match="extra"):

        class Model(RootModel[int]):
            model_config = ConfigDict(extra=extra_value)


def test_root_model_default_value() -> None:
    class Model(RootModel):
        root: Any = 42

    m: RootModel = Model()
    assert m.root == 42
    assert m.model_dump() == 42
    assert m.__pydantic_fields_set__ == set()


def test_root_model_default_factory() -> None:
    class Model(RootModel):
        root: Any = Field(default_factory=lambda: 42)

    m: RootModel = Model()
    assert m.root == 42
    assert m.model_dump() == 42
    assert m.__pydantic_fields_set__ == set()


def test_root_model_wrong_default_value_without_validate_default() -> None:
    class Model(RootModel):
        root: Any = "42"

    assert Model().root == "42"


def test_root_model_default_value_with_validate_default() -> None:
    class Model(RootModel):
        model_config = ConfigDict(validate_default=True)
        root: Any = "42"

    m: RootModel = Model()
    assert m.root == 42
    assert m.model_dump() == 42
    assert m.__pydantic_fields_set__ == set()


def test_root_model_default_value_with_validate_default_on_field() -> None:
    class Model(RootModel):
        pass

    m: RootModel = Model()
    assert m.root == 42
    assert m.model_dump() == 42
    assert m.__pydantic_fields_set__ == set()


def test_root_model_as_attr_with_validate_default() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(validate_default=True)
        rooted_value: Any = 42

    m: BaseModel = Model()
    assert m.rooted_value == RootModel[int](42)
    assert m.model_dump() == {"rooted_value": 42}
    assert m.rooted_value.__pydantic_fields_set__ == {"root"}


def test_root_model_in_root_model_default() -> None:
    class Nested(RootModel):
        root: Any = 42

    class Model(RootModel):
        root: Any = Nested()

    m: RootModel = Model()
    assert m.root.root == 42
    assert m.__pydantic_fields_set__ == set()
    assert m.root.__pydantic_fields_set__ == set()


def test_nested_root_model_naive_default() -> None:
    class Nested(RootModel):
        root: Any = 42

    class Model(BaseModel):
        value: Any

    m: Model = Model(value=Nested())
    assert m.value.root == 42
    assert m.value.__pydantic_fields_set__ == set()


def test_nested_root_model_proper_default() -> None:
    class Nested(RootModel):
        root: Any = 42

    class Model(BaseModel):
        value: Any = Field(default_factory=Nested)

    m: Model = Model()
    assert m.value.root == 42
    assert m.value.__pydantic_fields_set__ == set()


def test_root_model_json_schema_meta() -> None:
    ParametrizedModel: Type[RootModel[int]] = RootModel[int]

    class SubclassedModel(RootModel):
        """Subclassed Model docstring"""

    parametrized_json_schema: Dict[str, Any] = ParametrizedModel.model_json_schema()
    subclassed_json_schema: Dict[str, Any] = SubclassedModel.model_json_schema()
    assert parametrized_json_schema.get("title") == "RootModel[int]"
    assert parametrized_json_schema.get("description") is None
    assert subclassed_json_schema.get("title") == "SubclassedModel"
    assert subclassed_json_schema.get("description") == "Subclassed Model docstring"


@pytest.mark.parametrize("order", ["BR", "RB"])
def test_root_model_dump_with_base_model(order: str) -> None:
    class BModel(BaseModel):
        pass

    class RModel(RootModel):
        pass

    if order == "BR":

        class Model(RootModel):
            pass

    elif order == "RB":

        class Model(RootModel):
            pass

    m: RootModel[Any] = Model([1, 2, {"value": "abc"}])
    assert m.root == [RModel(1), RModel(2), BModel.model_construct(value="abc")]
    assert m.model_dump() == [1, 2, {"value": "abc"}]
    assert m.model_dump_json() == '[1,2,{"value":"abc"}]'


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"kind": "IModel", "int_value": 42}, id="IModel"),
        pytest.param({"kind": "SModel", "str_value": "abc"}, id="SModel"),
    ],
)
def test_mixed_discriminated_union(data: Dict[str, Any]) -> None:
    class IModel(BaseModel):
        pass

    class RModel(RootModel):
        pass

    class SModel(BaseModel):
        pass

    class Model(RootModel):
        root: Any = Field(discriminator="kind")

    if data["kind"] == "IModel":
        with pytest.warns(UserWarning, match="Failed to get discriminator value for tagged union serialization"):
            assert Model(data).model_dump() == data
            assert Model(**data).model_dump() == data
    else:
        assert Model(data).model_dump() == data
        assert Model(**data).model_dump() == data


def test_list_rootmodel() -> None:
    class A(BaseModel):
        pass

    class B(BaseModel):
        pass

    class D(RootModel[Union[A, B]]):
        pass

    LD: Type[RootModel[List[D]]] = RootModel[list[D]]  # type: ignore
    obj: RootModel[List[D]] = LD.model_validate([{"type": "a", "a": "a"}, {"type": "b", "b": "b"}])
    assert obj.model_dump() == [{"type": "a", "a": "a"}, {"type": "b", "b": "b"}]


def test_root_and_data_error() -> None:
    class BModel(BaseModel):
        pass

    Model: Type[RootModel[BModel]] = RootModel[BModel]
    with pytest.raises(ValueError, match='"RootModel.__init__" accepts either a single positional argument or arbitrary keyword arguments'):
        Model({"value": 42}, other_value="abc")


def test_pickle_root_model(create_module: Callable[[Any], Any]) -> None:
    @create_module
    def module() -> None:
        from pydantic import RootModel

        class MyRootModel(RootModel[str]):
            pass

    MyRootModel: Any = module.MyRootModel
    assert MyRootModel(root="abc") == pickle.loads(pickle.dumps(MyRootModel(root="abc")))


def test_json_schema_extra_on_model() -> None:
    class Model(RootModel):
        model_config = ConfigDict(json_schema_extra={"schema key": "schema value"})

    assert Model.model_json_schema() == {"schema key": "schema value", "title": "Model", "type": "string"}


def test_json_schema_extra_on_field() -> None:
    class Model(RootModel):
        root: Any = Field(json_schema_extra={"schema key": "schema value"})

    assert Model.model_json_schema() == {"schema key": "schema value", "title": "Model", "type": "string"}


def test_json_schema_extra_on_model_and_on_field() -> None:
    class Model(RootModel):
        model_config = ConfigDict(json_schema_extra={"schema key on model": "schema value on model"})
        root: Any = Field(json_schema_extra={"schema key on field": "schema value on field"})

    with pytest.raises(ValueError, match="json_schema_extra.*?must not be set simultaneously"):
        Model.model_json_schema()


def test_help(create_module: Callable[[Any], Any]) -> None:
    module = create_module(
        "\nimport pydoc\n\nfrom pydantic import RootModel\n\n\nhelp_result_string = pydoc.render_doc(RootModel)\n"
    )
    assert "class RootModel" in module.help_result_string


def test_copy_preserves_equality() -> None:
    model: RootModel = RootModel()
    copied: RootModel = model.__copy__()
    assert model == copied
    deepcopied: RootModel = model.__deepcopy__()
    assert model == deepcopied


@pytest.mark.parametrize(
    "root_type,input_value,expected,raises_match,strict",
    [
        (bool, "true", True, None, False),
        (bool, "true", True, None, True),
        (bool, "false", False, None, False),
        (bool, "e", ValidationError, "type=bool_parsing", False),
        (int, "1", 1, None, False),
        (int, "1", 1, None, True),
        (int, "xxx", ValidationError, "type=int_parsing", True),
        (float, "1.1", 1.1, None, False),
        (float, "1.10", 1.1, None, False),
        (float, "1.1", 1.1, None, True),
        (float, "1.10", 1.1, None, True),
        (date, "2017-01-01", date(2017, 1, 1), None, False),
        (date, "2017-01-01", date(2017, 1, 1), None, True),
        (date, "2017-01-01T12:13:14.567", ValidationError, "type=date_from_datetime_inexact", False),
        (date, "2017-01-01T12:13:14.567", ValidationError, "type=date_parsing", True),
        (date, "2017-01-01T00:00:00", date(2017, 1, 1), None, False),
        (date, "2017-01-01T00:00:00", ValidationError, "type=date_parsing", True),
        (datetime, "2017-01-01T12:13:14.567", datetime(2017, 1, 1, 12, 13, 14, 567000), None, False),
        (datetime, "2017-01-01T12:13:14.567", datetime(2017, 1, 1, 12, 13, 14, 567000), None, True),
    ],
    ids=repr,
)
def test_model_validate_strings(
    root_type: Type[Any],
    input_value: Any,
    expected: Any,
    raises_match: Optional[str],
    strict: bool,
) -> None:
    Model: Type[RootModel[Any]] = RootModel[root_type]
    if raises_match is not None:
        with pytest.raises(expected, match=raises_match):
            Model.model_validate_strings(input_value, strict=strict)
    else:
        assert Model.model_validate_strings(input_value, strict=strict).root == expected


def test_model_construction_with_invalid_generic_specification() -> None:
    T_ = TypeVar("T_", bound=BaseModel)
    with pytest.raises(TypeError, match="You should parametrize RootModel directly"):

        class GenericRootModel(RootModel, Generic[T_]):  # type: ignore
            pass


def test_model_with_field_description() -> None:
    class AModel(RootModel):
        root: Any = Field(description="abc")

    assert AModel.model_json_schema() == {"title": "AModel", "type": "integer", "description": "abc"}


def test_model_with_both_docstring_and_field_description() -> None:
    """Check if the docstring is used as the description when both are present."""
    class AModel(RootModel):
        """More detailed description"""
        root: Any = Field(description="abc")

    assert AModel.model_json_schema() == {"title": "AModel", "type": "integer", "description": "More detailed description"}