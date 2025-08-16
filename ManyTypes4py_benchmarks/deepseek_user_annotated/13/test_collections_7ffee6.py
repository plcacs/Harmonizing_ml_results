import io
import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast

import pydantic
import pytest

from prefect.utilities.annotations import BaseAnnotation, quote
from prefect.utilities.collections import (
    AutoEnum,
    StopVisiting,
    deep_merge,
    deep_merge_dicts,
    dict_to_flatdict,
    flatdict_to_dict,
    get_from_dict,
    isiterable,
    remove_nested_keys,
    set_in_dict,
    visit_collection,
)


class ExampleAnnotation(BaseAnnotation):
    pass


class Color(AutoEnum):
    RED = AutoEnum.auto()
    BLUE = AutoEnum.auto()


class TestAutoEnum:
    def test_autoenum_generates_string_values(self) -> None:
        assert Color.RED.value == "RED"
        assert Color.BLUE.value == "BLUE"

    def test_autoenum_repr(self) -> None:
        assert repr(Color.RED) == str(Color.RED) == "Color.RED"

    def test_autoenum_can_be_json_serialized_with_default_encoder(self) -> None:
        assert json.dumps(Color.RED) == '"RED"'


@pytest.mark.parametrize(
    "d, expected",
    [
        (
            {1: 2},
            {(1,): 2},
        ),
        (
            {1: 2, 2: {1: 2, 3: 4}, 3: {1: 2, 3: {4: 5, 6: {7: 8}}}},
            {
                (1,): 2,
                (2, 1): 2,
                (2, 3): 4,
                (3, 1): 2,
                (3, 3, 4): 5,
                (3, 3, 6, 7): 8,
            },
        ),
        (
            {1: 2, 3: {}, 4: {5: {}}},
            {(1,): 2, (3,): {}, (4, 5): {}},
        ),
    ],
)
def test_flatdict_conversion(d: Dict[Any, Any], expected: Dict[Tuple[Any, ...], Any]) -> None:
    flat = dict_to_flatdict(d)
    assert flat == expected
    assert flatdict_to_dict(flat) == d


def negative_even_numbers(x: Any) -> Any:
    print("Function called on", x)
    if isinstance(x, int) and x % 2 == 0:
        return -x
    return x


def all_negative_numbers(x: Any) -> Any:
    print("Function called on", x)
    if isinstance(x, int):
        return -x
    return x


EVEN: Set[int] = set()


def visit_even_numbers(x: Any) -> Any:
    if isinstance(x, int) and x % 2 == 0:
        EVEN.add(x)
    return x


VISITED: List[Any] = list()


def add_to_visited_list(x: Any) -> Any:
    VISITED.append(x)


@pytest.fixture(autouse=True)
def clear_sets() -> None:
    EVEN.clear()
    VISITED.clear()


@dataclass
class SimpleDataclass:
    x: int
    y: int


class SimplePydantic(pydantic.BaseModel):
    x: int
    y: int


class ExtraPydantic(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    x: int


class PrivatePydantic(pydantic.BaseModel):
    """Pydantic model with private attrs"""

    model_config = pydantic.ConfigDict(extra="forbid")

    x: int
    _y: int  # this is an implicit private attribute
    _z: Any = pydantic.PrivateAttr()  # this is an explicit private attribute


class ImmutablePrivatePydantic(PrivatePydantic):
    model_config = pydantic.ConfigDict(frozen=True)


class PydanticWithDefaults(pydantic.BaseModel):
    name: str
    val: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
    num: int = 0


@dataclass
class Foo:
    x: Any


@dataclass
class Bar:
    y: Any
    z: int = 2


class TestPydanticObjects:
    """
    Checks that the Pydantic test objects defined in this file behave as expected.

    These tests do not cover Prefect functionality and may break if Pydantic introduces
    breaking changes.
    """

    def test_private_pydantic_behaves_as_expected(self) -> None:
        input = PrivatePydantic(x=1)

        # Public attr accessible immediately
        assert input.x == 1

        # Extras not allowed
        with pytest.raises(ValueError):
            input.a = 1

        # Private attrs accessible after setting
        input._y = 4
        input._z = 5
        assert input._y == 4
        assert input._z == 5

    def test_immutable_pydantic_behaves_as_expected(self) -> None:
        input = ImmutablePrivatePydantic(x=1)

        # Public attr accessible immediately
        assert input.x == 1
        # Extras not allowed
        with pytest.raises(ValueError):
            input.a = 1
        # Private attr not accessible until set
        with pytest.raises(AttributeError):
            input._y

        # Private attrs accessible after setting
        input._y = 4
        input._z = 5
        assert input._y == 4
        assert input._z == 5

        # Mutating not allowed because frozen=True
        with pytest.raises(pydantic.ValidationError):
            input.x = 2

        # Can still mutate private attrs
        input._y = 6


class TestVisitCollection:
    @pytest.mark.parametrize(
        "inp,expected",
        [
            (3, 3),
            (4, -4),
            ([3, 4], [3, -4]),
            ((3, 4), (3, -4)),
            ([3, 4, [5, [6]]], [3, -4, [5, [-6]]]),
            ({3: 4, 6: 7}, {3: -4, -6: 7}),
            ({3: [4, {6: 7}]}, {3: [-4, {-6: 7}]}),
            ({3, 4, 5}, {3, -4, 5}),
            (SimpleDataclass(x=1, y=2), SimpleDataclass(x=1, y=-2)),
            (SimplePydantic(x=1, y=2), SimplePydantic(x=1, y=-2)),
            (ExtraPydantic(x=1, y=2, z=3), ExtraPydantic(x=1, y=-2, z=3)),
            (ExampleAnnotation(4), ExampleAnnotation(-4)),
        ],
    )
    def test_visit_collection_and_transform_data(self, inp: Any, expected: Any) -> None:
        result = visit_collection(inp, visit_fn=negative_even_numbers, return_data=True)
        assert result == expected

    @pytest.mark.parametrize(
        "inp,expected",
        [
            (3, set()),
            (4, {4}),
            ([3, 4], {4}),
            ((3, 4), {4}),
            ([3, 4, [5, [6]]], {4, 6}),
            ({3: 4, 6: 7}, {4, 6}),
            ({3: [4, {6: 7}]}, {4, 6}),
            ({3, 4, 5}, {4}),
            (SimpleDataclass(x=1, y=2), {2}),
            (SimplePydantic(x=1, y=2), {2}),
            (ExtraPydantic(x=1, y=2, z=4), {2, 4}),
            (ExtraPydantic(x=1, y=2, z=4).model_copy(), {2, 4}),
            (ExampleAnnotation(4), {4}),
        ],
    )
    def test_visit_collection(self, inp: Any, expected: Set[int]) -> None:
        result = visit_collection(inp, visit_fn=visit_even_numbers, return_data=False)
        assert result is None
        assert EVEN == expected

    def test_visit_collection_does_not_consume_generators(self) -> None:
        def f() -> Any:
            yield from [1, 2, 3]

        result = visit_collection([f()], visit_fn=visit_even_numbers, return_data=False)
        assert result is None
        assert not EVEN

    def test_visit_collection_does_not_consume_generators_when_returning_data(self) -> None:
        def f() -> Any:
            yield from [1, 2, 3]

        val = [f()]
        result = visit_collection(val, visit_fn=visit_even_numbers, return_data=True)
        assert result is val
        assert not EVEN

    @pytest.mark.parametrize(
        "inp,expected",
        [
            ({"x": 1}, [{"x": 1}, "x", 1]),
            (SimpleDataclass(x=1, y=2), [SimpleDataclass(x=1, y=2), 1, 2]),
        ],
    )
    def test_visit_collection_visits_nodes(self, inp: Any, expected: List[Any]) -> None:
        result = visit_collection(inp, visit_fn=add_to_visited_list, return_data=False)
        assert result is None
        assert VISITED == expected

    @pytest.mark.parametrize(
        "inp,expected",
        [
            (sorted([1, 2, 3]), [1, -2, 3]),
            # Not treated as iterators:
            ("test", "test"),
            (b"test", b"test"),
        ],
    )
    def test_visit_collection_iterators(self, inp: Any, expected: Any) -> None:
        result = visit_collection(inp, visit_fn=negative_even_numbers, return_data=True)
        assert result == expected

    @pytest.mark.parametrize(
        "inp",
        [
            io.StringIO("test"),
            io.BytesIO(b"test"),
        ],
    )
    def test_visit_collection_io_iterators(self, inp: Any) -> None:
        result = visit_collection(inp, visit_fn=lambda x: x, return_data=True)
        assert result is inp

    def test_visit_collection_allows_mutation_of_nodes(self) -> None:
        def collect_and_drop_x_from_dicts(node: Any) -> Any:
            add_to_visited_list(node)
            if isinstance(node, dict):
                return {key: value for key, value in node.items() if key != "x"}
            return node

        result = visit_collection(
            {"x": 1, "y": 2}, visit_fn=collect_and_drop_x_from_dicts, return_data=True
        )
        assert result == {"y": 2}
        assert VISITED == [{"x": 1, "y": 2}, "y", 2]

    def test_visit_collection_with_private_pydantic_attributes(self) -> None:
        """
        We should not visit private fields on Pydantic models.
        """
        input = PrivatePydantic(x=2)
        input._y = 3
        input._z = 4

        result = visit_collection(input, visit_fn=visit_even_numbers, return_data=False)
        assert EVEN == {2}, "Only the public field should be visited"
        assert result is None, "Data should not be returned"

        # The original model should not be mutated
        assert input._y == 3
        assert input._z == 4

    def test_visit_collection_includes_unset_pydantic_fields(self) -> None:
        class RandomPydantic(pydantic.BaseModel):
            val: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)

        input_model = RandomPydantic()
        output_model = visit_collection(
            input_model, visit_fn=visit_even_numbers, return_data=True
        )

        assert output_model.val == input_model.val, (
            "The fields value should be used, not the default factory"
        )

    @pytest.mark.parametrize(
        "input",
        [
            {"name": "prefect"},
            {"name": "prefect", "num": 1},
            {"name": "prefect", "num": 1, "val": uuid.UUID(int=0)},
            {"name": "prefect", "val": uuid.UUID(int=0)},
        ],
    )
    def test_visit_collection_remembers_unset_pydantic_fields(self, input: Dict[str, Any]) -> None:
        input_model = PydanticWithDefaults(**input)
        output_model = visit_collection(
            input_model, visit_fn=visit_even_numbers, return_data=True
        )
        assert output_model.model_dump(exclude_unset=True) == input, (
            "Unset fields values should be remembered and preserved"
        )

    @pytest.mark.parametrize("immutable", [True, False])
    def test_visit_collection_mutation_with_private_pydantic_attributes(
        self, immutable: bool
    ) -> None:
        model = ImmutablePrivatePydantic if immutable else PrivatePydantic
        model_instance = model(x=2)
        model_instance._y = 3
        model_instance._z = 4

        result = visit_collection(
            model_instance, visit_fn=negative_even_numbers, return_data=True
        )

        assert isinstance(result, model), "The model should be returned"

        assert result.x == -2, "The public attribute should be modified"

        # Verify that private attributes are retained
        assert getattr(result, "_y") == 3
        assert getattr(result, "_z") == 4

        # Verify fields set indirectly by checking the expected fields are still set
        for field in model_instance.model_fields_set:
            assert hasattr(result, field), (
                f"The field '{field}' should be set in the result"
            )

    def test_visit_collection_recursive_1(self) -> None:
        obj: Dict[str, Any] = dict()
        obj["a"] = obj
        # this would raise a RecursionError if we didn't handle it properly
        val = visit_collection(obj, lambda x: x, return_data=True)
        assert val is obj

    def test_visit_recursive_collection_2(self) -> None:
        # Create references to each other
        foo = Foo(x=None)
        bar = Bar(y=foo)
        foo.x = bar

        val = [foo, bar]

        result = visit_collection(val, lambda x: x, return_data=True)
        assert result is val

    def test_visit_collection_works_with_field_alias(self) -> None:
        class TargetConfigs(pydantic.BaseModel):
            type: str
            schema_: str = pydantic.Field(alias="schema")
            threads: int = 4

        target_configs = TargetConfigs(
            type="a_type", schema="a working schema", threads=1
        )
        result = visit_collection(
            target_configs, visit_fn=negative_even_numbers, return_data=True
        )

        assert result == target_configs

    @pytest.mark.parametrize(
        "inp,depth,expected",
        [
            (1, 0, -1),
            ([1, [2, [3, [4]]]], 0, [1, [2, [3, [4]]]]),
            ([1, [2, [3, [4]]]], 1, [-1, [2, [3, [4]]]]),
            ([1, [2, [3, [4]]]], 2, [-1, [-2, [3, [4]]]]),
            ([1, 1, 1, [2, 2, 2]], 1, [-1, -1, -1, [2, 2, 2]]),
        ],
    )
    def test_visit_collection_max_depth(self, inp: Any, depth: int, expected: Any) -> None:
        result = visit_collection(
            inp, visit_fn=all_negative_numbers, return_data=True, max_depth=depth
        )
        assert result == expected

    def test_visit_collection_context(self) -> None:
        # Create a list of integers with various levels of nesting
        foo = [1, 2, [3, 4], [5, [6, 7]], 8, 9]

        def visit(expr: Any, context: Dict[str, Any]) -> Any:
            # When visiting a list, add one to the depth and return the list
            if isinstance(expr, list):
                context["depth"] += 1
                return expr
            # When visiting an integer, return it plus the depth
            else:
                return expr + context["depth"]

        result = visit_collection(foo, visit, context={"depth": 0}, return_data=True)
        # Seeded with a depth of 0, we expect all of the items in the root list to be
        # incremented by one, items in a nested list to be incremented by one, etc.
        # We confirm that integers in the root list visited after the nested lists see
        # the depth of one
        assert result == [2, 3, [5, 6], [7, [9, 10]], 9, 10]

    def test_visit_collection_context_from_annotation(self) -> None:
        foo = quote([1, 2, [3]])

        def visit(expr: Any, context: Dict[str, Any]) -> Any:
            # If we're not visiting the first expression...
            if not isinstance(expr, quote):
                assert isinstance(context.get("annotation"), quote)
            return expr

        result = visit