import io
import json
import uuid
from dataclasses import dataclass
from typing import Any
import pydantic
import pytest
from prefect.utilities.annotations import BaseAnnotation, quote
from prefect.utilities.collections import AutoEnum, StopVisiting, deep_merge, deep_merge_dicts, dict_to_flatdict, flatdict_to_dict, get_from_dict, isiterable, remove_nested_keys, set_in_dict, visit_collection

class ExampleAnnotation(BaseAnnotation):
    pass

class Color(AutoEnum):
    RED = AutoEnum.auto()
    BLUE = AutoEnum.auto()

class TestAutoEnum:

    def test_autoenum_generates_string_values(self) -> None:
        assert Color.RED.value == 'RED'
        assert Color.BLUE.value == 'BLUE'

    def test_autoenum_repr(self) -> None:
        assert repr(Color.RED) == str(Color.RED) == 'Color.RED'

    def test_autoenum_can_be_json_serialized_with_default_encoder(self) -> None:
        json.dumps(Color.RED) == 'RED'

@pytest.mark.parametrize('d, expected', [({1: 2}, {(1,): 2}), ({1: 2, 2: {1: 2, 3: 4}, 3: {1: 2, 3: {4: 5, 6: {7: 8}}}}, {(1,): 2, (2, 1): 2, (2, 3): 4, (3, 1): 2, (3, 3, 4): 5, (3, 3, 6, 7): 8}), ({1: 2, 3: {}, 4: {5: {}}}, {(1,): 2, (3,): {}, (4, 5): {}})])
def test_flatdict_conversion(d: bool, expected: Union[bool, list[int]]) -> None:
    flat = dict_to_flatdict(d)
    assert flat == expected
    assert flatdict_to_dict(flat) == d

def negative_even_numbers(x: Union[T, float]) -> Union[float, T]:
    print('Function called on', x)
    if isinstance(x, int) and x % 2 == 0:
        return -x
    return x

def all_negative_numbers(x: Union[float, T, None]) -> Union[int, float, T, None]:
    print('Function called on', x)
    if isinstance(x, int):
        return -x
    return x
EVEN = set()

def visit_even_numbers(x: Union[int, slice, float]) -> Union[int, slice, float]:
    if isinstance(x, int) and x % 2 == 0:
        EVEN.add(x)
    return x
VISITED = list()

def add_to_visited_list(x: Union[list, list[str], None]) -> None:
    VISITED.append(x)

@pytest.fixture(autouse=True)
def clear_sets() -> None:
    EVEN.clear()
    VISITED.clear()

@dataclass
class SimpleDataclass:
    pass

class SimplePydantic(pydantic.BaseModel):
    pass

class ExtraPydantic(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

class PrivatePydantic(pydantic.BaseModel):
    """Pydantic model with private attrs"""
    model_config = pydantic.ConfigDict(extra='forbid')
    _z = pydantic.PrivateAttr()

class ImmutablePrivatePydantic(PrivatePydantic):
    model_config = pydantic.ConfigDict(frozen=True)

class PydanticWithDefaults(pydantic.BaseModel):
    val = pydantic.Field(default_factory=uuid.uuid4)
    num = 0

@dataclass
class Foo:
    pass

@dataclass
class Bar:
    z = 2

class TestPydanticObjects:
    """
    Checks that the Pydantic test objects defined in this file behave as expected.

    These tests do not cover Prefect functionality and may break if Pydantic introduces
    breaking changes.
    """

    def test_private_pydantic_behaves_as_expected(self) -> None:
        input = PrivatePydantic(x=1)
        assert input.x == 1
        with pytest.raises(ValueError):
            input.a = 1
        input._y = 4
        input._z = 5
        assert input._y == 4
        assert input._z == 5

    def test_immutable_pydantic_behaves_as_expected(self) -> None:
        input = ImmutablePrivatePydantic(x=1)
        assert input.x == 1
        with pytest.raises(ValueError):
            input.a = 1
        with pytest.raises(AttributeError):
            input._y
        input._y = 4
        input._z = 5
        assert input._y == 4
        assert input._z == 5
        with pytest.raises(pydantic.ValidationError):
            input.x = 2
        input._y = 6

class TestVisitCollection:

    @pytest.mark.parametrize('inp,expected', [(3, 3), (4, -4), ([3, 4], [3, -4]), ((3, 4), (3, -4)), ([3, 4, [5, [6]]], [3, -4, [5, [-6]]]), ({3: 4, 6: 7}, {3: -4, -6: 7}), ({3: [4, {6: 7}]}, {3: [-4, {-6: 7}]}), ({3, 4, 5}, {3, -4, 5}), (SimpleDataclass(x=1, y=2), SimpleDataclass(x=1, y=-2)), (SimplePydantic(x=1, y=2), SimplePydantic(x=1, y=-2)), (ExtraPydantic(x=1, y=2, z=3), ExtraPydantic(x=1, y=-2, z=3)), (ExampleAnnotation(4), ExampleAnnotation(-4))])
    def test_visit_collection_and_transform_data(self, inp: Union[bool, tuple[int]], expected: Union[int, list[str], float]) -> None:
        result = visit_collection(inp, visit_fn=negative_even_numbers, return_data=True)
        assert result == expected

    @pytest.mark.parametrize('inp,expected', [(3, set()), (4, {4}), ([3, 4], {4}), ((3, 4), {4}), ([3, 4, [5, [6]]], {4, 6}), ({3: 4, 6: 7}, {4, 6}), ({3: [4, {6: 7}]}, {4, 6}), ({3, 4, 5}, {4}), (SimpleDataclass(x=1, y=2), {2}), (SimplePydantic(x=1, y=2), {2}), (ExtraPydantic(x=1, y=2, z=4), {2, 4}), (ExtraPydantic(x=1, y=2, z=4).model_copy(), {2, 4}), (ExampleAnnotation(4), {4})])
    def test_visit_collection(self, inp: bool, expected: Union[bool, tuple[int]]) -> None:
        result = visit_collection(inp, visit_fn=visit_even_numbers, return_data=False)
        assert result is None
        assert EVEN == expected

    def test_visit_collection_does_not_consume_generators(self) -> None:

        def f() -> typing.Generator[list[int]]:
            yield from [1, 2, 3]
        result = visit_collection([f()], visit_fn=visit_even_numbers, return_data=False)
        assert result is None
        assert not EVEN

    def test_visit_collection_does_not_consume_generators_when_returning_data(self) -> None:

        def f() -> typing.Generator[list[int]]:
            yield from [1, 2, 3]
        val = [f()]
        result = visit_collection(val, visit_fn=visit_even_numbers, return_data=True)
        assert result is val
        assert not EVEN

    @pytest.mark.parametrize('inp,expected', [({'x': 1}, [{'x': 1}, 'x', 1]), (SimpleDataclass(x=1, y=2), [SimpleDataclass(x=1, y=2), 1, 2])])
    def test_visit_collection_visits_nodes(self, inp: bool, expected: int) -> None:
        result = visit_collection(inp, visit_fn=add_to_visited_list, return_data=False)
        assert result is None
        assert VISITED == expected

    @pytest.mark.parametrize('inp,expected', [(sorted([1, 2, 3]), [1, -2, 3]), ('test', 'test'), (b'test', b'test')])
    def test_visit_collection_iterators(self, inp: bool, expected: Union[int, list[int]]) -> None:
        result = visit_collection(inp, visit_fn=negative_even_numbers, return_data=True)
        assert result == expected

    @pytest.mark.parametrize('inp', [io.StringIO('test'), io.BytesIO(b'test')])
    def test_visit_collection_io_iterators(self, inp: Union[str, typing.Callable[None, bool], float]) -> None:
        result = visit_collection(inp, visit_fn=lambda x: x, return_data=True)
        assert result is inp

    def test_visit_collection_allows_mutation_of_nodes(self) -> None:

        def collect_and_drop_x_from_dicts(node: Any) -> dict:
            add_to_visited_list(node)
            if isinstance(node, dict):
                return {key: value for key, value in node.items() if key != 'x'}
            return node
        result = visit_collection({'x': 1, 'y': 2}, visit_fn=collect_and_drop_x_from_dicts, return_data=True)
        assert result == {'y': 2}
        assert VISITED == [{'x': 1, 'y': 2}, 'y', 2]

    def test_visit_collection_with_private_pydantic_attributes(self) -> None:
        """
        We should not visit private fields on Pydantic models.
        """
        input = PrivatePydantic(x=2)
        input._y = 3
        input._z = 4
        result = visit_collection(input, visit_fn=visit_even_numbers, return_data=False)
        assert EVEN == {2}, 'Only the public field should be visited'
        assert result is None, 'Data should not be returned'
        assert input._y == 3
        assert input._z == 4

    def test_visit_collection_includes_unset_pydantic_fields(self) -> None:

        class RandomPydantic(pydantic.BaseModel):
            val = pydantic.Field(default_factory=uuid.uuid4)
        input_model = RandomPydantic()
        output_model = visit_collection(input_model, visit_fn=visit_even_numbers, return_data=True)
        assert output_model.val == input_model.val, 'The fields value should be used, not the default factory'

    @pytest.mark.parametrize('input', [{'name': 'prefect'}, {'name': 'prefect', 'num': 1}, {'name': 'prefect', 'num': 1, 'val': uuid.UUID(int=0)}, {'name': 'prefect', 'val': uuid.UUID(int=0)}])
    def test_visit_collection_remembers_unset_pydantic_fields(self, input: str) -> None:
        input_model = PydanticWithDefaults(**input)
        output_model = visit_collection(input_model, visit_fn=visit_even_numbers, return_data=True)
        assert output_model.model_dump(exclude_unset=True) == input, 'Unset fields values should be remembered and preserved'

    @pytest.mark.parametrize('immutable', [True, False])
    def test_visit_collection_mutation_with_private_pydantic_attributes(self, immutable: typing.Callable) -> None:
        model = ImmutablePrivatePydantic if immutable else PrivatePydantic
        model_instance = model(x=2)
        model_instance._y = 3
        model_instance._z = 4
        result = visit_collection(model_instance, visit_fn=negative_even_numbers, return_data=True)
        assert isinstance(result, model), 'The model should be returned'
        assert result.x == -2, 'The public attribute should be modified'
        assert getattr(result, '_y') == 3
        assert getattr(result, '_z') == 4
        for field in model_instance.model_fields_set:
            assert hasattr(result, field), f"The field '{field}' should be set in the result"

    def test_visit_collection_recursive_1(self) -> None:
        obj = dict()
        obj['a'] = obj
        val = visit_collection(obj, lambda x: x, return_data=True)
        assert val is obj

    def test_visit_recursive_collection_2(self) -> None:
        foo = Foo(x=None)
        bar = Bar(y=foo)
        foo.x = bar
        val = [foo, bar]
        result = visit_collection(val, lambda x: x, return_data=True)
        assert result is val

    def test_visit_collection_works_with_field_alias(self) -> None:

        class TargetConfigs(pydantic.BaseModel):
            schema_ = pydantic.Field(alias='schema')
            threads = 4
        target_configs = TargetConfigs(type='a_type', schema='a working schema', threads=1)
        result = visit_collection(target_configs, visit_fn=negative_even_numbers, return_data=True)
        assert result == target_configs

    @pytest.mark.parametrize('inp,depth,expected', [(1, 0, -1), ([1, [2, [3, [4]]]], 0, [1, [2, [3, [4]]]]), ([1, [2, [3, [4]]]], 1, [-1, [2, [3, [4]]]]), ([1, [2, [3, [4]]]], 2, [-1, [-2, [3, [4]]]]), ([1, 1, 1, [2, 2, 2]], 1, [-1, -1, -1, [2, 2, 2]])])
    def test_visit_collection_max_depth(self, inp: Union[int, float], depth: Union[int, float], expected: Union[int, None]) -> None:
        result = visit_collection(inp, visit_fn=all_negative_numbers, return_data=True, max_depth=depth)
        assert result == expected

    def test_visit_collection_context(self) -> None:
        foo = [1, 2, [3, 4], [5, [6, 7]], 8, 9]

        def visit(expr: Any, context: Any) -> list:
            if isinstance(expr, list):
                context['depth'] += 1
                return expr
            else:
                return expr + context['depth']
        result = visit_collection(foo, visit, context={'depth': 0}, return_data=True)
        assert result == [2, 3, [5, 6], [7, [9, 10]], 9, 10]

    def test_visit_collection_context_from_annotation(self) -> None:
        foo = quote([1, 2, [3]])

        def visit(expr: Any, context: Any) -> list:
            if not isinstance(expr, quote):
                assert isinstance(context.get('annotation'), quote)
            return expr
        result = visit_collection(foo, visit, context={}, return_data=True)
        assert result == quote([1, 2, [3]])

    def test_visit_collection_remove_annotations(self) -> None:
        foo = quote([1, 2, quote([3])])

        def visit(expr: Any, context: Any) -> list:
            if isinstance(expr, int):
                return expr + 1
            return expr
        result = visit_collection(foo, visit, context={}, return_data=True, remove_annotations=True)
        assert result == [2, 3, [4]]

    def test_visit_collection_stop_visiting(self) -> None:
        foo = [1, 2, quote([3, [4, 5, 6]])]

        def visit(expr: Any, context: Any) -> list:
            if isinstance(context.get('annotation'), quote):
                raise StopVisiting()
            if isinstance(expr, int):
                return expr + 1
            else:
                return expr
        result = visit_collection(foo, visit, context={}, return_data=True, remove_annotations=True)
        assert result == [2, 3, [3, [4, 5, 6]]]

    @pytest.mark.parametrize('val', [1, [1, 2, 3], SimplePydantic(x=1, y=2), {'x': 1}])
    def test_visit_collection_simple_identity(self, val: Any) -> None:
        """test that visit collection does not modify an object at all in the identity case"""
        result = visit_collection(val, lambda x: x, return_data=True)
        assert result is val

    def test_visit_collection_only_modify_changed_objects_1(self) -> None:
        val = [[1, 2], [3, 5]]
        result = visit_collection(val, negative_even_numbers, return_data=True)
        assert result == [[1, -2], [3, 5]]
        assert result is not val
        assert result[0] is not val[0]
        assert result[1] is val[1]

    def test_visit_collection_only_modify_changed_objects_2(self) -> None:
        val = [[[1], {2: 3}], [3, 5]]
        result = visit_collection(val, negative_even_numbers, return_data=True)
        assert result == [[[1], {-2: 3}], [3, 5]]
        assert result[0] is not val[0]
        assert result[0][0] is val[0][0]
        assert result[0][1] is not val[0][1]
        assert result[1] is val[1]

    def test_visit_collection_only_modify_changed_objects_3(self) -> None:

        class Foo(pydantic.BaseModel):
            pass
        val = Foo(x=[[1, 2], [3, 5]], y=dict(a=dict(b=1, c=2), d=dict(e=3, f=5)))
        result = visit_collection(val, negative_even_numbers, return_data=True)
        assert result is not val
        assert result.x[0] is not val.x[0]
        assert result.x[1] is val.x[1]
        assert result.y['a'] is not val.y['a']
        assert result.y['d'] is val.y['d']

class TestRemoveKeys:

    def test_remove_single_key(self) -> None:
        obj = {'a': 'a', 'b': 'b', 'c': 'c'}
        assert remove_nested_keys(['a'], obj) == {'b': 'b', 'c': 'c'}

    def test_remove_multiple_keys(self) -> None:
        obj = {'a': 'a', 'b': 'b', 'c': 'c'}
        assert remove_nested_keys(['a', 'b'], obj) == {'c': 'c'}

    def test_remove_keys_recursively(self) -> None:
        obj = {'title': 'Test', 'description': 'This is a docstring', 'type': 'object', 'properties': {'a': {'title': 'A', 'description': 'A field', 'type': 'string'}}, 'required': ['a'], 'block_type_name': 'Test', 'block_schema_references': {}}
        assert remove_nested_keys(['description'], obj) == {'title': 'Test', 'type': 'object', 'properties': {'a': {'title': 'A', 'type': 'string'}}, 'required': ['a'], 'block_type_name': 'Test', 'block_schema_references': {}}

    def test_passes_through_non_dict(self) -> None:
        assert remove_nested_keys(['foo'], 1) == 1
        assert remove_nested_keys(['foo'], 'foo') == 'foo'
        assert remove_nested_keys(['foo'], b'foo') == b'foo'

class TestIsIterable:

    @pytest.mark.parametrize('obj', [[1, 2, 3], (1, 2, 3)])
    def test_is_iterable(self, obj: Union[typing.Callable, typing.Type, str]) -> None:
        assert isiterable(obj)

    @pytest.mark.parametrize('obj', [5, Exception(), True, 'hello', bytes()])
    def test_not_iterable(self, obj: Union[str, typing.Callable, Node]) -> None:
        assert not isiterable(obj)

class TestGetFromDict:

    @pytest.mark.parametrize('dct, keys, expected, default', [({}, 'a.b.c', None, None), ({'a': {'b': {'c': [1, 2, 3, 4]}}}, 'a.b.c[1]', 2, None), ({'a': {'b': {'c': [1, 2, 3, 4]}}}, 'a.b.c.1', 2, None), ({'a': {'b': [0, {'c': [1, 2]}]}}, 'a.b.1.c.1', 2, None), ({'a': {'b': [0, {'c': [1, 2]}]}}, ['a', 'b', 1, 'c', 1], 2, None), ({'a': {'b': [0, {'c': [1, 2]}]}}, 'a.b.1.c.2', None, None), ({'a': {'b': [0, {'c': [1, 2]}]}}, 'a.b.1.c.2', 'default_value', 'default_value')])
    def test_get_from_dict(self, dct: Union[str, dict, typing.Callable], keys: Union[str, dict, typing.Callable], expected: Union[str, dict, typing.Callable], default: Union[str, dict, typing.Callable]) -> None:
        assert get_from_dict(dct, keys, default) == expected

class TestSetInDict:

    @pytest.mark.parametrize('dct, keys, value, expected', [({}, 'a.b.c', 1, {'a': {'b': {'c': 1}}}), ({'a': {'b': {'c': 1}}}, 'a.b.c', 2, {'a': {'b': {'c': 2}}}), ({'a': {'b': {'c': 1}}}, 'a.b.d', 2, {'a': {'b': {'c': 1, 'd': 2}}}), ({'a': {'b': {'c': 1}}}, 'a', 2, {'a': 2}), ({'a': {'b': {'c': 1}}}, ['a', 'b', 'd'], 2, {'a': {'b': {'c': 1, 'd': 2}}})])
    def test_set_in_dict(self, dct: Union[dict, str, int], keys: Union[str, Exception, dict[str, typing.Union[bool,str]]], value: Union[str, Exception, dict[str, typing.Union[bool,str]]], expected: Union[str, dict[str, typing.Any], int]) -> None:
        set_in_dict(dct, keys, value)
        assert dct == expected

    def test_set_in_dict_raises_key_error(self) -> None:
        with pytest.raises(TypeError, match='Key path exists and contains a non-dict value'):
            set_in_dict({'a': {'b': [2]}}, ['a', 'b', 'c'], 1)

class TestDeepMerge:

    @pytest.mark.parametrize('dct, merge, expected', [({'a': 1}, {'b': 2}, {'a': 1, 'b': 2}), ({'a': 1}, {'a': 2}, {'a': 2}), ({'a': {'b': 1}}, {'a': {'c': 2}}, {'a': {'b': 1, 'c': 2}}), ({'a': {'b': 2}}, {'a': {'c': {'d': 1}}}, {'a': {'b': 2, 'c': {'d': 1}}})])
    def test_deep_merge(self, dct: Union[dict, list], merge: Union[dict, list], expected: Union[dict, list]) -> None:
        assert deep_merge(dct, merge) == expected

class TestDeepMergeDicts:

    @pytest.mark.parametrize('dicts, expected', [([{'a': 1}, {'b': 2}], {'a': 1, 'b': 2}), ([{'a': 1}, {'a': 2}], {'a': 2}), ([{'a': {'b': 1}}, {'a': {'c': 2}}], {'a': {'b': 1, 'c': 2}}), ([{'a': {'b': 2}}, {'a': {'c': {'d': 1}}}], {'a': {'b': 2, 'c': {'d': 1}}}), ([{'a': {'b': 2}}, {'a': {'c': {'d': 1}}}, {'a': {'c': {'d': 3}}}], {'a': {'b': 2, 'c': {'d': 3}}})])
    def test_deep_merge_dicts(self, dicts: Union[list, list[dict[str, typing.Any]], list[tuple]], expected: Union[list, list[dict[str, typing.Any]], list[tuple]]) -> None:
        assert deep_merge_dicts(*dicts) == expected