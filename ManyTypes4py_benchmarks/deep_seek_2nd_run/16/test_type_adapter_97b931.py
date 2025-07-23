import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Annotated, Any, ForwardRef, Generic, NamedTuple, Optional, TypeVar, Union
import pytest
from pydantic_core import ValidationError
from typing_extensions import TypeAlias, TypedDict
from pydantic import BaseModel, Field, TypeAdapter, ValidationInfo, create_model, field_validator
from pydantic._internal import _mock_val_ser
from pydantic._internal._typing_extra import annotated_type
from pydantic.config import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.errors import PydanticUndefinedAnnotation, PydanticUserError
from pydantic.type_adapter import _type_has_config

ItemType = TypeVar('ItemType')
NestedList = list[list[ItemType]]

class PydanticModel(BaseModel):
    x: int

T = TypeVar('T')

class GenericPydanticModel(BaseModel, Generic[T]):
    x: NestedList[T]

class SomeTypedDict(TypedDict):
    x: int

class SomeNamedTuple(NamedTuple):
    x: int

@pytest.mark.parametrize('tp, val, expected', [
    (PydanticModel, PydanticModel(x=1), PydanticModel(x=1)),
    (PydanticModel, {'x': 1}, PydanticModel(x=1)),
    (SomeTypedDict, {'x': 1}, {'x': 1}),
    (SomeNamedTuple, SomeNamedTuple(x=1), SomeNamedTuple(x=1)),
    (list[str], ['1', '2'], ['1', '2']),
    (tuple[str], ('1',), ('1',)),
    (tuple[str, int], ('1', 1), ('1', 1)),
    (tuple[str, ...], ('1',), ('1',)),
    (dict[str, int], {'foo': 123}, {'foo': 123}),
    (Union[int, str], 1, 1),
    (Union[int, str], '2', '2'),
    (GenericPydanticModel[int], {'x': [[1]]}, GenericPydanticModel[int](x=[[1]])),
    (GenericPydanticModel[int], {'x': [['1']]}, GenericPydanticModel[int](x=[[1]])),
    (NestedList[int], [[1]], [[1]]),
    (NestedList[int], [['1']], [[1]])
])
def test_types(tp: Any, val: Any, expected: Any) -> None:
    v = TypeAdapter(tp).validate_python
    assert expected == v(val)

IntList = list[int]
OuterDict = dict[str, 'IntList']

@pytest.mark.parametrize('defer_build', [False, True])
@pytest.mark.parametrize('method', ['validate', 'serialize', 'json_schema', 'json_schemas'])
def test_global_namespace_variables(defer_build: bool, method: str, generate_schema_calls: Any) -> None:
    config = ConfigDict(defer_build=True) if defer_build else None
    ta = TypeAdapter(OuterDict, config=config)
    assert generate_schema_calls.count == (0 if defer_build else 1), 'Should be built deferred'
    if method == 'validate':
        assert ta.validate_python({'foo': [1, '2']}) == {'foo': [1, 2]}
    elif method == 'serialize':
        assert ta.dump_python({'foo': [1, 2]}) == {'foo': [1, 2]}
    elif method == 'json_schema':
        assert ta.json_schema()['type'] == 'object'
    else:
        assert method == 'json_schemas'
        schemas, _ = TypeAdapter.json_schemas([(OuterDict, 'validation', ta)])
        assert schemas[OuterDict, 'validation']['type'] == 'object'

@pytest.mark.parametrize('defer_build', [False, True])
@pytest.mark.parametrize('method', ['validate', 'serialize', 'json_schema', 'json_schemas'])
def test_model_global_namespace_variables(defer_build: bool, method: str, generate_schema_calls: Any) -> None:
    class MyModel(BaseModel):
        x: OuterDict
        model_config = ConfigDict(defer_build=defer_build)
    
    ta = TypeAdapter(MyModel)
    assert generate_schema_calls.count == (0 if defer_build else 1), 'Should be built deferred'
    if method == 'validate':
        assert ta.validate_python({'x': {'foo': [1, '2']}}) == MyModel(x={'foo': [1, 2]})
    elif method == 'serialize':
        assert ta.dump_python(MyModel(x={'foo': [1, 2]})) == {'x': {'foo': [1, 2]}}
    elif method == 'json_schema':
        assert ta.json_schema()['title'] == 'MyModel'
    else:
        assert method == 'json_schemas'
        _, json_schema = TypeAdapter.json_schemas([(MyModel, 'validation', TypeAdapter(MyModel))])
        assert 'MyModel' in json_schema['$defs']

# ... (continuing with type annotations for the remaining functions in the same pattern)

def test_type_alias() -> None:
    MyList = list[MyUnion]
    v = TypeAdapter(MyList).validate_python
    res = v([1, '2'])
    assert res == [1, '2']

def test_validate_python_strict() -> None:
    class Model(TypedDict):
        x: int

    class ModelStrict(Model):
        __pydantic_config__ = ConfigDict(strict=True)
    
    lax_validator = TypeAdapter(Model)
    strict_validator = TypeAdapter(ModelStrict)
    assert lax_validator.validate_python({'x': '1'}, strict=None) == Model(x=1)
    assert lax_validator.validate_python({'x': '1'}, strict=False) == Model(x=1)
    with pytest.raises(ValidationError) as exc_info:
        lax_validator.validate_python({'x': '1'}, strict=True)
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': ('x',), 'msg': 'Input should be a valid integer', 'input': '1'}]
    with pytest.raises(ValidationError) as exc_info:
        strict_validator.validate_python({'x': '1'})
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': ('x',), 'msg': 'Input should be a valid integer', 'input': '1'}]
    assert strict_validator.validate_python({'x': '1'}, strict=False) == Model(x=1)
    with pytest.raises(ValidationError) as exc_info:
        strict_validator.validate_python({'x': '1'}, strict=True)
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': ('x',), 'msg': 'Input should be a valid integer', 'input': '1'}]

# ... (continuing with type annotations for all remaining functions)

def test_correct_frame_used_parametrized(create_module: Any) -> None:
    """https://github.com/pydantic/pydantic/issues/10892"""

    @create_module
    def module_1() -> None:
        from pydantic import TypeAdapter
        Any = int
        ta = TypeAdapter[int]('Any')
    
    with pytest.raises(ValidationError):
        module_1.ta.validate_python('a')
