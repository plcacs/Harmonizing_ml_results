from typing import Annotated, Any, ForwardRef, Generic, NamedTuple, Optional, TypeVar, Union, list, dict
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, ConfigDict, create_model, annotated_type, dataclass
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.errors import PydanticUndefinedAnnotation, PydanticUserError

ItemType = TypeVar('ItemType')
NestedList = list[list[ItemType]]

class PydanticModel(BaseModel):
    pass

T = TypeVar('T')

class GenericPydanticModel(BaseModel, Generic[T]):
    pass

class SomeTypedDict(TypedDict):
    pass

class SomeNamedTuple(NamedTuple):
    pass

IntList = list[int]
OuterDict = dict[str, 'IntList']

def test_types(tp: Any, val: Any, expected: Any):
    v = TypeAdapter(tp).validate_python
    assert expected == v(val)

def test_global_namespace_variables(defer_build: bool, method: str, generate_schema_calls: Any):
    config = ConfigDict(defer_build=True) if defer_build else None
    ta = TypeAdapter(OuterDict, config=config)
    assert generate_schema_calls.count == (0 if defer_build else 1)
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

# Add the rest of the functions with appropriate type annotations
