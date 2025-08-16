from typing import Annotated, Any, ForwardRef, Generic, NamedTuple, Optional, TypeVar, Union, list, dict
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.type_adapter import TypeAdapter
from pydantic._internal import _mock_val_ser
from pydantic._internal._typing_extra import annotated_type
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

def test_types(tp: Any, val: Any, expected: Any) -> None:
    v = TypeAdapter(tp).validate_python
    assert expected == v(val)

IntList = list[int]
OuterDict = dict[str, 'IntList']

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

# Add more test functions here with appropriate type annotations
