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
    pass

T = TypeVar('T')

class GenericPydanticModel(BaseModel, Generic[T]):
    pass

class SomeTypedDict(TypedDict):
    pass

class SomeNamedTuple(NamedTuple):
    pass

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
def test_types(tp: type, val: Any, expected: Any) -> None:
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

@pytest.mark.parametrize('defer_build', [False, True])
@pytest.mark.parametrize('method', ['validate', 'serialize', 'json_schema', 'json_schemas'])
def test_local_namespace_variables(defer_build: bool, method: str, generate_schema_calls: Any) -> None:
    IntList = list[int]
    OuterDict = dict[str, 'IntList']
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
def test_model_local_namespace_variables(defer_build: bool, method: str, generate_schema_calls: Any) -> None:
    IntList = list[int]

    class MyModel(BaseModel):
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
        _, json_schema = TypeAdapter.json_schemas([(MyModel, 'validation', ta)])
        assert 'MyModel' in json_schema['$defs']

@pytest.mark.parametrize('defer_build', [False, True])
@pytest.mark.parametrize('method', ['validate', 'serialize', 'json_schema', 'json_schemas'])
def test_top_level_fwd_ref(defer_build: bool, method: str, generate_schema_calls: Any) -> None:
    config = ConfigDict(defer_build=True) if defer_build else None
    FwdRef = ForwardRef('OuterDict', module=__name__)
    ta = TypeAdapter(FwdRef, config=config)
    assert generate_schema_calls.count == (0 if defer_build else 1), 'Should be built deferred'
    if method == 'validate':
        assert ta.validate_python({'foo': [1, '2']}) == {'foo': [1, 2]}
    elif method == 'serialize':
        assert ta.dump_python({'foo': [1, 2]}) == {'foo': [1, 2]}
    elif method == 'json_schema':
        assert ta.json_schema()['type'] == 'object'
    else:
        assert method == 'json_schemas'
        schemas, _ = TypeAdapter.json_schemas([(FwdRef, 'validation', ta)])
        assert schemas[FwdRef, 'validation']['type'] == 'object'

MyUnion: TypeAlias = 'Union[str, int]'

def test_type_alias() -> None:
    MyList = list[MyUnion]
    v = TypeAdapter(MyList).validate_python
    res = v([1, '2'])
    assert res == [1, '2']

def test_validate_python_strict() -> None:
    class Model(TypedDict):
        pass

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

@pytest.mark.xfail(reason='Need to fix this in https://github.com/pydantic/pydantic/pull/5944')
def test_validate_json_strict() -> None:
    class Model(TypedDict):
        pass

    class ModelStrict(Model):
        __pydantic_config__ = ConfigDict(strict=True)
    
    lax_validator = TypeAdapter(Model, config=ConfigDict(strict=False))
    strict_validator = TypeAdapter(ModelStrict)
    assert lax_validator.validate_json(json.dumps({'x': '1'}), strict=None) == Model(x=1)
    assert lax_validator.validate_json(json.dumps({'x': '1'}), strict=False) == Model(x=1)
    with pytest.raises(ValidationError) as exc_info:
        lax_validator.validate_json(json.dumps({'x': '1'}), strict=True)
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': ('x',), 'msg': 'Input should be a valid integer', 'input': '1'}]
    with pytest.raises(ValidationError) as exc_info:
        strict_validator.validate_json(json.dumps({'x': '1'}), strict=None)
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': ('x',), 'msg': 'Input should be a valid integer', 'input': '1'}]
    assert strict_validator.validate_json(json.dumps({'x': '1'}), strict=False) == Model(x=1)
    with pytest.raises(ValidationError) as exc_info:
        strict_validator.validate_json(json.dumps({'x': '1'}), strict=True)
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': ('x',), 'msg': 'Input should be a valid integer', 'input': '1'}]

def test_validate_python_context() -> None:
    contexts = [None, None, {'foo': 'bar'}]

    class Model(BaseModel):
        @field_validator('x')
        def val_x(cls, v: Any, info: ValidationInfo) -> Any:
            assert info.context == contexts.pop(0)
            return v
    
    validator = TypeAdapter(Model)
    validator.validate_python({'x': 1})
    validator.validate_python({'x': 1}, context=None)
    validator.validate_python({'x': 1}, context={'foo': 'bar'})
    assert contexts == []

def test_validate_json_context() -> None:
    contexts = [None, None, {'foo': 'bar'}]

    class Model(BaseModel):
        @field_validator('x')
        def val_x(cls, v: Any, info: ValidationInfo) -> Any:
            assert info.context == contexts.pop(0)
            return v
    
    validator = TypeAdapter(Model)
    validator.validate_json(json.dumps({'x': 1}))
    validator.validate_json(json.dumps({'x': 1}), context=None)
    validator.validate_json(json.dumps({'x': 1}), context={'foo': 'bar'})
    assert contexts == []

def test_validate_python_from_attributes() -> None:
    class Model(BaseModel):
        pass

    class ModelFromAttributesTrue(Model):
        model_config = ConfigDict(from_attributes=True)

    class ModelFromAttributesFalse(Model):
        model_config = ConfigDict(from_attributes=False)

    @dataclass
    class UnrelatedClass:
        x = 1
    
    input = UnrelatedClass(1)
    ta = TypeAdapter(Model)
    for from_attributes in (False, None):
        with pytest.raises(ValidationError) as exc_info:
            ta.validate_python(UnrelatedClass(), from_attributes=from_attributes)
        assert exc_info.value.errors(include_url=False) == [{'type': 'model_type', 'loc': (), 'msg': 'Input should be a valid dictionary or instance of Model', 'input': input, 'ctx': {'class_name': 'Model'}}]
    res = ta.validate_python(UnrelatedClass(), from_attributes=True)
    assert res == Model(x=1)
    ta = TypeAdapter(ModelFromAttributesTrue)
    with pytest.raises(ValidationError) as exc_info:
        ta.validate_python(UnrelatedClass(), from_attributes=False)
    assert exc_info.value.errors(include_url=False) == [{'type': 'model_type', 'loc': (), 'msg': 'Input should be a valid dictionary or instance of ModelFromAttributesTrue', 'input': input, 'ctx': {'class_name': 'ModelFromAttributesTrue'}}]
    for from_attributes in (True, None):
        res = ta.validate_python(UnrelatedClass(), from_attributes=from_attributes)
        assert res == ModelFromAttributesTrue(x=1)
    ta = TypeAdapter(ModelFromAttributesFalse)
    for from_attributes in (False, None):
        with pytest.raises(ValidationError) as exc_info:
            ta.validate_python(UnrelatedClass(), from_attributes=from_attributes)
        assert exc_info.value.errors(include_url=False) == [{'type': 'model_type', 'loc': (), 'msg': 'Input should be a valid dictionary or instance of ModelFromAttributesFalse', 'input': input, 'ctx': {'class_name': 'ModelFromAttributesFalse'}}]
    res = ta.validate_python(UnrelatedClass(), from_attributes=True)
    assert res == ModelFromAttributesFalse(x=1)

@pytest.mark.parametrize('field_type,input_value,expected,raises_match,strict', [
    (bool, 'true', True, None, False),
    (bool, 'true', True, None, True),
    (bool, 'false', False, None, False),
    (bool, 'e', ValidationError, 'type=bool_parsing', False),
    (int, '1', 1, None, False),
    (int, '1', 1, None, True),
    (int, 'xxx', ValidationError, 'type=int_parsing', True),
    (float, '1.1', 1.1, None, False),
    (float, '1.10', 1.1, None, False),
    (float, '1.1', 1.1, None, True),
    (float, '1.10', 1.1, None, True),
    (date, '2017-01-01', date(2017, 1, 1), None, False),
    (date, '2017-01-01', date(2017, 1, 1), None, True),
    (date, '2017-01-01T12:13:14.567', ValidationError, 'type=date_from_datetime_inexact', False),
    (date, '2017-01-01T12:13:14.567', ValidationError, 'type=date_parsing', True),
    (date, '2017-01-01T00:00:00', date(2017, 1, 1), None, False),
    (date, '2017-01-01T00:00:00', ValidationError, 'type=date_parsing', True),
    (datetime, '2017-01-01