from collections import deque
from typing import Any, List, Optional, Union, Dict, Tuple, TypeVar, Generic, cast
import jsonschema
import pytest
from prefect.utilities.schema_tools.hydration import HydrationError, InvalidJSON, Placeholder, ValueNotFound
from prefect.utilities.schema_tools.validation import CircularSchemaRefError, build_error_obj, is_valid, preprocess_schema, prioritize_placeholder_errors, validate

T = TypeVar('T')

class MockValidationError(jsonschema.exceptions.ValidationError):
    def __init__(self, message: str, relative_path: List[Union[str, int]], instance: Any = None, validator: Optional[str] = None) -> None:
        self.message = message
        self.relative_path = deque(relative_path)
        self.instance = instance
        self.validator = validator

async def test_hydration_error_causes_validation_error() -> None:
    error_msg = 'Something went real wrong!'

    class CatastrophicError(HydrationError):
        @property
        def message(self) -> str:
            return error_msg

    schema: Dict[str, Any] = {'title': 'Parameters', 'type': 'object', 'properties': {'param': {'title': 'user', 'position': 0}}}
    values: Dict[str, Any] = {'param': CatastrophicError()}
    errors: List[jsonschema.exceptions.ValidationError] = validate(values, schema)
    assert len(errors) == 1
    assert errors[0].message == error_msg

async def test_circular_schema_ref() -> None:
    circular_schema: Dict[str, Any] = {'title': 'Parameters', 'type': 'object', 'properties': {'param': {'title': 'param', 'position': 0, 'allOf': [{'$ref': '#/definitions/City'}]}}, 'required': ['param'], 'definitions': {'City': {'title': 'City', 'properties': {'population': {'title': 'Population', 'type': 'integer'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['population', 'name'], 'allOf': [{'$ref': '#/definitions/City'}]}}}
    with pytest.raises(CircularSchemaRefError):
        validate({'param': {"maybe a city, but we'll never know"}}, circular_schema)

async def test_ignore_required() -> None:
    schema: Dict[str, Any] = {'title': 'Parameters', 'type': 'object', 'properties': {'param': {'title': 'param', 'position': 0, 'type': 'integer'}}, 'required': ['param']}
    values: Dict[str, Any] = {}
    res: List[jsonschema.exceptions.ValidationError] = validate(values, schema, ignore_required=False)
    assert len(res) == 1
    assert res[0].message == "'param' is a required property"
    res = validate(values, schema, ignore_required=True)
    assert len(res) == 0

class TestNumber:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        return {'title': 'Parameters', 'type': 'object', 'properties': {'param': {'title': 'param', 'position': 0, 'type': 'integer'}}, 'required': ['param']}

    @pytest.mark.parametrize('obj, expected', [({'param': 10}, True), ({'param': 'not an integer'}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': 10}, []), ({'param': 'not an integer'}, ["'not an integer' is not of type 'integer'"]), ({}, ["'param' is a required property"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors

class TestBoolean:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        return {'title': 'Parameters', 'type': 'object', 'properties': {'param': {'title': 'param', 'position': 0, 'type': 'boolean'}}, 'required': ['param']}

    @pytest.mark.parametrize('obj, expected', [({'param': True}, True), ({'param': False}, True), ({'param': 'not a boolean'}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': True}, []), ({'param': False}, []), ({'param': 'not a boolean'}, ["'not a boolean' is not of type 'boolean'"]), ({}, ["'param' is a required property"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors

class TestString:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        return {'title': 'Parameters', 'type': 'object', 'properties': {'param': {'title': 'param', 'position': 0, 'type': 'string'}}, 'required': ['param']}

    @pytest.mark.parametrize('obj, expected', [({'param': 'test string'}, True), ({'param': 123}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': 'test string'}, []), ({'param': 123}, ["123 is not of type 'string'"]), ({}, ["'param' is a required property"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors

class TestDate:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        return {'title': 'Parameters', 'type': 'object', 'properties': {'value': {'title': 'value', 'position': 0, 'type': 'string', 'format': 'date'}}, 'required': ['value']}

    @pytest.mark.parametrize('obj, expected', [({'value': '2023-01-01'}, True), ({'value': 'not a date'}, False), ({'value': 123}, False), ({}, False), ({'value': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'value': '2023-01-01'}, []), ({'value': 'not a date'}, ["'not a date' is not a 'date'"]), ({'value': 123}, ["123 is not of type 'string'"]), ({}, ["'value' is a required property"]), ({'value': None}, ["None is not of type 'string'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors

class TestDateTime:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        return {'title': 'Parameters', 'type': 'object', 'properties': {'param': {'title': 'param', 'position': 0, 'type': 'string', 'format': 'date-time'}}, 'required': ['param']}

    @pytest.mark.parametrize('obj, expected', [({'param': '2023-01-01T12:00:00Z'}, True), ({'param': 'not a datetime'}, False), ({'param': 123}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': '2023-01-01T12:00:00Z'}, []), ({'param': 'not a date'}, ["'not a date' is not a 'date-time'"]), ({'param': 123}, ["123 is not of type 'string'"]), ({}, ["'param' is a required property"]), ({'param': None}, ["None is not of type 'string'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors

class TestDict:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        return {'title': 'Parameters', 'type': 'object', 'properties': {'param': {'title': 'param', 'position': 0, 'type': 'object'}}, 'required': ['param']}

    @pytest.mark.parametrize('obj, expected', [({'param': {'key': 'value'}}, True), ({'param': 'not a dict'}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': {'key': 'value'}}, []), ({'param': 'not a dict'}, ["'not a dict' is not of type 'object'"]), ({}, ["'param' is a required property"]), ({'param': None}, ["None is not of type 'object'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors

class TestObject:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        return {'title': 'Parameters', 'type': 'object', 'properties': {'param': {'title': 'param', 'position': 0, 'allOf': [{'$ref': '#/definitions/City'}]}}, 'required': ['param'], 'definitions': {'City': {'title': 'City', 'type': 'object', 'properties': {'population': {'title': 'Population', 'type': 'integer'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['population', 'name']}}}

    @pytest.mark.parametrize('obj, expected', [({'param': {'population': 1, 'name': 'string'}}, True), ({'param': {'population': 'not an integer', 'name': 'string'}}, False), ({'param': {'population': 1}}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': {'population': 1, 'name': 'string'}}, []), ({'param': {'population': 'not an integer', 'name': 'string'}}, ["'not an integer' is not of type 'integer'"]), ({'param': {'population': 1}}, ["'name' is a required property"]), ({}, ["'param' is a required property"]), ({'param': None}, ["None is not of type 'object'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors

class TestObjectOptionalParameters:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        return {'title': 'Parameters', 'type': 'object', 'properties': {'param': {'title': 'param', 'position': 0, 'allOf': [{'$ref': '#/definitions/City'}]}}, 'required': ['param'], 'definitions': {'City': {'title': 'City', 'type': 'object', 'properties': {'population': {'title': 'Population', 'type': 'integer'}, 'name': {'title': 'Name', 'type': 'string'}}}}}

    @pytest.mark.parametrize('obj, expected', [({'param': {'population': 100000, 'name': 'Example City'}}, True), ({'param': {'population': 100000}}, True), ({'param': {'name': 'Example City'}}, True), ({'param': {}}, True), ({'param': {'population': 'not an integer', 'name': 'Example City'}}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': {'population': 100000, 'name': 'Example City'}}, []), ({'param': {'population': 100000}}, []), ({'param': {'name': 'Example City'}}, []), ({'param': {}}, []), ({'param': {'population': 'not an integer', 'name': 'Example City'}}, ["'not an integer' is not valid under any of the given schemas"]), ({}, ["'param' is a required property"]), ({'param': None}, ["None is not of type 'object'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors

class TestArray:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        return {'title': 'Parameters', 'type': 'object', 'properties': {'param': {'title': 'param', 'position': 0, 'type': 'array', 'items': {}}}, 'required': ['param']}

    @pytest.mark.parametrize('obj, expected', [({'param': [1, 2, 3]}, True), ({'param': 'not an array'}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': [1, 2, 3]}, []), ({'param': 'not an array'}, ["'not an array' is not of type 'array'"]), ({}, ["'param' is a required property"]), ({'param': None}, ["None is not of type 'array'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors

class TestArrayOfStrings:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        return {'title': 'Parameters', 'type': 'object', 'properties': {'param': {'title': 'param', 'position': 0, 'type': 'array', 'items': {'type': 'string'}}}, 'required': ['param']}

    @pytest.mark.parametrize('obj, expected', [({'param': ['str1', 'str2']}, True), ({'param': [1, 2, 3]}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': ['str1', 'str2']}, []), ({'param': [1, 2, 3]}, ["1 is not of type 'string'", "2 is not of type 'string'", "3 is not of type 'string'"]), ({}, ["'param' is a required property"]), ({'param': None}, ["None is not of type 'array'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
