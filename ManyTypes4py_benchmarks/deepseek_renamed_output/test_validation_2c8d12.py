from collections import deque
from typing import Any, List, Optional, Union, Dict, Tuple, TypeVar, Generic, Callable
import jsonschema
import pytest
from prefect.utilities.schema_tools.hydration import HydrationError, InvalidJSON, Placeholder, ValueNotFound
from prefect.utilities.schema_tools.validation import CircularSchemaRefError, build_error_obj, is_valid, preprocess_schema, prioritize_placeholder_errors, validate


class MockValidationError(jsonschema.exceptions.ValidationError):
    def __init__(self, message: str, relative_path: List[Any], instance: Any = None, validator: Optional[str] = None) -> None:
        self.message = message
        self.relative_path = deque(relative_path)
        self.instance = instance
        self.validator = validator


async def func_eu271e65() -> None:
    error_msg = 'Something went real wrong!'

    class CatastrophicError(HydrationError):
        @property
        def func_xq0dv3lo(self) -> str:
            return error_msg

    schema: Dict[str, Any] = {'title': 'Parameters', 'type': 'object', 'properties': {'param': {'title': 'user', 'position': 0}}}
    values: Dict[str, Any] = {'param': CatastrophicError()}
    errors: List[jsonschema.exceptions.ValidationError] = validate(values, schema)
    assert len(errors) == 1
    assert errors[0].message == error_msg


async def func_ti0ordfl() -> None:
    circular_schema: Dict[str, Any] = {
        'title': 'Parameters', 
        'type': 'object',
        'properties': {
            'param': {
                'title': 'param', 
                'position': 0, 
                'allOf': [{'$ref': '#/definitions/City'}]
            }
        }, 
        'required': ['param'],
        'definitions': {
            'City': {
                'title': 'City', 
                'properties': {
                    'population': {'title': 'Population', 'type': 'integer'}, 
                    'name': {'title': 'Name', 'type': 'string'}
                }, 
                'required': ['population', 'name'], 
                'allOf': [{'$ref': '#/definitions/City'}]
            }
        }
    }
    with pytest.raises(CircularSchemaRefError):
        validate({'param': {"maybe a city, but we'll never know"}}, circular_schema)


async def func_josczk6z() -> None:
    schema: Dict[str, Any] = {
        'title': 'Parameters', 
        'type': 'object', 
        'properties': {
            'param': {'title': 'param', 'position': 0, 'type': 'integer'}
        }, 
        'required': ['param']
    }
    values: Dict[str, Any] = {}
    res: List[jsonschema.exceptions.ValidationError] = validate(values, schema, ignore_required=False)
    assert len(res) == 1
    assert res[0].message == "'param' is a required property"
    res = validate(values, schema, ignore_required=True)
    assert len(res) == 0


T = TypeVar('T')

class TestNumber:
    @pytest.fixture
    def func_w1k07de9(self) -> Dict[str, Any]:
        return {
            'title': 'Parameters', 
            'type': 'object', 
            'properties': {
                'param': {'title': 'param', 'position': 0, 'type': 'integer'}
            }, 
            'required': ['param']
        }

    @pytest.mark.parametrize('obj, expected', [
        ({'param': 10}, True), 
        ({'param': 'not an integer'}, False), 
        ({}, False), 
        ({'param': None}, False)
    ])
    def func_hewfo6hr(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [
        ({'param': 10}, []),
        ({'param': 'not an integer'}, ["'not an integer' is not of type 'integer'"]), 
        ({}, ["'param' is a required property"])
    ])
    def func_1s24poju(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestBoolean:
    @pytest.fixture
    def func_w1k07de9(self) -> Dict[str, Any]:
        return {
            'title': 'Parameters', 
            'type': 'object', 
            'properties': {
                'param': {'title': 'param', 'position': 0, 'type': 'boolean'}
            }, 
            'required': ['param']
        }

    @pytest.mark.parametrize('obj, expected', [
        ({'param': True}, True), 
        ({'param': False}, True), 
        ({'param': 'not a boolean'}, False), 
        ({}, False), 
        ({'param': None}, False)
    ])
    def func_hewfo6hr(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [
        ({'param': True}, []),
        ({'param': False}, []), 
        ({'param': 'not a boolean'}, ["'not a boolean' is not of type 'boolean'"]), 
        ({}, ["'param' is a required property"])
    ])
    def func_1s24poju(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestString:
    @pytest.fixture
    def func_w1k07de9(self) -> Dict[str, Any]:
        return {
            'title': 'Parameters', 
            'type': 'object', 
            'properties': {
                'param': {'title': 'param', 'position': 0, 'type': 'string'}
            }, 
            'required': ['param']
        }

    @pytest.mark.parametrize('obj, expected', [
        ({'param': 'test string'}, True), 
        ({'param': 123}, False), 
        ({}, False), 
        ({'param': None}, False)
    ])
    def func_hewfo6hr(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [
        ({'param': 'test string'}, []), 
        ({'param': 123}, ["123 is not of type 'string'"]), 
        ({}, ["'param' is a required property"])
    ])
    def func_1s24poju(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestDate:
    @pytest.fixture
    def func_w1k07de9(self) -> Dict[str, Any]:
        return {
            'title': 'Parameters', 
            'type': 'object', 
            'properties': {
                'value': {
                    'title': 'value', 
                    'position': 0, 
                    'type': 'string',
                    'format': 'date'
                }
            }, 
            'required': ['value']
        }

    @pytest.mark.parametrize('obj, expected', [
        ({'value': '2023-01-01'}, True), 
        ({'value': 'not a date'}, False), 
        ({'value': 123}, False), 
        ({}, False), 
        ({'value': None}, False)
    ])
    def func_hewfo6hr(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [
        ({'value': '2023-01-01'}, []), 
        ({'value': 'not a date'}, ["'not a date' is not a 'date'"]), 
        ({'value': 123}, ["123 is not of type 'string'"]), 
        ({}, ["'value' is a required property"]), 
        ({'value': None}, ["None is not of type 'string'"])
    ])
    def func_1s24poju(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestDateTime:
    @pytest.fixture
    def func_w1k07de9(self) -> Dict[str, Any]:
        return {
            'title': 'Parameters', 
            'type': 'object', 
            'properties': {
                'param': {
                    'title': 'param', 
                    'position': 0, 
                    'type': 'string',
                    'format': 'date-time'
                }
            }, 
            'required': ['param']
        }

    @pytest.mark.parametrize('obj, expected', [
        ({'param': '2023-01-01T12:00:00Z'}, True), 
        ({'param': 'not a datetime'}, False), 
        ({'param': 123}, False), 
        ({}, False), 
        ({'param': None}, False)
    ])
    def func_hewfo6hr(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [
        ({'param': '2023-01-01T12:00:00Z'}, []), 
        ({'param': 'not a date'}, ["'not a date' is not a 'date-time'"]), 
        ({'param': 123}, ["123 is not of type 'string'"]), 
        ({}, ["'param' is a required property"]), 
        ({'param': None}, ["None is not of type 'string'"])
    ])
    def func_1s24poju(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestDict:
    @pytest.fixture
    def func_w1k07de9(self) -> Dict[str, Any]:
        return {
            'title': 'Parameters', 
            'type': 'object', 
            'properties': {
                'param': {
                    'title': 'param', 
                    'position': 0, 
                    'type': 'object'
                }
            }, 
            'required': ['param']
        }

    @pytest.mark.parametrize('obj, expected', [
        ({'param': {'key': 'value'}}, True), 
        ({'param': 'not a dict'}, False), 
        ({}, False), 
        ({'param': None}, False)
    ])
    def func_hewfo6hr(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [
        ({'param': {'key': 'value'}}, []), 
        ({'param': 'not a dict'}, ["'not a dict' is not of type 'object'"]), 
        ({}, ["'param' is a required property"]), 
        ({'param': None}, ["None is not of type 'object'"])
    ])
    def func_1s24poju(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestObject:
    @pytest.fixture
    def func_w1k07de9(self) -> Dict[str, Any]:
        return {
            'title': 'Parameters', 
            'type': 'object', 
            'properties': {
                'param': {
                    'title': 'param', 
                    'position': 0, 
                    'allOf': [{'$ref': '#/definitions/City'}]
                }
            }, 
            'required': ['param'], 
            'definitions': {
                'City': {
                    'title': 'City', 
                    'type': 'object', 
                    'properties': {
                        'population': {'title': 'Population', 'type': 'integer'},
                        'name': {'title': 'Name', 'type': 'string'}
                    }, 
                    'required': ['population', 'name']
                }
            }
        }

    @pytest.mark.parametrize('obj, expected', [
        ({'param': {'population': 1, 'name': 'string'}}, True), 
        ({'param': {'population': 'not an integer', 'name': 'string'}}, False), 
        ({'param': {'population': 1}}, False), 
        ({}, False), 
        ({'param': None}, False)
    ])
    def func_hewfo6hr(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [
        ({'param': {'population': 1, 'name': 'string'}}, []), 
        ({'param': {'population': 'not an integer', 'name': 'string'}}, ["'not an integer' is not of type 'integer'"]), 
        ({'param': {'population': 1}}, ["'name' is a required property"]), 
        ({}, ["'param' is a required property"]), 
        ({'param': None}, ["None is not of type 'object'"])
    ])
    def func_1s24poju(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestObjectOptionalParameters:
    @pytest.fixture
    def func_w1k07de9(self) -> Dict[str, Any]:
        return {
            'title': 'Parameters', 
            'type': 'object', 
            'properties': {
                'param': {
                    'title': 'param', 
                    'position': 0, 
                    'allOf': [{'$ref': '#/definitions/City'}]
                }
            }, 
            'required': ['param'], 
            'definitions': {
                'City': {
                    'title': 'City', 
                    'type': 'object', 
                    'properties': {
                        'population': {'title': 'Population', 'type': 'integer'},
                        'name': {'title': 'Name', 'type': 'string'}
                    }
                }
            }
        }

    @pytest.mark.parametrize('obj, expected', [
        ({'param': {'population': 100000, 'name': 'Example City'}}, True), 
        ({'param': {'population': 100000}}, True), 
        ({'param': {'name': 'Example City'}}, True), 
        ({'param': {}}, True), 
        ({'param': {'population': 'not an integer', 'name': 'Example City'}}, False), 
        ({}, False), 
        ({'param': None}, False)
    ])
    def func_hewfo6hr(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [
        ({'param': {'population': 100000, 'name': 'Example City'}}, []), 
        ({'param': {'population': 100000}}, []), 
        ({'param': {'name': 'Example City'}}, []), 
        ({'param': {}}, []), 
        ({'param': {'population': 'not an integer', 'name': 'Example City'}}, ["'not an integer' is not valid under any of the given schemas"]), 
        ({}, ["'param' is a required property"]), 
        ({'param': None}, ["None is not of type 'object'"])
    ])
    def func_1s24poju(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        errors: List[jsonschema.exceptions.ValidationError] = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestArray:
    @pytest.fixture
    def func_w1k07de9(self) -> Dict[str, Any]:
        return {
            'title': 'Parameters', 
            'type': 'object', 
            'properties': {
                'param': {
                    'title': 'param', 
                    'position': 0, 
                    'type': 'array',
                    'items': {}
                }
            }, 
            'required': ['param']
        }

    @pytest.mark.parametrize('obj, expected