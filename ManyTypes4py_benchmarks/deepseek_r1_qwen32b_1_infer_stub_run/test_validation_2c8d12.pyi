from collections import deque
from typing import Any, Dict, List, Optional, Union
import jsonschema
import pytest

from prefect.utilities.schema_tools.hydration import (
    HydrationError,
    InvalidJSON,
    Placeholder,
    ValueNotFound,
)
from prefect.utilities.schema_tools.validation import (
    CircularSchemaRefError,
    build_error_obj,
    is_valid,
    preprocess_schema,
    prioritize_placeholder_errors,
    validate,
)

class MockValidationError(jsonschema.exceptions.ValidationError):
    message: str
    relative_path: deque
    instance: Any
    validator: Any

    def __init__(self, message: str, relative_path: deque, instance: Any = None, validator: Any = None) -> None:
        ...

class CatastrophicError(HydrationError):
    @property
    def message(self) -> str:
        ...

async def test_hydration_error_causes_validation_error() -> None:
    ...

async def test_circular_schema_ref() -> None:
    ...

async def test_ignore_required() -> None:
    ...

class TestNumber:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        ...

    @pytest.mark.parametrize('obj, expected', [({'param': 10}, True), ({'param': 'not an integer'}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        ...

    @pytest.mark.parametrize('obj, expected_errors', [({'param': 10}, []), ({'param': 'not an integer'}, ["'not an integer' is not of type 'integer'"]), ({}, ["'param' is a required property"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        ...

class TestBoolean:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        ...

    @pytest.mark.parametrize('obj, expected', [({'param': True}, True), ({'param': False}, True), ({'param': 'not a boolean'}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        ...

    @pytest.mark.parametrize('obj, expected_errors', [({'param': True}, []), ({'param': False}, []), ({'param': 'not a boolean'}, ["'not a boolean' is not of type 'boolean'"]), ({}, ["'param' is a required property"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        ...

class TestString:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        ...

    @pytest.mark.parametrize('obj, expected', [({'param': 'test string'}, True), ({'param': 123}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        ...

    @pytest.mark.parametrize('obj, expected_errors', [({'param': 'test string'}, []), ({'param': 123}, ["123 is not of type 'string'"]), ({}, ["'param' is a required property"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        ...

class TestDate:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        ...

    @pytest.mark.parametrize('obj, expected', [({'value': '2023-01-01'}, True), ({'value': 'not a date'}, False), ({'value': 123}, False), ({}, False), ({'value': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        ...

    @pytest.mark.parametrize('obj, expected_errors', [({'value': '2023-01-01'}, []), ({'value': 'not a date'}, ["'not a date' is not a 'date'"]), ({'value': 123}, ["123 is not of type 'string'"]), ({}, ["'value' is a required property"]), ({'value': None}, ["None is not of type 'string'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        ...

class TestDateTime:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        ...

    @pytest.mark.parametrize('obj, expected', [({'param': '2023-01-01T12:00:00Z'}, True), ({'param': 'not a datetime'}, False), ({'param': 123}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        ...

    @pytest.mark.parametrize('obj, expected_errors', [({'param': '2023-01-01T12:00:00Z'}, []), ({'param': 'not a date'}, ["'not a date' is not a 'date-time'"]), ({'param': 123}, ["123 is not of type 'string'"]), ({}, ["'param' is a required property"]), ({'param': None}, ["None is not of type 'string'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        ...

class TestDict:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        ...

    @pytest.mark.parametrize('obj, expected', [({'param': {'key': 'value'}}, True), ({'param': 'not a dict'}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        ...

    @pytest.mark.parametrize('obj, expected_errors', [({'param': {'key': 'value'}}, []), ({'param': 'not a dict'}, ["'not a dict' is not of type 'object'"]), ({}, ["'param' is a required property"]), ({'param': None}, ["None is not of type 'object'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        ...

class TestObject:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        ...

    @pytest.mark.parametrize('obj, expected', [({'param': {'population': 1, 'name': 'string'}}, True), ({'param': {'population': 'not an integer', 'name': 'string'}}, False), ({'param': {'population': 1}}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        ...

    @pytest.mark.parametrize('obj, expected_errors', [({'param': {'population': 1, 'name': 'string'}}, []), ({'param': {'population': 'not an integer', 'name': 'string'}}, ["'not an integer' is not of type 'integer'"]), ({'param': {'population': 1}}, ["'name' is a required property"]), ({}, ["'param' is a required property"]), ({'param': None}, ["None is not of type 'object'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        ...

class TestObjectOptionalParameters:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        ...

    @pytest.mark.parametrize('obj, expected', [({'param': {'population': 100000, 'name': 'Example City'}}, True), ({'param': {'population': 100000}}, True), ({'param': {'name': 'Example City'}}, True), ({'param': {}}, True), ({'param': {'population': 'not an integer', 'name': 'Example City'}}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        ...

    @pytest.mark.parametrize('obj, expected_errors', [({'param': {'population': 100000, 'name': 'Example City'}}, []), ({'param': {'population': 100000}}, []), ({'param': {'name': 'Example City'}}, []), ({'param': {}}, []), ({'param': {'population': 'not an integer', 'name': 'Example City'}}, ["'not an integer' is not valid under any of the given schemas"]), ({}, ["'param' is a required property"]), ({'param': None}, ["None is not of type 'object'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        ...

class TestArray:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        ...

    @pytest.mark.parametrize('obj, expected', [({'param': [1, 2, 3]}, True), ({'param': 'not an array'}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        ...

    @pytest.mark.parametrize('obj, expected_errors', [({'param': [1, 2, 3]}, []), ({'param': 'not an array'}, ["'not an array' is not of type 'array'"]), ({}, ["'param' is a required property"]), ({'param': None}, ["None is not of type 'array'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        ...

class TestArrayOfStrings:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        ...

    @pytest.mark.parametrize('obj, expected', [({'param': ['str1', 'str2']}, True), ({'param': [1, 2, 3]}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        ...

    @pytest.mark.parametrize('obj, expected_errors', [({'param': ['str1', 'str2']}, []), ({'param': [1, 2, 3]}, ["1 is not of type 'string'", "2 is not of type 'string'", "3 is not of type 'string'"]), ({}, ["'param' is a required property"]), ({'param': None}, ["None is not of type 'array'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        ...

class TestArrayOfObjects:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        ...

    @pytest.mark.parametrize('obj, expected', [({'param': [{'population': 1, 'name': 'string'}]}, True), ({'param': [{'population': 1}]}, False), ({'param': [{'population': 'string', 'name': 1}]}, False), ({'param': [1]}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        ...

    @pytest.mark.parametrize('obj, expected_errors', [({'param': [{'population': 1, 'name': 'string'}]}, []), ({'param': [{'population': 1}]}, ["'name' is a required property"]), ({'param': [{'population': 'string', 'name': 1}]}, ["'string' is not of type 'integer'", "1 is not of type 'string'"]), ({'param': [1]}, ["1 is not of type 'object'"]), ({}, ["'param' is a required property"]), ({'param': None}, ["None is not of type 'array'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        ...

class TestNestedObject:
    @pytest.fixture
    def schema(self) -> Dict[str, Any]:
        ...

    @pytest.mark.parametrize('obj, expected', [({'param': {'population': 100000, 'name': 'Raccoon City', 'state': {'name': 'South Dakota', 'bird': 'Blue Jay'}}}, True), ({'param': {'population': 'not an integer', 'name': 'Raccoon City', 'state': {'name': 'South Dakota', 'bird': 'Blue Jay'}}}, False), ({'param': {'population': 100000, 'name': 'Raccoon City', 'state': {'name': 'South Dakota'}}}, False), ({}, False), ({'param': None}, False)])
    def test_is_valid(self, schema: Dict[str, Any], obj: Dict[str, Any], expected: bool) -> None:
        ...

    @pytest.mark.parametrize('obj, expected_errors', [({'param': {'population': 100000, 'name': 'Raccoon City', 'state': {'name': 'South Dakota', 'bird': 'Blue Jay'}}}, []), ({'param': {'population': 'not an integer', 'name': 'Raccoon City', 'state': {'name': 'South Dakota', 'bird': 'Blue Jay'}}}, ["'not an integer' is not of type 'integer'"]), ({'param': {'population': 100000, 'name': 'Raccoon City', 'state': {'name': 'South Dakota'}}}, ["'bird' is a required property"]), ({}, ["'param' is a required property"]), ({'param': None}, ["None is not of type 'object'"])])
    def test_validate(self, schema: Dict[str, Any], obj: Dict[str, Any], expected_errors: List[str]) -> None:
        ...

class TestPrioritizePlaceholderErrors:
    def test_prioritize_placeholder_errors(self) -> None:
        ...

class TestBuildErrorObject:
    def test_field_missing(self) -> None:
        ...

    def test_single_field_error(self) -> None:
        ...

    def test_multiple_field_errors(self) -> None:
        ...

    def test_array_of_strings(self) -> None:
        ...

    def test_array_of_objects(self) -> None:
        ...

    async def test_root_level_error(self) -> None:
        ...

class TestBuildErrorObjectWithPlaceholders:
    def test_non_error_placeholder(self) -> None:
        ...

    def test_invalid_json(self) -> None:
        ...

    def test_invalid_json_with_detail(self) -> None:
        ...

    def test_value_not_found(self) -> None:
        ...

class TestPreprocessSchemaPydanticV1NullTypes:
    def test_pydantic_v1_required_int(self) -> None:
        ...

    def test_pydantic_v1_optional_int(self) -> None:
        ...

    def test_pydantic_v1_optional_int_default_none(self) -> None:
        ...

    def test_pydantic_v1_required_int_or_none(self) -> None:
        ...

    def test_pydantic_v1_optional_int_or_none_default_none(self) -> None:
        ...

    def test_pydantic_v1_optional_int_or_none_default_int(self) -> None:
        ...

    def test_pydantic_v1_model_required_int(self) -> None:
        ...

    def test_pydantic_v1_model_optional_int(self) -> None:
        ...

    def test_pydantic_v1_model_required_int_or_none(self) -> None:
        ...

    def test_pydantic_v1_model_optional_int_or_none_default_none(self) -> None:
        ...

    def test_pydantic_v1_model_optional_int_or_none_default_int(self) -> None:
        ...

class TestPreprocessSchemaPydanticV2NullTypes:
    def test_pydantic_v2_required_int(self) -> None:
        ...

    def test_pydantic_v2_optional_int(self) -> None:
        ...

    def test_pydantic_v2_optional_int_default_none(self) -> None:
        ...

    def test_pydantic_v2_required_int_or_none(self) -> None:
        ...

    def test_pydantic_v2_optional_int_or_none_default_none(self) -> None:
        ...

    def test_pydantic_v2_optional_int_or_none_default_int(self) -> None:
        ...

    def test_pydantic_v2_model_required_int(self) -> None:
        ...

    def test_pydantic_v2_model_optional_int(self) -> None:
        ...

    def test_pydantic_v2_model_optional_int_default_none(self) -> None:
        ...

    def test_pydantic_v2_model_required_int_or_none(self) -> None:
        ...

    def test_pydantic_v2_model_optional_int_or_none_default_none(self) -> None:
        ...

    def test_pydantic_v2_model_optional_int_or_none_default_int(self) -> None:
        ...

class TestPreprocessSchemaPydanticV1Tuples:
    async def test_pydantic_v1_single_type_tuple(self) -> None:
        ...

    async def test_pydantic_v1_union_type_tuple(self) -> None:
        ...

    async def test_pydantic_v1_ordered_multi_type_tuple(self) -> None:
        ...

    async def test_pydantic_v1_model_single_type_tuple(self) -> None:
        ...

    async def test_pydantic_v1_model_union_type_tuple(self) -> None:
        ...

    async def test_pydantic_v1_model_ordered_multi_type_tuple(self) -> None:
        ...

class TestPreprocessSchemaPydanticV2Tuples:
    def test_pydantic_v2_single_type_tuple(self) -> None:
        ...

    def test_pydantic_v2_union_type_tuple(self) -> None:
        ...

    def test_pydantic_v2_ordered_multi_type_tuple(self) -> None:
        ...

    def test_pydantic_v2_model_single_type_tuple(self) -> None:
        ...

    def test_pydantic_v2_model_union_type_tuple(self) -> None:
        ...

    def test_pydantic_v2_model_ordered_multi_type_tuple(self) -> None:
        ...