from collections import deque
from typing import Any, List, Optional, Union
import jsonschema
import pytest
from prefect.utilities.schema_tools.hydration import HydrationError, InvalidJSON, Placeholder, ValueNotFound
from prefect.utilities.schema_tools.validation import CircularSchemaRefError, build_error_obj, is_valid, preprocess_schema, prioritize_placeholder_errors, validate


class MockValidationError(jsonschema.exceptions.ValidationError):

    def __init__(self, message, relative_path, instance=None, validator=None):
        self.message = message
        self.relative_path = deque(relative_path)
        self.instance = instance
        self.validator = validator


async def func_eu271e65():
    error_msg = 'Something went real wrong!'


    class CatastrophicError(HydrationError):

        @property
        def func_xq0dv3lo(self):
            return error_msg
    schema = {'title': 'Parameters', 'type': 'object', 'properties': {
        'param': {'title': 'user', 'position': 0}}}
    values = {'param': CatastrophicError()}
    errors = validate(values, schema)
    assert len(errors) == 1
    assert errors[0].message == error_msg


async def func_ti0ordfl():
    circular_schema = {'title': 'Parameters', 'type': 'object',
        'properties': {'param': {'title': 'param', 'position': 0, 'allOf':
        [{'$ref': '#/definitions/City'}]}}, 'required': ['param'],
        'definitions': {'City': {'title': 'City', 'properties': {
        'population': {'title': 'Population', 'type': 'integer'}, 'name': {
        'title': 'Name', 'type': 'string'}}, 'required': ['population',
        'name'], 'allOf': [{'$ref': '#/definitions/City'}]}}}
    with pytest.raises(CircularSchemaRefError):
        validate({'param': {"maybe a city, but we'll never know"}},
            circular_schema)


async def func_josczk6z():
    schema = {'title': 'Parameters', 'type': 'object', 'properties': {
        'param': {'title': 'param', 'position': 0, 'type': 'integer'}},
        'required': ['param']}
    values = {}
    res = validate(values, schema, ignore_required=False)
    assert len(res) == 1
    assert res[0].message == "'param' is a required property"
    res = validate(values, schema, ignore_required=True)
    assert len(res) == 0


class TestNumber:

    @pytest.fixture
    def func_w1k07de9(self):
        return {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'type': 'integer'}},
            'required': ['param']}

    @pytest.mark.parametrize('obj, expected', [({'param': 10}, True), ({
        'param': 'not an integer'}, False), ({}, False), ({'param': None}, 
        False)])
    def func_hewfo6hr(self, schema, obj, expected):
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': 10}, []),
        ({'param': 'not an integer'}, [
        "'not an integer' is not of type 'integer'"]), ({}, [
        "'param' is a required property"])])
    def func_1s24poju(self, schema, obj, expected_errors):
        errors = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestBoolean:

    @pytest.fixture
    def func_w1k07de9(self):
        return {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'type': 'boolean'}},
            'required': ['param']}

    @pytest.mark.parametrize('obj, expected', [({'param': True}, True), ({
        'param': False}, True), ({'param': 'not a boolean'}, False), ({}, 
        False), ({'param': None}, False)])
    def func_hewfo6hr(self, schema, obj, expected):
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': True}, []),
        ({'param': False}, []), ({'param': 'not a boolean'}, [
        "'not a boolean' is not of type 'boolean'"]), ({}, [
        "'param' is a required property"])])
    def func_1s24poju(self, schema, obj, expected_errors):
        errors = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestString:

    @pytest.fixture
    def func_w1k07de9(self):
        return {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'type': 'string'}},
            'required': ['param']}

    @pytest.mark.parametrize('obj, expected', [({'param': 'test string'}, 
        True), ({'param': 123}, False), ({}, False), ({'param': None}, False)])
    def func_hewfo6hr(self, schema, obj, expected):
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param':
        'test string'}, []), ({'param': 123}, [
        "123 is not of type 'string'"]), ({}, [
        "'param' is a required property"])])
    def func_1s24poju(self, schema, obj, expected_errors):
        errors = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestDate:

    @pytest.fixture
    def func_w1k07de9(self):
        return {'title': 'Parameters', 'type': 'object', 'properties': {
            'value': {'title': 'value', 'position': 0, 'type': 'string',
            'format': 'date'}}, 'required': ['value']}

    @pytest.mark.parametrize('obj, expected', [({'value': '2023-01-01'}, 
        True), ({'value': 'not a date'}, False), ({'value': 123}, False), (
        {}, False), ({'value': None}, False)])
    def func_hewfo6hr(self, schema, obj, expected):
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'value':
        '2023-01-01'}, []), ({'value': 'not a date'}, [
        "'not a date' is not a 'date'"]), ({'value': 123}, [
        "123 is not of type 'string'"]), ({}, [
        "'value' is a required property"]), ({'value': None}, [
        "None is not of type 'string'"])])
    def func_1s24poju(self, schema, obj, expected_errors):
        errors = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestDateTime:

    @pytest.fixture
    def func_w1k07de9(self):
        return {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'type': 'string',
            'format': 'date-time'}}, 'required': ['param']}

    @pytest.mark.parametrize('obj, expected', [({'param':
        '2023-01-01T12:00:00Z'}, True), ({'param': 'not a datetime'}, False
        ), ({'param': 123}, False), ({}, False), ({'param': None}, False)])
    def func_hewfo6hr(self, schema, obj, expected):
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param':
        '2023-01-01T12:00:00Z'}, []), ({'param': 'not a date'}, [
        "'not a date' is not a 'date-time'"]), ({'param': 123}, [
        "123 is not of type 'string'"]), ({}, [
        "'param' is a required property"]), ({'param': None}, [
        "None is not of type 'string'"])])
    def func_1s24poju(self, schema, obj, expected_errors):
        errors = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestDict:

    @pytest.fixture
    def func_w1k07de9(self):
        return {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'type': 'object'}},
            'required': ['param']}

    @pytest.mark.parametrize('obj, expected', [({'param': {'key': 'value'}},
        True), ({'param': 'not a dict'}, False), ({}, False), ({'param':
        None}, False)])
    def func_hewfo6hr(self, schema, obj, expected):
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': {'key':
        'value'}}, []), ({'param': 'not a dict'}, [
        "'not a dict' is not of type 'object'"]), ({}, [
        "'param' is a required property"]), ({'param': None}, [
        "None is not of type 'object'"])])
    def func_1s24poju(self, schema, obj, expected_errors):
        errors = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestObject:

    @pytest.fixture
    def func_w1k07de9(self):
        return {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'allOf': [{'$ref':
            '#/definitions/City'}]}}, 'required': ['param'], 'definitions':
            {'City': {'title': 'City', 'type': 'object', 'properties': {
            'population': {'title': 'Population', 'type': 'integer'},
            'name': {'title': 'Name', 'type': 'string'}}, 'required': [
            'population', 'name']}}}

    @pytest.mark.parametrize('obj, expected', [({'param': {'population': 1,
        'name': 'string'}}, True), ({'param': {'population':
        'not an integer', 'name': 'string'}}, False), ({'param': {
        'population': 1}}, False), ({}, False), ({'param': None}, False)])
    def func_hewfo6hr(self, schema, obj, expected):
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': {
        'population': 1, 'name': 'string'}}, []), ({'param': {'population':
        'not an integer', 'name': 'string'}}, [
        "'not an integer' is not of type 'integer'"]), ({'param': {
        'population': 1}}, ["'name' is a required property"]), ({}, [
        "'param' is a required property"]), ({'param': None}, [
        "None is not of type 'object'"])])
    def func_1s24poju(self, schema, obj, expected_errors):
        errors = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestObjectOptionalParameters:

    @pytest.fixture
    def func_w1k07de9(self):
        return {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'allOf': [{'$ref':
            '#/definitions/City'}]}}, 'required': ['param'], 'definitions':
            {'City': {'title': 'City', 'type': 'object', 'properties': {
            'population': {'title': 'Population', 'type': 'integer'},
            'name': {'title': 'Name', 'type': 'string'}}}}}

    @pytest.mark.parametrize('obj, expected', [({'param': {'population': 
        100000, 'name': 'Example City'}}, True), ({'param': {'population': 
        100000}}, True), ({'param': {'name': 'Example City'}}, True), ({
        'param': {}}, True), ({'param': {'population': 'not an integer',
        'name': 'Example City'}}, False), ({}, False), ({'param': None}, 
        False)])
    def func_hewfo6hr(self, schema, obj, expected):
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': {
        'population': 100000, 'name': 'Example City'}}, []), ({'param': {
        'population': 100000}}, []), ({'param': {'name': 'Example City'}},
        []), ({'param': {}}, []), ({'param': {'population':
        'not an integer', 'name': 'Example City'}}, [
        "'not an integer' is not valid under any of the given schemas"]), (
        {}, ["'param' is a required property"]), ({'param': None}, [
        "None is not of type 'object'"])])
    def func_1s24poju(self, schema, obj, expected_errors):
        errors = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestArray:

    @pytest.fixture
    def func_w1k07de9(self):
        return {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'type': 'array',
            'items': {}}}, 'required': ['param']}

    @pytest.mark.parametrize('obj, expected', [({'param': [1, 2, 3]}, True),
        ({'param': 'not an array'}, False), ({}, False), ({'param': None}, 
        False)])
    def func_hewfo6hr(self, schema, obj, expected):
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': [1, 2, 3]},
        []), ({'param': 'not an array'}, [
        "'not an array' is not of type 'array'"]), ({}, [
        "'param' is a required property"]), ({'param': None}, [
        "None is not of type 'array'"])])
    def func_1s24poju(self, schema, obj, expected_errors):
        errors = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestArrayOfStrings:

    @pytest.fixture
    def func_w1k07de9(self):
        return {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'type': 'array',
            'items': {'type': 'string'}}}, 'required': ['param']}

    @pytest.mark.parametrize('obj, expected', [({'param': ['str1', 'str2']},
        True), ({'param': [1, 2, 3]}, False), ({}, False), ({'param': None},
        False)])
    def func_hewfo6hr(self, schema, obj, expected):
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': ['str1',
        'str2']}, []), ({'param': [1, 2, 3]}, ["1 is not of type 'string'",
        "2 is not of type 'string'", "3 is not of type 'string'"]), ({}, [
        "'param' is a required property"]), ({'param': None}, [
        "None is not of type 'array'"])])
    def func_1s24poju(self, schema, obj, expected_errors):
        errors = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestArrayOfObjects:

    @pytest.fixture
    def func_w1k07de9(self):
        return {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'type': 'array',
            'items': {'$ref': '#/definitions/City'}}}, 'required': ['param'
            ], 'definitions': {'City': {'title': 'City', 'type': 'object',
            'properties': {'population': {'title': 'Population', 'type':
            'integer'}, 'name': {'title': 'Name', 'type': 'string'}},
            'required': ['population', 'name']}}}

    @pytest.mark.parametrize('obj, expected', [({'param': [{'population': 1,
        'name': 'string'}]}, True), ({'param': [{'population': 1}]}, False),
        ({'param': [{'population': 'string', 'name': 1}]}, False), ({
        'param': [1]}, False), ({}, False), ({'param': None}, False)])
    def func_hewfo6hr(self, schema, obj, expected):
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': [{
        'population': 1, 'name': 'string'}]}, []), ({'param': [{
        'population': 1}]}, ["'name' is a required property"]), ({'param':
        [{'population': 'string', 'name': 1}]}, [
        "'string' is not of type 'integer'", "1 is not of type 'string'"]),
        ({'param': [1]}, ["1 is not of type 'object'"]), ({}, [
        "'param' is a required property"]), ({'param': None}, [
        "None is not of type 'array'"])])
    def func_1s24poju(self, schema, obj, expected_errors):
        errors = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestNestedObject:

    @pytest.fixture
    def func_w1k07de9(self):
        return {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'allOf': [{'$ref':
            '#/definitions/City'}]}}, 'required': ['param'], 'definitions':
            {'State': {'title': 'State', 'type': 'object', 'properties': {
            'name': {'title': 'Name', 'type': 'string'}, 'bird': {'title':
            'Bird', 'type': 'string'}}, 'required': ['name', 'bird']},
            'City': {'title': 'City', 'type': 'object', 'properties': {
            'population': {'title': 'Population', 'type': 'integer'},
            'name': {'title': 'Name', 'type': 'string'}, 'state': {'$ref':
            '#/definitions/State'}}, 'required': ['population', 'name',
            'state']}}}

    @pytest.mark.parametrize('obj, expected', [({'param': {'population': 
        100000, 'name': 'Raccoon City', 'state': {'name': 'South Dakota',
        'bird': 'Blue Jay'}}}, True), ({'param': {'population':
        'not an integer', 'name': 'Raccoon City', 'state': {'name':
        'South Dakota', 'bird': 'Blue Jay'}}}, False), ({'param': {
        'population': 100000, 'name': 'Raccoon City', 'state': {'name':
        'South Dakota'}}}, False), ({}, False), ({'param': None}, False)])
    def func_hewfo6hr(self, schema, obj, expected):
        assert is_valid(obj, schema) == expected

    @pytest.mark.parametrize('obj, expected_errors', [({'param': {
        'population': 100000, 'name': 'Raccoon City', 'state': {'name':
        'South Dakota', 'bird': 'Blue Jay'}}}, []), ({'param': {
        'population': 'not an integer', 'name': 'Raccoon City', 'state': {
        'name': 'South Dakota', 'bird': 'Blue Jay'}}}, [
        "'not an integer' is not of type 'integer'"]), ({'param': {
        'population': 100000, 'name': 'Raccoon City', 'state': {'name':
        'South Dakota'}}}, ["'bird' is a required property"]), ({}, [
        "'param' is a required property"]), ({'param': None}, [
        "None is not of type 'object'"])])
    def func_1s24poju(self, schema, obj, expected_errors):
        errors = validate(obj, schema)
        assert [e.message for e in errors] == expected_errors


class TestPrioritizePlaceholderErrors:

    def func_w79gd9hg(self):
        errors = [MockValidationError(message=
            "InvalidJSON() is not of type 'string", relative_path=['x'],
            instance=InvalidJSON(), validator='type'), MockValidationError(
            message=
            'Invalid JSON: Unterminated string starting at: line 1 column 1 (char 0)'
            , relative_path=['x'], instance=InvalidJSON(), validator=
            '_placeholders'), MockValidationError(message=
            "1 is not of type 'string", relative_path=['y'], instance=1,
            validator='type')]
        prioritized_errors = prioritize_placeholder_errors(errors)
        assert len(prioritized_errors) == 2
        assert prioritized_errors[0].validator == '_placeholders'
        assert prioritized_errors[1].validator == 'type'
        assert prioritized_errors[1].instance == 1


class TestBuildErrorObject:

    def func_2bt4hmvx(self):
        errors = [MockValidationError(message=
            "'param' is a required property", relative_path=[], instance={
            'not param': 1}, validator='required')]
        error_obj = build_error_obj(errors)
        assert error_obj == {'valid': False, 'errors': [{'property':
            'param', 'errors': ["'param' is a required property"]}]}

    def func_yele5gv3(self):
        errors = [MockValidationError(message=
            "'not an integer' is not of type 'integer'", relative_path=[
            'param'], instance='not an integer')]
        error_obj = build_error_obj(errors)
        assert error_obj == {'valid': False, 'errors': [{'property':
            'param', 'errors': ["'not an integer' is not of type 'integer'"]}]}

    def func_qpqmt4v5(self):
        errors = [MockValidationError(message="1 is not of type 'string'",
            relative_path=['param']), MockValidationError(message=
            "2 is not of type 'string'", relative_path=['other_param'])]
        error_obj = build_error_obj(errors)
        assert error_obj == {'valid': False, 'errors': [{'property':
            'param', 'errors': ["1 is not of type 'string'"]}, {'property':
            'other_param', 'errors': ["2 is not of type 'string'"]}]}

    def func_lq12edbs(self):
        errors = [MockValidationError(message="2 is not of type 'string'",
            relative_path=['param', 1]), MockValidationError(message=
            "3 is not of type 'string'", relative_path=['param', 2])]
        error_obj = build_error_obj(errors)
        assert error_obj == {'valid': False, 'errors': [{'property':
            'param', 'errors': [{'index': 1, 'errors': [
            "2 is not of type 'string'"]}, {'index': 2, 'errors': [
            "3 is not of type 'string'"]}]}]}

    def func_lfdm6l56(self):
        errors = [MockValidationError(message=
            "'not an integer' is not of type 'integer'", relative_path=[
            'param', 0, 'population'], instance='not an integer'),
            MockValidationError(message="1 is not of type 'string'",
            relative_path=['param', 0, 'name'], instance=1)]
        error_obj = build_error_obj(errors)
        assert error_obj == {'valid': False, 'errors': [{'property':
            'param', 'errors': [{'index': 0, 'errors': [{'property':
            'population', 'errors': [
            "'not an integer' is not of type 'integer'"]}, {'property':
            'name', 'errors': ["1 is not of type 'string'"]}]}]}]}

    async def func_1g8682d0(self):
        errors = [MockValidationError(message='Root level error!!',
            relative_path=[], instance=None)]
        error_obj = build_error_obj(errors)
        assert error_obj == {'valid': False, 'errors': ['Root level error!!']}


class TestBuildErrorObjectWithPlaceholders:

    def func_05srz1fa(self):


        class ValidPlaceholder(Placeholder):
            pass
        placeholder = ValidPlaceholder()
        assert not placeholder.is_error
        errors = [MockValidationError(message=
            "'object at XXX is not of type 'string'", relative_path=[
            'param'], instance=placeholder)]
        error_obj = build_error_obj(errors)
        assert error_obj == {'valid': True, 'errors': []}

    def func_8agmft9r(self):
        errors = [MockValidationError(message=InvalidJSON().message,
            relative_path=['param'], instance=InvalidJSON())]
        error_obj = build_error_obj(errors)
        assert error_obj == {'valid': False, 'errors': [{'property':
            'param', 'errors': ['Invalid JSON']}]}

    def func_gcj0b943(self):
        errors = [MockValidationError(message=InvalidJSON(detail=
            'error at char 5').message, relative_path=['param'], instance=
            InvalidJSON(detail='error at char 5'))]
        error_obj = build_error_obj(errors)
        assert error_obj == {'valid': False, 'errors': [{'property':
            'param', 'errors': ['Invalid JSON: error at char 5']}]}

    def func_gqtskfx1(self):
        errors = [MockValidationError(message=ValueNotFound().message,
            relative_path=['param'], instance=ValueNotFound())]
        error_obj = build_error_obj(errors)
        assert error_obj == {'valid': False, 'errors': [{'property':
            'param', 'errors': ["Missing 'value' key in __prefect object"]}]}


class TestPreprocessSchemaPydanticV1NullTypes:

    def func_2ha9c3kq(self):
        """
        required_int: int
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'required_int': {'title': 'required_int', 'position': 0, 'type':
            'integer'}}, 'required': ['required_int']}
        preprocessed_schema = preprocess_schema(schema)
        assert schema == preprocessed_schema

    def func_jdw9u11z(self):
        """
        optional_int: int = 10
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'optional_int': {'title': 'optional_int', 'default': 10,
            'position': 0, 'type': 'integer'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_rut4mzip(self):
        """
        optional_int_default_none:Optional[int] = None
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'optional_int_default_none': {'title':
            'optional_int_default_none', 'position': 0, 'type': 'integer'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == {'title': 'Parameters', 'type':
            'object', 'properties': {'optional_int_default_none': {'title':
            'optional_int_default_none', 'position': 0, 'anyOf': [{'type':
            'integer'}, {'type': 'null'}]}}}

    def func_889gd9j7(self):
        """
        required_int_or_none: int | None
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'required_int_or_none': {'title': 'required_int_or_none',
            'position': 0, 'type': 'integer'}}, 'required': [
            'required_int_or_none']}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_ckyr2est(self):
        """
        optional_int_or_none_default_none: int | None = None
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'optional_int_or_none_default_none': {'title':
            'optional_int_or_none_default_none', 'position': 0, 'type':
            'integer'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == {'title': 'Parameters', 'type':
            'object', 'properties': {'optional_int_or_none_default_none': {
            'title': 'optional_int_or_none_default_none', 'position': 0,
            'anyOf': [{'type': 'integer'}, {'type': 'null'}]}}}

    def func_51y3hdeu(self):
        """
        optional_int_or_none_default_int: int | None = 10
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'optional_int_or_none_default_int': {'title':
            'optional_int_or_none_default_int', 'default': 10, 'position': 
            0, 'type': 'integer'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_gfxm5pmy(self):
        """
        class MyModel(BaseModel):
            required_int: int
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'allOf': [{'$ref':
            '#/definitions/MyModel'}]}}, 'required': ['param'],
            'definitions': {'MyModel': {'title': 'MyModel', 'type':
            'object', 'properties': {'required_int': {'title':
            'Required Int', 'type': 'integer'}}, 'required': ['required_int']}}
            }
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_h1m668e4(self):
        """
        class MyModel(BaseModel):
            optional_int: int = 10
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'allOf': [{'$ref':
            '#/definitions/MyModel'}]}}, 'required': ['param'],
            'definitions': {'MyModel': {'title': 'MyModel', 'type':
            'object', 'properties': {'optional_int': {'title':
            'Optional Int', 'default': 10, 'type': 'integer'}}}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_xtzjlhyt(self):
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'allOf': [{'$ref':
            '#/definitions/MyModel'}]}}, 'required': ['param'],
            'definitions': {'MyModel': {'title': 'MyModel', 'type':
            'object', 'properties': {'required_int_or_none': {'title':
            'Required Int Or None', 'type': 'integer'}}}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == {'title': 'Parameters', 'type':
            'object', 'properties': {'param': {'title': 'param', 'position':
            0, 'allOf': [{'$ref': '#/definitions/MyModel'}]}}, 'required':
            ['param'], 'definitions': {'MyModel': {'title': 'MyModel',
            'type': 'object', 'properties': {'required_int_or_none': {
            'title': 'Required Int Or None', 'anyOf': [{'type': 'integer'},
            {'type': 'null'}]}}}}}

    def func_cpqs11sn(self):
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'allOf': [{'$ref':
            '#/definitions/MyModel'}]}}, 'required': ['param'],
            'definitions': {'MyModel': {'title': 'MyModel', 'type':
            'object', 'properties': {'optional_int_or_none_default_none': {
            'title': 'Optional Int Or None Default None', 'type': 'integer'}}}}
            }
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == {'title': 'Parameters', 'type':
            'object', 'properties': {'param': {'title': 'param', 'position':
            0, 'allOf': [{'$ref': '#/definitions/MyModel'}]}}, 'required':
            ['param'], 'definitions': {'MyModel': {'title': 'MyModel',
            'type': 'object', 'properties': {
            'optional_int_or_none_default_none': {'title':
            'Optional Int Or None Default None', 'anyOf': [{'type':
            'integer'}, {'type': 'null'}]}}}}}

    def func_wk0rqf0f(self):
        """
        class MyModel(BaseModel):
            optional_int_or_none_default_int: int | None = 10
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'allOf': [{'$ref':
            '#/definitions/MyModel'}]}}, 'required': ['param'],
            'definitions': {'MyModel': {'title': 'MyModel', 'type':
            'object', 'properties': {'optional_int_or_none_default_int': {
            'title': 'Optional Int Or None Default Int', 'default': 10,
            'type': 'integer'}}}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema


class TestPreprocessSchemaPydanticV2NullTypes:

    def func_r5nfjgng(self):
        """
        required_int: int
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'required_int': {'position': 0, 'title': 'required_int', 'type':
            'integer'}}, 'required': ['required_int']}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_68qyemiq(self):
        """
        optional_int: int = 10
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'optional_int': {'default': 10, 'position': 0, 'title':
            'optional_int', 'type': 'integer'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_ve1beile(self):
        """
        optional_int_default_none:Optional[int] = None
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'optional_int_default_none': {'default': None, 'position': 0,
            'title': 'optional_int_default_none', 'type': 'integer'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_9lkwlxqq(self):
        """
        required_int_or_none: int | None
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'required_int_or_none': {'anyOf': [{'type': 'integer'}, {'type':
            'null'}], 'position': 0, 'title': 'required_int_or_none'}},
            'required': ['required_int_or_none']}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_satamu3e(self):
        """
        optional_int_or_none_default_none: int | None = None
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'optional_int_or_none_default_none': {'anyOf': [{'type':
            'integer'}, {'type': 'null'}], 'default': None, 'position': 0,
            'title': 'optional_int_or_none_default_none'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_08uaen80(self):
        """
        optional_int_or_none_default_int: int | None = 10
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'optional_int_or_none_default_int': {'anyOf': [{'type':
            'integer'}, {'type': 'null'}], 'default': 10, 'position': 0,
            'title': 'optional_int_or_none_default_int'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_m6wdnrc8(self):
        """
        required_int: int
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'allOf': [{'$ref': '#/definitions/MyModel'}],
            'position': 0, 'title': 'param'}}, 'required': ['param'],
            'definitions': {'MyModel': {'properties': {'required_int': {
            'title': 'Required Int', 'type': 'integer'}}, 'required': [
            'required_int'], 'title': 'MyModel', 'type': 'object'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_6ryk3c3f(self):
        """
        optional_int: int = 10
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'allOf': [{'$ref': '#/definitions/MyModel'}],
            'position': 0, 'title': 'param'}}, 'required': ['param'],
            'definitions': {'MyModel': {'properties': {'optional_int': {
            'default': 10, 'title': 'Optional Int', 'type': 'integer'}},
            'title': 'MyModel', 'type': 'object'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_7dkj3zum(self):
        """
        optional_int_default_none:Optional[int] = None
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'allOf': [{'$ref': '#/definitions/MyModel'}],
            'position': 0, 'title': 'param'}}, 'required': ['param'],
            'definitions': {'MyModel': {'properties': {
            'optional_int_default_none': {'default': None, 'title':
            'Optional Int Default None', 'type': 'integer'}}, 'title':
            'MyModel', 'type': 'object'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_7k19ddx0(self):
        """
        required_int_or_none: int | None
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'allOf': [{'$ref': '#/definitions/MyModel'}],
            'position': 0, 'title': 'param'}}, 'required': ['param'],
            'definitions': {'MyModel': {'properties': {
            'required_int_or_none': {'anyOf': [{'type': 'integer'}, {'type':
            'null'}], 'title': 'Required Int Or None'}}, 'required': [
            'required_int_or_none'], 'title': 'MyModel', 'type': 'object'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_ldje4688(self):
        """
        optional_int_or_none_default_none: int | None = None
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'allOf': [{'$ref': '#/definitions/MyModel'}],
            'position': 0, 'title': 'param'}}, 'required': ['param'],
            'definitions': {'MyModel': {'properties': {
            'optional_int_or_none_default_none': {'anyOf': [{'type':
            'integer'}, {'type': 'null'}], 'default': None, 'title':
            'Optional Int Or None Default None'}}, 'title': 'MyModel',
            'type': 'object'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_eb927ctw(self):
        """
        optional_int_or_none_default_int: int | None = 10
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'allOf': [{'$ref': '#/definitions/MyModel'}],
            'position': 0, 'title': 'param'}}, 'required': ['param'],
            'definitions': {'MyModel': {'properties': {
            'optional_int_or_none_default_int': {'anyOf': [{'type':
            'integer'}, {'type': 'null'}], 'default': 10, 'title':
            'Optional Int Or None Default Int'}}, 'title': 'MyModel',
            'type': 'object'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema


class TestPreprocessSchemaPydanticV1Tuples:

    async def func_hpjzczdk(self):
        """
        single_type_tuple: tuple[str]
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'single_type_tuple': {'title': 'single_type_tuple', 'position':
            0, 'type': 'array', 'minItems': 1, 'maxItems': 1, 'items': [{
            'type': 'string'}]}}, 'required': ['single_type_tuple']}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == {'title': 'Parameters', 'type':
            'object', 'properties': {'single_type_tuple': {'title':
            'single_type_tuple', 'position': 0, 'type': 'array', 'minItems':
            1, 'maxItems': 1, 'prefixItems': [{'type': 'string'}]}},
            'required': ['single_type_tuple']}

    async def func_c6ysld64(self):
        """
        union_type_tuple: [str | int]
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'union_type_tuple': {'title': 'union_type_tuple', 'position': 0,
            'type': 'array', 'minItems': 1, 'maxItems': 1, 'items': [{
            'anyOf': [{'type': 'string'}, {'type': 'integer'}]}]}},
            'required': ['union_type_tuple']}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == {'title': 'Parameters', 'type':
            'object', 'properties': {'union_type_tuple': {'title':
            'union_type_tuple', 'position': 0, 'type': 'array', 'minItems':
            1, 'maxItems': 1, 'prefixItems': [{'anyOf': [{'type': 'string'},
            {'type': 'integer'}]}]}}, 'required': ['union_type_tuple']}

    async def func_tg864xg9(self):
        """
        ordered_multi_type_tuple: tuple[str, int]
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'ordered_multi_type_tuple': {'title':
            'ordered_multi_type_tuple', 'position': 0, 'type': 'array',
            'minItems': 2, 'maxItems': 2, 'items': [{'type': 'string'}, {
            'type': 'integer'}]}}, 'required': ['ordered_multi_type_tuple']}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == {'title': 'Parameters', 'type':
            'object', 'properties': {'ordered_multi_type_tuple': {'title':
            'ordered_multi_type_tuple', 'position': 0, 'type': 'array',
            'minItems': 2, 'maxItems': 2, 'prefixItems': [{'type': 'string'
            }, {'type': 'integer'}]}}, 'required': ['ordered_multi_type_tuple']
            }

    async def func_bs63qasv(self):
        """
        class MyModel(BaseModel):
            single_type_tuple: tuple[str]
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'allOf': [{'$ref':
            '#/definitions/MyModel'}]}}, 'required': ['param'],
            'definitions': {'MyModel': {'title': 'MyModel', 'type':
            'object', 'properties': {'single_type_tuple': {'title':
            'Single Type Tuple', 'type': 'array', 'minItems': 1, 'maxItems':
            1, 'items': [{'type': 'string'}]}}, 'required': [
            'single_type_tuple']}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == {'title': 'Parameters', 'type':
            'object', 'properties': {'param': {'title': 'param', 'position':
            0, 'allOf': [{'$ref': '#/definitions/MyModel'}]}}, 'required':
            ['param'], 'definitions': {'MyModel': {'title': 'MyModel',
            'type': 'object', 'properties': {'single_type_tuple': {'title':
            'Single Type Tuple', 'type': 'array', 'minItems': 1, 'maxItems':
            1, 'prefixItems': [{'type': 'string'}]}}, 'required': [
            'single_type_tuple']}}}

    async def func_mqwl41c6(self):
        """
        class MyModel(BaseModel):
            union_type_tuple: tuple[str | int]
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'allOf': [{'$ref':
            '#/definitions/MyModel'}]}}, 'required': ['param'],
            'definitions': {'MyModel': {'title': 'MyModel', 'type':
            'object', 'properties': {'union_type_tuple': {'title':
            'Union Type Tuple', 'type': 'array', 'minItems': 1, 'maxItems':
            1, 'items': [{'anyOf': [{'type': 'string'}, {'type': 'integer'}
            ]}]}}, 'required': ['union_type_tuple']}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == {'title': 'Parameters', 'type':
            'object', 'properties': {'param': {'title': 'param', 'position':
            0, 'allOf': [{'$ref': '#/definitions/MyModel'}]}}, 'required':
            ['param'], 'definitions': {'MyModel': {'title': 'MyModel',
            'type': 'object', 'properties': {'union_type_tuple': {'title':
            'Union Type Tuple', 'type': 'array', 'minItems': 1, 'maxItems':
            1, 'prefixItems': [{'anyOf': [{'type': 'string'}, {'type':
            'integer'}]}]}}, 'required': ['union_type_tuple']}}}

    async def func_0qdze8we(self):
        """
        class MyModel(BaseModel):
            ordered_multi_type_tuple: tuple[str, int]
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'title': 'param', 'position': 0, 'allOf': [{'$ref':
            '#/definitions/MyModel'}]}}, 'required': ['param'],
            'definitions': {'MyModel': {'title': 'MyModel', 'type':
            'object', 'properties': {'ordered_multi_type_tuple': {'title':
            'Ordered Multi Type Tuple', 'type': 'array', 'minItems': 2,
            'maxItems': 2, 'items': [{'type': 'string'}, {'type': 'integer'
            }]}}, 'required': ['ordered_multi_type_tuple']}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == {'title': 'Parameters', 'type':
            'object', 'properties': {'param': {'title': 'param', 'position':
            0, 'allOf': [{'$ref': '#/definitions/MyModel'}]}}, 'required':
            ['param'], 'definitions': {'MyModel': {'title': 'MyModel',
            'type': 'object', 'properties': {'ordered_multi_type_tuple': {
            'title': 'Ordered Multi Type Tuple', 'type': 'array',
            'minItems': 2, 'maxItems': 2, 'prefixItems': [{'type': 'string'
            }, {'type': 'integer'}]}}, 'required': [
            'ordered_multi_type_tuple']}}}


class TestPreprocessSchemaPydanticV2Tuples:

    def func_uc98jyal(self):
        """
        single_type_tuple: tuple[str]
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'single_type_tuple': {'maxItems': 1, 'minItems': 1, 'position':
            0, 'prefixItems': [{'type': 'string'}], 'title':
            'single_type_tuple', 'type': 'array'}}, 'required': [
            'single_type_tuple']}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_enbco6a3(self):
        """
        union_type_tuple: tuple[str | int]
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'union_type_tuple': {'maxItems': 1, 'minItems': 1, 'position': 
            0, 'prefixItems': [{'anyOf': [{'type': 'string'}, {'type':
            'integer'}]}], 'title': 'union_type_tuple', 'type': 'array'}},
            'required': ['union_type_tuple']}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_911ow2hq(self):
        """
        ordered_multi_type_tuple: tuple[str, int]
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'ordered_multi_type_tuple': {'maxItems': 2, 'minItems': 2,
            'position': 0, 'prefixItems': [{'type': 'string'}, {'type':
            'integer'}], 'title': 'ordered_multi_type_tuple', 'type':
            'array'}}, 'required': ['ordered_multi_type_tuple']}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_sopkbfwb(self):
        """
        class MyModel(BaseModel):
            single_type_tuple: tuple[str]
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'allOf': [{'$ref': '#/definitions/MyModel'}],
            'position': 0, 'title': 'param'}}, 'required': ['param'],
            'definitions': {'MyModel': {'properties': {'single_type_tuple':
            {'maxItems': 1, 'minItems': 1, 'prefixItems': [{'type':
            'string'}], 'title': 'Single Type Tuple', 'type': 'array'}},
            'required': ['single_type_tuple'], 'title': 'MyModel', 'type':
            'object'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_rkyj5ibh(self):
        """
        class MyModel(BaseModel):
            union_type_tuple: tuple[str | int]
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'allOf': [{'$ref': '#/definitions/MyModel'}],
            'position': 0, 'title': 'param'}}, 'required': ['param'],
            'definitions': {'MyModel': {'properties': {'union_type_tuple':
            {'maxItems': 1, 'minItems': 1, 'prefixItems': [{'anyOf': [{
            'type': 'string'}, {'type': 'integer'}]}], 'title':
            'Union Type Tuple', 'type': 'array'}}, 'required': [
            'union_type_tuple'], 'title': 'MyModel', 'type': 'object'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema

    def func_4qbtd9db(self):
        """
        class MyModel(BaseModel):
            ordered_multi_type_tuple: tuple[str, int]
        """
        schema = {'title': 'Parameters', 'type': 'object', 'properties': {
            'param': {'allOf': [{'$ref': '#/definitions/MyModel'}],
            'position': 0, 'title': 'param'}}, 'required': ['param'],
            'definitions': {'MyModel': {'properties': {
            'ordered_multi_type_tuple': {'maxItems': 2, 'minItems': 2,
            'prefixItems': [{'type': 'string'}, {'type': 'integer'}],
            'title': 'Ordered Multi Type Tuple', 'type': 'array'}},
            'required': ['ordered_multi_type_tuple'], 'title': 'MyModel',
            'type': 'object'}}}
        preprocessed_schema = preprocess_schema(schema)
        assert preprocessed_schema == schema
