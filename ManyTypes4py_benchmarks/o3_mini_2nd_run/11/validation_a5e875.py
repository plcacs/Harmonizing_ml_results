from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast, Type, List, Dict, Tuple
import jsonschema
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from jsonschema.validators import Draft202012Validator, create, Validator
from referencing.jsonschema import ObjectSchema, Schema
from prefect.utilities.collections import remove_nested_keys
from prefect.utilities.schema_tools.hydration import HydrationError, Placeholder

if TYPE_CHECKING:
    from jsonschema.validators import _Validator  # type: ignore

class CircularSchemaRefError(Exception):
    pass

class ValidationError(Exception):
    pass

PLACEHOLDERS_VALIDATOR_NAME = '_placeholders'

def _build_validator() -> Type[Validator]:
    def _applicable_validators(schema: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
        assert not isinstance(schema, bool)
        schema = {**schema, PLACEHOLDERS_VALIDATOR_NAME: None}
        return schema.items()

    def _placeholders(
        _validator: Any, _property: str, instance: Any, _schema: Any
    ) -> Iterator[JSONSchemaValidationError]:
        if isinstance(instance, HydrationError):
            yield JSONSchemaValidationError(instance.message)

    validators: Dict[str, Any] = dict(Draft202012Validator.VALIDATORS)
    validators.update({PLACEHOLDERS_VALIDATOR_NAME: _placeholders})
    return create(
        meta_schema=Draft202012Validator.META_SCHEMA,
        validators=validators,
        version='prefect',
        type_checker=Draft202012Validator.TYPE_CHECKER,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
        id_of=cast(Callable[[Schema], str], Draft202012Validator.ID_OF),
        applicable_validators=_applicable_validators,
    )

_VALIDATOR: Type[Validator] = _build_validator()

def is_valid_schema(schema: Schema, preprocess: bool = True) -> None:
    if preprocess:
        schema = preprocess_schema(schema)  # type: ignore
    try:
        _VALIDATOR.check_schema(schema, format_checker=_VALIDATOR.FORMAT_CHECKER)
    except jsonschema.SchemaError as exc:
        raise ValueError(f'Invalid schema: {exc.message}') from exc

def validate(
    obj: Any,
    schema: Schema,
    raise_on_error: bool = False,
    preprocess: bool = True,
    ignore_required: bool = False,
    allow_none_with_default: bool = False,
) -> List[JSONSchemaValidationError]:
    if preprocess:
        schema = preprocess_schema(schema, allow_none_with_default)  # type: ignore
    if ignore_required:
        schema = remove_nested_keys(['required'], schema)
    if raise_on_error:
        try:
            jsonschema.validate(obj, schema, _VALIDATOR)
        except RecursionError:
            raise CircularSchemaRefError
        except JSONSchemaValidationError as exc:
            if exc.json_path == '$':
                error_message = 'Validation failed.'
            else:
                error_message = f'Validation failed for field {exc.json_path.replace("$.","")!r}.'
            error_message += f' Failure reason: {exc.message}'
            raise ValidationError(error_message) from exc
        return []
    else:
        try:
            validator = _VALIDATOR(schema, format_checker=_VALIDATOR.FORMAT_CHECKER)
            errors = list(validator.iter_errors(obj))
        except RecursionError:
            raise CircularSchemaRefError
        return errors

def is_valid(obj: Any, schema: Schema) -> bool:
    errors = validate(obj, schema)
    return not errors

def prioritize_placeholder_errors(
    errors: Iterable[JSONSchemaValidationError]
) -> List[JSONSchemaValidationError]:
    errors_by_path: Dict[str, List[JSONSchemaValidationError]] = defaultdict(list)
    for error in errors:
        path_str = '->'.join(str(p) for p in error.relative_path)
        errors_by_path[path_str].append(error)
    filtered_errors: List[JSONSchemaValidationError] = []
    for grouped_errors in errors_by_path.values():
        placeholders_errors = [
            error for error in grouped_errors if error.validator == PLACEHOLDERS_VALIDATOR_NAME
        ]
        if placeholders_errors:
            filtered_errors.extend(placeholders_errors)
        else:
            filtered_errors.extend(grouped_errors)
    return filtered_errors

def build_error_obj(
    errors: Iterable[JSONSchemaValidationError]
) -> Dict[str, Any]:
    error_response: Dict[str, Any] = {'errors': []}
    errors = prioritize_placeholder_errors(errors)
    for error in errors:
        if isinstance(error.instance, Placeholder) and (not error.instance.is_error):
            continue
        path = deque(error.relative_path)
        if error.validator == 'required':
            required_field = error.message.partition(' ')[0].strip("'")
            path.append(required_field)
        current = error_response['errors']
        if not path:
            current.append(error.message)
        while path:
            part = path.popleft()
            if isinstance(part, int):
                if not path:
                    current.append({'index': part, 'errors': [error.message]})
                else:
                    for entry in current:
                        if entry.get('index') == part:
                            current = cast(List[Any], entry['errors'])
                            break
                    else:
                        new_entry = {'index': part, 'errors': []}
                        current.append(new_entry)
                        current = new_entry['errors']
            elif not path:
                current.append({'property': part, 'errors': [error.message]})
            else:
                for entry in current:
                    if entry.get('property') == part:
                        current = entry.get('errors', [])
                        break
                else:
                    new_entry = {'property': part, 'errors': []}
                    current.append(new_entry)
                    current = new_entry['errors']
    valid = not bool(error_response['errors'])
    error_response['valid'] = valid
    return error_response

def _fix_null_typing(
    key: str,
    schema: Dict[str, Any],
    required_fields: List[str],
    allow_none_with_default: bool = False,
) -> None:
    """
    Pydantic V1 does not generate a valid Draft2020-12 schema for null types.
    """
    if (
        key not in required_fields
        and 'type' in schema
        and (schema.get('type') != 'null')
        and ('default' not in schema or allow_none_with_default)
    ):
        schema['anyOf'] = [{'type': schema['type']}, {'type': 'null'}]
        del schema['type']

def _fix_tuple_items(schema: Dict[str, Any]) -> None:
    """
    Pydantic V1 does not generate a valid Draft2020-12 schema for tuples.
    """
    if schema.get('items') and isinstance(schema['items'], list) and (not schema.get('prefixItems')):
        schema['prefixItems'] = deepcopy(cast(List[Any], schema['items']))
        del schema['items']

def process_properties(
    properties: Dict[str, Any],
    required_fields: List[str],
    allow_none_with_default: bool = False,
) -> None:
    for key, subschema in properties.items():
        _fix_null_typing(key, subschema, required_fields, allow_none_with_default)
        _fix_tuple_items(subschema)
        if 'properties' in subschema:
            sub_required_fields = subschema.get('required', [])
            process_properties(subschema['properties'], sub_required_fields, allow_none_with_default)

def preprocess_schema(
    schema: Dict[str, Any],
    allow_none_with_default: bool = False,
) -> Dict[str, Any]:
    schema = deepcopy(schema)
    if 'properties' in schema:
        required_fields = schema.get('required', [])
        process_properties(schema['properties'], required_fields, allow_none_with_default)
    if 'definitions' in schema:
        definitions = cast(Dict[str, Any], schema['definitions'])
        for definition in definitions.values():
            if 'properties' in definition:
                required_fields = definition.get('required', [])
                process_properties(definition['properties'], required_fields, allow_none_with_default)
            if 'block_type_slug' in definition:
                schema['definitions'][definition['title']] = {
                    'oneOf': [
                        definition,
                        {
                            'type': 'object',
                            'properties': {
                                '$ref': {
                                    'oneOf': [
                                        {'type': 'string', 'format': 'uuid'},
                                        {
                                            'type': 'object',
                                            'additionalProperties': {'type': 'string'},
                                            'minProperties': 1,
                                        },
                                    ]
                                }
                            },
                            'required': ['$ref'],
                        },
                    ]
                }
    return schema
