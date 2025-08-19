import json
import os
import re
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import orjson
from openapi_core import OpenAPI
from openapi_core.protocols import Request, Response
from openapi_core.testing import MockRequest, MockResponse
from openapi_core.validation.exceptions import ValidationError as OpenAPIValidationError
from pydantic import BaseModel

OPENAPI_SPEC_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '../openapi/zulip.yaml'))
EXCLUDE_UNDOCUMENTED_ENDPOINTS: Set[Tuple[str, str]] = {('/users', 'patch')}
EXCLUDE_DOCUMENTED_ENDPOINTS: Set[Tuple[str, str]] = set()


def naively_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    ret: Dict[str, Any] = a.copy()
    for key, b_value in b.items():
        if key == 'example' or key not in ret:
            ret[key] = b_value
            continue
        a_value = ret[key]
        if isinstance(b_value, list):
            assert isinstance(a_value, list)
            ret[key] = a_value + b_value
        elif isinstance(b_value, dict):
            assert isinstance(a_value, dict)
            ret[key] = naively_merge(a_value, b_value)
    return ret


def naively_merge_allOf(obj: Any) -> Any:
    if isinstance(obj, dict):
        return naively_merge_allOf_dict(obj)
    elif isinstance(obj, list):
        return list(map(naively_merge_allOf, obj))
    else:
        return obj


def naively_merge_allOf_dict(obj: Dict[str, Any]) -> Dict[str, Any]:
    if 'allOf' in obj:
        ret: Dict[str, Any] = obj.copy()
        subschemas = ret.pop('allOf')
        ret = naively_merge_allOf_dict(ret)
        assert isinstance(subschemas, list)
        for subschema in subschemas:
            assert isinstance(subschema, dict)
            ret = naively_merge(ret, naively_merge_allOf_dict(subschema))
        return ret
    return {key: naively_merge_allOf(value) for key, value in obj.items()}


class OpenAPISpec:
    def __init__(self, openapi_path: str) -> None:
        self.openapi_path: str = openapi_path
        self.mtime: Optional[float] = None
        self._openapi: Dict[str, Any] = {}
        self._endpoints_dict: Dict[str, str] = {}
        self._spec: Optional[OpenAPI] = None

    def check_reload(self) -> None:
        import yaml
        from jsonref import JsonRef

        with open(self.openapi_path) as f:
            mtime = os.fstat(f.fileno()).st_mtime
            if self.mtime == mtime:
                return
            openapi: Dict[str, Any] = yaml.load(f, Loader=yaml.CSafeLoader)
        spec = OpenAPI.from_dict(openapi)
        self._spec = spec
        self._openapi = naively_merge_allOf_dict(JsonRef.replace_refs(openapi))
        self.create_endpoints_dict()
        self.mtime = mtime

    def create_endpoints_dict(self) -> None:
        email_regex: str = '([a-zA-Z0-9_\\-\\.]+)@([a-zA-Z0-9_\\-\\.]+)\\.([a-zA-Z]{2,5})'
        self._endpoints_dict = {}
        for endpoint in self._openapi['paths']:
            if '{' not in endpoint:
                continue
            path_regex = '^' + endpoint + '$'
            path_regex = re.sub('{[^}]*id}', '[0-9]*', path_regex)
            path_regex = re.sub('{[^}]*email}', email_regex, path_regex)
            path_regex = re.sub('{[^}]*}', '[^\\/]*', path_regex)
            path_regex = path_regex.replace('/', '\\/')
            self._endpoints_dict[path_regex] = endpoint

    def openapi(self) -> Dict[str, Any]:
        """Reload the OpenAPI file if it has been modified after the last time
        it was read, and then return the parsed data.
        """
        self.check_reload()
        assert len(self._openapi) > 0
        return self._openapi

    def endpoints_dict(self) -> Dict[str, str]:
        """Reload the OpenAPI file if it has been modified after the last time
        it was read, and then return the parsed data.
        """
        self.check_reload()
        assert len(self._endpoints_dict) > 0
        return self._endpoints_dict

    def spec(self) -> OpenAPI:
        """Reload the OpenAPI file if it has been modified after the last time
        it was read, and then return the openapi_core validator object. Similar
        to preceding functions. Used for proper access to OpenAPI objects.
        """
        self.check_reload()
        assert self._spec is not None
        return self._spec


class SchemaError(Exception):
    pass


openapi_spec: OpenAPISpec = OpenAPISpec(OPENAPI_SPEC_PATH)


def get_schema(endpoint: str, method: str, status_code: str) -> Mapping[str, Any]:
    if len(status_code) == 3 and 'oneOf' in openapi_spec.openapi()['paths'][endpoint][method.lower()]['responses'][status_code]['content']['application/json']['schema']:
        status_code += '_0'
    if len(status_code) == 3:
        schema = openapi_spec.openapi()['paths'][endpoint][method.lower()]['responses'][status_code]['content']['application/json']['schema']
        return schema
    else:
        subschema_index = int(status_code[4])
        status_code = status_code[0:3]
        schema = openapi_spec.openapi()['paths'][endpoint][method.lower()]['responses'][status_code]['content']['application/json']['schema']['oneOf'][subschema_index]
        return schema


def get_openapi_fixture(endpoint: str, method: str, status_code: str = '200') -> Iterable[Mapping[str, Any]]:
    """Fetch a fixture from the full spec object."""
    if 'example' not in get_schema(endpoint, method, status_code):
        return openapi_spec.openapi()['paths'][endpoint][method.lower()]['responses'][status_code]['content']['application/json']['examples'].values()
    return [{'description': get_schema(endpoint, method, status_code)['description'], 'value': get_schema(endpoint, method, status_code)['example']}]


def get_curl_include_exclude(endpoint: str, method: str) -> List[Mapping[str, Any]]:
    """Fetch all the kinds of parameters required for curl examples."""
    if 'x-curl-examples-parameters' not in openapi_spec.openapi()['paths'][endpoint][method.lower()]:
        return [{'type': 'exclude', 'parameters': {'enum': ['']}}]
    return openapi_spec.openapi()['paths'][endpoint][method.lower()]['x-curl-examples-parameters']['oneOf']


def check_requires_administrator(endpoint: str, method: str) -> bool:
    """Fetch if the endpoint requires admin config."""
    return openapi_spec.openapi()['paths'][endpoint][method.lower()].get('x-requires-administrator', False)


def check_additional_imports(endpoint: str, method: str) -> Optional[str]:
    """Fetch the additional imports required for an endpoint."""
    return openapi_spec.openapi()['paths'][endpoint][method.lower()].get('x-python-examples-extra-imports', None)


def get_responses_description(endpoint: str, method: str) -> str:
    """Fetch responses description of an endpoint."""
    return openapi_spec.openapi()['paths'][endpoint][method.lower()].get('x-response-description', '')


def get_parameters_description(endpoint: str, method: str) -> str:
    """Fetch parameters description of an endpoint."""
    return openapi_spec.openapi()['paths'][endpoint][method.lower()].get('x-parameter-description', '')


def generate_openapi_fixture(endpoint: str, method: str) -> List[str]:
    """Generate fixture to be rendered"""
    fixture: List[str] = []
    for status_code in sorted(openapi_spec.openapi()['paths'][endpoint][method.lower()]['responses']):
        if 'oneOf' in openapi_spec.openapi()['paths'][endpoint][method.lower()]['responses'][status_code]['content']['application/json']['schema']:
            subschema_count = len(openapi_spec.openapi()['paths'][endpoint][method.lower()]['responses'][status_code]['content']['application/json']['schema']['oneOf'])
        else:
            subschema_count = 1
        for subschema_index in range(subschema_count):
            if subschema_count != 1:
                subschema_status_code = status_code + '_' + str(subschema_index)
            else:
                subschema_status_code = status_code
            fixture_dict = get_openapi_fixture(endpoint, method, subschema_status_code)
            for example in fixture_dict:
                fixture_json = json.dumps(example['value'], indent=4, sort_keys=True, separators=(',', ': '))
                if 'description' in example:
                    fixture.extend(str(example['description']).strip().splitlines())
                fixture.append('``` json')
                fixture.extend(fixture_json.splitlines())
                fixture.append('```')
    return fixture


def get_openapi_description(endpoint: str, method: str) -> str:
    """Fetch a description from the full spec object."""
    endpoint_documentation = openapi_spec.openapi()['paths'][endpoint][method.lower()]
    endpoint_description: str = endpoint_documentation['description']
    check_deprecated_consistency(endpoint_documentation.get('deprecated', False), endpoint_description)
    return endpoint_description


def get_openapi_summary(endpoint: str, method: str) -> str:
    """Fetch a summary from the full spec object."""
    return openapi_spec.openapi()['paths'][endpoint][method.lower()]['summary']


def get_endpoint_from_operationid(operationid: str) -> Tuple[str, str]:
    for endpoint in openapi_spec.openapi()['paths']:
        for method in openapi_spec.openapi()['paths'][endpoint]:
            operationId = openapi_spec.openapi()['paths'][endpoint][method].get('operationId')
            if operationId == operationid:
                return (endpoint, method)
    raise AssertionError('No such page exists in OpenAPI data.')


def get_openapi_paths() -> Set[str]:
    return set(openapi_spec.openapi()['paths'].keys())


NO_EXAMPLE: object = object()


ParameterKind = Union['path', 'query', 'header', 'cookie', 'formData']


class Parameter(BaseModel):
    kind: ParameterKind
    name: str
    description: str
    json_encoded: bool
    value_schema: Mapping[str, Any]
    example: Any = NO_EXAMPLE
    required: bool
    deprecated: bool


def get_openapi_parameters(endpoint: str, method: str, include_url_parameters: bool = True) -> List[Parameter]:
    operation = openapi_spec.openapi()['paths'][endpoint][method.lower()]
    parameters: List[Parameter] = []
    for parameter in operation.get('parameters', []):
        if not include_url_parameters and parameter['in'] == 'path':
            continue
        json_encoded = 'content' in parameter
        if json_encoded:
            schema = parameter['content']['application/json']['schema']
        else:
            schema = parameter['schema']
        if 'example' in parameter:
            example = parameter['example']
        elif json_encoded and 'example' in parameter['content']['application/json']:
            example = parameter['content']['application/json']['example']
        else:
            example = schema.get('example', NO_EXAMPLE)
        parameters.append(
            Parameter(
                kind=parameter['in'],
                name=parameter['name'],
                description=parameter['description'],
                json_encoded=json_encoded,
                value_schema=schema,
                example=example,
                required=parameter.get('required', False),
                deprecated=parameter.get('deprecated', False),
            )
        )
    if 'requestBody' in operation and 'application/x-www-form-urlencoded' in (content := operation['requestBody']['content']):
        media_type = content['application/x-www-form-urlencoded']
        required = media_type['schema'].get('required', [])
        for key, schema in media_type['schema']['properties'].items():
            json_encoded = 'encoding' in media_type and key in (encodings := media_type['encoding']) and (encodings[key].get('contentType') == 'application/json') or schema.get('type') == 'object'
            parameters.append(
                Parameter(
                    kind='formData',
                    name=key,
                    description=schema['description'],
                    json_encoded=json_encoded,
                    value_schema=schema,
                    example=schema.get('example'),
                    required=key in required,
                    deprecated=schema.get('deprecated', False),
                )
            )
    return parameters


def get_openapi_return_values(endpoint: str, method: str) -> Dict[str, Any]:
    operation = openapi_spec.openapi()['paths'][endpoint][method.lower()]
    schema = operation['responses']['200']['content']['application/json']['schema']
    assert 'properties' in schema
    return schema['properties']


def find_openapi_endpoint(path: str) -> Optional[str]:
    for path_regex, endpoint in openapi_spec.endpoints_dict().items():
        matches = re.match(path_regex, path)
        if matches:
            return endpoint
    return None


def validate_against_openapi_schema(content: Mapping[str, Any], path: str, method: str, status_code: str) -> bool:
    mock_request = MockRequest('http://localhost:9991/', method, '/api/v1' + path)
    mock_response = MockResponse(orjson.dumps(content), status_code=int(status_code))
    return validate_test_response(mock_request, mock_response)


def validate_test_response(request: Request, response: Response) -> bool:
    """Compare a "content" dict with the defined schema for a specific method
    in an endpoint. Return true if validated and false if skipped.
    """
    if request.path.startswith('/json/'):
        path = request.path.removeprefix('/json')
    elif request.path.startswith('/api/v1/'):
        path = request.path.removeprefix('/api/v1')
    else:
        return False
    assert request.method is not None
    method = request.method.lower()
    status_code = str(response.status_code)
    if path not in openapi_spec.openapi()['paths']:
        endpoint = find_openapi_endpoint(path)
        if endpoint is None:
            return False
    else:
        endpoint = path
    if (endpoint, method) in EXCLUDE_UNDOCUMENTED_ENDPOINTS:
        return False
    if (endpoint, method) in EXCLUDE_DOCUMENTED_ENDPOINTS:
        return True
    if status_code.startswith('4'):
        return True
    try:
        openapi_spec.spec().validate_response(request, response)
    except OpenAPIValidationError as error:
        message = f'Response validation error at {method} /api/v1{path} ({status_code}):'
        message += f'\n\n{type(error).__name__}: {error}'
        message += '\n\nFor help debugging these errors see: https://zulip.readthedocs.io/en/latest/documentation/api.html#debugging-schema-validation-errors'
        raise SchemaError(message) from None
    return True


def validate_schema(schema: Mapping[str, Any]) -> None:
    """Check if opaque objects are present in the OpenAPI spec; this is an
    important part of our policy for ensuring every detail of Zulip's
    API responses is correct.

    This is done by checking for the presence of the
    `additionalProperties` attribute for all objects (dictionaries).
    """
    if 'oneOf' in schema:
        for subschema in schema['oneOf']:
            validate_schema(subschema)
    elif schema['type'] == 'array':
        validate_schema(schema['items'])
    elif schema['type'] == 'object':
        if 'additionalProperties' not in schema:
            raise SchemaError('additionalProperties needs to be defined for objects to make sure they have no additional properties left to be documented.')
        for property_schema in schema.get('properties', {}).values():
            validate_schema(property_schema)
        if schema['additionalProperties']:
            validate_schema(schema['additionalProperties'])


def deprecated_note_in_description(description: str) -> bool:
    if '**Changes**: Deprecated' in description:
        return True
    return '**Deprecated**' in description


def check_deprecated_consistency(deprecated: bool, description: str) -> None:
    if deprecated_note_in_description(description):
        assert deprecated, f'Missing `deprecated: true` despite being described as deprecated:\n\n{description}\n'
    if deprecated:
        assert deprecated_note_in_description(description), f"Marked as `deprecated: true`, but changes documentation doesn't properly explain as **Deprecated** in the standard format\n\n:{description}\n"


SKIP_JSON: Set[Tuple[str, str]] = {('/fetch_api_key', 'post')}


def validate_request(
    url: str,
    method: str,
    data: Dict[str, Any],
    http_headers: Mapping[str, str],
    json_url: bool,
    status_code: str,
    intentionally_undocumented: bool = False,
) -> None:
    assert isinstance(data, dict)
    mock_request = MockRequest('http://localhost:9991/', method, '/api/v1' + url, headers=http_headers, args={k: str(v) for k, v in data.items()})
    validate_test_request(mock_request, status_code, intentionally_undocumented)


def validate_test_request(request: Request, status_code: str, intentionally_undocumented: bool = False) -> None:
    assert request.method is not None
    method = request.method.lower()
    if request.path.startswith('/json/'):
        url = request.path.removeprefix('/json')
        if (url, method) in SKIP_JSON:
            return
    else:
        assert request.path.startswith('/api/v1/')
        url = request.path.removeprefix('/api/v1')
    if url == '/user_uploads' or url.startswith('/realm/emoji/'):
        return
    if status_code.startswith('4'):
        return
    if status_code.startswith('2') and intentionally_undocumented:
        return
    try:
        openapi_spec.spec().validate_request(request)
    except OpenAPIValidationError as error:
        msg = f'\n\nError!  The OpenAPI schema for {method} {url} is not consistent\nwith the parameters passed in this HTTP request.  Consider:\n\n* Updating the OpenAPI schema defined in zerver/openapi/zulip.yaml\n* Adjusting the test to pass valid parameters.  If the test\n  fails due to intentionally_undocumented features, you need to pass\n  `intentionally_undocumented=True` to self.client_{method.lower()} or\n  self.api_{method.lower()} to document your intent.\n\nSee https://zulip.readthedocs.io/en/latest/documentation/api.html for help.\n\nThe error logged by the OpenAPI validator is below:\n{error}\n'
        raise SchemaError(msg)