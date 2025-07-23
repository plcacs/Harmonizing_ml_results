import json
import os
import re
from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import orjson
from openapi_core import OpenAPI
from openapi_core.protocols import Request, Response
from openapi_core.testing import MockRequest, MockResponse
from openapi_core.validation.exceptions import ValidationError as OpenAPIValidationError
from pydantic import BaseModel
from typing_extensions import TypedDict

OPENAPI_SPEC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../openapi/zulip.yaml'))
EXCLUDE_UNDOCUMENTED_ENDPOINTS: Set[Tuple[str, str]] = {('/users', 'patch')}
EXCLUDE_DOCUMENTED_ENDPOINTS: Set[Tuple[str, str]] = set()

class ParameterDict(TypedDict, total=False):
    example: Any
    description: str
    required: bool
    deprecated: bool
    schema: Dict[str, Any]
    content: Dict[str, Any]
    name: str
    in: str

class SchemaDict(TypedDict, total=False):
    type: str
    properties: Dict[str, Any]
    items: Dict[str, Any]
    additionalProperties: Union[bool, Dict[str, Any]]
    oneOf: List[Dict[str, Any]]
    description: str
    example: Any
    required: List[str]
    deprecated: bool

class ExampleDict(TypedDict, total=False):
    description: str
    value: Any

class OperationDict(TypedDict, total=False):
    parameters: List[ParameterDict]
    requestBody: Dict[str, Any]
    responses: Dict[str, Dict[str, Any]]
    description: str
    summary: str
    deprecated: bool
    x-curl-examples-parameters: Dict[str, Any]
    x-requires-administrator: bool
    x-python-examples-extra-imports: Optional[List[str]]
    x-response-description: str
    x-parameter-description: str

class PathItemDict(TypedDict, total=False):
    get: OperationDict
    post: OperationDict
    patch: OperationDict
    put: OperationDict
    delete: OperationDict
    head: OperationDict
    options: OperationDict

class OpenAPIDict(TypedDict, total=False):
    paths: Dict[str, PathItemDict]
    components: Dict[str, Any]

def naively_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    ret = a.copy()
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

def naively_merge_allOf(obj: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
    if isinstance(obj, dict):
        return naively_merge_allOf_dict(obj)
    elif isinstance(obj, list):
        return list(map(naively_merge_allOf, obj))
    else:
        return obj

def naively_merge_allOf_dict(obj: Dict[str, Any]) -> Dict[str, Any]:
    if 'allOf' in obj:
        ret = obj.copy()
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
        self.openapi_path = openapi_path
        self.mtime: Optional[float] = None
        self._openapi: OpenAPIDict = {}
        self._endpoints_dict: Dict[str, str] = {}
        self._spec: Optional[OpenAPI] = None

    def check_reload(self) -> None:
        import yaml
        from jsonref import JsonRef
        with open(self.openapi_path) as f:
            mtime = os.fstat(f.fileno()).st_mtime
            if self.mtime == mtime:
                return
            openapi = yaml.load(f, Loader=yaml.CSafeLoader)
        spec = OpenAPI.from_dict(openapi)
        self._spec = spec
        self._openapi = naively_merge_allOf_dict(JsonRef.replace_refs(openapi))
        self.create_endpoints_dict()
        self.mtime = mtime

    def create_endpoints_dict(self) -> None:
        email_regex = '([a-zA-Z0-9_\\-\\.]+)@([a-zA-Z0-9_\\-\\.]+)\\.([a-zA-Z]{2,5})'
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

    def openapi(self) -> OpenAPIDict:
        self.check_reload()
        assert len(self._openapi) > 0
        return self._openapi

    def endpoints_dict(self) -> Dict[str, str]:
        self.check_reload()
        assert len(self._endpoints_dict) > 0
        return self._endpoints_dict

    def spec(self) -> OpenAPI:
        self.check_reload()
        assert self._spec is not None
        return self._spec

class SchemaError(Exception):
    pass

openapi_spec = OpenAPISpec(OPENAPI_SPEC_PATH)

def get_schema(endpoint: str, method: str, status_code: str) -> SchemaDict:
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

def get_openapi_fixture(endpoint: str, method: str, status_code: str = '200') -> List[ExampleDict]:
    if 'example' not in get_schema(endpoint, method, status_code):
        return list(openapi_spec.openapi()['paths'][endpoint][method.lower()]['responses'][status_code]['content']['application/json']['examples'].values())
    schema = get_schema(endpoint, method, status_code)
    return [{'description': schema['description'], 'value': schema['example']}]

def get_curl_include_exclude(endpoint: str, method: str) -> List[Dict[str, Any]]:
    if 'x-curl-examples-parameters' not in openapi_spec.openapi()['paths'][endpoint][method.lower()]:
        return [{'type': 'exclude', 'parameters': {'enum': ['']}}]
    return openapi_spec.openapi()['paths'][endpoint][method.lower()]['x-curl-examples-parameters']['oneOf']

def check_requires_administrator(endpoint: str, method: str) -> bool:
    return openapi_spec.openapi()['paths'][endpoint][method.lower()].get('x-requires-administrator', False)

def check_additional_imports(endpoint: str, method: str) -> Optional[List[str]]:
    return openapi_spec.openapi()['paths'][endpoint][method.lower()].get('x-python-examples-extra-imports', None)

def get_responses_description(endpoint: str, method: str) -> str:
    return openapi_spec.openapi()['paths'][endpoint][method.lower()].get('x-response-description', '')

def get_parameters_description(endpoint: str, method: str) -> str:
    return openapi_spec.openapi()['paths'][endpoint][method.lower()].get('x-parameter-description', '')

def generate_openapi_fixture(endpoint: str, method: str) -> List[str]:
    fixture = []
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
                    fixture.extend(example['description'].strip().splitlines())
                fixture.append('