import json
import os
import re
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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

def naively_merge_allOf(obj: Union[Dict[str, Any], List[Any], Any]) -> Any:
    if isinstance(obj, dict):
        return naively_merge_allOf_dict(obj)
    elif isinstance(obj, list):
        return list(map(naively_merge_allOf, obj))
    else:
        return obj

def naively_merge_allOf_dict(obj: Dict[str, Any]) -> Dict[str, Any]:
    if 'allOf' in obj:
        ret: Dict[str, Any] = obj.copy()
        subschemas: Any = ret.pop('allOf')
        ret = naively_merge_allOf_dict(ret)
        assert isinstance(subschemas, list)
        for subschema in subschemas:
            assert isinstance(subschema, dict)
            ret = naively_merge(ret, naively_merge_allOf_dict(subschema))
        return ret
    return {key: naively_merge_allOf(value) for key, value in obj.items()}

class OpenAPISpec:
    openapi_path: str
    mtime: Optional[float]
    _openapi: Dict[str, Any]
    _endpoints_dict: Dict[str, str]
    _spec: Optional[OpenAPI]

    def __init__(self, openapi_path: str) -> None:
        self.openapi_path = openapi_path
        self.mtime = None
        self._openapi = {}
        self._endpoints_dict = {}
        self._spec = None

    def check_reload(self) -> None:
        import yaml
        from jsonref import JsonRef
        with open(self.openapi_path, 'r') as f:
            mtime: float = os.fstat(f.fileno()).st_mtime
            if self.mtime == mtime:
                return
            openapi: Dict[str, Any] = yaml.load(f, Loader=yaml.CSafeLoader)
        spec: OpenAPI = OpenAPI.from_dict(openapi)
        self._spec = spec
        self._openapi = naively_merge_allOf_dict(JsonRef.replace_refs(openapi))
        self.create_endpoints_dict()
        self.mtime = mtime

    def create_endpoints_dict(self) -> None:
        email_regex: str = '([a-zA-Z0-9_\\-\\.]+)@([a-zA-Z0-9_\\-\\.]+)\\.([a-zA-Z]{2,5})'
        self._endpoints_dict = {}
        for endpoint in self._openapi.get('paths', {}):
            if '{' not in endpoint:
                continue
            path_regex: str = '^' + endpoint + '$'
            path_regex = re.sub(r'{[^}]*id}', r'[0-9]*', path_regex)
            path_regex = re.sub(r'{[^}]*email}', email_regex, path_regex)
            path_regex = re.sub(r'{[^}]*}', r'[^\\/]*', path_regex)
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

def get_schema(endpoint: str, method: str, status_code: str) -> Any:
    if len(status_code) == 3 and 'oneOf' in openapi_spec.openapi()['paths'][endpoint][method.lower()]['responses'][status_code]['content']['application/json']['schema']:
        status_code += '_0'
    if len(status_code) == 3:
        schema: Any = openapi_spec.openapi()['paths'][endpoint][method.lower()]['responses'][status_code]['content']['application/json']['schema']
        return schema
    else:
        subschema_index: int = int(status_code[4])
        status_code = status_code[0:3]
        schema = openapi_spec.openapi()['paths'][endpoint][method.lower()]['responses'][status_code]['content']['application/json']['schema']['oneOf'][subschema_index]
        return schema

def get_openapi_fixture(endpoint: str, method: str, status_code: str = '200') -> List[Dict[str, Any]]:
    """Fetch a fixture from the full spec object."""
    schema = get_schema(endpoint, method, status_code)
    if 'example' not in schema:
        examples = openapi_spec.openapi()['paths'][endpoint][method.lower()]['responses'][status_code]['content']['application/json']['examples']
        return list(examples.values())
    return [{'description': schema['description'], 'value': schema['example']}]

def get_curl_include_exclude(endpoint: str, method: str) -> List[Dict[str, Any]]:
    """Fetch all the kinds of parameters required for curl examples."""
    path_item = openapi_spec.openapi()['paths'][endpoint][method.lower()]
    if 'x-curl-examples-parameters' not in path_item:
        return [{'type': 'exclude', 'parameters': {'enum': ['']}}]
    return path_item['x-curl-examples-parameters']['oneOf']

def check_requires_administrator(endpoint: str, method: str) -> bool:
    """Fetch if the endpoint requires admin config."""
    return openapi_spec.openapi()['paths'][endpoint][method.lower()].get('x-requires-administrator', False)

def check_additional_imports(endpoint: str, method: str) -> Optional[Any]:
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
    responses = openapi_spec.openapi()['paths'][endpoint][method.lower()]['responses']
    for status_code in sorted(responses):
        response = responses[status_code]
        schema = response['content']['application/json']['schema']
        if 'oneOf' in schema:
            subschema_count = len(schema['oneOf'])
        else:
            subschema_count = 1
        for subschema_index in range(subschema_count):
            if subschema_count != 1:
                subschema_status_code = f"{status_code}_{subschema_index}"
            else:
                subschema_status_code = status_code
            fixture_dict = get_openapi_fixture(endpoint, method, subschema_status_code)
            for example in fixture_dict:
                fixture_json = json.dumps(example['value'], indent=4, sort_keys=True, separators=(',', ': '))
                if 'description' in example:
                    fixture.extend(example['description'].strip().splitlines())
                fixture.append('