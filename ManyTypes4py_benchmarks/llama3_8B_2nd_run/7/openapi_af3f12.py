import json
import os
import re
from collections.abc import Mapping
from typing import Any, Literal
import orjson
from openapi_core import OpenAPI
from openapi_core.protocols import Request, Response
from openapi_core.testing import MockRequest, MockResponse
from openapi_core.validation.exceptions import ValidationError as OpenAPIValidationError
from pydantic import BaseModel
OPENAPI_SPEC_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '../openapi/zulip.yaml'))
EXCLUDE_UNDOCUMENTED_ENDPOINTS: dict[tuple[str, str], None] = {('/users', 'patch')}
EXCLUDE_DOCUMENTED_ENDPOINTS: set[str] = set()

class OpenAPISpec:
    def __init__(self, openapi_path: str):
        self.openapi_path: str = openapi_path
        self.mtime: int | None = None
        self._openapi: dict[str, Any] | None = None
        self._endpoints_dict: dict[str, str] | None = None
        self._spec: OpenAPI | None = None

    # ... (rest of the class definition)

def get_schema(endpoint: str, method: str, status_code: str) -> dict[str, Any]:
    # ... (rest of the function definition)

def get_openapi_fixture(endpoint: str, method: str, status_code: str = '200') -> list[dict[str, Any]]:
    # ... (rest of the function definition)

def get_curl_include_exclude(endpoint: str, method: str) -> list[dict[str, Any]]:
    # ... (rest of the function definition)

def check_requires_administrator(endpoint: str, method: str) -> bool:
    # ... (rest of the function definition)

def check_additional_imports(endpoint: str, method: str) -> list[str] | None:
    # ... (rest of the function definition)

def get_responses_description(endpoint: str, method: str) -> str | None:
    # ... (rest of the function definition)

def get_parameters_description(endpoint: str, method: str) -> str | None:
    # ... (rest of the function definition)

def generate_openapi_fixture(endpoint: str, method: str) -> list[str]:
    # ... (rest of the function definition)

def get_openapi_description(endpoint: str, method: str) -> str | None:
    # ... (rest of the function definition)

def get_openapi_summary(endpoint: str, method: str) -> str | None:
    # ... (rest of the function definition)

def get_endpoint_from_operationid(operationid: str) -> tuple[str, str]:
    # ... (rest of the function definition)

def get_openapi_paths() -> set[str]:
    # ... (rest of the function definition)

class Parameter(BaseModel):
    kind: str
    name: str
    description: str
    json_encoded: bool
    value_schema: dict[str, Any]
    example: Any | None
    required: bool
    deprecated: bool

def get_openapi_parameters(endpoint: str, method: str, include_url_parameters: bool = True) -> list[Parameter]:
    # ... (rest of the function definition)

def get_openapi_return_values(endpoint: str, method: str) -> dict[str, Any]:
    # ... (rest of the function definition)

def find_openapi_endpoint(path: str) -> str | None:
    # ... (rest of the function definition)

def validate_against_openapi_schema(content: dict[str, Any], path: str, method: str, status_code: str) -> bool:
    # ... (rest of the function definition)

def validate_test_response(request: Request, response: Response) -> bool:
    # ... (rest of the function definition)

def validate_schema(schema: dict[str, Any]) -> None:
    # ... (rest of the function definition)

def deprecated_note_in_description(description: str) -> bool:
    # ... (rest of the function definition)

def check_deprecated_consistency(deprecated: bool, description: str) -> None:
    # ... (rest of the function definition)

def validate_request(url: str, method: str, data: dict[str, Any], http_headers: dict[str, str], json_url: str, status_code: str, intentionally_undocumented: bool = False) -> None:
    # ... (rest of the function definition)

def validate_test_request(request: Request, status_code: str, intentionally_undocumented: bool = False) -> None:
    # ... (rest of the function definition)
