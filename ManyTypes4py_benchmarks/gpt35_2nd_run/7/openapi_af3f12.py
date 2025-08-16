def naively_merge(a: Mapping, b: Mapping) -> Mapping:
def naively_merge_allOf(obj: Any) -> Any:
def naively_merge_allOf_dict(obj: Any) -> Any:
class OpenAPISpec:
    def __init__(self, openapi_path: str) -> None:
    def check_reload(self) -> None:
    def create_endpoints_dict(self) -> None:
    def openapi(self) -> Any:
    def endpoints_dict(self) -> Any:
    def spec(self) -> Any:
def get_schema(endpoint: str, method: str, status_code: str) -> Any:
def get_openapi_fixture(endpoint: str, method: str, status_code: str = '200') -> Any:
def get_curl_include_exclude(endpoint: str, method: str) -> Any:
def check_requires_administrator(endpoint: str, method: str) -> bool:
def check_additional_imports(endpoint: str, method: str) -> Any:
def get_responses_description(endpoint: str, method: str) -> str:
def get_parameters_description(endpoint: str, method: str) -> str:
def generate_openapi_fixture(endpoint: str, method: str) -> Any:
def get_openapi_description(endpoint: str, method: str) -> str:
def get_openapi_summary(endpoint: str, method: str) -> str:
def get_endpoint_from_operationid(operationid: str) -> Any:
def get_openapi_paths() -> set:
def validate_against_openapi_schema(content: Any, path: str, method: str, status_code: str) -> Any:
def validate_test_response(request: Request, response: Response) -> bool:
def validate_schema(schema: Any) -> None:
def deprecated_note_in_description(description: str) -> bool:
def check_deprecated_consistency(deprecated: bool, description: str) -> None:
def validate_request(url: str, method: str, data: dict, http_headers: dict, json_url: str, status_code: str, intentionally_undocumented: bool = False) -> None:
def validate_test_request(request: Request, status_code: str, intentionally_undocumented: bool = False) -> None:
