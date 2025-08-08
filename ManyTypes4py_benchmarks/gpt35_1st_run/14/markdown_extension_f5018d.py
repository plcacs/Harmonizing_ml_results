from typing import List, Dict

def extract_code_example(source: List[str], snippet: List[str], example_regex: Pattern) -> List[str]:
def render_python_code_example(function: str, admin_config: bool = False, **kwargs: Any) -> List[str]:
def render_javascript_code_example(function: str, admin_config: bool = False, **kwargs: Any) -> List[str]:
def curl_method_arguments(endpoint: str, method: str, api_url: str) -> List[str]:
def get_openapi_param_example_value_as_string(endpoint: str, method: str, parameter: Parameter, curl_argument: bool = False) -> str:
def generate_curl_example(endpoint: str, method: str, api_url: str, auth_email: str = DEFAULT_AUTH_EMAIL, auth_api_key: str = DEFAULT_AUTH_API_KEY, exclude: List[str] = None, include: List[str] = None) -> List[str]:
def render_curl_example(function: str, api_url: str, admin_config: bool = False) -> List[str]:
def render(self, function: str) -> List[str]:
def makeExtension(*args, **kwargs) -> APIMarkdownExtension:
