#!/usr/bin/env python3
from collections.abc import Mapping
from typing import Any, Dict, List, Optional
import inspect
import json
import re
import shlex
from re import Match, Pattern
from textwrap import dedent
import markdown
from django.conf import settings
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from typing_extensions import override
import zerver.openapi.python_examples
from zerver.lib.markdown.priorities import PREPROCESSOR_PRIORITIES
from zerver.openapi.openapi import NO_EXAMPLE, Parameter, check_additional_imports, check_requires_administrator, generate_openapi_fixture, get_curl_include_exclude, get_openapi_description, get_openapi_parameters, get_openapi_summary, get_parameters_description, get_responses_description, openapi_spec

API_ENDPOINT_NAME: str = '/[a-z_\\-/-{}]+:[a-z]+'
API_LANGUAGE: str = '\\w+'
API_KEY_TYPE: str = 'fixture|example'
MACRO_REGEXP: Pattern[str] = re.compile(
    f'\n        {{\n          generate_code_example\n          (?: \\( \\s* ({API_LANGUAGE}) \\s* \\) )?\n          \\|\n          \\s* ({API_ENDPOINT_NAME}) \\s*\n          \\|\n          \\s* ({API_KEY_TYPE}) \\s*\n        }}\n    ',
    re.VERBOSE,
)
PYTHON_EXAMPLE_REGEX: Pattern[str] = re.compile('\\# \\{code_example\\|\\s*(start|end)\\s*\\}')
JS_EXAMPLE_REGEX: Pattern[str] = re.compile('\\/\\/ \\{code_example\\|\\s*(start|end)\\s*\\}')
MACRO_REGEXP_HEADER: Pattern[str] = re.compile(f'{{generate_api_header\\(\\s*({API_ENDPOINT_NAME})\\s*\\)}}')
MACRO_REGEXP_RESPONSE_DESC: Pattern[str] = re.compile(f'{{generate_response_description\\(\\s*({API_ENDPOINT_NAME})\\s*\\)}}')
MACRO_REGEXP_PARAMETER_DESC: Pattern[str] = re.compile(f'{{generate_parameter_description\\(\\s*({API_ENDPOINT_NAME})\\s*\\)}}')
PYTHON_CLIENT_CONFIG: str = (
    '\n#!/usr/bin/env python3\n\nimport zulip\n\n# Pass the path to your zuliprc file here.\nclient = zulip.Client(config_file="~/zuliprc")\n\n'
)
PYTHON_CLIENT_ADMIN_CONFIG: str = (
    '\n#!/usr/bin/env python\n\nimport zulip\n\n# The user for this zuliprc file must be an organization administrator\nclient = zulip.Client(config_file="~/zuliprc-admin")\n\n'
)
JS_CLIENT_CONFIG: str = (
    '\nconst zulipInit = require("zulip-js");\n\n// Pass the path to your zuliprc file here.\nconst config = { zuliprc: "zuliprc" };\n\n'
)
JS_CLIENT_ADMIN_CONFIG: str = (
    '\nconst zulipInit = require("zulip-js");\n\n// The user for this zuliprc file must be an organization administrator.\nconst config = { zuliprc: "zuliprc-admin" };\n\n'
)
DEFAULT_AUTH_EMAIL: str = 'BOT_EMAIL_ADDRESS'
DEFAULT_AUTH_API_KEY: str = 'BOT_API_KEY'
DEFAULT_EXAMPLE: Dict[str, Any] = {'integer': 1, 'string': 'demo', 'boolean': False}
ADMIN_CONFIG_LANGUAGES: List[str] = ['python', 'javascript']

def extract_code_example(source: List[str], snippet: List[List[str]], example_regex: Pattern[str]) -> List[List[str]]:
    start: int = -1
    end: int = -1
    for line in source:
        match: Optional[Match[str]] = example_regex.search(line)
        if match:
            if match.group(1) == 'start':
                start = source.index(line)
            elif match.group(1) == 'end':
                end = source.index(line)
                break
    if start == -1 and end == -1:
        return snippet
    snippet.append(source[start + 1:end])
    source = source[end + 1:]
    return extract_code_example(source, snippet, example_regex)

def render_python_code_example(function: str, admin_config: bool = False, **kwargs: Any) -> List[str]:
    if function not in zerver.openapi.python_examples.TEST_FUNCTIONS:
        return []
    method = zerver.openapi.python_examples.TEST_FUNCTIONS[function]
    function_source_lines: List[str] = inspect.getsourcelines(method)[0]
    if admin_config:
        config_string: str = PYTHON_CLIENT_ADMIN_CONFIG
    else:
        config_string = PYTHON_CLIENT_CONFIG
    endpoint, endpoint_method = function.split(':')
    extra_imports = check_additional_imports(endpoint, endpoint_method)
    if extra_imports:
        extra_imports = sorted([*extra_imports, 'zulip'])
        extra_imports = [f'import {each_import}' for each_import in extra_imports]
        config_string = config_string.replace('import zulip', '\n'.join(extra_imports))
    config: List[str] = config_string.splitlines()
    snippets: List[List[str]] = extract_code_example(function_source_lines, [], PYTHON_EXAMPLE_REGEX)
    return ['{tab|python}\n', '```python', *config, *(line.removeprefix('    ').rstrip() for snippet in snippets for line in snippet), 'print(result)', '\n', '```']

def render_javascript_code_example(function: str, admin_config: bool = False, **kwargs: Any) -> List[str]:
    pattern: str = (
        f'^add_example\\(\\s*"[^"]*",\\s*{re.escape(json.dumps(function))},\\s*\\d+,\\s*async \\(client, console\\) => \\{{\\n'
        f'(.*?)^(?:\\}}| *\\}},\\n)\\);$'
    )
    with open('zerver/openapi/javascript_examples.js') as f:
        file_content: str = f.read()
    m: Optional[Match[str]] = re.search(pattern, file_content, re.MULTILINE | re.DOTALL)
    if m is None:
        return []
    function_source_lines: List[str] = dedent(m.group(1)).splitlines()
    snippets: List[List[str]] = extract_code_example(function_source_lines, [], JS_EXAMPLE_REGEX)
    if admin_config:
        config_lines: List[str] = JS_CLIENT_ADMIN_CONFIG.splitlines()
    else:
        config_lines = JS_CLIENT_CONFIG.splitlines()
    code_example: List[str] = ['{tab|js}\n', 'More examples and documentation can be found [here](https://github.com/zulip/zulip-js).']
    code_example.append('```js')
    code_example.extend(config_lines)
    code_example.append('(async () => {')
    code_example.append('    const client = await zulipInit(config);')
    for snippet in snippets:
        code_example.append('')
        code_example.extend(('    ' + line.rstrip() for line in snippet))
    code_example.append('})();')
    code_example.append('```')
    return code_example

def curl_method_arguments(endpoint: str, method: str, api_url: str) -> List[str]:
    method = method.upper()
    url: str = f'{api_url}/v1{endpoint}'
    valid_methods: List[str] = ['GET', 'POST', 'DELETE', 'PUT', 'PATCH', 'OPTIONS']
    if method == 'GET':
        return ['-sSX', 'GET', '-G', url]
    elif method in valid_methods:
        return ['-sSX', method, url]
    else:
        msg: str = f'The request method {method} is not one of {valid_methods}'
        raise ValueError(msg)

def get_openapi_param_example_value_as_string(endpoint: str, method: str, parameter: Parameter, curl_argument: bool = False) -> str:
    if 'type' in parameter.value_schema:  # type: ignore
        param_type: str = parameter.value_schema['type']  # type: ignore
    else:
        param_type = parameter.value_schema['oneOf'][0]['type']  # type: ignore
    if param_type in ['object', 'array']:
        if parameter.example is NO_EXAMPLE:  # type: ignore
            msg: str = (
                f'All array and object type request parameters must have\n'
                f'concrete examples. The openAPI documentation for {endpoint}/{method} is missing an example\n'
                f'value for the {parameter.name} parameter. Without this we cannot automatically generate a\n'
                f'cURL example.'
            )
            raise ValueError(msg)
        ordered_ex_val_str: str = json.dumps(parameter.example, sort_keys=True)  # type: ignore
        assert parameter.json_encoded  # type: ignore
        if curl_argument:
            return '    --data-urlencode ' + shlex.quote(f'{parameter.name}={ordered_ex_val_str}')
        return ordered_ex_val_str
    else:
        if parameter.example is NO_EXAMPLE:  # type: ignore
            example_value: Any = DEFAULT_EXAMPLE[param_type]
        else:
            example_value = parameter.example  # type: ignore
        if parameter.json_encoded or isinstance(example_value, (bool, float, int)):  # type: ignore
            example_value = json.dumps(example_value)
        else:
            assert isinstance(example_value, str)
        if curl_argument:
            return '    --data-urlencode ' + shlex.quote(f'{parameter.name}={example_value}')
        return example_value

def generate_curl_example(
    endpoint: str,
    method: str,
    api_url: str,
    auth_email: str = DEFAULT_AUTH_EMAIL,
    auth_api_key: str = DEFAULT_AUTH_API_KEY,
    exclude: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
) -> List[str]:
    lines: List[str] = ['```curl']
    operation: str = endpoint + ':' + method.lower()
    operation_entry: Dict[str, Any] = openapi_spec.openapi()['paths'][endpoint][method.lower()]  # type: ignore
    global_security: Any = openapi_spec.openapi()['security']  # type: ignore
    parameters: List[Parameter] = get_openapi_parameters(endpoint, method)
    operation_request_body: Optional[Any] = operation_entry.get('requestBody', None)
    operation_security: Optional[Any] = operation_entry.get('security', None)
    if settings.RUNNING_OPENAPI_CURL_TEST:
        from zerver.openapi.curl_param_value_generators import patch_openapi_example_values
        parameters, operation_request_body = patch_openapi_example_values(operation, parameters, operation_request_body)
    format_dict: Dict[str, str] = {}
    for parameter in parameters:
        if parameter.kind != 'path':  # type: ignore
            continue
        example_value: str = get_openapi_param_example_value_as_string(endpoint, method, parameter)
        format_dict[parameter.name] = example_value  # type: ignore
    example_endpoint: str = endpoint.format_map(format_dict)
    curl_first_line_parts: List[str] = ['curl', *curl_method_arguments(example_endpoint, method, api_url)]
    lines.append(shlex.join(curl_first_line_parts))
    insecure_operations: List[str] = ['/dev_fetch_api_key:post', '/fetch_api_key:post']
    if operation_security is None:
        if global_security == [{'basicAuth': []}]:
            authentication_required: bool = True
        else:
            raise AssertionError('Unhandled global securityScheme. Please update the code to handle this scheme.')
    elif operation_security == []:
        if operation in insecure_operations:
            authentication_required = False
        else:
            raise AssertionError('Unknown operation without a securityScheme. Please update insecure_operations.')
    else:
        raise AssertionError('Unhandled securityScheme. Please update the code to handle this scheme.')
    if authentication_required:
        lines.append('    -u ' + shlex.quote(f'{auth_email}:{auth_api_key}'))
    for parameter in parameters:
        if parameter.kind == 'path':  # type: ignore
            continue
        if include is not None and parameter.name not in include:  # type: ignore
            continue
        if exclude is not None and parameter.name in exclude:  # type: ignore
            continue
        example_value = get_openapi_param_example_value_as_string(endpoint, method, parameter, curl_argument=True)
        lines.append(example_value)
    if 'requestBody' in operation_entry and 'multipart/form-data' in (content := operation_entry['requestBody']['content']):  # type: ignore
        properties: Dict[str, Any] = content['multipart/form-data']['schema']['properties']
        for key, property in properties.items():
            lines.append('    -F ' + shlex.quote('{}=@{}'.format(key, property['example'])))
    for i in range(1, len(lines) - 1):
        lines[i] += ' \\'
    lines.append('```')
    return lines

def render_curl_example(function: str, api_url: str, admin_config: bool = False) -> List[str]:
    """A simple wrapper around generate_curl_example."""
    parts: List[str] = function.split(':')
    endpoint: str = parts[0]
    method: str = parts[1]
    kwargs: Dict[str, Any] = {}
    if len(parts) > 2:
        kwargs['auth_email'] = parts[2]
    if len(parts) > 3:
        kwargs['auth_api_key'] = parts[3]
    kwargs['api_url'] = api_url
    rendered_example: List[str] = []
    for element in get_curl_include_exclude(endpoint, method):
        kwargs['include'] = None
        kwargs['exclude'] = None
        if element['type'] == 'include':
            kwargs['include'] = element['parameters']['enum']
        if element['type'] == 'exclude':
            kwargs['exclude'] = element['parameters']['enum']
        if 'description' in element:
            rendered_example.extend(element['description'].splitlines())
        rendered_example += generate_curl_example(endpoint, method, **kwargs)
    return rendered_example

SUPPORTED_LANGUAGES: Dict[str, Dict[str, Any]] = {
    'python': {
        'client_config': PYTHON_CLIENT_CONFIG,
        'admin_config': PYTHON_CLIENT_ADMIN_CONFIG,
        'render': render_python_code_example,
    },
    'curl': {
        'render': render_curl_example,
    },
    'javascript': {
        'client_config': JS_CLIENT_CONFIG,
        'admin_config': JS_CLIENT_ADMIN_CONFIG,
        'render': render_javascript_code_example,
    },
}

class APIMarkdownExtension(Extension):
    def __init__(self, api_url: str) -> None:
        self.config: Dict[str, List[Any]] = {'api_url': [api_url, 'API URL to use when rendering curl examples']}
        super().__init__()

    @override
    def extendMarkdown(self, md: markdown.Markdown) -> None:
        md.preprocessors.register(
            APICodeExamplesPreprocessor(md, self.getConfigs()),
            'generate_code_example',
            PREPROCESSOR_PRIORITIES['generate_code_example']
        )
        md.preprocessors.register(
            APIHeaderPreprocessor(md, self.getConfigs()),
            'generate_api_header',
            PREPROCESSOR_PRIORITIES['generate_api_header']
        )
        md.preprocessors.register(
            ResponseDescriptionPreprocessor(md, self.getConfigs()),
            'generate_response_description',
            PREPROCESSOR_PRIORITIES['generate_response_description']
        )
        md.preprocessors.register(
            ParameterDescriptionPreprocessor(md, self.getConfigs()),
            'generate_parameter_description',
            PREPROCESSOR_PRIORITIES['generate_parameter_description']
        )

class BasePreprocessor(Preprocessor):
    def __init__(self, regexp: Pattern[str], md: markdown.Markdown, config: Mapping[str, Any]) -> None:
        super().__init__(md)
        self.api_url: str = config['api_url']
        self.REGEXP: Pattern[str] = regexp

    @override
    def run(self, lines: List[str]) -> List[str]:
        done: bool = False
        while not done:
            for line in lines:
                loc: int = lines.index(line)
                match: Optional[Match[str]] = self.REGEXP.search(line)
                if match:
                    text: List[str] = self.generate_text(match)
                    line_split: List[str] = self.REGEXP.split(line, maxsplit=0)
                    preceding: str = line_split[0]
                    following: str = line_split[-1]
                    text = [preceding, *text, following]
                    lines = lines[:loc] + text + lines[loc + 1:]
                    break
            else:
                done = True
        return lines

    def generate_text(self, match: Match[str]) -> List[str]:
        function: str = match.group(1)
        text: List[str] = self.render(function)
        return text

    def render(self, function: str) -> List[str]:
        raise NotImplementedError('Must be overridden by a child class')

class APICodeExamplesPreprocessor(BasePreprocessor):
    def __init__(self, md: markdown.Markdown, config: Mapping[str, Any]) -> None:
        super().__init__(MACRO_REGEXP, md, config)

    @override
    def generate_text(self, match: Match[str]) -> List[str]:
        language: str = match.group(1) or ''
        function: str = match.group(2)
        key: str = match.group(3)
        if self.api_url is None:
            raise AssertionError('Cannot render curl API examples without API URL set.')
        if key == 'fixture':
            text: List[str] = self.render(function)
        elif key == 'example':
            path, method = function.rsplit(':', 1)
            admin_config: bool = language in ADMIN_CONFIG_LANGUAGES and check_requires_administrator(path, method)
            text = SUPPORTED_LANGUAGES[language]['render'](function, api_url=self.api_url, admin_config=admin_config)
        else:
            text = []
        return text

    @override
    def render(self, function: str) -> List[str]:
        path, method = function.rsplit(':', 1)
        return generate_openapi_fixture(path, method)

class APIHeaderPreprocessor(BasePreprocessor):
    def __init__(self, md: markdown.Markdown, config: Mapping[str, Any]) -> None:
        super().__init__(MACRO_REGEXP_HEADER, md, config)

    @override
    def render(self, function: str) -> List[str]:
        path, method = function.rsplit(':', 1)
        raw_title: str = get_openapi_summary(path, method)
        description_dict: str = get_openapi_description(path, method)
        header_lines: List[str] = [*('# ' + line for line in raw_title.splitlines())]
        if check_requires_administrator(path, method):
            header_lines.extend(['{!api-admin-only.md!}'])
        header_lines.extend(['', f'`{method.upper()} {self.api_url}/v1{path}`', '', *description_dict.splitlines()])
        return header_lines

class ResponseDescriptionPreprocessor(BasePreprocessor):
    def __init__(self, md: markdown.Markdown, config: Mapping[str, Any]) -> None:
        super().__init__(MACRO_REGEXP_RESPONSE_DESC, md, config)

    @override
    def render(self, function: str) -> List[str]:
        path, method = function.rsplit(':', 1)
        raw_description: str = get_responses_description(path, method)
        return raw_description.splitlines()

class ParameterDescriptionPreprocessor(BasePreprocessor):
    def __init__(self, md: markdown.Markdown, config: Mapping[str, Any]) -> None:
        super().__init__(MACRO_REGEXP_PARAMETER_DESC, md, config)

    @override
    def render(self, function: str) -> List[str]:
        path, method = function.rsplit(':', 1)
        raw_description: str = get_parameters_description(path, method)
        return raw_description.splitlines()

def makeExtension(*args: Any, **kwargs: Any) -> APIMarkdownExtension:
    return APIMarkdownExtension(*args, **kwargs)