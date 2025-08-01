import inspect
import json
import re
import shlex
from collections.abc import Mapping
from re import Match, Pattern
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import markdown
from django.conf import settings
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from typing_extensions import override
import zerver.openapi.python_examples
from zerver.lib.markdown.priorities import PREPROCESSOR_PRIORITIES
from zerver.openapi.openapi import NO_EXAMPLE, Parameter, check_additional_imports, check_requires_administrator, generate_openapi_fixture, get_curl_include_exclude, get_openapi_description, get_openapi_parameters, get_openapi_summary, get_parameters_description, get_responses_description, openapi_spec
from typing import Literal, TypedDict

API_ENDPOINT_NAME = '/[a-z_\\-/-{}]+:[a-z]+'
API_LANGUAGE = '\\w+'
API_KEY_TYPE = 'fixture|example'
MACRO_REGEXP = re.compile(f'\n        {{\n          generate_code_example\n          (?: \\( \\s* ({API_LANGUAGE}) \\s* \\) )?\n          \\|\n          \\s* ({API_ENDPOINT_NAME}) \\s*\n          \\|\n          \\s* ({API_KEY_TYPE}) \\s*\n        }}\n    ', re.VERBOSE)
PYTHON_EXAMPLE_REGEX = re.compile('\\# \\{code_example\\|\\s*(start|end)\\s*\\}')
JS_EXAMPLE_REGEX = re.compile('\\/\\/ \\{code_example\\|\\s*(start|end)\\s*\\}')
MACRO_REGEXP_HEADER = re.compile(f'{{generate_api_header\\(\\s*({API_ENDPOINT_NAME})\\s*\\)}}')
MACRO_REGEXP_RESPONSE_DESC = re.compile(f'{{generate_response_description\\(\\s*({API_ENDPOINT_NAME})\\s*\\)}}')
MACRO_REGEXP_PARAMETER_DESC = re.compile(f'{{generate_parameter_description\\(\\s*({API_ENDPOINT_NAME})\\s*\\)}}')
PYTHON_CLIENT_CONFIG = '\n#!/usr/bin/env python3\n\nimport zulip\n\n# Pass the path to your zuliprc file here.\nclient = zulip.Client(config_file="~/zuliprc")\n\n'
PYTHON_CLIENT_ADMIN_CONFIG = '\n#!/usr/bin/env python\n\nimport zulip\n\n# The user for this zuliprc file must be an organization administrator\nclient = zulip.Client(config_file="~/zuliprc-admin")\n\n'
JS_CLIENT_CONFIG = '\nconst zulipInit = require("zulip-js");\n\n// Pass the path to your zuliprc file here.\nconst config = { zuliprc: "zuliprc" };\n\n'
JS_CLIENT_ADMIN_CONFIG = '\nconst zulipInit = require("zulip-js");\n\n// The user for this zuliprc file must be an organization administrator.\nconst config = { zuliprc: "zuliprc-admin" };\n\n'
DEFAULT_AUTH_EMAIL = 'BOT_EMAIL_ADDRESS'
DEFAULT_AUTH_API_KEY = 'BOT_API_KEY'
DEFAULT_EXAMPLE = {'integer': 1, 'string': 'demo', 'boolean': False}
ADMIN_CONFIG_LANGUAGES = ['python', 'javascript']

class IncludeExcludeElement(TypedDict):
    type: str
    parameters: Dict[str, List[str]]
    description: Optional[str]

def extract_code_example(source: List[str], snippet: List[List[str]], example_regex: Pattern[str]) -> List[List[str]]:
    start = -1
    end = -1
    for line in source:
        match = example_regex.search(line)
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
    function_source_lines = inspect.getsourcelines(method)[0]
    if admin_config:
        config_string = PYTHON_CLIENT_ADMIN_CONFIG
    else:
        config_string = PYTHON_CLIENT_CONFIG
    endpoint, endpoint_method = function.split(':')
    extra_imports = check_additional_imports(endpoint, endpoint_method)
    if extra_imports:
        extra_imports = sorted([*extra_imports, 'zulip'])
        extra_imports = [f'import {each_import}' for each_import in extra_imports]
        config_string = config_string.replace('import zulip', '\n'.join(extra_imports))
    config = config_string.splitlines()
    snippets = extract_code_example(function_source_lines, [], PYTHON_EXAMPLE_REGEX)
    return ['{tab|python}\n', '