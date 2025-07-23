import inspect
import json
import re
import shlex
from collections.abc import Mapping
from re import Match, Pattern
from textwrap import dedent
from typing import Any, Dict, List, Optional
import markdown
from django.conf import settings
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from typing_extensions import override
import zerver.openapi.python_examples
from zerver.lib.markdown.priorities import PREPROCESSOR_PRIORITIES
from zerver.openapi.openapi import (
    NO_EXAMPLE,
    Parameter,
    check_additional_imports,
    check_requires_administrator,
    generate_openapi_fixture,
    get_curl_include_exclude,
    get_openapi_description,
    get_openapi_parameters,
    get_openapi_summary,
    get_parameters_description,
    get_responses_description,
    openapi_spec,
)

API_ENDPOINT_NAME: str = '/[a-z_\\-/-{}]+:[a-z]+'
API_LANGUAGE: str = '\\w+'
API_KEY_TYPE: str = 'fixture|example'
MACRO_REGEXP: Pattern = re.compile(
    f'''
        {{
          generate_code_example
          (?: \( \s* ({API_LANGUAGE}) \s* \) )
          \|
          \s* ({API_ENDPOINT_NAME}) \s*
          \|
          \s* ({API_KEY_TYPE}) \s*
        }}
    ''',
    re.VERBOSE,
)
PYTHON_EXAMPLE_REGEX: Pattern = re.compile(r'\# \{code_example\|\s*(start|end)\s*\}')
JS_EXAMPLE_REGEX: Pattern = re.compile(r'\/\/ \{code_example\|\s*(start|end)\s*\}')
MACRO_REGEXP_HEADER: Pattern = re.compile(f'{{generate_api_header\\(\\s*({API_ENDPOINT_NAME})\\s*\\)}}')
MACRO_REGEXP_RESPONSE_DESC: Pattern = re.compile(f'{{generate_response_description\\(\\s*({API_ENDPOINT_NAME})\\s*\\)}}')
MACRO_REGEXP_PARAMETER_DESC: Pattern = re.compile(f'{{generate_parameter_description\\(\\s*({API_ENDPOINT_NAME})\\s*\\)}}')
PYTHON_CLIENT_CONFIG: str = '''
#!/usr/bin/env python3

import zulip

# Pass the path to your zuliprc file here.
client = zulip.Client(config_file="~/zuliprc")

'''
PYTHON_CLIENT_ADMIN_CONFIG: str = '''
#!/usr/bin/env python

import zulip

# The user for this zuliprc file must be an organization administrator
client = zulip.Client(config_file="~/zuliprc-admin")

'''
JS_CLIENT_CONFIG: str = '''
const zulipInit = require("zulip-js");

// Pass the path to your zuliprc file here.
const config = { zuliprc: "zuliprc" };

'''
JS_CLIENT_ADMIN_CONFIG: str = '''
const zulipInit = require("zulip-js");

// The user for this zuliprc file must be an organization administrator.
const config = { zuliprc: "zuliprc-admin" };

'''
DEFAULT_AUTH_EMAIL: str = 'BOT_EMAIL_ADDRESS'
DEFAULT_AUTH_API_KEY: str = 'BOT_API_KEY'
DEFAULT_EXAMPLE: Dict[str, Any] = {'integer': 1, 'string': 'demo', 'boolean': False}
ADMIN_CONFIG_LANGUAGES: List[str] = ['python', 'javascript']


def extract_code_example(
    source: List[str], snippet: List[List[str]], example_regex: Pattern
) -> List[List[str]]:
    start: int = -1
    end: int = -1
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
    snippet.append(source[start + 1 : end])
    source = source[end + 1 :]
    return extract_code_example(source, snippet, example_regex)


def render_python_code_example(function: str, admin_config: bool = False, **kwargs: Any) -> List[str]:
    if function not in zerver.openapi.python_examples.TEST_FUNCTIONS:
        return []
    method = zerver.openapi.python_examples.TEST_FUNCTIONS[function]
    function_source_lines: List[str] = inspect.getsourcelines(method)[0]
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
    snippets: List[List[str]] = extract_code_example(function_source_lines, [], PYTHON_EXAMPLE_REGEX)
    return [
        '{tab|python}\n',
        '