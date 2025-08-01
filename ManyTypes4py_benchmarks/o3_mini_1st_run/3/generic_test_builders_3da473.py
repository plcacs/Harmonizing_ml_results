#!/usr/bin/env python3
import re
from copy import deepcopy
from typing import Any, Dict, Generic, List, Optional, Tuple, Union
from dbt.artifacts.resources import NodeVersion
from dbt.clients.jinja import GENERIC_TEST_KWARGS_NAME, get_rendered
from dbt.contracts.graph.nodes import UnpatchedSourceDefinition
from dbt.contracts.graph.unparsed import UnparsedModelUpdate, UnparsedNodeUpdate
from dbt.exceptions import (
    CustomMacroPopulatingConfigValueError, 
    SameKeyNestedError, 
    TagNotStringError, 
    TagsNotListOfStringsError, 
    TestArgIncludesModelError, 
    TestArgsNotDictError, 
    TestDefinitionDictLengthError, 
    TestNameNotStringError, 
    TestTypeError, 
    UnexpectedTestNamePatternError
)
from dbt.parser.common import Testable
from dbt.utils import md5
from dbt_common.exceptions.macros import UndefinedMacroError

def synthesize_generic_test_names(test_type: str, test_name: str, args: Dict[str, Any]) -> Tuple[str, str]:
    flat_args: List[str] = []
    for arg_name in sorted(args):
        if arg_name == 'model':
            continue
        arg_val = args[arg_name]
        if isinstance(arg_val, dict):
            parts = list(arg_val.values())
        elif isinstance(arg_val, (list, tuple)):
            parts = list(arg_val)
        else:
            parts = [arg_val]
        flat_args.extend([str(part) for part in parts])
    clean_flat_args = [re.sub('[^0-9a-zA-Z_]+', '_', arg) for arg in flat_args]
    unique = '__'.join(clean_flat_args)
    test_identifier = '{}_{}'.format(test_type, test_name)
    full_name = '{}_{}'.format(test_identifier, unique)
    if len(full_name) >= 64:
        test_trunc_identifier = test_identifier[:30]
        label = md5(full_name)
        short_name = '{}_{}'.format(test_trunc_identifier, label)
    else:
        short_name = full_name
    return (short_name, full_name)

class TestBuilder(Generic[Testable]):
    """An object to hold assorted test settings and perform basic parsing

    Test names have the following pattern:
        - the test name itself may be namespaced (package.test)
        - or it may not be namespaced (test)
    """
    TEST_NAME_PATTERN = re.compile(r'((?P<test_namespace>([a-zA-Z_][0-9a-zA-Z_]*))\.)?(?P<test_name>([a-zA-Z_][0-9a-zA-Z_]*))')
    CONFIG_ARGS = (
        'severity', 'tags', 'enabled', 'where', 'limit', 'warn_if', 'error_if',
        'fail_calc', 'store_failures', 'store_failures_as', 'meta', 'database', 
        'schema', 'alias'
    )

    def __init__(
        self, 
        data_test: Any, 
        target: Union[UnparsedModelUpdate, UnparsedNodeUpdate, UnpatchedSourceDefinition],
        package_name: str, 
        render_ctx: Dict[str, Any], 
        column_name: Optional[str] = None, 
        version: Optional[str] = None
    ) -> None:
        test_name, test_args = self.extract_test_args(data_test, column_name)
        self.args: Dict[str, Any] = test_args
        if 'model' in self.args:
            raise TestArgIncludesModelError()
        self.package_name: str = package_name
        self.target: Union[UnparsedModelUpdate, UnparsedNodeUpdate, UnpatchedSourceDefinition] = target
        self.version: Optional[str] = version
        self.render_ctx: Dict[str, Any] = render_ctx
        self.column_name: Optional[str] = column_name
        self.args['model'] = self.build_model_str()
        match = self.TEST_NAME_PATTERN.match(test_name)
        if match is None:
            raise UnexpectedTestNamePatternError(test_name)
        groups = match.groupdict()
        self.name: str = groups['test_name']  # type: ignore
        self.namespace: Optional[str] = groups.get('test_namespace')
        self.config: Dict[str, Any] = {}
        self.config.update(self._process_legacy_args())
        if 'config' in self.args:
            self.config.update(self._render_values(self.args.pop('config', {})))
        if self.namespace is not None:
            self.package_name = self.namespace
        self.description: str = ''
        if 'description' in self.args:
            self.description = self.args['description']
            del self.args['description']
        self.compiled_name: str = ''
        self.fqn_name: str = ''
        if 'name' in self.args:
            self.compiled_name = self.args['name']
            self.fqn_name = self.args['name']
            del self.args['name']
        else:
            short_name, full_name = self.get_synthetic_test_names()
            self.compiled_name = short_name
            self.fqn_name = full_name
            if short_name != full_name and 'alias' not in self.config:
                self.config['alias'] = short_name

    def _process_legacy_args(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for key in self.CONFIG_ARGS:
            value = self.args.pop(key, None)
            if value and 'config' in self.args and (key in self.args['config']):
                raise SameKeyNestedError()
            if not value and 'config' in self.args:
                value = self.args['config'].pop(key, None)
            config[key] = value
        return self._render_values(config)

    def _render_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        rendered_config: Dict[str, Any] = {}
        for key, value in config.items():
            if isinstance(value, str):
                try:
                    value = get_rendered(value, self.render_ctx, native=True)
                except UndefinedMacroError as e:
                    raise CustomMacroPopulatingConfigValueError(
                        target_name=self.target.name, 
                        column_name=self.column_name, 
                        name=self.name, 
                        key=key, 
                        err_msg=e.msg
                    )
            if value is not None:
                rendered_config[key] = value
        return rendered_config

    def _bad_type(self) -> Exception:
        return TypeError('invalid target type "{}"'.format(type(self.target)))

    @staticmethod
    def extract_test_args(data_test: Any, name: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        if not isinstance(data_test, dict):
            raise TestTypeError(data_test)
        if 'test_name' in data_test.keys():
            test_name = data_test.pop('test_name')
            test_args = data_test
        else:
            data_test_items = list(data_test.items())
            if len(data_test_items) != 1:
                raise TestDefinitionDictLengthError(data_test)
            test_name, test_args = data_test_items[0]
        if not isinstance(test_args, dict):
            raise TestArgsNotDictError(test_args)
        if not isinstance(test_name, str):
            raise TestNameNotStringError(test_name)
        test_args = deepcopy(test_args)
        if name is not None:
            test_args['column_name'] = name
        return (test_name, test_args)

    def tags(self) -> List[str]:
        tags = self.config.get('tags', [])
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, list):
            raise TagsNotListOfStringsError(tags)
        for tag in tags:
            if not isinstance(tag, str):
                raise TagNotStringError(tag)
        return tags[:]

    def macro_name(self) -> str:
        macro_name = 'test_{}'.format(self.name)
        if self.namespace is not None:
            macro_name = '{}.{}'.format(self.namespace, macro_name)
        return macro_name

    def get_synthetic_test_names(self) -> Tuple[str, str]:
        target_name: str = self.target.name
        if isinstance(self.target, UnparsedModelUpdate):
            name = self.name
            if self.version:
                target_name = f'{self.target.name}_v{self.version}'
        elif isinstance(self.target, UnparsedNodeUpdate):
            name = self.name
        elif isinstance(self.target, UnpatchedSourceDefinition):
            name = 'source_' + self.name
        else:
            raise self._bad_type()
        if self.namespace is not None:
            name = '{}_{}'.format(self.namespace, name)
        return synthesize_generic_test_names(name, target_name, self.args)

    def construct_config(self) -> str:
        configs = ','.join([
            f'{key}=' + ('"' + value.replace(\'"\', \'\\\\"\') + '"' if isinstance(value, str) else str(value))
            for key, value in self.config.items() if value is not None
        ])
        if configs:
            return f'{{{{ config({configs}) }}}}'
        else:
            return ''

    def build_raw_code(self) -> str:
        return '{{{{ {macro}(**{kwargs_name}) }}}}{config}'.format(
            macro=self.macro_name(), 
            config=self.construct_config(), 
            kwargs_name=GENERIC_TEST_KWARGS_NAME
        )

    def build_model_str(self) -> str:
        targ = self.target
        if isinstance(self.target, UnparsedModelUpdate):
            if self.version:
                target_str = f"ref('{targ.name}', version='{self.version}')"
            else:
                target_str = f"ref('{targ.name}')"
        elif isinstance(self.target, UnparsedNodeUpdate):
            target_str = f"ref('{targ.name}')"
        elif isinstance(self.target, UnpatchedSourceDefinition):
            target_str = f"source('{targ.source.name}', '{targ.table.name}')"
        else:
            target_str = ''
        return f'{{{{ get_where_subquery({target_str}) }}}}'