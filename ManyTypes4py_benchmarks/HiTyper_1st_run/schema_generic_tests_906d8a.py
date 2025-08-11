import itertools
import os
import pathlib
from typing import Any, Dict, List, Optional, Union
from dbt.adapters.factory import get_adapter, get_adapter_package_names
from dbt.artifacts.resources import NodeVersion, RefArgs
from dbt.clients.jinja import add_rendered_test_kwargs, get_rendered
from dbt.context.configured import SchemaYamlVars, generate_schema_yml_context
from dbt.context.context_config import ContextConfig
from dbt.context.macro_resolver import MacroResolver
from dbt.context.providers import generate_test_context
from dbt.contracts.files import FileHash
from dbt.contracts.graph.nodes import GenericTestNode, GraphMemberNode, ManifestNode, UnpatchedSourceDefinition
from dbt.contracts.graph.unparsed import UnparsedColumn, UnparsedNodeUpdate
from dbt.exceptions import CompilationError, ParsingError, SchemaConfigError, TestConfigError
from dbt.node_types import NodeType
from dbt.parser.base import SimpleParser
from dbt.parser.common import GenericTestBlock, Testable, TestBlock, TestDef, VersionedTestBlock, trimmed
from dbt.parser.generic_test_builders import TestBuilder
from dbt.parser.search import FileBlock
from dbt.utils import get_pseudo_test_path, md5
from dbt_common.dataclass_schema import ValidationError

class SchemaGenericTestParser(SimpleParser):

    def __init__(self, project: str, manifest: Union[str, pathlib.Path, Directory, None], root_project: str) -> None:
        super().__init__(project, manifest, root_project)
        self.schema_yaml_vars = SchemaYamlVars()
        self.render_ctx = generate_schema_yml_context(self.root_project, self.project.project_name, self.schema_yaml_vars)
        internal_package_names = get_adapter_package_names(self.root_project.credentials.type)
        self.macro_resolver = MacroResolver(self.manifest.macros, self.root_project.project_name, internal_package_names)

    @property
    def resource_type(self):
        return NodeType.Test

    @classmethod
    def get_compiled_path(cls: Union[T, bool, dict[str, typing.Any]], block: Union[dbparser.search.FileBlock, dict[str, bool], None, typing.Type]):
        return block.path.relative_path

    def parse_file(self, block: Union[bytes, dict[str, typing.Any], dict], dct: Union[None, bytes, dict[str, typing.Any], dict]=None) -> None:
        pass

    def parse_from_dict(self, dct: Union[dict, bool, AbstractSetIntStr, MappingIntStrAny], validate: bool=True):
        if validate:
            GenericTestNode.validate(dct)
        return GenericTestNode.from_dict(dct)

    def parse_column_tests(self, block: Union[int, list[str], str, None], column: Union[int, typing.Callable, tuple[float]], version: Union[int, list[str], str, None]) -> None:
        if not column.data_tests:
            return
        for data_test in column.data_tests:
            self.parse_test(block, data_test, column, version)

    def create_test_node(self, target: Union[str, None, dict[str, typing.Union[str,int]]], path: Union[str, None, dict[str, typing.Union[str,int]]], config: Union[str, None, dict[str, typing.Union[str,int]]], tags: Union[str, None, dict[str, typing.Union[str,int]]], fqn: Union[str, None, dict[str, typing.Union[str,int]]], name: Union[str, None, bool], raw_code: Union[str, None, dict[str, typing.Union[str,int]]], test_metadata: Union[str, bool, dict[str, str], None], file_key_name: Union[str, None, dict[str, typing.Union[str,int]]], column_name: Union[str, None, dict[str, typing.Union[str,int]]], description: Union[str, None, dict[str, typing.Union[str,int]]]):
        HASH_LENGTH = 10

        def get_hashable_md(data: Any) -> Union[dict[list, ], list, str]:
            if type(data) == dict:
                return {k: get_hashable_md(data[k]) for k in sorted(data.keys())}
            elif type(data) == list:
                return [get_hashable_md(val) for val in data]
            else:
                return str(data)
        hashable_metadata = repr(get_hashable_md(test_metadata))
        hash_string = ''.join([name, hashable_metadata])
        test_hash = md5(hash_string)[-HASH_LENGTH:]
        dct = {'alias': name, 'schema': self.default_schema, 'database': self.default_database, 'fqn': fqn, 'name': name, 'resource_type': self.resource_type, 'tags': tags, 'path': path, 'original_file_path': target.original_file_path, 'package_name': self.project.project_name, 'raw_code': raw_code, 'language': 'sql', 'unique_id': self.generate_unique_id(name, test_hash), 'config': self.config_dict(config), 'test_metadata': test_metadata, 'column_name': column_name, 'checksum': FileHash.empty().to_dict(omit_none=True), 'file_key_name': file_key_name, 'description': description}
        try:
            GenericTestNode.validate(dct)
            return GenericTestNode.from_dict(dct)
        except ValidationError as exc:
            node = self._create_error_node(name=target.name, path=path, original_file_path=target.original_file_path, raw_code=raw_code)
            raise TestConfigError(exc, node)

    def parse_generic_test(self, target: Union[str, None, typing.Iterable[str], Testable], data_test: Union[str, None, typing.Iterable[str], Testable], tags: Union[typing.Iterable[str], None, str], column_name: Union[str, None, typing.Iterable[str], Testable], schema_file_id: Union[str, None, int], version: Union[str, None, typing.Iterable[str], Testable]) -> Union[str, list, list[list[typing.Any]]]:
        try:
            builder = TestBuilder(data_test=data_test, target=target, column_name=column_name, version=version, package_name=target.package_name, render_ctx=self.render_ctx)
            if self.schema_yaml_vars.env_vars:
                self.store_env_vars(target, schema_file_id, self.schema_yaml_vars.env_vars)
                self.schema_yaml_vars.env_vars = {}
        except ParsingError as exc:
            context = trimmed(str(target))
            msg = 'Invalid test config given in {}:\n\t{}\n\t@: {}'.format(target.original_file_path, exc.msg, context)
            raise ParsingError(msg) from exc
        except CompilationError as exc:
            context = trimmed(str(target))
            msg = f'Invalid generic test configuration given in {target.original_file_path}: \n{exc.msg}\n\t@: {context}'
            raise CompilationError(msg) from exc
        original_name = os.path.basename(target.original_file_path)
        compiled_path = get_pseudo_test_path(builder.compiled_name, original_name)
        path = pathlib.Path(target.original_file_path)
        relative_path = str(path.relative_to(*path.parts[:1]))
        fqn = self.get_fqn(relative_path, builder.fqn_name)
        config = self.initial_config(fqn)
        config.add_config_call(builder.config)
        metadata = {'namespace': builder.namespace, 'name': builder.name, 'kwargs': builder.args}
        tags = sorted(set(itertools.chain(tags, builder.tags())))
        if isinstance(target, UnpatchedSourceDefinition):
            file_key_name = f'{target.source.yaml_key}.{target.source.name}'
        else:
            file_key_name = f'{target.yaml_key}.{target.name}'
        node = self.create_test_node(target=target, path=compiled_path, config=config, fqn=fqn, tags=tags, name=builder.fqn_name, raw_code=builder.build_raw_code(), column_name=column_name, test_metadata=metadata, file_key_name=file_key_name, description=builder.description)
        self.render_test_update(node, config, builder, schema_file_id)
        return node

    def _lookup_attached_node(self, target: Union[str, None, dict[str, str]], version: str) -> Union[None, set[str], set, dict[str, typing.Any]]:
        """Look up attached node for Testable target nodes other than sources. Can be None if generic test attached to SQL node with no corresponding .sql file."""
        attached_node = None
        if not isinstance(target, UnpatchedSourceDefinition):
            attached_node_unique_id = self.manifest.ref_lookup.get_unique_id(target.name, target.package_name, version)
            if attached_node_unique_id:
                attached_node = self.manifest.nodes[attached_node_unique_id]
            else:
                disabled_node = self.manifest.disabled_lookup.find(target.name, None) or self.manifest.disabled_lookup.find(target.name.upper(), None)
                if disabled_node:
                    attached_node = self.manifest.disabled[disabled_node[0].unique_id][0]
        return attached_node

    def store_env_vars(self, target: str, schema_file_id: str, env_vars: dict) -> None:
        self.manifest.env_vars.update(env_vars)
        if schema_file_id in self.manifest.files:
            schema_file = self.manifest.files[schema_file_id]
            if isinstance(target, UnpatchedSourceDefinition):
                search_name = target.source.name
                yaml_key = target.source.yaml_key
                if '.' in search_name:
                    search_name, _ = search_name.split('.')
            else:
                search_name = target.name
                yaml_key = target.yaml_key
            for var in env_vars.keys():
                schema_file.add_env_var(var, yaml_key, search_name)

    def render_test_update(self, node: Union[dict, str, tuple[typing.Union[str,...]], dict[str, typing.Any]], config: Union[cmk.utils.type_defs.HostAddress, None, dict, str], builder: Union[dict, typing.Mapping, typing.Type], schema_file_id: Union[str, typing.Type, typing.IO]) -> None:
        macro_unique_id = self.macro_resolver.get_macro_id(node.package_name, 'test_' + builder.name)
        node.depends_on.add_macro(macro_unique_id)
        if macro_unique_id in ['macro.dbt.test_not_null', 'macro.dbt.test_unique']:
            config_call_dict = builder.config
            config._config_call_dict = config_call_dict
            self.update_parsed_node_config(node, config)
            if isinstance(builder.target, UnpatchedSourceDefinition):
                sources = [builder.target.fqn[-2], builder.target.fqn[-1]]
                node.sources.append(sources)
            else:
                node.refs.append(RefArgs(name=builder.target.name, version=builder.version))
        else:
            try:
                context = generate_test_context(node, self.root_project, self.manifest, config, self.macro_resolver)
                add_rendered_test_kwargs(context, node, capture_macros=True)
                get_rendered(node.raw_code, context, node, capture_macros=True)
                self.update_parsed_node_config(node, config)
            except ValidationError as exc:
                raise SchemaConfigError(exc, node=node) from exc
        attached_node = self._lookup_attached_node(builder.target, builder.version)
        if attached_node:
            node.attached_node = attached_node.unique_id
            node.group, node.group = (attached_node.group, attached_node.group)

    def parse_node(self, block: Union[mypy.nodes.Block, dbparser.schema_test_builders.TestBlock, typing.Iterable[object]]):
        """In schema parsing, we rewrite most of the part of parse_node that
        builds the initial node to be parsed, but rendering is basically the
        same
        """
        node = self.parse_generic_test(target=block.target, data_test=block.data_test, tags=block.tags, column_name=block.column_name, schema_file_id=block.file.file_id, version=block.version)
        self.add_test_node(block, node)
        return node

    def add_test_node(self, block: dbparser.search.FileBlock, node: Union[mypy.nodes.Var, mypy.nodes.FuncDef, mypy.nodes.OverloadedFuncDef, dict[str, tuple[str]]]) -> None:
        test_from = {'key': block.target.yaml_key, 'name': block.target.name}
        if node.config.enabled:
            self.manifest.add_node(block.file, node, test_from)
        else:
            self.manifest.add_disabled(block.file, node, test_from)

    def render_with_context(self, node: Union[typing.Mapping, dict, dbcontexcontext_config.ContextConfig], config: Union[dict[str, typing.Any], typing.OrderedDict, dict]) -> None:
        """Given the parsed node and a ContextConfig to use during
        parsing, collect all the refs that might be squirreled away in the test
        arguments. This includes the implicit "model" argument.
        """
        context = self._context_for(node, config)
        add_rendered_test_kwargs(context, node, capture_macros=True)
        get_rendered(node.raw_code, context, node, capture_macros=True)

    def parse_test(self, target_block: Union[str, tuple[typing.Union[int,str]], None], data_test: Union[Cell, None, typing.IO, T], column: Union[str, None, int], version: Union[int, list[dict[str, str]], list[int]]) -> None:
        if isinstance(data_test, str):
            data_test = {data_test: {}}
        if column is None:
            column_name = None
            column_tags = []
        else:
            column_name = column.name
            should_quote = column.quote or (column.quote is None and target_block.quote_columns)
            if should_quote:
                column_name = get_adapter(self.root_project).quote(column_name)
            column_tags = column.tags
        block = GenericTestBlock.from_test_block(src=target_block, data_test=data_test, column_name=column_name, tags=column_tags, version=version)
        self.parse_node(block)

    def parse_tests(self, block: Union[mypy.nodes.Block, mypy.nodes.FuncItem]) -> None:
        for column in block.columns:
            self.parse_column_tests(block, column, None)
        for data_test in block.data_tests:
            self.parse_test(block, data_test, None, None)

    def parse_versioned_tests(self, block: mypy.nodes.Block) -> None:
        if not block.target.versions:
            self.parse_tests(block)
        else:
            for version in block.target.versions:
                for column in block.target.get_columns_for_version(version.v):
                    self.parse_column_tests(block, column, version.v)
                for test in block.target.get_tests_for_version(version.v):
                    self.parse_test(block, test, None, version.v)

    def generate_unique_id(self, resource_name: str, hash: Union[None, str]=None) -> str:
        return '.'.join(filter(None, [self.resource_type, self.project.project_name, resource_name, hash]))