import itertools
import os
import pathlib
from typing import Any, Dict, List, Optional, Union, Set, Tuple, cast
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

    def __init__(self, project: Any, manifest: Any, root_project: Any) -> None:
        super().__init__(project, manifest, root_project)
        self.schema_yaml_vars: SchemaYamlVars = SchemaYamlVars()
        self.render_ctx: Dict[str, Any] = generate_schema_yml_context(self.root_project,
            self.project.project_name, self.schema_yaml_vars)
        internal_package_names: List[str] = get_adapter_package_names(self.
            root_project.credentials.type)
        self.macro_resolver: MacroResolver = MacroResolver(self.manifest.macros, self.
            root_project.project_name, internal_package_names)

    @property
    def func_bpvy90p6(self) -> NodeType:
        return NodeType.Test

    @classmethod
    def func_6n204gmv(cls, block: GenericTestBlock) -> str:
        return block.path.relative_path

    def func_oakaxmvf(self, block: GenericTestBlock, dct: Optional[Dict[str, Any]] = None) -> None:
        pass

    def func_zqdykt0y(self, dct: Dict[str, Any], validate: bool = True) -> GenericTestNode:
        if validate:
            GenericTestNode.validate(dct)
        return GenericTestNode.from_dict(dct)

    def func_953wk6l3(self, block: GenericTestBlock, column: UnparsedColumn, version: Optional[NodeVersion]) -> None:
        if not column.data_tests:
            return
        for data_test in column.data_tests:
            self.parse_test(block, data_test, column, version)

    def func_ykzc860u(self, target: Union[Testable, UnpatchedSourceDefinition], path: str, config: ContextConfig, tags: List[str], fqn: List[str], name: str,
        raw_code: str, test_metadata: Dict[str, Any], file_key_name: str, column_name: Optional[str], description: Optional[str]) -> GenericTestNode:
        HASH_LENGTH: int = 10

        def func_cmsshqqk(data: Any) -> Any:
            if type(data) == dict:
                return {k: func_cmsshqqk(data[k]) for k in sorted(data.keys())}
            elif type(data) == list:
                return [func_cmsshqqk(val) for val in data]
            else:
                return str(data)
        hashable_metadata: str = repr(func_cmsshqqk(test_metadata))
        hash_string: str = ''.join([name, hashable_metadata])
        test_hash: str = md5(hash_string)[-HASH_LENGTH:]
        dct: Dict[str, Any] = {'alias': name, 'schema': self.default_schema, 'database':
            self.default_database, 'fqn': fqn, 'name': name,
            'resource_type': self.resource_type, 'tags': tags, 'path': path,
            'original_file_path': target.original_file_path, 'package_name':
            self.project.project_name, 'raw_code': raw_code, 'language':
            'sql', 'unique_id': self.generate_unique_id(name, test_hash),
            'config': self.config_dict(config), 'test_metadata':
            test_metadata, 'column_name': column_name, 'checksum': FileHash
            .empty().to_dict(omit_none=True), 'file_key_name':
            file_key_name, 'description': description}
        try:
            GenericTestNode.validate(dct)
            return GenericTestNode.from_dict(dct)
        except ValidationError as exc:
            node: GenericTestNode = self._create_error_node(name=target.name, path=path,
                original_file_path=target.original_file_path, raw_code=raw_code
                )
            raise TestConfigError(exc, node)

    def func_19dtn2ee(self, target: Union[Testable, UnpatchedSourceDefinition], data_test: Union[str, Dict[str, Any]], tags: List[str], column_name: Optional[str],
        schema_file_id: str, version: Optional[NodeVersion]) -> GenericTestNode:
        try:
            builder: TestBuilder = TestBuilder(data_test=data_test, target=target,
                column_name=column_name, version=version, package_name=
                target.package_name, render_ctx=self.render_ctx)
            if self.schema_yaml_vars.env_vars:
                self.store_env_vars(target, schema_file_id, self.
                    schema_yaml_vars.env_vars)
                self.schema_yaml_vars.env_vars = {}
        except ParsingError as exc:
            context: str = trimmed(str(target))
            msg: str = 'Invalid test config given in {}:\n\t{}\n\t@: {}'.format(
                target.original_file_path, exc.msg, context)
            raise ParsingError(msg) from exc
        except CompilationError as exc:
            context = trimmed(str(target))
            msg = f"""Invalid generic test configuration given in {target.original_file_path}: 
{exc.msg}
	@: {context}"""
            raise CompilationError(msg) from exc
        original_name: str = os.path.basename(target.original_file_path)
        compiled_path: str = get_pseudo_test_path(builder.compiled_name,
            original_name)
        path: pathlib.Path = pathlib.Path(target.original_file_path)
        relative_path: str = str(path.relative_to(*path.parts[:1]))
        fqn: List[str] = self.get_fqn(relative_path, builder.fqn_name)
        config: ContextConfig = self.initial_config(fqn)
        config.add_config_call(builder.config)
        metadata: Dict[str, Any] = {'namespace': builder.namespace, 'name': builder.name,
            'kwargs': builder.args}
        tags = sorted(set(itertools.chain(tags, builder.tags())))
        if isinstance(target, UnpatchedSourceDefinition):
            file_key_name: str = f'{target.source.yaml_key}.{target.source.name}'
        else:
            file_key_name = f'{target.yaml_key}.{target.name}'
        node: GenericTestNode = self.create_test_node(target=target, path=compiled_path,
            config=config, fqn=fqn, tags=tags, name=builder.fqn_name,
            raw_code=builder.build_raw_code(), column_name=column_name,
            test_metadata=metadata, file_key_name=file_key_name,
            description=builder.description)
        self.render_test_update(node, config, builder, schema_file_id)
        return node

    def func_7bq8v6t1(self, target: Union[Testable, UnpatchedSourceDefinition], version: Optional[NodeVersion]) -> Optional[GraphMemberNode]:
        """Look up attached node for Testable target nodes other than sources. Can be None if generic test attached to SQL node with no corresponding .sql file."""
        attached_node: Optional[GraphMemberNode] = None
        if not isinstance(target, UnpatchedSourceDefinition):
            attached_node_unique_id: Optional[str] = self.manifest.ref_lookup.get_unique_id(
                target.name, target.package_name, version)
            if attached_node_unique_id:
                attached_node = self.manifest.nodes[attached_node_unique_id]
            else:
                disabled_node: Optional[Tuple[ManifestNode, ...]] = self.manifest.disabled_lookup.find(target.
                    name, None) or self.manifest.disabled_lookup.find(target
                    .name.upper(), None)
                if disabled_node:
                    attached_node = self.manifest.disabled[disabled_node[0]
                        .unique_id][0]
        return attached_node

    def func_bsyt568l(self, target: Union[Testable, UnpatchedSourceDefinition], schema_file_id: str, env_vars: Dict[str, str]) -> None:
        self.manifest.env_vars.update(env_vars)
        if schema_file_id in self.manifest.files:
            schema_file: Any = self.manifest.files[schema_file_id]
            if isinstance(target, UnpatchedSourceDefinition):
                search_name: str = target.source.name
                yaml_key: str = target.source.yaml_key
                if '.' in search_name:
                    search_name, _ = search_name.split('.')
            else:
                search_name = target.name
                yaml_key = target.yaml_key
            for var in env_vars.keys():
                schema_file.add_env_var(var, yaml_key, search_name)

    def func_xzv0sk10(self, node: GenericTestNode, config: ContextConfig, builder: TestBuilder, schema_file_id: str) -> None:
        macro_unique_id: str = self.macro_resolver.get_macro_id(node.
            package_name, 'test_' + builder.name)
        node.depends_on.add_macro(macro_unique_id)
        if macro_unique_id in ['macro.dbt.test_not_null',
            'macro.dbt.test_unique']:
            config_call_dict: Dict[str, Any] = builder.config
            config._config_call_dict = config_call_dict
            self.update_parsed_node_config(node, config)
            if isinstance(builder.target, UnpatchedSourceDefinition):
                sources: List[str] = [builder.target.fqn[-2], builder.target.fqn[-1]]
                node.sources.append(sources)
            else:
                node.refs.append(RefArgs(name=builder.target.name, version=
                    builder.version))
        else:
            try:
                context: Dict[str, Any] = generate_test_context(node, self.root_project,
                    self.manifest, config, self.macro_resolver)
                add_rendered_test_kwargs(context, node, capture_macros=True)
                get_rendered(node.raw_code, context, node, capture_macros=True)
                self.update_parsed_node_config(node, config)
            except ValidationError as exc:
                raise SchemaConfigError(exc, node=node) from exc
        attached_node: Optional[GraphMemberNode] = self._lookup_attached_node(builder.target, builder.
            version)
        if attached_node:
            node.attached_node = attached_node.unique_id
            node.group, node.group = attached_node.group, attached_node.group

    def func_6mvicwdc(self, block: GenericTestBlock) -> GenericTestNode:
        """In schema parsing, we rewrite most of the part of parse_node that
        builds the initial node to be parsed, but rendering is basically the
        same
        """
        node: GenericTestNode = self.parse_generic_test(target=block.target, data_test=block
            .data_test, tags=block.tags, column_name=block.column_name,
            schema_file_id=block.file.file_id, version=block.version)
        self.add_test_node(block, node)
        return node

    def func_3llpyfh0(self, block: GenericTestBlock, node: GenericTestNode) -> None:
        test_from: Dict[str, str] = {'key': block.target.yaml_key, 'name': block.target.name}
        if node.config.enabled:
            self.manifest.add_node(block.file, node, test_from)
        else:
            self.manifest.add_disabled(block.file, node, test_from)

    def func_zcqxkc8m(self, node: GenericTestNode, config: ContextConfig) -> None:
        """Given the parsed node and a ContextConfig to use during
        parsing, collect all the refs that might be squirreled away in the test
        arguments. This includes the implicit "model" argument.
        """
        context: Dict[str, Any] = self._context_for(node, config)
        add_rendered_test_kwargs(context, node, capture_macros=True)
        get_rendered(node.raw_code, context, node, capture_macros=True)

    def func_5fu9201u(self, target_block: TestBlock, data_test: Union[str, Dict[str, Any]], column: Optional[UnparsedColumn], version: Optional[NodeVersion]) -> None:
        if isinstance(data_test, str):
            data_test = {data_test: {}}
        if column is None:
            column_name: Optional[str] = None
            column_tags: List[str] = []
        else:
            column_name = column.name
            should_quote: bool = (column.quote or column.quote is None and
                target_block.quote_columns)
            if should_quote:
                column_name = get_adapter(self.root_project).quote(column_name)
            column_tags = column.tags
        block: GenericTestBlock = GenericTestBlock.from_test_block(src=target_block,
            data_test=data_test, column_name=column_name, tags=column_tags,
            version=version)
        self.parse_node(block)

    def func_vgle3cc3(self, block: TestBlock) -> None:
        for column in block.columns:
            self.parse_column_tests(block, column, None)
        for data_test in block.data_tests:
            self.parse_test(block, data_test, None, None)

    def func_qzzub2vo(self, block: TestBlock) -> None:
        if not block.target.versions:
            self.parse_tests(block)
        else:
            for version in block.target.versions:
                for column in block.target.get_columns_for_version(version.v):
                    self.parse_column_tests(block, column, version.v)
                for test in block.target.get_tests_for_version(version.v):
                    self.parse_test(block, test, None, version.v)

    def func_uu6wl9be(self, resource_name: str, hash: Optional[str] = None) -> str:
        return '.'.join(filter(None, [self.resource_type, self.project.
            project_name, resource_name, hash]))
