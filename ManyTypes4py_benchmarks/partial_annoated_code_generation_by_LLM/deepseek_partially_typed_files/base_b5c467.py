import abc
import itertools
import os
from typing import Any, Dict, Generic, List, Optional, TypeVar, cast
from dbt import hooks, utils
from dbt.adapters.factory import get_adapter
from dbt.artifacts.resources import Contract
from dbt.clients.jinja import MacroGenerator, get_rendered
from dbt.config import RuntimeConfig
from dbt.context.context_config import ContextConfig
from dbt.context.providers import generate_generate_name_macro_context, generate_parser_model_context
from dbt.contracts.files import SchemaSourceFile
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import BaseNode, ManifestNode
from dbt.contracts.graph.unparsed import Docs, UnparsedNode
from dbt.exceptions import ConfigUpdateError, DbtInternalError, DictParseError, InvalidAccessTypeError
from dbt.flags import get_flags
from dbt.node_types import AccessType, ModelLanguage, NodeType
from dbt.parser.common import resource_types_to_schema_file_keys
from dbt.parser.search import FileBlock
from dbt_common.dataclass_schema import ValidationError
from dbt_common.utils import deep_merge

FinalValue = TypeVar('FinalValue', bound=BaseNode)
IntermediateValue = TypeVar('IntermediateValue', bound=BaseNode)
FinalNode = TypeVar('FinalNode', bound=ManifestNode)
ConfiguredBlockType = TypeVar('ConfiguredBlockType', bound=FileBlock)

class BaseParser(Generic[FinalValue]):

    def __init__(self, project: RuntimeConfig, manifest: Manifest) -> None:
        self.project: RuntimeConfig = project
        self.manifest: Manifest = manifest

    @abc.abstractmethod
    def parse_file(self, block: FileBlock) -> None:
        pass

    @abc.abstractproperty
    def resource_type(self) -> NodeType:
        pass

    def generate_unique_id(self, resource_name: str, hash: Optional[str] = None) -> str:
        """Returns a unique identifier for a resource
        An optional hash may be passed in to ensure uniqueness for edge cases"""
        return '.'.join(filter(None, [self.resource_type, self.project.project_name, resource_name, hash]))

class Parser(BaseParser[FinalValue], Generic[FinalValue]):

    def __init__(self, project: RuntimeConfig, manifest: Manifest, root_project: RuntimeConfig) -> None:
        super().__init__(project, manifest)
        self.root_project: RuntimeConfig = root_project

class RelationUpdate:

    def __init__(self, config: RuntimeConfig, manifest: Manifest, component: str) -> None:
        default_macro = manifest.find_generate_macro_by_name(component=component, root_project_name=config.project_name)
        if default_macro is None:
            raise DbtInternalError(f'No macro with name generate_{component}_name found')
        default_macro_context = generate_generate_name_macro_context(default_macro, config, manifest)
        self.default_updater = MacroGenerator(default_macro, default_macro_context)
        package_names = config.dependencies.keys() if config.dependencies else {}
        package_updaters: Dict[str, MacroGenerator] = {}
        for package_name in package_names:
            package_macro = manifest.find_generate_macro_by_name(component=component, root_project_name=config.project_name, imported_package=package_name)
            if package_macro:
                imported_macro_context = generate_generate_name_macro_context(package_macro, config, manifest)
                package_updaters[package_macro.package_name] = MacroGenerator(package_macro, imported_macro_context)
        self.package_updaters: Dict[str, MacroGenerator] = package_updaters
        self.component: str = component

    def __call__(self, parsed_node: ManifestNode, override: Optional[str]) -> None:
        if getattr(parsed_node, 'package_name', None) in self.package_updaters:
            new_value = self.package_updaters[parsed_node.package_name](override, parsed_node)
        else:
            new_value = self.default_updater(override, parsed_node)
        if isinstance(new_value, str):
            new_value = new_value.strip()
        setattr(parsed_node, self.component, new_value)

class ConfiguredParser(Parser[FinalNode], Generic[ConfiguredBlockType, FinalNode]):

    def __init__(self, project: RuntimeConfig, manifest: Manifest, root_project: RuntimeConfig) -> None:
        super().__init__(project, manifest, root_project)
        self._update_node_database: RelationUpdate = RelationUpdate(manifest=manifest, config=root_project, component='database')
        self._update_node_schema: RelationUpdate = RelationUpdate(manifest=manifest, config=root_project, component='schema')
        self._update_node_alias: RelationUpdate = RelationUpdate(manifest=manifest, config=root_project, component='alias')

    @classmethod
    @abc.abstractmethod
    def get_compiled_path(cls, block: ConfiguredBlockType) -> str:
        pass

    @abc.abstractmethod
    def parse_from_dict(self, dict: Dict[str, Any], validate: bool = True) -> FinalNode:
        pass

    @abc.abstractproperty
    def resource_type(self) -> NodeType:
        pass

    @property
    def default_schema(self) -> Optional[str]:
        return self.root_project.credentials.schema

    @property
    def default_database(self) -> Optional[str]:
        return self.root_project.credentials.database

    def get_fqn_prefix(self, path: str) -> List[str]:
        no_ext = os.path.splitext(path)[0]
        fqn = [self.project.project_name]
        fqn.extend(utils.split_path(no_ext)[:-1])
        return fqn

    def get_fqn(self, path: str, name: str) -> List[str]:
        """Get the FQN for the node. This impacts node selection and config
        application.
        """
        fqn = self.get_fqn_prefix(path)
        fqn.append(name)
        return fqn

    def _mangle_hooks(self, config: Dict[str, Any]) -> None:
        """Given a config dict that may have `pre-hook`/`post-hook` keys,
        convert it from the yucky maybe-a-string, maybe-a-dict to a dict.
        """
        for key in hooks.ModelHookType:
            if key in config:
                config[key] = [hooks.get_hook_dict(h) for h in config[key]]

    def _create_error_node(self, name: str, path: str, original_file_path: str, raw_code: str, language: str = 'sql') -> UnparsedNode:
        """If we hit an error before we've actually parsed a node, provide some
        level of useful information by attaching this to the exception.
        """
        return UnparsedNode(name=name, resource_type=self.resource_type, path=path, original_file_path=original_file_path, package_name=self.project.project_name, raw_code=raw_code, language=language)

    def _create_parsetime_node(self, block: ConfiguredBlockType, path: str, config: ContextConfig, fqn: List[str], name: Optional[str] = None, **kwargs: Any) -> FinalNode:
        """Create the node that will be passed in to the parser context for
        "rendering". Some information may be partial, as it'll be updated by
        config() and any ref()/source() calls discovered during rendering.
        """
        if name is None:
            name = block.name
        if block.path.relative_path.endswith('.py'):
            language = ModelLanguage.python
        else:
            language = ModelLanguage.sql
        dct: Dict[str, Any] = {'alias': name, 'schema': self.default_schema, 'database': self.default_database, 'fqn': fqn, 'name': name, 'resource_type': self.resource_type, 'path': path, 'original_file_path': block.path.original_file_path, 'package_name': self.project.project_name, 'raw_code': block.contents, 'language': language, 'unique_id': self.generate_unique_id(name), 'config': self.config_dict(config), 'checksum': block.file.checksum.to_dict(omit_none=True)}
        dct.update(kwargs)
        try:
            return self.parse_from_dict(dct, validate=True)
        except ValidationError as exc:
            node = self._create_error_node(name=block.name, path=path, original_file_path=block.path.original_file_path, raw_code=block.contents)
            raise DictParseError(exc, node=node)

    def _context_for(self, parsed_node: FinalNode, config: ContextConfig) -> Dict[str, Any]:
        return generate_parser_model_context(parsed_node, self.root_project, self.manifest, config)

    def render_with_context(self, parsed_node: FinalNode, config: ContextConfig) -> Dict[str, Any]:
        context = self._context_for(parsed_node, config)
        get_rendered(parsed_node.raw_code, context, parsed_node, capture_macros=True)
        return context

    def update_parsed_node_config_dict(self, parsed_node: FinalNode, config_dict: Dict[str, Any]) -> None:
        final_config_dict = parsed_node.config.to_dict(omit_none=True)
        final_config_dict.update({k.strip(): v for (k, v) in config_dict.items()})
        self._mangle_hooks(final_config_dict)
        parsed_node.config = parsed_node.config.from_dict(final_config_dict)

    def update_parsed_node_relation_names(self, parsed_node: FinalNode, config_dict: Dict[str, Any]) -> None:
        self._update_node_database(parsed_node, config_dict.get('database'))
        self._update_node_schema(parsed_node, config_dict.get('schema'))
        self._update_node_alias(parsed_node, config_dict.get('alias'))
        if getattr(parsed_node, 'resource_type', None) == NodeType.Snapshot:
            if 'target_database' in config_dict and config_dict['target_database']:
                parsed_node.database = config_dict['target_database']
            if 'target_schema' in config_dict and config_dict['target_schema']:
                parsed_node.schema = config_dict['target_schema']
        self._update_node_relation_name(parsed_node)

    def update_parsed_node_config(self, parsed_node: FinalNode, config: ContextConfig, context: Optional[Dict[str, Any]] = None, patch_config_dict: Optional[Dict[str, Any]] = None, patch_file_id: Optional[str] = None) -> None:
        """Given the ContextConfig used for parsing and the parsed node,
        generate and set the true values to use, overriding the temporary parse
        values set in _build_intermediate_parsed_node.
        """
        if not patch_config_dict:
            patch_config_dict = {}
        if parsed_node.resource_type == NodeType.Model and parsed_node.language == ModelLanguage.python:
            if 'materialized' not in patch_config_dict:
                patch_config_dict['materialized'] = 'table'
        config_dict = config.build_config_dict(patch_config_dict=patch_config_dict)
        model_tags = config_dict.get('tags', [])
        for tag in model_tags:
            if tag not in parsed_node.tags:
                parsed_node.tags.append(tag)
        if 'meta' in config_dict and config_dict['meta']:
            parsed_node.meta = config_dict['meta']
        if 'group' in config_dict and config_dict['group']:
            parsed_node.group = config_dict['group']
        if parsed_node.resource_type == NodeType.Model and config_dict.get('access', None):
            if AccessType.is_valid(config_dict['access']):
                assert hasattr(parsed_node, 'access')
                parsed_node.access = AccessType(config_dict['access'])
            else:
                raise InvalidAccessTypeError(unique_id=parsed_node.unique_id, field_value=config_dict['access'])
        if 'docs' in config_dict and config_dict['docs']:
            docs_show = config_dict['docs']['show'] if 'show' in config_dict['docs'] else parsed_node.docs.show
            if 'node_color' in config_dict['docs']:
                parsed_node.docs = Docs(show=docs_show, node_color=config_dict['docs']['node_color'])
            else:
                parsed_node.docs = Docs(show=docs_show)
        if 'contract' in config_dict and config_dict['contract']:
            contract_dct = config_dict['contract']
            Contract.validate(contract_dct)
            if hasattr(parsed_node, 'contract'):
                parsed_node.contract = Contract.from_dict(contract_dct)
        if get_flags().state_modified_compare_more_unrendered_values:
            if patch_file_id:
                patch_file = self.manifest.files.get(patch_file_id, None)
                if patch_file and isinstance(patch_file, SchemaSourceFile):
                    schema_key = resource_types_to_schema_file_keys[parsed_node.resource_type]
                    if (unrendered_patch_config := patch_file.get_unrendered_config(schema_key, parsed_node.name, getattr(parsed_node, 'version', None))):
                        patch_config_dict = deep_merge(patch_config_dict, unrendered_patch_config)
        parsed_node.unrendered_config = config.build_config_dict(rendered=False, patch_config_dict=patch_config_dict)
        parsed_node.config_call_dict = config._config_call_dict
        parsed_node.unrendered_config_call_dict = config._unrendered_config_call_dict
        self.update_parsed_node_config_dict(parsed_node, config_dict)
        self.update_parsed_node_relation_names(parsed_node, config_dict)
        if parsed_node.resource_type == NodeType.Test:
            return
        assert hasattr(parsed_node.config, 'pre_hook') and hasattr(parsed_node.config, 'post_hook')
        hooks_list = list(itertools.chain(parsed_node.config.pre_hook, parsed_node.config.post_hook))
        if not hooks_list:
            return
        if not context:
            context = self._context_for(parsed_node, config)
        for hook in hooks_list:
            get_rendered(hook.sql, context, parsed_node, capture_macros=True)

    def initial_config(self, fqn: List[str]) -> ContextConfig:
        config_version = min([self.project.config_version, self.root_project.config_version])
        if config_version == 2:
            return ContextConfig(self.root_project, fqn, self.resource_type, self.project.project_name)
        else:
            raise DbtInternalError(f'Got an unexpected project version={config_version}, expected 2')

    def config_dict(self, config: ContextConfig) -> Dict[str, Any]:
        config_dict = config.build_config_dict(base=True)
        self._mangle_hooks(config_dict)
        return config_dict

    def render_update(self, node: FinalNode, config: ContextConfig) -> None:
        try:
            context = self.render_with_context(node, config)
            self.update_parsed_node_config(node, config, context=context)
        except ValidationError as exc:
            raise ConfigUpdateError(exc, node=node) from exc

    def add_result_node(self, block: FileBlock, node: ManifestNode) -> None:
        if node.config.enabled:
            self.manifest.add_node(block.file, node)
        else:
            self.manifest.add_disabled(block.file, node)

    def parse_node(self, block: ConfiguredBlockType) -> FinalNode:
        compiled_path: str = self.get_compiled_path(block)
        fqn = self.get_fqn(compiled_path, block.name)
        config: ContextConfig = self.initial_config(fqn)
        node = self._create_parsetime_node(block=block, path=compiled_path, config=config, fqn=fqn)
        self.render_update(node, config)
        self.add_result_node(block, node)
        return node

    def _update_node_relation_name(self, node: ManifestNode) -> None:
        if getattr(node, 'is_relational', None) and (not getattr(node, 'is_ephemeral_model', None)):
            adapter = get_adapter(self.root_project)
            relation_cls = adapter.Relation
            node.relation_name = str(relation_cls.create_from(self.root_project, node))
        else:
            node.relation_name = None

    @abc.abstractmethod
    def parse_file(self, file_block: FileBlock) -> None:
        pass

class SimpleParser(ConfiguredParser[ConfiguredBlockType, FinalNode], Generic[ConfiguredBlockType, FinalNode]):
    pass

class SQLParser(ConfiguredParser[FileBlock, FinalNode], Generic[FinalNode]):

    def parse_file(self, file_block: FileBlock) -> None:
        self.parse_node(file_block)

class SimpleSQLParser(SQLParser[FinalNode]):
    pass
