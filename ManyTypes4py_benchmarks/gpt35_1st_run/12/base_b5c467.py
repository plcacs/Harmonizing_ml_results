import abc
import itertools
import os
from typing import Any, Dict, Generic, List, Optional, TypeVar
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

    def __init__(self, project: RuntimeConfig, manifest: Manifest):
        self.project = project
        self.manifest = manifest

    @abc.abstractmethod
    def parse_file(self, block: FileBlock) -> None:
        pass

    @abc.abstractproperty
    def resource_type(self) -> str:
        pass

    def generate_unique_id(self, resource_name: str, hash: Optional[str] = None) -> str:
        return '.'.join(filter(None, [self.resource_type, self.project.project_name, resource_name, hash]))

class Parser(BaseParser[FinalValue], Generic[FinalValue]):

    def __init__(self, project: RuntimeConfig, manifest: Manifest, root_project: RuntimeConfig):
        super().__init__(project, manifest)
        self.root_project = root_project

class RelationUpdate:

    def __init__(self, config: RuntimeConfig, manifest: Manifest, component: str):
        ...

    def __call__(self, parsed_node: Any, override: Any) -> None:
        ...

class ConfiguredParser(Parser[FinalNode], Generic[ConfiguredBlockType, FinalNode]):

    def __init__(self, project: RuntimeConfig, manifest: Manifest, root_project: RuntimeConfig):
        ...

    @classmethod
    @abc.abstractmethod
    def get_compiled_path(cls, block: ConfiguredBlockType) -> str:
        pass

    @abc.abstractmethod
    def parse_from_dict(self, dict: Dict[str, Any], validate: bool = True) -> None:
        pass

    @abc.abstractproperty
    def resource_type(self) -> str:
        pass

    @property
    def default_schema(self) -> str:
        ...

    @property
    def default_database(self) -> str:
        ...

    def get_fqn_prefix(self, path: str) -> List[str]:
        ...

    def get_fqn(self, path: str, name: str) -> List[str]:
        ...

    def _mangle_hooks(self, config: Dict[str, Any]) -> None:
        ...

    def _create_error_node(self, name: str, path: str, original_file_path: str, raw_code: str, language: str = 'sql') -> UnparsedNode:
        ...

    def _create_parsetime_node(self, block: FileBlock, path: str, config: ContextConfig, fqn: List[str], name: Optional[str] = None, **kwargs: Any) -> Any:
        ...

    def _context_for(self, parsed_node: Any, config: RuntimeConfig) -> Any:
        ...

    def render_with_context(self, parsed_node: Any, config: RuntimeConfig) -> Any:
        ...

    def update_parsed_node_config_dict(self, parsed_node: Any, config_dict: Dict[str, Any]) -> None:
        ...

    def update_parsed_node_relation_names(self, parsed_node: Any, config_dict: Dict[str, Any]) -> None:
        ...

    def update_parsed_node_config(self, parsed_node: Any, config: ContextConfig, context: Any = None, patch_config_dict: Optional[Dict[str, Any]] = None, patch_file_id: Optional[str] = None) -> None:
        ...

    def initial_config(self, fqn: List[str]) -> ContextConfig:
        ...

    def config_dict(self, config: ContextConfig) -> Dict[str, Any]:
        ...

    def render_update(self, node: Any, config: ContextConfig) -> None:
        ...

    def add_result_node(self, block: FileBlock, node: Any) -> None:
        ...

    def parse_node(self, block: FileBlock) -> Any:
        ...

    def _update_node_relation_name(self, node: Any) -> None:
        ...

    @abc.abstractmethod
    def parse_file(self, file_block: ConfiguredBlockType) -> None:
        pass

class SimpleParser(ConfiguredParser[ConfiguredBlockType, FinalNode], Generic[ConfiguredBlockType, FinalNode]):
    pass

class SQLParser(ConfiguredParser[FileBlock, FinalNode], Generic[FinalNode]):

    def parse_file(self, file_block: FileBlock) -> None:
        ...

class SimpleSQLParser(SQLParser[FinalNode]):
    pass
