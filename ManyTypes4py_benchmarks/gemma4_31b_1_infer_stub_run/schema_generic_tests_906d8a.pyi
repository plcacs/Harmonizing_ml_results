import itertools
import os
import pathlib
from typing import Any, Dict, List, Optional, Union, overload
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
    schema_yaml_vars: SchemaYamlVars
    render_ctx: Any
    macro_resolver: MacroResolver

    def __init__(self, project: Any, manifest: Any, root_project: Any) -> None: ...

    @property
    def resource_type(self) -> NodeType: ...

    @classmethod
    def get_compiled_path(cls, block: Any) -> str: ...

    def parse_file(self, block: Any, dct: Optional[Dict[str, Any]] = None) -> None: ...

    def parse_from_dict(self, dct: Dict[str, Any], validate: bool = True) -> GenericTestNode: ...

    def parse_column_tests(self, block: Any, column: Any, version: Optional[str]) -> None: ...

    def create_test_node(
        self,
        target: Any,
        path: str,
        config: Any,
        tags: List[str],
        fqn: str,
        name: str,
        raw_code: str,
        test_metadata: Dict[str, Any],
        file_key_name: str,
        column_name: Optional[str],
        description: Optional[str],
    ) -> GenericTestNode: ...

    def parse_generic_test(
        self,
        target: Any,
        data_test: Union[str, Dict[str, Any]],
        tags: List[str],
        column_name: Optional[str],
        schema_file_id: str,
        version: Optional[str],
    ) -> GenericTestNode: ...

    def _lookup_attached_node(self, target: Any, version: Optional[str]) -> Optional[ManifestNode]: ...

    def store_env_vars(self, target: Any, schema_file_id: str, env_vars: Dict[str, Any]) -> None: ...

    def render_test_update(
        self,
        node: GenericTestNode,
        config: Any,
        builder: TestBuilder,
        schema_file_id: str,
    ) -> None: ...

    def parse_node(self, block: GenericTestBlock) -> GenericTestNode: ...

    def add_test_node(self, block: GenericTestBlock, node: GenericTestNode) -> None: ...

    def render_with_context(self, node: GenericTestNode, config: ContextConfig) -> None: ...

    def parse_test(
        self,
        target_block: Any,
        data_test: Union[str, Dict[str, Any]],
        column: Optional[Any],
        version: Optional[str],
    ) -> None: ...

    def parse_tests(self, block: Any) -> None: ...

    def parse_versioned_tests(self, block: Any) -> None: ...

    def generate_unique_id(self, resource_name: str, hash: Optional[str] = None) -> str: ...