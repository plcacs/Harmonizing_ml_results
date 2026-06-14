from typing import Any, Dict, List, Optional, Union

from dbt.artifacts.resources import NodeVersion
from dbt.clients.jinja import add_rendered_test_kwargs, get_rendered
from dbt.context.configured import SchemaYamlVars, generate_schema_yml_context
from dbt.context.context_config import ContextConfig
from dbt.context.macro_resolver import MacroResolver
from dbt.context.providers import generate_test_context
from dbt.contracts.graph.nodes import GenericTestNode, ManifestNode, UnpatchedSourceDefinition
from dbt.contracts.graph.unparsed import UnparsedColumn
from dbt.node_types import NodeType
from dbt.parser.base import SimpleParser
from dbt.parser.common import GenericTestBlock, TestBlock, Testable, VersionedTestBlock
from dbt.parser.generic_test_builders import TestBuilder
from dbt.parser.search import FileBlock


class SchemaGenericTestParser(SimpleParser):
    def __init__(self, project: Any, manifest: Any, root_project: Any) -> None: ...
    @property
    def resource_type(self) -> NodeType: ...
    @classmethod
    def get_compiled_path(cls, block: FileBlock) -> str: ...
    def parse_file(self, block: FileBlock, dct: Optional[Dict[str, Any]] = ...) -> None: ...
    def parse_from_dict(self, dct: Dict[str, Any], validate: bool = ...) -> GenericTestNode: ...
    def parse_column_tests(self, block: TestBlock[Testable], column: UnparsedColumn, version: Optional[NodeVersion]) -> None: ...
    def create_test_node(
        self,
        target: Testable,
        path: str,
        config: ContextConfig,
        tags: List[str],
        fqn: List[str],
        name: str,
        raw_code: str,
        test_metadata: Dict[str, Any],
        file_key_name: str,
        column_name: Optional[str],
        description: Optional[str],
    ) -> GenericTestNode: ...
    def parse_generic_test(
        self,
        target: Testable,
        data_test: Dict[str, Any],
        tags: List[str],
        column_name: Optional[str],
        schema_file_id: str,
        version: Optional[NodeVersion],
    ) -> GenericTestNode: ...
    def _lookup_attached_node(self, target: Testable, version: Optional[NodeVersion]) -> Optional[ManifestNode]: ...
    def store_env_vars(self, target: Testable, schema_file_id: str, env_vars: Dict[str, Any]) -> None: ...
    def render_test_update(self, node: GenericTestNode, config: ContextConfig, builder: TestBuilder[Testable], schema_file_id: str) -> None: ...
    def parse_node(self, block: GenericTestBlock[Testable]) -> GenericTestNode: ...
    def add_test_node(self, block: GenericTestBlock[Testable], node: GenericTestNode) -> None: ...
    def render_with_context(self, node: GenericTestNode, config: ContextConfig) -> None: ...
    def parse_test(
        self,
        target_block: TestBlock[Testable],
        data_test: Union[str, Dict[str, Any]],
        column: Optional[UnparsedColumn],
        version: Optional[NodeVersion],
    ) -> None: ...
    def parse_tests(self, block: TestBlock[Testable]) -> None: ...
    def parse_versioned_tests(self, block: VersionedTestBlock[Testable]) -> None: ...
    def generate_unique_id(self, resource_name: str, hash: Optional[str] = ...) -> str: ...