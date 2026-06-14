import itertools
import os
import pathlib
from typing import Any, Dict, List, Optional, Union

from dbt.adapters.factory import get_adapter, get_adapter_package_names
from dbt.artifacts.resources import NodeVersion, RefArgs
from dbt.context.configured import SchemaYamlVars
from dbt.context.context_config import ContextConfig
from dbt.context.macro_resolver import MacroResolver
from dbt.contracts.files import FileHash
from dbt.contracts.graph.nodes import GenericTestNode, GraphMemberNode, ManifestNode, UnpatchedSourceDefinition
from dbt.contracts.graph.unparsed import UnparsedColumn, UnparsedNodeUpdate
from dbt.node_types import NodeType
from dbt.parser.base import SimpleParser
from dbt.parser.common import GenericTestBlock, Testable, TestBlock, TestDef, VersionedTestBlock
from dbt.parser.generic_test_builders import TestBuilder
from dbt.parser.search import FileBlock


class SchemaGenericTestParser(SimpleParser):

    def __init__(self, project: Any, manifest: Any, root_project: Any) -> None: ...

    schema_yaml_vars: SchemaYamlVars
    render_ctx: Dict[str, Any]
    macro_resolver: MacroResolver

    @property
    def resource_type(self) -> NodeType: ...

    @classmethod
    def get_compiled_path(cls, block: FileBlock) -> str: ...

    def parse_file(self, block: FileBlock, dct: Optional[Dict[str, Any]] = ...) -> None: ...

    def parse_from_dict(self, dct: Dict[str, Any], validate: bool = ...) -> GenericTestNode: ...

    def parse_column_tests(
        self,
        block: Union[TestBlock, VersionedTestBlock],
        column: UnparsedColumn,
        version: Optional[NodeVersion],
    ) -> None: ...

    def create_test_node(
        self,
        target: Any,
        path: str,
        config: ContextConfig,
        tags: List[str],
        fqn: List[str],
        name: str,
        raw_code: str,
        test_metadata: Dict[str, Any],
        file_key_name: str,
        column_name: Optional[str],
        description: str,
    ) -> GenericTestNode: ...

    def parse_generic_test(
        self,
        target: Union[Testable, UnpatchedSourceDefinition],
        data_test: TestDef,
        tags: List[str],
        column_name: Optional[str],
        schema_file_id: str,
        version: Optional[NodeVersion],
    ) -> GenericTestNode: ...

    def _lookup_attached_node(
        self,
        target: Union[Testable, UnpatchedSourceDefinition],
        version: Optional[NodeVersion],
    ) -> Optional[ManifestNode]: ...

    def store_env_vars(
        self,
        target: Union[Testable, UnpatchedSourceDefinition],
        schema_file_id: str,
        env_vars: Dict[str, Any],
    ) -> None: ...

    def render_test_update(
        self,
        node: GenericTestNode,
        config: ContextConfig,
        builder: TestBuilder,
        schema_file_id: str,
    ) -> None: ...

    def parse_node(self, block: GenericTestBlock) -> GenericTestNode: ...

    def add_test_node(self, block: GenericTestBlock, node: GenericTestNode) -> None: ...

    def render_with_context(self, node: GenericTestNode, config: ContextConfig) -> None: ...

    def parse_test(
        self,
        target_block: Union[TestBlock, VersionedTestBlock],
        data_test: TestDef,
        column: Optional[UnparsedColumn],
        version: Optional[NodeVersion],
    ) -> None: ...

    def parse_tests(self, block: TestBlock) -> None: ...

    def parse_versioned_tests(self, block: VersionedTestBlock) -> None: ...

    def generate_unique_id(self, resource_name: str, hash: Optional[str] = ...) -> str: ...