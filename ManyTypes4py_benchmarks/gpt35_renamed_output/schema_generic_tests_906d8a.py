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

    def __init__(self, project: Any, manifest: Any, root_project: Any) -> None:
        self.schema_yaml_vars: SchemaYamlVars = SchemaYamlVars()
        self.render_ctx: Any = generate_schema_yml_context(self.root_project,
            self.project.project_name, self.schema_yaml_vars)

    @property
    def func_bpvy90p6(self) -> NodeType:
        return NodeType.Test

    @classmethod
    def func_6n204gmv(cls, block: Any) -> Any:
        return block.path.relative_path

    def func_oakaxmvf(self, block: Any, dct: Optional[Dict] = None) -> None:
        pass

    def func_zqdykt0y(self, dct: Dict, validate: bool = True) -> GenericTestNode:
        if validate:
            GenericTestNode.validate(dct)
        return GenericTestNode.from_dict(dct)

    def func_953wk6l3(self, block: Any, column: Any, version: Any) -> None:
        pass

    def func_ykzc860u(self, target: Any, path: str, config: Dict, tags: List, fqn: str, name: str, raw_code: str,
        test_metadata: Dict, file_key_name: str, column_name: str, description: str) -> GenericTestNode:
        pass

    def func_19dtn2ee(self, target: Any, data_test: Any, tags: List, column_name: str,
        schema_file_id: str, version: Any) -> Any:
        pass

    def func_7bq8v6t1(self, target: Any, version: Any) -> Any:
        pass

    def func_bsyt568l(self, target: Any, schema_file_id: str, env_vars: Dict) -> None:
        pass

    def func_xzv0sk10(self, node: Any, config: Any, builder: Any, schema_file_id: str) -> None:
        pass

    def func_6mvicwdc(self, block: Any) -> Any:
        pass

    def func_3llpyfh0(self, block: Any, node: Any) -> None:
        pass

    def func_zcqxkc8m(self, node: Any, config: Any) -> None:
        pass

    def func_5fu9201u(self, target_block: Any, data_test: Any, column: Any, version: Any) -> None:
        pass

    def func_vgle3cc3(self, block: Any) -> None:
        pass

    def func_qzzub2vo(self, block: Any) -> None:
        pass

    def func_uu6wl9be(self, resource_name: str, hash: Optional[str] = None) -> str:
        return '.'.join(filter(None, [self.resource_type, self.project.
            project_name, resource_name, hash]))
