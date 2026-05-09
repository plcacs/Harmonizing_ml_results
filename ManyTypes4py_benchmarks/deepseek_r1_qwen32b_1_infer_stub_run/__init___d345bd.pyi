"""Unit test utility functions."""

import os
import string
from unittest import TestCase
from unittest.mock import MagicMock
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
from dataclasses import dataclass
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import Macro, SourceDefinition, ModelNode, SeedNode, Documentation
from dbt.node_types import NodeType
from dbt.config.project import PartialProject
from dbt.config.profile import Profile
from dbt.config.renderer import DbtProjectYamlRenderer
from dbt.adapters.factory import BaseAdapter, Plugin
from dbt_common.dataclass_schema import ValidationError
import agate
from agate import TypeTester, Table

def normalize(path: str) -> str: ...

class Obj:
    which: str
    single_threaded: bool

def mock_connection(name: str, state: str = 'open') -> MagicMock: ...

def profile_from_dict(
    profile: Dict[str, Any],
    profile_name: str,
    cli_vars: Union[str, Dict[str, Any]] = '{}'
) -> Profile: ...

def project_from_dict(
    project: Dict[str, Any],
    profile: Profile,
    packages: Optional[Dict[str, Any]] = None,
    selectors: Optional[Dict[str, Any]] = None,
    cli_vars: Union[str, Dict[str, Any]] = '{}'
) -> PartialProject: ...

def config_from_parts_or_dicts(
    project: Union[Dict[str, Any], Project],
    profile: Union[Dict[str, Any], Profile],
    packages: Optional[Dict[str, Any]] = None,
    selectors: Optional[Dict[str, Any]] = None,
    cli_vars: Dict[str, Any] = {}
) -> RuntimeConfig: ...

def inject_plugin(plugin: Plugin) -> None: ...

def inject_plugin_for(config: RuntimeConfig) -> BaseAdapter: ...

def inject_adapter(value: BaseAdapter, plugin: Plugin) -> None: ...

def clear_plugin(plugin: Plugin) -> None: ...

class ContractTestCase(TestCase):
    ContractType: Optional[type] = None

    def setUp(self) -> None: ...
    def assert_to_dict(self, obj: Any, dct: Dict[str, Any]) -> None: ...
    def assert_from_dict(self, obj: Any, dct: Dict[str, Any], cls: Optional[type] = None) -> None: ...
    def assert_symmetric(self, obj: Any, dct: Dict[str, Any], cls: Optional[type] = None) -> None: ...
    def assert_fails_validation(self, dct: Dict[str, Any], cls: Optional[type] = None) -> None: ...

def compare_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> None: ...

def assert_from_dict(obj: Any, dct: Dict[str, Any], cls: Optional[type] = None) -> None: ...
def assert_to_dict(obj: Any, dct: Dict[str, Any]) -> None: ...
def assert_symmetric(obj: Any, dct: Dict[str, Any], cls: Optional[type] = None) -> None: ...
def assert_fails_validation(dct: Dict[str, Any], cls: type) -> None: ...

def generate_name_macros(package: str) -> Generator[Macro, None, None]: ...

class TestAdapterConversions(TestCase):
    def _get_tester_for(self, column_type: type) -> TypeTester: ...
    def _make_table_of(
        self,
        rows: List[List[Any]],
        column_types: Union[type, List[type]]
    ) -> Table: ...

def MockMacro(package: str, name: str = 'my_macro', **kwargs: Any) -> Macro: ...
def MockMaterialization(package: str, name: str = 'my_materialization', adapter_type: str = 'default', **kwargs: Any) -> Macro: ...
def MockGenerateMacro(package: str, component: str = 'some_component', **kwargs: Any) -> Macro: ...
def MockSource(package: str, source_name: str, name: str, **kwargs: Any) -> SourceDefinition: ...
def MockNode(
    package: str,
    name: str,
    resource_type: NodeType = NodeType.Model,
    **kwargs: Any
) -> Union[ModelNode, SeedNode]: ...
def MockDocumentation(package: str, name: str, **kwargs: Any) -> Documentation: ...

def load_internal_manifest_macros(config: RuntimeConfig, macro_hook: Optional[Callable[[Macro], None]] = None) -> Manifest: ...

def dict_replace(dct: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]: ...
def replace_config(n: Any, **kwargs: Any) -> Any: ...

def make_manifest(
    nodes: List[Any] = [],
    sources: List[Any] = [],
    macros: List[Any] = [],
    docs: List[Any] = []
) -> Manifest: ...