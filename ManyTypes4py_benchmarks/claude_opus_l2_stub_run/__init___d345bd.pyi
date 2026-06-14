import os
from typing import Any, Dict, Generator, List, Optional, Type, Union
from unittest import TestCase, mock

import agate

from dbt.config import Profile, Project, RuntimeConfig
from dbt.config.project import PartialProject
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import (
    Documentation,
    Macro,
    ModelNode,
    SeedNode,
    SourceDefinition,
)
from dbt.node_types import NodeType
from dbt_common.dataclass_schema import ValidationError


def normalize(path: str) -> str: ...


class Obj:
    which: str
    single_threaded: bool


def mock_connection(name: str, state: str = 'open') -> mock.MagicMock: ...


def profile_from_dict(
    profile: dict,
    profile_name: str,
    cli_vars: Union[str, Dict[str, Any]] = '{}',
) -> Profile: ...


def project_from_dict(
    project: dict,
    profile: Profile,
    packages: Optional[dict] = None,
    selectors: Optional[dict] = None,
    cli_vars: Union[str, Dict[str, Any]] = '{}',
) -> Project: ...


def config_from_parts_or_dicts(
    project: Union[Project, dict],
    profile: Union[Profile, dict],
    packages: Optional[dict] = None,
    selectors: Optional[dict] = None,
    cli_vars: Dict[str, Any] = {},
) -> RuntimeConfig: ...


def inject_plugin(plugin: Any) -> None: ...


def inject_plugin_for(config: Any) -> Any: ...


def inject_adapter(value: Any, plugin: Any) -> None: ...


def clear_plugin(plugin: Any) -> None: ...


class ContractTestCase(TestCase):
    ContractType: Optional[Any]
    maxDiff: Optional[int]

    def setUp(self) -> None: ...
    def assert_to_dict(self, obj: Any, dct: dict) -> None: ...
    def assert_from_dict(self, obj: Any, dct: dict, cls: Optional[Any] = None) -> None: ...
    def assert_symmetric(self, obj: Any, dct: dict, cls: Optional[Any] = None) -> None: ...
    def assert_fails_validation(self, dct: dict, cls: Optional[Any] = None) -> None: ...


def compare_dicts(dict1: dict, dict2: dict) -> None: ...


def assert_from_dict(obj: Any, dct: dict, cls: Optional[type] = None) -> None: ...


def assert_to_dict(obj: Any, dct: dict) -> None: ...


def assert_symmetric(obj: Any, dct: dict, cls: Optional[type] = None) -> None: ...


def assert_fails_validation(dct: dict, cls: Any) -> None: ...


def generate_name_macros(package: str) -> Generator[Macro, None, None]: ...


class TestAdapterConversions(TestCase):
    def _get_tester_for(self, column_type: type) -> Any: ...
    def _make_table_of(
        self,
        rows: List[list],
        column_types: Union[type, List[type]],
    ) -> agate.Table: ...


def MockMacro(package: str, name: str = 'my_macro', **kwargs: Any) -> mock.MagicMock: ...


def MockMaterialization(
    package: str,
    name: str = 'my_materialization',
    adapter_type: Optional[str] = None,
    **kwargs: Any,
) -> mock.MagicMock: ...


def MockGenerateMacro(
    package: str,
    component: str = 'some_component',
    **kwargs: Any,
) -> mock.MagicMock: ...


def MockSource(
    package: str,
    source_name: str,
    name: str,
    **kwargs: Any,
) -> mock.MagicMock: ...


def MockNode(
    package: str,
    name: str,
    resource_type: Optional[NodeType] = None,
    **kwargs: Any,
) -> mock.MagicMock: ...


def MockDocumentation(
    package: str,
    name: str,
    **kwargs: Any,
) -> mock.MagicMock: ...


def load_internal_manifest_macros(
    config: Any,
    macro_hook: Any = ...,
) -> Manifest: ...


def dict_replace(dct: dict, **kwargs: Any) -> dict: ...


def replace_config(n: Any, **kwargs: Any) -> Any: ...


def make_manifest(
    nodes: list = [],
    sources: list = [],
    macros: list = [],
    docs: list = [],
) -> Manifest: ...