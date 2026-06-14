import os
from typing import Any, Callable, Dict, Generator, List, Optional, Type, Union
from unittest import TestCase, mock

import agate

from dbt.config import Profile, Project, RuntimeConfig
from dbt.config.project import PartialProject
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import Macro
from dbt_common.dataclass_schema import ValidationError


def normalize(path: str) -> str: ...


class Obj:
    which: str
    single_threaded: bool


def mock_connection(name: str, state: str = 'open') -> mock.MagicMock: ...


def profile_from_dict(
    profile: dict[str, Any],
    profile_name: str,
    cli_vars: Union[str, dict[str, Any]] = '{}',
) -> Profile: ...


def project_from_dict(
    project: dict[str, Any],
    profile: Profile,
    packages: Optional[dict[str, Any]] = None,
    selectors: Optional[dict[str, Any]] = None,
    cli_vars: Union[str, dict[str, Any]] = '{}',
) -> Project: ...


def config_from_parts_or_dicts(
    project: Union[Project, dict[str, Any]],
    profile: Union[Profile, dict[str, Any]],
    packages: Optional[dict[str, Any]] = None,
    selectors: Optional[dict[str, Any]] = None,
    cli_vars: dict[str, Any] = {},
) -> RuntimeConfig: ...


def inject_plugin(plugin: Any) -> None: ...


def inject_plugin_for(config: Any) -> Any: ...


def inject_adapter(value: Any, plugin: Any) -> None: ...


def clear_plugin(plugin: Any) -> None: ...


class ContractTestCase(TestCase):
    ContractType: Optional[Type[Any]]

    def setUp(self) -> None: ...
    def assert_to_dict(self, obj: Any, dct: dict[str, Any]) -> None: ...
    def assert_from_dict(
        self, obj: Any, dct: dict[str, Any], cls: Optional[Type[Any]] = None
    ) -> None: ...
    def assert_symmetric(
        self, obj: Any, dct: dict[str, Any], cls: Optional[Type[Any]] = None
    ) -> None: ...
    def assert_fails_validation(
        self, dct: dict[str, Any], cls: Optional[Type[Any]] = None
    ) -> None: ...


def compare_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> None: ...


def assert_from_dict(
    obj: Any, dct: dict[str, Any], cls: Optional[Type[Any]] = None
) -> None: ...


def assert_to_dict(obj: Any, dct: dict[str, Any]) -> None: ...


def assert_symmetric(
    obj: Any, dct: dict[str, Any], cls: Optional[Type[Any]] = None
) -> None: ...


def assert_fails_validation(dct: dict[str, Any], cls: Type[Any]) -> None: ...


def generate_name_macros(package: str) -> Generator[Macro, None, None]: ...


class TestAdapterConversions(TestCase):
    def _get_tester_for(self, column_type: Type[Any]) -> Any: ...
    def _make_table_of(
        self,
        rows: List[List[Any]],
        column_types: Union[Type[Any], List[Type[Any]]],
    ) -> agate.Table: ...


def MockMacro(
    package: str, name: str = 'my_macro', **kwargs: Any
) -> mock.MagicMock: ...


def MockMaterialization(
    package: str,
    name: str = 'my_materialization',
    adapter_type: Optional[str] = None,
    **kwargs: Any,
) -> mock.MagicMock: ...


def MockGenerateMacro(
    package: str, component: str = 'some_component', **kwargs: Any
) -> mock.MagicMock: ...


def MockSource(
    package: str, source_name: str, name: str, **kwargs: Any
) -> mock.MagicMock: ...


def MockNode(
    package: str,
    name: str,
    resource_type: Optional[Any] = None,
    **kwargs: Any,
) -> mock.MagicMock: ...


def MockDocumentation(
    package: str, name: str, **kwargs: Any
) -> mock.MagicMock: ...


def load_internal_manifest_macros(
    config: Any, macro_hook: Callable[[Any], None] = ...
) -> Manifest: ...


def dict_replace(dct: dict[str, Any], **kwargs: Any) -> dict[str, Any]: ...


def replace_config(n: Any, **kwargs: Any) -> Any: ...


def make_manifest(
    nodes: List[Any] = [],
    sources: List[Any] = [],
    macros: List[Any] = [],
    docs: List[Any] = [],
) -> Manifest: ...