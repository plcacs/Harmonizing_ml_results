```python
import os
import string
from typing import Any, Callable, Dict, Generator, List, Optional, Type, Union
from unittest import TestCase, mock

import agate
import pytest

from dbt.config.project import PartialProject
from dbt.contracts.graph.manifest import Manifest
from dbt_common.dataclass_schema import ValidationError


def normalize(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


class Obj:
    which: str = "blah"
    single_threaded: bool = False


def mock_connection(name: str, state: str = "open") -> mock.MagicMock:
    conn = mock.MagicMock()
    conn.name = name
    conn.state = state
    return conn


def profile_from_dict(profile: Dict[str, Any], profile_name: str, cli_vars: Union[str, Dict[str, Any]] = "{}") -> Any:
    from dbt.config import Profile
    from dbt.config.renderer import ProfileRenderer
    from dbt.config.utils import parse_cli_vars

    if not isinstance(cli_vars, dict):
        cli_vars = parse_cli_vars(cli_vars)

    renderer = ProfileRenderer(cli_vars)
    return Profile.from_raw_profile_info(
        profile,
        profile_name,
        renderer,
    )


def project_from_dict(
    project: Dict[str, Any],
    profile: Any,
    packages: Optional[Dict[str, Any]] = None,
    selectors: Optional[Dict[str, Any]] = None,
    cli_vars: Union[str, Dict[str, Any]] = "{}"
) -> Any:
    from dbt.config.renderer import DbtProjectYamlRenderer
    from dbt.config.utils import parse_cli_vars

    if not isinstance(cli_vars, dict):
        cli_vars = parse_cli_vars(cli_vars)

    renderer = DbtProjectYamlRenderer(profile, cli_vars)

    project_root = project.pop("project-root", os.getcwd())

    partial = PartialProject.from_dicts(
        project_root=project_root,
        project_dict=project,
        packages_dict=packages,
        selectors_dict=selectors,
    )
    return partial.render(renderer)


def config_from_parts_or_dicts(
    project: Union[Dict[str, Any], Any],
    profile: Union[Dict[str, Any], Any],
    packages: Optional[Dict[str, Any]] = None,
    selectors: Optional[Dict[str, Any]] = None,
    cli_vars: Dict[str, Any] = {}
) -> Any:
    from copy import deepcopy

    from dbt.config import Profile, Project, RuntimeConfig

    if isinstance(project, Project):
        profile_name = project.profile_name
    else:
        profile_name = project.get("profile")

    if not isinstance(profile, Profile):
        profile = profile_from_dict(
            deepcopy(profile),
            profile_name,
            cli_vars,
        )

    if not isinstance(project, Project):
        project = project_from_dict(
            deepcopy(project),
            profile,
            packages,
            selectors,
            cli_vars,
        )

    args = Obj()
    args.vars = cli_vars
    args.profile_dir = "/dev/null"
    return RuntimeConfig.from_parts(project=project, profile=profile, args=args)


def inject_plugin(plugin: Any) -> None:
    from dbt.adapters.factory import FACTORY

    key = plugin.adapter.type()
    FACTORY.plugins[key] = plugin


def inject_plugin_for(config: Any) -> Any:
    from dbt.adapters.factory import FACTORY

    FACTORY.load_plugin(config.credentials.type)
    adapter = FACTORY.get_adapter(config)
    return adapter


def inject_adapter(value: Any, plugin: Any) -> None:
    inject_plugin(plugin)
    from dbt.adapters.factory import FACTORY

    key = value.type()
    FACTORY.adapters[key] = value


def clear_plugin(plugin: Any) -> None:
    from dbt.adapters.factory import FACTORY

    key = plugin.adapter.type()
    FACTORY.plugins.pop(key, None)
    FACTORY.adapters.pop(key, None)


class ContractTestCase(TestCase):
    ContractType: Optional[Type] = None

    def setUp(self) -> None:
        self.maxDiff = None
        super().setUp()

    def assert_to_dict(self, obj: Any, dct: Dict[str, Any]) -> None:
        self.assertEqual(obj.to_dict(omit_none=True), dct)

    def assert_from_dict(self, obj: Any, dct: Dict[str, Any], cls: Optional[Type] = None) -> None:
        if cls is None:
            cls = self.ContractType
        cls.validate(dct)
        self.assertEqual(cls.from_dict(dct), obj)

    def assert_symmetric(self, obj: Any, dct: Dict[str, Any], cls: Optional[Type] = None) -> None:
        self.assert_to_dict(obj, dct)
        self.assert_from_dict(obj, dct, cls)

    def assert_fails_validation(self, dct: Dict[str, Any], cls: Optional[Type] = None) -> None:
        if cls is None:
            cls = self.ContractType

        with self.assertRaises(ValidationError):
            cls.validate(dct)
            cls.from_dict(dct)


def compare_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> None:
    first_set = set(dict1.keys())
    second_set = set(dict2.keys())
    print(f"--- Difference between first and second keys: {first_set.difference(second_set)}")
    print(f"--- Difference between second and first keys: {second_set.difference(first_set)}")
    common_keys = set(first_set).intersection(set(second_set))
    found_differences = False
    for key in common_keys:
        if dict1[key] != dict2[key]:
            print(f"--- --- first dict: {key}: {str(dict1[key])}")
            print(f"--- --- second dict: {key}: {str(dict2[key])}")
            found_differences = True
    if found_differences:
        print("--- Found differences in dictionaries")
    else:
        print("--- Found no differences in dictionaries")


def assert_from_dict(obj: Any, dct: Dict[str, Any], cls: Optional[Type] = None) -> None:
    if cls is None:
        cls = obj.__class__
    cls.validate(dct)

    obj_from_dict = cls.from_dict(dct)

    if hasattr(obj, "created_at"):
        obj_from_dict.created_at = 1
        obj.created_at = 1

    assert obj_from_dict == obj


def assert_to_dict(obj: Any, dct: Dict[str, Any]) -> None:
    obj_to_dict = obj.to_dict(omit_none=True)
    if "created_at" in obj_to_dict:
        obj_to_dict["created_at"] = 1
    if "created_at" in dct:
        dct["created_at"] = 1
    if obj_to_dict != dct:
        compare_dicts(obj_to_dict, dct)
    assert obj_to_dict == dct


def assert_symmetric(obj: Any, dct: Dict[str, Any], cls: Optional[Type] = None) -> None:
    assert_to_dict(obj, dct)
    assert_from_dict(obj, dct, cls)


def assert_fails_validation(dct: Dict[str, Any], cls: Type) -> None:
    with pytest.raises(ValidationError):
        cls.validate(dct)
        cls.from_dict(dct)


def generate_name_macros(package: str) -> Generator[Any, None, None]:
    from dbt.contracts.graph.nodes import Macro
    from dbt.node_types import NodeType

    name_sql = {}
    for component in ("database", "schema", "alias"):
        if component == "alias":
            source = "node.name"
        else:
            source = f"target.{component}"
        name = f"generate_{component}_name"
        sql = f"{{% macro {name}(value, node) %}} {{% if value %}} {{{{ value }}}} {{% else %}} {{{{ {source} }}}} {{% endif %}} {{% endmacro %}}"
        name_sql[name] = sql

    for name, sql in name_sql.items():
        pm = Macro(
            name=name,
            resource_type=NodeType.Macro,
            unique_id=f"macro.{package}.{name}",
            package_name=package,
            original_file_path=normalize("macros/macro.sql"),
            path=normalize("macros/macro.sql"),
            macro_sql=sql,
        )
        yield pm


class TestAdapterConversions(TestCase):
    def _get_tester_for(self, column_type: Type) -> Any:
        from dbt_common.clients import agate_helper

        if column_type is agate.TimeDelta:
            return agate.TimeDelta()

        for instance in agate_helper.DEFAULT_TYPE_TESTER._possible_types:
            if isinstance(instance, column_type):
                return instance

        raise ValueError(f"no tester for {column_type}")

    def _make_table_of(self, rows: List[List[Any]], column_types: Union[Type, List[Type]]) -> agate.Table:
        column_names = list(string.ascii_letters[: len(rows[0])])
        if isinstance(column_types, type):
            column_types = [self._get_tester_for(column_types) for _ in column_names]
        else:
            column_types = [self._get_tester_for(typ) for typ in column_types]
        table = agate.Table(rows, column_names=column_names, column_types=column_types)
        return table


def MockMacro(package: str, name: str = "my_macro", **kwargs: Any) -> mock.MagicMock:
    from dbt.contracts.graph.nodes import Macro
    from dbt.node_types import NodeType

    mock_kwargs = dict(
        resource_type=NodeType.Macro,
        package_name=package,
        unique_id=f"macro.{package}.{name}",
        original_file_path="/dev/null",
    )

    mock_kwargs.update(kwargs)

    macro = mock.MagicMock(spec=Macro, **mock_kwargs)
    macro.name = name
    return macro


def MockMaterialization(package: str, name: str = "my_materialization", adapter_type: Optional[str] = None, **kwargs: Any) -> mock.MagicMock:
    if adapter_type is None:
        adapter_type = "default"
    kwargs["adapter_type"] = adapter_type
    return MockMacro(package, f"materialization_{name}_{adapter_type}", **kwargs)


def MockGenerateMacro(package: str, component: str = "some_component", **kwargs: Any) -> mock.MagicMock:
    name = f"generate_{component}_name"
    return MockMacro(package, name=name, **kwargs)


def MockSource(package: str, source_name: str, name: str, **kwargs: Any) -> mock.MagicMock:
    from dbt.contracts.graph.nodes import SourceDefinition
    from dbt.node_types import NodeType

    src = mock.MagicMock(
        __class__=SourceDefinition,
        resource_type=NodeType.Source,
        source_name=source_name,
        package_name=package,
        unique_id=f"source.{package}.{source_name}.{name}",
        search_name=f"{source_name}.{name}",
        **kwargs,
    )
    src.name = name
    return src


def MockNode(package: str, name: str, resource_type: Optional[NodeType] = None, **kwargs: Any) -> mock.MagicMock:
    from dbt.contracts.graph.nodes import ModelNode, SeedNode
    from dbt.node_types import NodeType

    if resource_type is None:
        resource_type = NodeType.Model
    if resource_type == NodeType.Model:
        cls = ModelNode
    elif resource_type == NodeType.Seed:
        cls = SeedNode
    else:
        raise ValueError(f"I do not know how to handle {resource_type}")

    version = kwargs.get("version")
    search_name = name if version is None else f"{name}.v{version}"
    unique_id = f"{str(resource_type)}.{package}.{search_name}"
    node = mock.MagicMock(
        __class__=cls,
        resource_type=resource_type,
        package_name=package,
        unique_id=unique_id,
        search_name=search_name,
        **kwargs,
    )
    node.name = name
    node.is_versioned = resource_type is NodeType.Model and version is not None
    return node


def MockDocumentation(package: str, name: str, **kwargs: Any) -> mock.MagicMock:
    from dbt.contracts.graph.nodes import Documentation
    from dbt.node_types import NodeType

    doc = mock.MagicMock(
        __class__=Documentation,
        resource_type=NodeType.Documentation,
        package_name=package,
        search_name=name,
        unique_id=f"{package}.{name}",
        **kwargs,
    )
    doc.name = name
    return doc


def load_internal_manifest_macros(config: Any, macro_hook: Callable[[Any], None] = lambda m: None) -> Any:
    from dbt.parser.manifest import ManifestLoader

    return ManifestLoader.load_macros(config, macro_hook)


def dict_replace(dct: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    dct = dct.copy()
    dct.update(kwargs)
    return dct


def replace_config(n: Any, **kwargs: Any) -> Any:
    from dataclasses import replace

    return replace(
        n,
        config=n.config.replace(**kwargs),
        unrendered_config=dict_replace(n.unrendered_config, **kwargs),
    )


def make_manifest(
    nodes: List[Any] = [],
    sources: List[Any] = [],
    macros: List[Any] = [],
    docs: List[Any] = []
) -> Manifest:
    return Manifest(
        nodes={n.unique_id: n for n in nodes},
        macros={m.unique_id: m for m in macros},
        sources={s.unique_id: s for s in sources},
        docs={d.unique_id: d for d in docs},
        disabled={},
        files={},
        exposures={},
        metrics={},
        selectors={},
    )
```