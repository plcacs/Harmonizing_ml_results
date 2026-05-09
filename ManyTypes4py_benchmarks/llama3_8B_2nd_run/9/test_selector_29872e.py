import string
from argparse import Namespace
from queue import Empty
from typing import List, Dict, Any
from unittest.mock import MagicMock
import networkx as nx
import pytest
import dbt.compilation
import dbt.config
import dbt.exceptions
import dbt.graph.cli as graph_cli
import dbt.graph.selector as graph_selector
import dbt.parser
import dbt.parser.manifest
import dbt.utils
import dbt_common.exceptions
from dbt.config.runtime import RuntimeConfig
from dbt.flags import set_from_args
from dbt.graph import NodeSelector, parse_difference
from dbt.node_types import NodeType

def _get_graph() -> nx.DiGraph:
    ...

def _get_manifest(graph: nx.DiGraph) -> MagicMock:
    ...

@pytest.fixture
def graph() -> nx.DiGraph:
    return _get_graph()

@pytest.fixture
def mock_manifest_with_mock_graph(graph: nx.DiGraph) -> MagicMock:
    return _get_manifest(graph)

def id_macro(arg: Any) -> str:
    ...

run_specs: List[Tuple[Any, Any, Any]] = [
    ...
]

@pytest.mark.parametrize('include, exclude, expected', run_specs, ids=id_macro)
def test_run_specs(include: Any, exclude: Any, expected: Any, graph: nx.DiGraph, mock_manifest_with_mock_graph: MagicMock) -> None:
    ...

param_specs: List[Tuple[Any, Any, Any, Any, Any, Any, Any, Any]] = [
    ...
]

@pytest.mark.parametrize('spec, parents, parents_depth, children, children_depth, filter_type, filter_value, childrens_parents', param_specs, ids=id_macro)
def test_parse_specs(spec: Any, parents: Any, parents_depth: Any, children: Any, children_depth: Any, filter_type: Any, filter_value: Any, childrens_parents: Any) -> None:
    ...

invalid_specs: List[str] = [
    ...
]

@pytest.mark.parametrize('invalid', invalid_specs, ids=lambda k: str(k))
def test_invalid_specs(invalid: str) -> None:
    ...

class TestCompiler:
    def test_single_model(self, runtime_config: RuntimeConfig) -> None:
        ...

    def test_two_models_simple_ref(self, runtime_config: RuntimeConfig) -> None:
        ...

class TestNodeSelector:
    def test_dependency_list(self, runtime_config: RuntimeConfig) -> None:
        ...

    def test_select_downstream_of_empty_model(self, runtime_config: RuntimeConfig) -> None:
        ...
