import string
from unittest import mock
import networkx as nx
import pytest
import dbt.graph.cli as graph_cli
import dbt.graph.selector as graph_selector
import dbt_common.exceptions
from dbt.node_types import NodeType

def _get_graph() -> nx.DiGraph:
    integer_graph = nx.balanced_tree(2, 2, nx.DiGraph())
    package_mapping = {i: 'm.' + ('X' if i % 2 == 0 else 'Y') + '.' + letter for i, letter in enumerate(string.ascii_lowercase)}
    return graph_selector.Graph(nx.relabel_nodes(integer_graph, package_mapping))

def _get_manifest(graph: graph_selector.Graph) -> mock.MagicMock:
    nodes = {}
    for unique_id in graph:
        fqn = unique_id.split('.')
        node = mock.MagicMock(unique_id=unique_id, fqn=fqn, package_name=fqn[0], tags=[], resource_type=NodeType.Model, empty=False, config=mock.MagicMock(enabled=True), is_versioned=False)
        nodes[unique_id] = node
    nodes['m.X.a'].tags = ['abc']
    nodes['m.Y.b'].tags = ['abc', 'bcef']
    nodes['m.X.c'].tags = ['abc', 'bcef']
    nodes['m.Y.d'].tags = []
    nodes['m.X.e'].tags = ['efg', 'bcef']
    nodes['m.Y.f'].tags = ['efg', 'bcef']
    nodes['m.X.g'].tags = ['efg']
    return mock.MagicMock(nodes=nodes)

@pytest.fixture
def graph() -> graph_selector.Graph:
    return _get_graph()

@pytest.fixture
def manifest(graph: graph_selector.Graph) -> mock.MagicMock:
    return _get_manifest(graph)

def id_macro(arg: str) -> str:
    if isinstance(arg, str):
        return arg
    try:
        return '_'.join(arg)
    except TypeError:
        return arg

run_specs = [(['X.a'], [], {'m.X.a'}), (['tag:abc'], [], {'m.X.a', 'm.Y.b', 'm.X.c'}), (['*'], ['tag:abc'], {'m.Y.d', 'm.X.e', 'm.Y.f', 'm.X.g'}), ...]

@pytest.mark.parametrize('include,exclude,expected', run_specs, ids=id_macro)
def test_run_specs(include: list[str], exclude: list[str], expected: set[str], graph: graph_selector.Graph, manifest: mock.MagicMock):
    selector = graph_selector.NodeSelector(graph, manifest)
    spec = graph_cli.parse_difference(include, exclude)
    selected, _ = selector.select_nodes(spec)
    assert selected == expected

param_specs = [('a', False, None, False, None, 'fqn', 'a', False), ('+a', True, None, False, None, 'fqn', 'a', False), ...]

@pytest.mark.parametrize('spec,parents,parents_depth,children,children_depth,filter_type,filter_value,childrens_parents', param_specs, ids=id_macro)
def test_parse_specs(spec: str, parents: bool, parents_depth: int, children: bool, children_depth: int, filter_type: str, filter_value: str, childrens_parents: bool):
    parsed = graph_selector.SelectionCriteria.from_single_spec(spec)
    assert parsed.parents == parents
    assert parsed.parents_depth == parents_depth
    assert parsed.children == children
    assert parsed.children_depth == children_depth
    assert parsed.method == filter_type
    assert parsed.value == filter_value
    assert parsed.childrens_parents == childrens_parents

invalid_specs = ['@a+', '@a.b+', '@a.b*+', '@tag:a+', '@source:a+']

@pytest.mark.parametrize('invalid', invalid_specs, ids=lambda k: str(k))
def test_invalid_specs(invalid: str):
    with pytest.raises(dbt_common.exceptions.DbtRuntimeError):
        graph_selector.SelectionCriteria.from_single_spec(invalid)
