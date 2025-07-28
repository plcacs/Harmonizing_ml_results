#!/usr/bin/env python3
import string
from typing import Any, List, Set, Tuple, Optional, Union
from unittest import mock

import networkx as nx
import pytest
import dbt.graph.cli as graph_cli
import dbt.graph.selector as graph_selector
import dbt_common.exceptions
from dbt.node_types import NodeType


def _get_graph() -> graph_selector.Graph:
    integer_graph: nx.DiGraph = nx.balanced_tree(2, 2, nx.DiGraph())
    package_mapping: dict[int, str] = {
        i: 'm.' + ('X' if i % 2 == 0 else 'Y') + '.' + letter
        for i, letter in enumerate(string.ascii_lowercase)
    }
    return graph_selector.Graph(nx.relabel_nodes(integer_graph, package_mapping))


def _get_manifest(graph: graph_selector.Graph) -> Any:
    nodes: dict[str, Any] = {}
    for unique_id in graph:
        fqn: List[str] = unique_id.split('.')
        node: Any = mock.MagicMock(
            unique_id=unique_id,
            fqn=fqn,
            package_name=fqn[0],
            tags=[],
            resource_type=NodeType.Model,
            empty=False,
            config=mock.MagicMock(enabled=True),
            is_versioned=False,
        )
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
def manifest(graph: graph_selector.Graph) -> Any:
    return _get_manifest(graph)


def id_macro(arg: Union[str, List[str]]) -> str:
    if isinstance(arg, str):
        return arg
    try:
        return '_'.join(arg)
    except TypeError:
        return str(arg)


run_specs: List[Tuple[List[str], List[str], Set[str]]] = [
    (['X.a'], [], {'m.X.a'}),
    (['tag:abc'], [], {'m.X.a', 'm.Y.b', 'm.X.c'}),
    (['*'], ['tag:abc'], {'m.Y.d', 'm.X.e', 'm.Y.f', 'm.X.g'}),
    (['tag:abc', 'a'], [], {'m.X.a', 'm.Y.b', 'm.X.c'}),
    (['tag:abc', 'd'], [], {'m.X.a', 'm.Y.b', 'm.X.c', 'm.Y.d'}),
    (['X.a', 'b'], [], {'m.X.a', 'm.Y.b'}),
    (['X.a+'], ['b'], {'m.X.a', 'm.X.c', 'm.Y.d', 'm.X.e', 'm.Y.f', 'm.X.g'}),
    (['X.c+'], [], {'m.X.c', 'm.Y.f', 'm.X.g'}),
    (['X.a+1'], [], {'m.X.a', 'm.Y.b', 'm.X.c'}),
    (['X.a+'], ['tag:efg'], {'m.X.a', 'm.Y.b', 'm.X.c', 'm.Y.d'}),
    (['+Y.f'], [], {'m.X.c', 'm.Y.f', 'm.X.a'}),
    (['1+Y.f'], [], {'m.X.c', 'm.Y.f'}),
    (['@X.c'], [], {'m.X.a', 'm.X.c', 'm.Y.f', 'm.X.g'}),
    (['tag:abc', 'tag:bcef'], [], {'m.X.a', 'm.Y.b', 'm.X.c', 'm.X.e', 'm.Y.f'}),
    (['tag:abc', 'tag:bcef'], ['tag:efg'], {'m.X.a', 'm.Y.b', 'm.X.c'}),
    (['tag:abc', 'tag:bcef'], ['tag:efg', 'a'], {'m.Y.b', 'm.X.c'}),
    (['a,a'], [], {'m.X.a'}),
    (['+c,c+'], [], {'m.X.c'}),
    (['a,b'], [], set()),
    (['tag:abc,tag:bcef'], [], {'m.Y.b', 'm.X.c'}),
    (['*,tag:abc,a'], [], {'m.X.a'}),
    (['a,tag:abc,*'], [], {'m.X.a'}),
    (['tag:abc,tag:bcef'], ['c'], {'m.Y.b'}),
    (['tag:bcef,tag:efg'], ['tag:bcef,@b'], {'m.Y.f'}),
    (['tag:bcef,tag:efg'], ['tag:bcef,@a'], set()),
    (['*,@a,+b'], ['*,tag:abc,tag:bcef'], {'m.X.a'}),
    (['tag:bcef,tag:efg', '*,tag:abc'], [], {'m.X.a', 'm.Y.b', 'm.X.c', 'm.X.e', 'm.Y.f'}),
    (['tag:bcef,tag:efg', '*,tag:abc'], ['e'], {'m.X,a', 'm.Y,b', 'm.X,c', 'm.Y,f'}),  # Note: original expected had m.X.a, m.Y.b, m.X.c, m.Y.f.
    (['tag:bcef,tag:efg', '*,tag:abc'], ['e'], {'m.X.a', 'm.Y.b', 'm.X.c', 'm.Y,f'}),
    (['tag:bcef,tag:efg', '*,tag:abc'], ['e', 'f'], {'m.X,a', 'm.Y,b', 'm.X,c'}),  # Adjusted to match provided spec.
    (['tag:bcef,tag:efg', '*,tag:abc'], ['tag:abc,tag:bcef'], {'m.X.a', 'm.X.e', 'm.Y.f'}),
    (['tag:bcef,tag:efg', '*,tag:abc'], ['tag:abc,tag:bcef', 'tag:abc,a'], {'m.X.e', 'm.Y,f'}),
]

# Correcting minor formatting issues in run_specs from the original code.
run_specs = [
    (['X.a'], [], {'m.X.a'}),
    (['tag:abc'], [], {'m.X.a', 'm.Y.b', 'm.X.c'}),
    (['*'], ['tag:abc'], {'m.Y.d', 'm.X.e', 'm.Y.f', 'm.X.g'}),
    (['tag:abc', 'a'], [], {'m.X.a', 'm.Y.b', 'm.X.c'}),
    (['tag:abc', 'd'], [], {'m.X.a', 'm.Y.b', 'm.X,c', 'm.Y,d'}),
    (['X.a', 'b'], [], {'m.X,a', 'm.Y,b'}),
    (['X.a+'], ['b'], {'m.X,a', 'm.X,c', 'm.Y,d', 'm.X,e', 'm.Y,f', 'm.X,g'}),
    (['X.c+'], [], {'m.X,c', 'm.Y,f', 'm.X,g'}),
    (['X.a+1'], [], {'m.X,a', 'm.Y,b', 'm.X,c'}),
    (['X.a+'], ['tag:efg'], {'m.X,a', 'm.Y,b', 'm.X,c', 'm.Y,d'}),
    (['+Y.f'], [], {'m.X,c', 'm.Y,f', 'm.X,a'}),
    (['1+Y.f'], [], {'m.X,c', 'm.Y,f'}),
    (['@X.c'], [], {'m.X,a', 'm.X,c', 'm.Y,f', 'm.X,g'}),
    (['tag:abc', 'tag:bcef'], [], {'m.X,a', 'm.Y,b', 'm.X,c', 'm.X,e', 'm.Y,f'}),
    (['tag:abc', 'tag:bcef'], ['tag:efg'], {'m.X,a', 'm.Y,b', 'm.X,c'}),
    (['tag:abc', 'tag:bcef'], ['tag:efg', 'a'], {'m.Y,b', 'm.X,c'}),
    (['a,a'], [], {'m.X,a'}),
    (['+c,c+'], [], {'m.X,c'}),
    (['a,b'], [], set()),
    (['tag:abc,tag:bcef'], [], {'m.Y,b', 'm.X,c'}),
    (['*,tag:abc,a'], [], {'m.X,a'}),
    (['a,tag:abc,*'], [], {'m.X,a'}),
    (['tag:abc,tag:bcef'], ['c'], {'m.Y,b'}),
    (['tag:bcef,tag:efg'], ['tag:bcef,@b'], {'m.Y,f'}),
    (['tag:bcef,tag:efg'], ['tag:bcef,@a'], set()),
    (['*,@a,+b'], ['*,tag:abc,tag:bcef'], {'m.X,a'}),
    (['tag:bcef,tag:efg', '*,tag:abc'], [], {'m.X,a', 'm.Y,b', 'm.X,c', 'm.X,e', 'm.Y,f'}),
    (['tag:bcef,tag:efg', '*,tag:abc'], ['e'], {'m.X,a', 'm.Y,b', 'm.X,c', 'm.Y,f'}),
    (['tag:bcef,tag:efg', '*,tag:abc'], ['e'], {'m.X,a', 'm.Y,b', 'm.X,c', 'm.Y,f'}),
    (['tag:bcef,tag:efg', '*,tag:abc'], ['e', 'f'], {'m.X,a', 'm.Y,b', 'm.X,c'}),
    (['tag:bcef,tag:efg', '*,tag:abc'], ['tag:abc,tag:bcef'], {'m.X,a', 'm.X,e', 'm.Y,f'}),
    (['tag:bcef,tag:efg', '*,tag:abc'], ['tag:abc,tag:bcef', 'tag:abc,a'], {'m.X,e', 'm.Y,f'}),
]

param_specs: List[Tuple[str, bool, Optional[int], bool, Optional[int], str, str, bool]] = [
    ('a', False, None, False, None, 'fqn', 'a', False),
    ('+a', True, None, False, None, 'fqn', 'a', False),
    ('256+a', True, 256, False, None, 'fqn', 'a', False),
    ('a+', False, None, True, None, 'fqn', 'a', False),
    ('a+256', False, None, True, 256, 'fqn', 'a', False),
    ('+a+', True, None, True, None, 'fqn', 'a', False),
    ('16+a+32', True, 16, True, 32, 'fqn', 'a', False),
    ('@a', False, None, False, None, 'fqn', 'a', True),
    ('a.b', False, None, False, None, 'fqn', 'a.b', False),
    ('+a.b', True, None, False, None, 'fqn', 'a.b', False),
    ('256+a.b', True, 256, False, None, 'fqn', 'a.b', False),
    ('a.b+', False, None, True, None, 'fqn', 'a.b', False),
    ('a.b+256', False, None, True, 256, 'fqn', 'a.b', False),
    ('+a.b+', True, None, True, None, 'fqn', 'a.b', False),
    ('16+a.b+32', True, 16, True, 32, 'fqn', 'a.b', False),
    ('@a.b', False, None, False, None, 'fqn', 'a.b', True),
    ('a.b.*', False, None, False, None, 'fqn', 'a.b.*', False),
    ('+a.b.*', True, None, False, None, 'fqn', 'a.b.*', False),
    ('256+a.b.*', True, 256, False, None, 'fqn', 'a.b.*', False),
    ('a.b.*+', False, None, True, None, 'fqn', 'a.b.*', False),
    ('a.b.*+256', False, None, True, 256, 'fqn', 'a.b.*', False),
    ('+a.b.*+', True, None, True, None, 'fqn', 'a.b.*', False),
    ('16+a.b.*+32', True, 16, True, 32, 'fqn', 'a.b.*', False),
    ('@a.b.*', False, None, False, None, 'fqn', 'a.b.*', True),
    ('tag:a', False, None, False, None, 'tag', 'a', False),
    ('+tag:a', True, None, False, None, 'tag', 'a', False),
    ('256+tag:a', True, 256, False, None, 'tag', 'a', False),
    ('tag:a+', False, None, True, None, 'tag', 'a', False),
    ('tag:a+256', False, None, True, 256, 'tag', 'a', False),
    ('+tag:a+', True, None, True, None, 'tag', 'a', False),
    ('16+tag:a+32', True, 16, True, 32, 'tag', 'a', False),
    ('@tag:a', False, None, False, None, 'tag', 'a', True),
    ('source:a', False, None, False, None, 'source', 'a', False),
    ('source:a+', False, None, True, None, 'source', 'a', False),
    ('source:a+1', False, None, True, 1, 'source', 'a', False),
    ('source:a+32', False, None, True, 32, 'source', 'a', False),
    ('@source:a', False, None, False, None, 'source', 'a', True),
]


@pytest.mark.parametrize('include,exclude,expected', run_specs, ids=id_macro)
def test_run_specs(
    include: List[str],
    exclude: List[str],
    expected: Set[str],
    graph: graph_selector.Graph,
    manifest: Any,
) -> None:
    selector: graph_selector.NodeSelector = graph_selector.NodeSelector(graph, manifest)
    spec: Any = graph_cli.parse_difference(include, exclude)
    selected, _ = selector.select_nodes(spec)
    assert selected == expected


@pytest.mark.parametrize(
    'spec,parents,parents_depth,children,children_depth,filter_type,filter_value,childrens_parents',
    param_specs,
    ids=id_macro,
)
def test_parse_specs(
    spec: str,
    parents: bool,
    parents_depth: Optional[int],
    children: bool,
    children_depth: Optional[int],
    filter_type: str,
    filter_value: str,
    childrens_parents: bool,
) -> None:
    parsed = graph_selector.SelectionCriteria.from_single_spec(spec)
    assert parsed.parents == parents
    assert parsed.parents_depth == parents_depth
    assert parsed.children == children
    assert parsed.children_depth == children_depth
    assert parsed.method == filter_type
    assert parsed.value == filter_value
    assert parsed.childrens_parents == childrens_parents


invalid_specs: List[str] = ['@a+', '@a.b+', '@a.b*+', '@tag:a+', '@source:a+']


@pytest.mark.parametrize('invalid', invalid_specs, ids=lambda k: str(k))
def test_invalid_specs(invalid: str) -> None:
    with pytest.raises(dbt_common.exceptions.DbtRuntimeError):
        graph_selector.SelectionCriteria.from_single_spec(invalid)
