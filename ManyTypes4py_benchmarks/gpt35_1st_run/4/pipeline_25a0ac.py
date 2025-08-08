from __future__ import annotations
from typing import TYPE_CHECKING, Any, Set, List, Dict
from collections import Counter, defaultdict
from graphlib import CycleError, TopologicalSorter
from itertools import chain
import kedro
from kedro.pipeline.node import Node, _to_list
from .transcoding import _strip_transcoding

if TYPE_CHECKING:
    from collections.abc import Iterable

class OutputNotUniqueError(Exception):
    pass

class ConfirmNotUniqueError(Exception):
    pass

class Pipeline:
    def __init__(self, nodes: Iterable[Node], *, tags: Set[str] = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __add__(self, other: Pipeline) -> Pipeline:
        ...

    def __radd__(self, other: int) -> Pipeline:
        ...

    def __sub__(self, other: Pipeline) -> Pipeline:
        ...

    def __and__(self, other: Pipeline) -> Pipeline:
        ...

    def __or__(self, other: Pipeline) -> Pipeline:
        ...

    def all_inputs(self) -> Set[str]:
        ...

    def all_outputs(self) -> Set[str]:
        ...

    def _remove_intermediates(self, datasets: Set[str]) -> Set[str]:
        ...

    def inputs(self) -> Set[str]:
        ...

    def outputs(self) -> Set[str]:
        ...

    def datasets(self) -> Set[str]:
        ...

    def _transcode_compatible_names(self) -> Set[str]:
        ...

    def describe(self, names_only: bool = True) -> str:
        ...

    @property
    def node_dependencies(self) -> Dict[Node, Set[Node]]:
        ...

    @property
    def nodes(self) -> List[Node]:
        ...

    @property
    def grouped_nodes(self) -> List[List[Node]]:
        ...

    @property
    def grouped_nodes_by_namespace(self) -> Dict[str, Dict[str, Any]]:
        ...

    def only_nodes(self, *node_names: str) -> Pipeline:
        ...

    def only_nodes_with_namespace(self, node_namespace: str) -> Pipeline:
        ...

    def _get_nodes_with_inputs_transcode_compatible(self, datasets: Set[str]) -> Set[Node]:
        ...

    def _get_nodes_with_outputs_transcode_compatible(self, datasets: Set[str]) -> Set[Node]:
        ...

    def only_nodes_with_inputs(self, *inputs: str) -> Pipeline:
        ...

    def from_inputs(self, *inputs: str) -> Pipeline:
        ...

    def only_nodes_with_outputs(self, *outputs: str) -> Pipeline:
        ...

    def to_outputs(self, *outputs: str) -> Pipeline:
        ...

    def from_nodes(self, *node_names: str) -> Pipeline:
        ...

    def to_nodes(self, *node_names: str) -> Pipeline:
        ...

    def only_nodes_with_tags(self, *tags: str) -> Pipeline:
        ...

    def filter(self, tags: Set[str] = None, from_nodes: List[str] = None, to_nodes: List[str] = None, node_names: List[str] = None, from_inputs: List[str] = None, to_outputs: List[str] = None, node_namespace: str = None) -> Pipeline:
        ...

    def tag(self, tags: Set[str]) -> Pipeline:
        ...

    def to_json(self) -> str:
        ...
