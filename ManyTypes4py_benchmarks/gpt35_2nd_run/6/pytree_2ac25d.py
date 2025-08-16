from typing import Dict, List, Tuple

_type_reprs: Dict[int, str] = {}

class Base:
    type: int = None
    parent: 'Base' = None
    children: Tuple['Base'] = ()
    was_changed: bool = False
    was_checked: bool = False

    def __eq__(self, other: 'Base') -> bool:
        ...

    def __ne__(self, other: 'Base') -> bool:
        ...

    def _eq(self, other: 'Base') -> bool:
        ...

    def clone(self) -> 'Base':
        ...

    def post_order(self) -> 'Base':
        ...

    def pre_order(self) -> 'Base':
        ...

    def set_prefix(self, prefix: str) -> None:
        ...

    def get_prefix(self) -> str:
        ...

    def replace(self, new: 'Base') -> None:
        ...

    def get_lineno(self) -> int:
        ...

    def changed(self) -> None:
        ...

    def remove(self) -> int:
        ...

    @property
    def next_sibling(self) -> 'Base':
        ...

    @property
    def prev_sibling(self) -> 'Base':
        ...

    def leaves(self) -> 'Base':
        ...

    def depth(self) -> int:
        ...

    def get_suffix(self) -> str:
        ...

    def __str__(self) -> bytes:
        ...

class Node(Base):
    def __init__(self, type: int, children: List['Base'], context=None, prefix=None, fixers_applied=None):
        ...

    def __repr__(self) -> str:
        ...

    def __unicode__(self) -> str:
        ...

    def _eq(self, other: 'Node') -> bool:
        ...

    def clone(self) -> 'Node':
        ...

    def post_order(self) -> 'Node':
        ...

    def pre_order(self) -> 'Node':
        ...

    def _prefix_getter(self) -> str:
        ...

    def _prefix_setter(self, prefix: str) -> None:
        ...

    def set_child(self, i: int, child: 'Base') -> None:
        ...

    def insert_child(self, i: int, child: 'Base') -> None:
        ...

    def append_child(self, child: 'Base') -> None:
        ...

class Leaf(Base):
    def __init__(self, type: int, value: str, context=None, prefix=None, fixers_applied=[]) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __unicode__(self) -> str:
        ...

    def _eq(self, other: 'Leaf') -> bool:
        ...

    def clone(self) -> 'Leaf':
        ...

    def leaves(self) -> 'Leaf':
        ...

    def post_order(self) -> 'Leaf':
        ...

    def pre_order(self) -> 'Leaf':
        ...

    def _prefix_getter(self) -> str:
        ...

    def _prefix_setter(self, prefix: str) -> None:
        ...

class BasePattern:
    type: int = None
    content: str = None
    name: str = None

    def optimize(self) -> 'BasePattern':
        ...

    def match(self, node: 'Base', results: Dict[str, 'Base'] = None) -> bool:
        ...

    def match_seq(self, nodes: List['Base'], results: Dict[str, 'Base'] = None) -> bool:
        ...

    def generate_matches(self, nodes: List['Base']) -> Tuple[int, Dict[str, 'Base']]:
        ...

class LeafPattern(BasePattern):
    def __init__(self, type: int = None, content: str = None, name: str = None) -> None:
        ...

    def match(self, node: 'Base', results: Dict[str, 'Base'] = None) -> bool:
        ...

    def _submatch(self, node: 'Base', results: Dict[str, 'Base'] = None) -> bool:
        ...

class NodePattern(BasePattern):
    wildcards: bool = False

    def __init__(self, type: int = None, content: List[BasePattern] = None, name: str = None) -> None:
        ...

    def _submatch(self, node: 'Base', results: Dict[str, 'Base'] = None) -> bool:
        ...

class WildcardPattern(BasePattern):
    def __init__(self, content: List[List[BasePattern]] = None, min: int = 0, max: int = HUGE, name: str = None) -> None:
        ...

    def optimize(self) -> 'WildcardPattern':
        ...

    def match(self, node: 'Base', results: Dict[str, 'Base'] = None) -> bool:
        ...

    def match_seq(self, nodes: List['Base'], results: Dict[str, 'Base'] = None) -> bool:
        ...

    def generate_matches(self, nodes: List['Base']) -> Tuple[int, Dict[str, 'Base']]:
        ...

class NegatedPattern(BasePattern):
    def __init__(self, content: BasePattern = None) -> None:
        ...

    def match(self, node: 'Base') -> bool:
        ...

    def match_seq(self, nodes: List['Base']) -> bool:
        ...

    def generate_matches(self, nodes: List['Base']) -> Tuple[int, Dict[str, 'Base']]:
        ...

def generate_matches(patterns: List[BasePattern], nodes: List['Base']) -> Tuple[int, Dict[str, 'Base']]:
    ...
