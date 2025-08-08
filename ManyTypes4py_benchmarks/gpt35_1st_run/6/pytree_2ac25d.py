from typing import Dict, List, Tuple, Union

_type_reprs: Dict[int, str] = {}

class Base:
    type: Union[int, None] = None
    parent: Union[Base, None] = None
    children: Tuple = ()
    was_changed: bool = False
    was_checked: bool = False

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
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

    def replace(self, new: Union['Base', List['Base']]) -> None:
        ...

    def get_lineno(self) -> Union[int, None]:
        ...

    def changed(self) -> None:
        ...

    def remove(self) -> int:
        ...

    @property
    def next_sibling(self) -> Union['Base', None]:
        ...

    @property
    def prev_sibling(self) -> Union['Base', None]:
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
    def __init__(self, type: int, children: List['Base'], context: Union[None, str] = None, prefix: Union[None, str] = None, fixers_applied: Union[None, List[str]] = None) -> None:
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
    _prefix: str = ''
    lineno: int = 0
    column: int = 0

    def __init__(self, type: int, value: str, context: Union[None, Tuple[str, Tuple[int, int]]] = None, prefix: Union[None, str] = None, fixers_applied: List[str] = []) -> None:
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
    type: Union[int, None] = None
    content: Union[str, None] = None
    name: Union[str, None] = None

    def optimize(self) -> 'BasePattern':
        ...

    def match(self, node: 'Base', results: Union[None, Dict[str, 'Base']]) -> bool:
        ...

    def match_seq(self, nodes: List['Base'], results: Union[None, Dict[str, 'Base']]) -> bool:
        ...

    def generate_matches(self, nodes: List['Base']) -> Tuple[int, Dict[str, 'Base']]:
        ...

class LeafPattern(BasePattern):
    def __init__(self, type: Union[None, int] = None, content: Union[None, str] = None, name: Union[None, str] = None) -> None:
        ...

    def match(self, node: 'Leaf', results: Union[None, Dict[str, 'Leaf']]) -> bool:
        ...

    def _submatch(self, node: 'Leaf', results: Union[None, Dict[str, 'Leaf']]) -> bool:
        ...

class NodePattern(BasePattern):
    wildcards: bool = False

    def __init__(self, type: Union[None, int] = None, content: Union[None, List[BasePattern]] = None, name: Union[None, str] = None) -> None:
        ...

    def _submatch(self, node: 'Node', results: Union[None, Dict[str, 'Node']]) -> bool:
        ...

class WildcardPattern(BasePattern):
    def __init__(self, content: Union[None, List[List[BasePattern]]] = None, min: int = 0, max: int = 2147483647, name: Union[None, str] = None) -> None:
        ...

    def optimize(self) -> 'WildcardPattern':
        ...

    def match(self, node: 'Base', results: Union[None, Dict[str, 'Base']]) -> bool:
        ...

    def match_seq(self, nodes: List['Base'], results: Union[None, Dict[str, 'Base']]) -> bool:
        ...

    def generate_matches(self, nodes: List['Base']) -> Tuple[int, Dict[str, 'Base']]:
        ...

class NegatedPattern(BasePattern):
    def __init__(self, content: Union[None, BasePattern]) -> None:
        ...

    def match(self, node: 'Base') -> bool:
        ...

    def match_seq(self, nodes: List['Base']) -> bool:
        ...

    def generate_matches(self, nodes: List['Base']) -> Tuple[int, Dict[str, 'Base']]:
        ...

def generate_matches(patterns: List[BasePattern], nodes: List['Base']) -> Tuple[int, Dict[str, 'Base']]:
    ...
