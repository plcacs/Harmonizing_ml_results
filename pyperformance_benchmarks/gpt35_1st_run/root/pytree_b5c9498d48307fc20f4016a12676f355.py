from typing import Dict, List, Tuple, Union

_type_reprs: Dict[int, str] = {}

class Base(object):
    type: Union[int, None] = None
    parent: Union[Base, None] = None
    children: Tuple = ()
    was_changed: bool = False
    was_checked: bool = False

    def __eq__(self, other: object) -> Union[bool, NotImplemented]:
        ...

    def _eq(self, other: 'Base') -> None:
        ...

    def clone(self) -> None:
        ...

    def post_order(self) -> None:
        ...

    def pre_order(self) -> None:
        ...

    def replace(self, new: Union['Base', List['Base']]) -> None:
        ...

    def get_lineno(self) -> None:
        ...

    def changed(self) -> None:
        ...

    def remove(self) -> None:
        ...

    @property
    def next_sibling(self) -> Union['Base', None]:
        ...

    @property
    def prev_sibling(self) -> Union['Base', None]:
        ...

    def leaves(self) -> None:
        ...

    def depth(self) -> int:
        ...

    def get_suffix(self) -> str:
        ...

    def __str__(self) -> bytes:
        ...

class Node(Base):
    type: int
    children: List['Base']

    def __init__(self, type: int, children: List['Base'], context: Union[None, str] = None, prefix: Union[None, str] = None, fixers_applied: Union[None, List] = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __unicode__(self) -> str:
        ...

    def _eq(self, other: 'Node') -> bool:
        ...

    def clone(self) -> 'Node':
        ...

    def post_order(self) -> None:
        ...

    def pre_order(self) -> None:
        ...

    @property
    def prefix(self) -> str:
        ...

    def set_child(self, i: int, child: 'Base') -> None:
        ...

    def insert_child(self, i: int, child: 'Base') -> None:
        ...

    def append_child(self, child: 'Base') -> None:
        ...

class Leaf(Base):
    type: int
    value: str

    def __init__(self, type: int, value: str, context: Union[None, Tuple] = None, prefix: Union[None, str] = None, fixers_applied: List = []) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __unicode__(self) -> str:
        ...

    def _eq(self, other: 'Leaf') -> bool:
        ...

    def clone(self) -> 'Leaf':
        ...

    def leaves(self) -> None:
        ...

    def post_order(self) -> None:
        ...

    def pre_order(self) -> None:
        ...

    @property
    def prefix(self) -> str:
        ...

class BasePattern(object):
    type: Union[int, None]
    content: Union[str, None]
    name: Union[str, None]

    def optimize(self) -> 'BasePattern':
        ...

    def match(self, node: 'Base', results: Union[None, Dict] = None) -> bool:
        ...

    def match_seq(self, nodes: List['Base'], results: Union[None, Dict] = None) -> bool:
        ...

    def generate_matches(self, nodes: List['Base']) -> None:
        ...

class LeafPattern(BasePattern):

    def __init__(self, type: Union[int, None] = None, content: Union[str, None] = None, name: Union[str, None] = None) -> None:
        ...

    def match(self, node: 'Base', results: Union[None, Dict] = None) -> bool:
        ...

    def _submatch(self, node: 'Base', results: Union[None, Dict] = None) -> bool:
        ...

class NodePattern(BasePattern):
    wildcards: bool

    def __init__(self, type: Union[int, None] = None, content: Union[List[BasePattern], None] = None, name: Union[str, None] = None) -> None:
        ...

    def _submatch(self, node: 'Base', results: Union[None, Dict] = None) -> bool:
        ...

class WildcardPattern(BasePattern):

    def __init__(self, content: Union[List[List[BasePattern]], None] = None, min: int = 0, max: int = 2147483647, name: Union[str, None] = None) -> None:
        ...

    def optimize(self) -> 'WildcardPattern':
        ...

    def match(self, node: 'Base', results: Union[None, Dict] = None) -> bool:
        ...

    def match_seq(self, nodes: List['Base'], results: Union[None, Dict] = None) -> bool:
        ...

    def generate_matches(self, nodes: List['Base']) -> None:
        ...

class NegatedPattern(BasePattern):

    def __init__(self, content: Union[BasePattern, None]) -> None:
        ...

    def match(self, node: 'Base') -> bool:
        ...

    def match_seq(self, nodes: List['Base']) -> bool:
        ...

    def generate_matches(self, nodes: List['Base']) -> None:
        ...

def generate_matches(patterns: List[BasePattern], nodes: List['Base']) -> None:
    ...
