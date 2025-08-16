from typing import Any, Optional, TypeVar, Union, Tuple, List, Dict

_type_reprs: Dict[int, str] = {}

def type_repr(type_num: int) -> str:
    global _type_reprs
    if not _type_reprs:
        ...
    return _type_reprs.setdefault(type_num, type_num)

_P = TypeVar('_P', bound='Base')
NL = Union['Node', 'Leaf']
Context = Tuple[str, Tuple[int, int]]
RawNode = Tuple[int, Optional[str], Optional[Context], Optional[List[NL]]]

class Base:
    parent: Optional['Base'] = None
    was_changed: bool = False
    was_checked: bool = False

    def __eq__(self, other: Any) -> Any:
        ...

    @property
    def prefix(self) -> str:
        ...

    def _eq(self, other: Any) -> Any:
        ...

    def __deepcopy__(self, memo: Any) -> Any:
        ...

    def clone(self) -> Any:
        ...

    def post_order(self) -> Iterator:
        ...

    def pre_order(self) -> Iterator:
        ...

    def replace(self, new: Any) -> None:
        ...

    def get_lineno(self) -> Optional[int]:
        ...

    def changed(self) -> None:
        ...

    def remove(self) -> Optional[int]:
        ...

    @property
    def next_sibling(self) -> Optional['Base']:
        ...

    @property
    def prev_sibling(self) -> Optional['Base']:
        ...

    def leaves(self) -> Iterator:
        ...

    def depth(self) -> int:
        ...

    def get_suffix(self) -> str:
        ...

class Node(Base):
    type: int
    children: List[NL]
    prefix: str
    fixers_applied: Optional[List[Any]]

    def __init__(self, type: int, children: List[NL], context: Optional[Context] = None, prefix: Optional[str] = None, fixers_applied: Optional[List[Any]] = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def _eq(self, other: Any) -> Any:
        ...

    def clone(self) -> Any:
        ...

    def post_order(self) -> Iterator:
        ...

    def pre_order(self) -> Iterator:
        ...

    @property
    def prefix(self) -> str:
        ...

    def set_child(self, i: int, child: Any) -> None:
        ...

    def insert_child(self, i: int, child: Any) -> None:
        ...

    def append_child(self, child: Any) -> None:
        ...

    def invalidate_sibling_maps(self) -> None:
        ...

    def update_sibling_maps(self) -> None:
        ...

class Leaf(Base):
    type: int
    value: str
    _prefix: str
    lineno: int
    column: int
    fmt_pass_converted_first_leaf: Optional[Any]

    def __init__(self, type: int, value: str, context: Optional[Context] = None, prefix: Optional[str] = None, fixers_applied: List[Any] = [], opening_bracket: Optional[Any] = None, fmt_pass_converted_first_leaf: Optional[Any] = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def _eq(self, other: Any) -> Any:
        ...

    def clone(self) -> Any:
        ...

    def leaves(self) -> Iterator:
        ...

    def post_order(self) -> Iterator:
        ...

    def pre_order(self) -> Iterator:
        ...

    @property
    def prefix(self) -> str:
        ...

class BasePattern:
    type: Optional[int]
    content: Optional[Any]
    name: Optional[str]

    def __repr__(self) -> str:
        ...

    def _submatch(self, node: Any, results: Optional[Dict[str, NL]] = None) -> Any:
        ...

    def optimize(self) -> Any:
        ...

    def match(self, node: Any, results: Optional[Dict[str, NL]] = None) -> bool:
        ...

    def match_seq(self, nodes: List[Any], results: Optional[Dict[str, NL]] = None) -> bool:
        ...

    def generate_matches(self, nodes: List[Any]) -> Iterator:
        ...

class LeafPattern(BasePattern):

    def __init__(self, type: Optional[int] = None, content: Optional[str] = None, name: Optional[str] = None) -> None:
        ...

    def match(self, node: Any, results: Optional[Dict[str, NL]] = None) -> bool:
        ...

    def _submatch(self, node: Any, results: Optional[Dict[str, NL]] = None) -> bool:
        ...

class NodePattern(BasePattern):
    wildcards: bool

    def __init__(self, type: Optional[int] = None, content: Optional[List[BasePattern]] = None, name: Optional[str] = None) -> None:
        ...

    def _submatch(self, node: Any, results: Optional[Dict[str, NL]] = None) -> bool:
        ...

class WildcardPattern(BasePattern):

    def __init__(self, content: Optional[List[List[BasePattern]]] = None, min: int = 0, max: int = HUGE, name: Optional[str] = None) -> None:
        ...

    def optimize(self) -> Any:
        ...

    def match(self, node: Any, results: Optional[Dict[str, NL]] = None) -> bool:
        ...

    def match_seq(self, nodes: List[Any], results: Optional[Dict[str, NL]] = None) -> bool:
        ...

    def generate_matches(self, nodes: List[Any]) -> Iterator:
        ...

class NegatedPattern(BasePattern):

    def __init__(self, content: Optional[BasePattern] = None) -> None:
        ...

    def match(self, node: Any, results: Optional[Dict[str, NL]] = None) -> bool:
        ...

    def match_seq(self, nodes: List[Any], results: Optional[Dict[str, NL]] = None) -> bool:
        ...

    def generate_matches(self, nodes: List[Any]) -> Iterator:
        ...
