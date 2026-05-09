from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union
from parso.utils import split_lines

def search_ancestor(node: NodeOrLeaf, *node_types: str) -> Optional[NodeOrLeaf]: ...

class NodeOrLeaf:
    __slots__ = ('parent',)
    parent: Optional[BaseNode]

    @abstractmethod
    def get_root_node(self) -> NodeOrLeaf: ...

    @abstractmethod
    def get_next_sibling(self) -> Optional[NodeOrLeaf]: ...

    @abstractmethod
    def get_previous_sibling(self) -> Optional[NodeOrLeaf]: ...

    @abstractmethod
    def get_previous_leaf(self) -> Optional[Leaf]: ...

    @abstractmethod
    def get_next_leaf(self) -> Optional[Leaf]: ...

    @abstractproperty
    def start_pos(self) -> Tuple[int, int]: ...

    @abstractproperty
    def end_pos(self) -> Tuple[int, int]: ...

    @abstractmethod
    def get_start_pos_of_prefix(self) -> Tuple[int, int]: ...

    @abstractmethod
    def get_first_leaf(self) -> Leaf: ...

    @abstractmethod
    def get_last_leaf(self) -> Leaf: ...

    @abstractmethod
    def get_code(self, include_prefix: bool = True) -> str: ...

    @abstractmethod
    def search_ancestor(self, *node_types: str) -> Optional[NodeOrLeaf]: ...

    def dump(self, *, indent: Union[int, str, None] = 4) -> str: ...

class Leaf(NodeOrLeaf):
    __slots__ = ('value', 'line', 'column', 'prefix')
    value: str
    line: int
    column: int
    prefix: str

    def __init__(self, value: str, start_pos: Tuple[int, int], prefix: str = '') -> None: ...

    @property
    def start_pos(self) -> Tuple[int, int]: ...
    @start_pos.setter
    def start_pos(self, value: Tuple[int, int]) -> None: ...

    def get_start_pos_of_prefix(self) -> Tuple[int, int]: ...

    def get_first_leaf(self) -> Leaf: ...

    def get_last_leaf(self) -> Leaf: ...

    def get_code(self, include_prefix: bool = True) -> str: ...

    @property
    def end_pos(self) -> Tuple[int, int]: ...

    def __repr__(self) -> str: ...

class TypedLeaf(Leaf):
    __slots__ = ('type',)
    type: str

    def __init__(self, type: str, value: str, start_pos: Tuple[int, int], prefix: str = '') -> None: ...

class BaseNode(NodeOrLeaf):
    __slots__ = ('children',)
    children: List[NodeOrLeaf]

    def __init__(self, children: List[NodeOrLeaf]) -> None: ...

    @property
    def start_pos(self) -> Tuple[int, int]: ...

    def get_start_pos_of_prefix(self) -> Tuple[int, int]: ...

    @property
    def end_pos(self) -> Tuple[int, int]: ...

    def _get_code_for_children(self, children: List[NodeOrLeaf], include_prefix: bool) -> str: ...

    def get_code(self, include_prefix: bool = True) -> str: ...

    def get_leaf_for_position(self, position: Tuple[int, int], include_prefixes: bool = False) -> Optional[Leaf]: ...

    def get_first_leaf(self) -> Leaf: ...

    def get_last_leaf(self) -> Leaf: ...

    def __repr__(self) -> str: ...

class Node(BaseNode):
    __slots__ = ('type',)
    type: str

    def __init__(self, type: str, children: List[NodeOrLeaf]) -> None: ...

    def __repr__(self) -> str: ...

class ErrorNode(BaseNode):
    __slots__ = ()
    type: str = 'error_node'

class ErrorLeaf(Leaf):
    __slots__ = ('token_type',)
    token_type: str
    type: str = 'error_leaf'

    def __init__(self, token_type: str, value: str, start_pos: Tuple[int, int], prefix: str = '') -> None: ...

    def __repr__(self) -> str: ...