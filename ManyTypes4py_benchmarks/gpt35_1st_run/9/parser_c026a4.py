from typing import Dict, Type, List, Tuple
from parso import tree
from parso.pgen2.generator import ReservedString

class ParserSyntaxError(Exception):
    def __init__(self, message: str, error_leaf: tree.Leaf):
        self.message = message
        self.error_leaf = error_leaf

class InternalParseError(Exception):
    def __init__(self, msg: str, type_: Type, value: str, start_pos: int):
        Exception.__init__(self, '%s: type=%r, value=%r, start_pos=%r' % (msg, type_.name, value, start_pos))
        self.msg = msg
        self.type = type_
        self.value = value
        self.start_pos = start_pos

class Stack(list):
    def _allowed_transition_names_and_token_types(self) -> List[str]:
        ...

class StackNode:
    def __init__(self, dfa):
        self.dfa = dfa
        self.nodes: List[tree.Node] = []

    @property
    def nonterminal(self) -> str:
        return self.dfa.from_rule

    def __repr__(self) -> str:
        return '%s(%s, %s)' % (self.__class__.__name__, self.dfa, self.nodes)

def _token_to_transition(grammar, type_, value) -> ReservedString:
    ...

class BaseParser:
    node_map: Dict[str, Type[tree.Node]] = {}
    default_node: Type[tree.Node] = tree.Node
    leaf_map: Dict[str, Type[tree.Leaf]] = {}
    default_leaf: Type[tree.Leaf] = tree.Leaf

    def __init__(self, pgen_grammar, start_nonterminal='file_input', error_recovery=False):
        self._pgen_grammar = pgen_grammar
        self._start_nonterminal = start_nonterminal
        self._error_recovery = error_recovery

    def parse(self, tokens: List[Tuple[str, str, int, str]]) -> tree.Node:
        ...

    def error_recovery(self, token: Tuple[str, str, int, str]) -> None:
        ...

    def convert_node(self, nonterminal: str, children: List[tree.Node]) -> tree.Node:
        ...

    def convert_leaf(self, type_: str, value: str, prefix: str, start_pos: int) -> tree.Leaf:
        ...

    def _add_token(self, token: Tuple[str, str, int, str]) -> None:
        ...

    def _pop(self) -> None:
        ...
