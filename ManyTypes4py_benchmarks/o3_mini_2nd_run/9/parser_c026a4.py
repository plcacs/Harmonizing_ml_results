#!/usr/bin/env python3
from typing import Any, Callable, Dict, List, Tuple, Type, Union, Iterable
from parso import tree
from parso.pgen2.generator import ReservedString

Token = Tuple[Any, str, Tuple[int, int], str]  # (type, value, start_pos, prefix)

class ParserSyntaxError(Exception):
    """
    Contains error information about the parser tree.

    May be raised as an exception.
    """
    def __init__(self, message: str, error_leaf: tree.ErrorLeaf) -> None:
        self.message: str = message
        self.error_leaf: tree.ErrorLeaf = error_leaf

class InternalParseError(Exception):
    """
    Exception to signal the parser is stuck and error recovery didn't help.
    Basically this shouldn't happen. It's a sign that something is really
    wrong.
    """
    def __init__(self, msg: str, type_: Any, value: Any, start_pos: Tuple[int, int]) -> None:
        Exception.__init__(self, '%s: type=%r, value=%r, start_pos=%r' % (msg, type_.name, value, start_pos))
        self.msg: str = msg
        self.type: Any = type_
        self.value: Any = value
        self.start_pos: Tuple[int, int] = start_pos

class Stack(List["StackNode"]):
    def _allowed_transition_names_and_token_types(self) -> List[Union[str, Any]]:
        def iterate() -> Iterable[Union[str, Any]]:
            for stack_node in reversed(self):
                for transition in stack_node.dfa.transitions:
                    if isinstance(transition, ReservedString):
                        yield transition.value
                    else:
                        yield transition
                if not stack_node.dfa.is_final:
                    break
        return list(iterate())

class StackNode:
    def __init__(self, dfa: Any) -> None:
        self.dfa: Any = dfa
        self.nodes: List[Any] = []

    @property
    def nonterminal(self) -> Any:
        return self.dfa.from_rule

    def __repr__(self) -> str:
        return '%s(%s, %s)' % (self.__class__.__name__, self.dfa, self.nodes)

def _token_to_transition(grammar: Any, type_: Any, value: str) -> Any:
    if type_.value.contains_syntax:
        try:
            return grammar.reserved_syntax_strings[value]
        except KeyError:
            pass
    return type_

class BaseParser:
    """Parser engine.

    A Parser instance contains state pertaining to the current token
    sequence, and should not be used concurrently by different threads
    to parse separate token sequences.

    See python/tokenize.py for how to get input tokens by a string.

    When a syntax error occurs, error_recovery() is called.
    """
    node_map: Dict[Any, Callable[[List[Any]], tree.Node]] = {}
    default_node: Type[tree.Node] = tree.Node
    leaf_map: Dict[Any, Callable[[str, Tuple[int, int], str], tree.Leaf]] = {}
    default_leaf: Type[tree.Leaf] = tree.Leaf

    def __init__(self, pgen_grammar: Any, start_nonterminal: str = 'file_input', error_recovery: bool = False) -> None:
        self._pgen_grammar: Any = pgen_grammar
        self._start_nonterminal: str = start_nonterminal
        self._error_recovery: bool = error_recovery
        self.stack: Stack = Stack()

    def parse(self, tokens: Iterable[Token]) -> tree.Node:
        first_dfa = self._pgen_grammar.nonterminal_to_dfas[self._start_nonterminal][0]
        self.stack = Stack([StackNode(first_dfa)])
        for token in tokens:
            self._add_token(token)
        while True:
            tos: StackNode = self.stack[-1]
            if not tos.dfa.is_final:
                # Using the last token from the loop as context
                raise InternalParseError('incomplete input', token[0], token[1], token[2])
            if len(self.stack) > 1:
                self._pop()
            else:
                return self.convert_node(tos.nonterminal, tos.nodes)

    def error_recovery(self, token: Token) -> None:
        if self._error_recovery:
            raise NotImplementedError('Error Recovery is not implemented')
        else:
            type_, value, start_pos, prefix = token
            error_leaf = tree.ErrorLeaf(type_, value, start_pos, prefix)
            raise ParserSyntaxError('SyntaxError: invalid syntax', error_leaf)

    def convert_node(self, nonterminal: Any, children: List[Any]) -> tree.Node:
        try:
            node = self.node_map[nonterminal](children)
        except KeyError:
            node = self.default_node(nonterminal, children)
        return node

    def convert_leaf(self, type_: Any, value: str, prefix: str, start_pos: Tuple[int, int]) -> tree.Leaf:
        try:
            return self.leaf_map[type_](value, start_pos, prefix)
        except KeyError:
            return self.default_leaf(value, start_pos, prefix)

    def _add_token(self, token: Token) -> None:
        """
        This is the only core function for parsing. Here happens basically
        everything. Everything is well prepared by the parser generator and we
        only apply the necessary steps here.
        """
        grammar: Any = self._pgen_grammar
        stack: Stack = self.stack
        type_, value, start_pos, prefix = token
        transition = _token_to_transition(grammar, type_, value)
        while True:
            try:
                plan = stack[-1].dfa.transitions[transition]
                break
            except KeyError:
                if stack[-1].dfa.is_final:
                    self._pop()
                else:
                    self.error_recovery(token)
                    return
            except IndexError:
                raise InternalParseError('too much input', type_, value, start_pos)
        stack[-1].dfa = plan.next_dfa
        for push in plan.dfa_pushes:
            stack.append(StackNode(push))
        leaf: tree.Leaf = self.convert_leaf(type_, value, prefix, start_pos)
        stack[-1].nodes.append(leaf)

    def _pop(self) -> None:
        tos: StackNode = self.stack.pop()
        if len(tos.nodes) == 1:
            new_node = tos.nodes[0]
        else:
            new_node = self.convert_node(tos.dfa.from_rule, tos.nodes)
        self.stack[-1].nodes.append(new_node)