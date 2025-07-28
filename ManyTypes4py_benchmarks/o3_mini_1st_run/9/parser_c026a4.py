#!/usr/bin/env python3
"""
The ``Parser`` tries to convert the available Python code in an easy to read
format, something like an abstract syntax tree. The classes who represent this
tree, are sitting in the :mod:`parso.tree` module.

The Python module ``tokenize`` is a very important part in the ``Parser``,
because it splits the code into different words (tokens).  Sometimes it looks a
bit messy. Sorry for that! You might ask now: "Why didn't you use the ``ast``
module for this? Well, ``ast`` does a very good job understanding proper Python
code, but fails to work as soon as there's a single line of broken code.

There's one important optimization that needs to be known: Statements are not
being parsed completely. ``Statement`` is just a representation of the tokens
within the statement. This lowers memory usage and cpu time and reduces the
complexity of the ``Parser`` (there's another parser sitting inside
``Statement``, which produces ``Array`` and ``Call``).
"""
from typing import Any, Callable, Dict, List, Tuple, Iterable, Union
from parso import tree
from parso.pgen2.generator import ReservedString

Token = Tuple[Any, str, Tuple[int, int], str]


class ParserSyntaxError(Exception):
    """
    Contains error information about the parser tree.

    May be raised as an exception.
    """

    def __init__(self, message: str, error_leaf: tree.Leaf) -> None:
        self.message: str = message
        self.error_leaf: tree.Leaf = error_leaf


class InternalParseError(Exception):
    """
    Exception to signal the parser is stuck and error recovery didn't help.
    Basically this shouldn't happen. It's a sign that something is really
    wrong.
    """

    def __init__(self, msg: str, type_: Any, value: str, start_pos: Tuple[int, int]) -> None:
        super().__init__('%s: type=%r, value=%r, start_pos=%r' % (msg, type_.name, value, start_pos))
        self.msg: str = msg
        self.type: Any = type_
        self.value: str = value
        self.start_pos: Tuple[int, int] = start_pos


class Stack(List[Any]):
    def _allowed_transition_names_and_token_types(self) -> List[Any]:
        def iterate() -> Iterable[Any]:
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
    if getattr(type_.value, 'contains_syntax', False):
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
    node_map: Dict[Any, Callable[[List[Any]], Any]] = {}
    default_node: Any = tree.Node
    leaf_map: Dict[Any, Callable[[str, Tuple[int, int], str], Any]] = {}
    default_leaf: Any = tree.Leaf

    def __init__(self, pgen_grammar: Any, start_nonterminal: str = 'file_input', error_recovery: bool = False) -> None:
        self._pgen_grammar: Any = pgen_grammar
        self._start_nonterminal: str = start_nonterminal
        self._error_recovery: bool = error_recovery
        self.stack: Stack = Stack()

    def parse(self, tokens: Iterable[Token]) -> Any:
        first_dfa = self._pgen_grammar.nonterminal_to_dfas[self._start_nonterminal][0]
        self.stack = Stack([StackNode(first_dfa)])
        for token in tokens:
            self._add_token(token)
        while True:
            tos: StackNode = self.stack[-1]
            if not tos.dfa.is_final:
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
            error_leaf: tree.ErrorLeaf = tree.ErrorLeaf(type_, value, start_pos, prefix)
            raise ParserSyntaxError('SyntaxError: invalid syntax', error_leaf)

    def convert_node(self, nonterminal: Any, children: List[Any]) -> Any:
        try:
            node = self.node_map[nonterminal](children)
        except KeyError:
            node = self.default_node(nonterminal, children)
        return node

    def convert_leaf(self, type_: Any, value: str, prefix: str, start_pos: Tuple[int, int]) -> Any:
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
        transition: Any = _token_to_transition(grammar, type_, value)
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
        leaf: Any = self.convert_leaf(type_, value, prefix, start_pos)
        stack[-1].nodes.append(leaf)

    def _pop(self) -> None:
        tos: StackNode = self.stack.pop()
        if len(tos.nodes) == 1:
            new_node: Any = tos.nodes[0]
        else:
            new_node = self.convert_node(tos.dfa.from_rule, tos.nodes)
        self.stack[-1].nodes.append(new_node)