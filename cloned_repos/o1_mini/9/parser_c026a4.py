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
from typing import Dict, Type, List, Union, Any, Optional, Tuple
from parso import tree
from parso.pgen2.generator import ReservedString
from parso.pgen2.grammar import Grammar, Symbol, DFA
from parso.pgen2.token import Token

class ParserSyntaxError(Exception):
    """
    Contains error information about the parser tree.

    May be raised as an exception.
    """

    def __init__(self, message: str, error_leaf: tree.ErrorLeaf) -> None:
        self.message: str = message
        self.error_leaf: tree.ErrorLeaf = error_leaf
        super().__init__(message)

class InternalParseError(Exception):
    """
    Exception to signal the parser is stuck and error recovery didn't help.
    Basically this shouldn't happen. It's a sign that something is really
    wrong.
    """

    def __init__(self, msg: str, type_: Token, value: str, start_pos: Tuple[int, int]) -> None:
        Exception.__init__(self, f'{msg}: type={type_.name!r}, value={value!r}, start_pos={start_pos!r}')
        self.msg: str = msg
        self.type: Token = type_
        self.value: str = value
        self.start_pos: Tuple[int, int] = start_pos

class Stack(list):
    def _allowed_transition_names_and_token_types(self) -> List[Union[str, Any]]:
        def iterate():
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
    dfa: DFA
    nodes: List[Union[tree.Node, tree.Leaf]]

    def __init__(self, dfa: DFA) -> None:
        self.dfa: DFA = dfa
        self.nodes: List[Union[tree.Node, tree.Leaf]] = []

    @property
    def nonterminal(self) -> Symbol:
        return self.dfa.from_rule

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.dfa}, {self.nodes})'

def _token_to_transition(grammar: Grammar, type_: Token, value: str) -> Union[ReservedString, Token]:
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
    node_map: Dict[str, Type[tree.Node]] = {}
    default_node: Type[tree.Node] = tree.Node
    leaf_map: Dict[int, Type[tree.Leaf]] = {}
    default_leaf: Type[tree.Leaf] = tree.Leaf

    def __init__(self, pgen_grammar: Grammar, start_nonterminal: str = 'file_input', error_recovery: bool = False) -> None:
        self._pgen_grammar: Grammar = pgen_grammar
        self._start_nonterminal: str = start_nonterminal
        self._error_recovery: bool = error_recovery
        self.stack: Stack

    def parse(self, tokens: List[Token]) -> tree.Node:
        first_dfa: DFA = self._pgen_grammar.nonterminal_to_dfas[self._start_nonterminal][0]
        self.stack = Stack([StackNode(first_dfa)])
        for token in tokens:
            self._add_token(token)
        while True:
            tos: StackNode = self.stack[-1]
            if not tos.dfa.is_final:
                raise InternalParseError('incomplete input', token.type, token.value, token.start_pos)
            if len(self.stack) > 1:
                self._pop()
            else:
                return self.convert_node(tos.nonterminal, tos.nodes)

    def error_recovery(self, token: Token) -> None:
        if self._error_recovery:
            raise NotImplementedError('Error Recovery is not implemented')
        else:
            type_, value, start_pos, prefix = token.type, token.value, token.start_pos, token.prefix
            error_leaf = tree.ErrorLeaf(type_, value, start_pos, prefix)
            raise ParserSyntaxError('SyntaxError: invalid syntax', error_leaf)

    def convert_node(self, nonterminal: Symbol, children: List[Union[tree.Node, tree.Leaf]]) -> tree.Node:
        try:
            node = self.node_map[nonterminal](children)
        except KeyError:
            node = self.default_node(nonterminal, children)
        return node

    def convert_leaf(self, type_: Token, value: str, prefix: str, start_pos: Tuple[int, int]) -> tree.Leaf:
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
        grammar: Grammar = self._pgen_grammar
        stack: Stack = self.stack
        type_: Token = token.type
        value: str = token.value
        start_pos: Tuple[int, int] = token.start_pos
        prefix: str = token.prefix
        transition: Union[ReservedString, Token] = _token_to_transition(grammar, type_, value)
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
            new_node: Union[tree.Node, tree.Leaf] = tos.nodes[0]
        else:
            new_node = self.convert_node(tos.dfa.from_rule, tos.nodes)
        self.stack[-1].nodes.append(new_node)
