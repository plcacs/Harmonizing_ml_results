from typing import Dict, Type, List, Union, Any
from parso import tree
from parso.pgen2.generator import ReservedString

class ParserSyntaxError(Exception):
    def __init__(self, message: str, error_leaf: tree.ErrorLeaf) -> None:
        self.message = message
        self.error_leaf = error_leaf

class InternalParseError(Exception):
    def __init__(self, msg: str, type_: Any, value: str, start_pos: Any) -> None:
        Exception.__init__(self, '%s: type=%r, value=%r, start_pos=%r' % (msg, type_.name, value, start_pos))
        self.msg = msg
        self.type = type_
        self.value = value
        self.start_pos = start_pos

class Stack(List['StackNode']):
    def _allowed_transition_names_and_token_types(self) -> List[Union[str, ReservedString]]:
        def iterate() -> Any:
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
        self.dfa = dfa
        self.nodes: List[Union[tree.Node, tree.Leaf]] = []

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
    node_map: Dict[str, Type[tree.Node]] = {}
    default_node: Type[tree.Node] = tree.Node
    leaf_map: Dict[str, Type[tree.Leaf]] = {}
    default_leaf: Type[tree.Leaf] = tree.Leaf

    def __init__(self, pgen_grammar: Any, start_nonterminal: str = 'file_input', error_recovery: bool = False) -> None:
        self._pgen_grammar = pgen_grammar
        self._start_nonterminal = start_nonterminal
        self._error_recovery = error_recovery

    def parse(self, tokens: List[Any]) -> tree.Node:
        first_dfa = self._pgen_grammar.nonterminal_to_dfas[self._start_nonterminal][0]
        self.stack = Stack([StackNode(first_dfa)])
        for token in tokens:
            self._add_token(token)
        while True:
            tos = self.stack[-1]
            if not tos.dfa.is_final:
                raise InternalParseError('incomplete input', token.type, token.string, token.start_pos)
            if len(self.stack) > 1:
                self._pop()
            else:
                return self.convert_node(tos.nonterminal, tos.nodes)

    def error_recovery(self, token: Any) -> None:
        if self._error_recovery:
            raise NotImplementedError('Error Recovery is not implemented')
        else:
            type_, value, start_pos, prefix = token
            error_leaf = tree.ErrorLeaf(type_, value, start_pos, prefix)
            raise ParserSyntaxError('SyntaxError: invalid syntax', error_leaf)

    def convert_node(self, nonterminal: Any, children: List[Union[tree.Node, tree.Leaf]]) -> tree.Node:
        try:
            node = self.node_map[nonterminal](children)
        except KeyError:
            node = self.default_node(nonterminal, children)
        return node

    def convert_leaf(self, type_: Any, value: str, prefix: str, start_pos: Any) -> tree.Leaf:
        try:
            return self.leaf_map[type_](value, start_pos, prefix)
        except KeyError:
            return self.default_leaf(value, start_pos, prefix)

    def _add_token(self, token: Any) -> None:
        grammar = self._pgen_grammar
        stack = self.stack
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
        leaf = self.convert_leaf(type_, value, prefix, start_pos)
        stack[-1].nodes.append(leaf)

    def _pop(self) -> None:
        tos = self.stack.pop()
        if len(tos.nodes) == 1:
            new_node = tos.nodes[0]
        else:
            new_node = self.convert_node(tos.dfa.from_rule, tos.nodes)
        self.stack[-1].nodes.append(new_node)
