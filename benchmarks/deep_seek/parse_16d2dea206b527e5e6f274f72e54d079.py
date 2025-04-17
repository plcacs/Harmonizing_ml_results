from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from . import token

class ParseError(Exception):
    'Exception to signal the parser is stuck.'

    def __init__(self, msg: str, type: int, value: Optional[str], context: Optional[Any]) -> None:
        Exception.__init__(self, ('%s: type=%r, value=%r, context=%r' % (msg, type, value, context)))
        self.msg: str = msg
        self.type: int = type
        self.value: Optional[str] = value
        self.context: Optional[Any] = context

    def __reduce__(self) -> Tuple[Type['ParseError'], Tuple[str, int, Optional[str], Optional[Any]]]:
        return (type(self), (self.msg, self.type, self.value, self.context))

class Parser(object):
    'Parser engine.'

    def __init__(self, grammar: Any, convert: Optional[Any] = None) -> None:
        self.grammar: Any = grammar
        self.convert: Any = (convert or (lambda grammar, node: node))
        self.stack: List[Tuple[Any, int, Tuple[Any, Optional[str], Optional[Any], List[Any]]]] = []
        self.rootnode: Optional[Any] = None
        self.used_names: Set[str] = set()

    def setup(self, start: Optional[int] = None) -> None:
        if start is None:
            start = self.grammar.start
        newnode: Tuple[int, None, None, List[Any]] = (start, None, None, [])
        stackentry: Tuple[Any, int, Tuple[Any, Optional[str], Optional[Any], List[Any]]] = (self.grammar.dfas[start], 0, newnode)
        self.stack = [stackentry]
        self.rootnode = None
        self.used_names = set()

    def addtoken(self, type: int, value: Optional[str], context: Optional[Any]) -> bool:
        ilabel = self.classify(type, value, context)
        while True:
            (dfa, state, node) = self.stack[-1]
            (states, first) = dfa
            arcs = states[state]
            for (i, newstate) in arcs:
                (t, v) = self.grammar.labels[i]
                if ilabel == i:
                    assert t < 256
                    self.shift(type, value, newstate, context)
                    state = newstate
                    while states[state] == [(0, state)]:
                        self.pop()
                        if not self.stack:
                            return True
                        (dfa, state, node) = self.stack[-1]
                        (states, first) = dfa
                    return False
                elif t >= 256:
                    itsdfa = self.grammar.dfas[t]
                    (itsstates, itsfirst) = itsdfa
                    if ilabel in itsfirst:
                        self.push(t, self.grammar.dfas[t], newstate, context)
                        break
            else:
                if (0, state) in arcs:
                    self.pop()
                    if not self.stack:
                        raise ParseError('too much input', type, value, context)
                else:
                    raise ParseError('bad input', type, value, context)

    def classify(self, type: int, value: Optional[str], context: Optional[Any]) -> int:
        if type == token.NAME:
            if value is not None:
                self.used_names.add(value)
                ilabel = self.grammar.keywords.get(value)
                if ilabel is not None:
                    return ilabel
        ilabel = self.grammar.tokens.get(type)
        if ilabel is None:
            raise ParseError('bad token', type, value, context)
        return ilabel

    def shift(self, type: int, value: Optional[str], newstate: int, context: Optional[Any]) -> None:
        (dfa, state, node) = self.stack[-1]
        newnode: Tuple[int, Optional[str], Optional[Any], None] = (type, value, context, None)
        newnode = self.convert(self.grammar, newnode)
        if newnode is not None:
            node[-1].append(newnode)
        self.stack[-1] = (dfa, newstate, node)

    def push(self, type: int, newdfa: Any, newstate: int, context: Optional[Any]) -> None:
        (dfa, state, node) = self.stack[-1]
        newnode: Tuple[int, None, Optional[Any], List[Any]] = (type, None, context, [])
        self.stack[-1] = (dfa, newstate, node)
        self.stack.append((newdfa, 0, newnode))

    def pop(self) -> None:
        (popdfa, popstate, popnode) = self.stack.pop()
        newnode = self.convert(self.grammar, popnode)
        if newnode is not None:
            if self.stack:
                (dfa, state, node) = self.stack[-1]
                node[-1].append(newnode)
            else:
                self.rootnode = newnode
                self.rootnode.used_names = self.used_names
