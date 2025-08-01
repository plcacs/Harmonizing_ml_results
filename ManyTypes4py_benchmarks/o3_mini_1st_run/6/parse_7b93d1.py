from __future__ import annotations
from typing import Any, Callable, Optional, Tuple, List, Set
from . import token

class ParseError(Exception):
    def __init__(self, msg: str, type: Any, value: Any, context: Any) -> None:
        super().__init__('%s: type=%r, value=%r, context=%r' % (msg, type, value, context))
        self.msg: str = msg
        self.type: Any = type
        self.value: Any = value
        self.context: Any = context

class Parser:
    def __init__(self, grammar: Any, convert: Optional[Callable[[Any, Any], Any]] = None) -> None:
        self.grammar: Any = grammar
        self.convert: Callable[[Any, Any], Any] = convert or (lambda grammar, node: node)
        self.stack: List[Tuple[Any, int, Tuple[Any, Any, Any, List[Any]]]] = []
        self.rootnode: Optional[Any] = None
        self.used_names: Set[str] = set()

    def setup(self, start: Optional[Any] = None) -> None:
        if start is None:
            start = self.grammar.start
        newnode: Tuple[Any, Any, Any, List[Any]] = (start, None, None, [])
        stackentry: Tuple[Any, int, Tuple[Any, Any, Any, List[Any]]] = (self.grammar.dfas[start], 0, newnode)
        self.stack = [stackentry]
        self.rootnode = None
        self.used_names = set()

    def addtoken(self, type: int, value: Any, context: Any) -> bool:
        ilabel: int = self.classify(type, value, context)
        while True:
            dfa, state, node = self.stack[-1]
            states, first = dfa
            arcs: List[Tuple[int, int]] = states[state]
            for i, newstate in arcs:
                t, v = self.grammar.labels[i]
                if ilabel == i:
                    assert t < 256
                    self.shift(type, value, newstate, context)
                    state = newstate
                    while states[state] == [(0, state)]:
                        self.pop()
                        if not self.stack:
                            return True
                        dfa, state, node = self.stack[-1]
                        states, first = dfa
                    return False
                elif t >= 256:
                    itsdfa = self.grammar.dfas[t]
                    itsstates, itsfirst = itsdfa
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

    def classify(self, type: int, value: Any, context: Any) -> int:
        if type == token.NAME:
            self.used_names.add(value)
            ilabel: Optional[int] = self.grammar.keywords.get(value)
            if ilabel is not None:
                return ilabel
        ilabel: Optional[int] = self.grammar.tokens.get(type)
        if ilabel is None:
            raise ParseError('bad token', type, value, context)
        return ilabel

    def shift(self, type: int, value: Any, newstate: int, context: Any) -> None:
        dfa, state, node = self.stack[-1]
        newnode: Tuple[int, Any, Any, Optional[Any]] = (type, value, context, None)
        newnode = self.convert(self.grammar, newnode)
        if newnode is not None:
            node[-1].append(newnode)
        self.stack[-1] = (dfa, newstate, node)

    def push(self, type: int, newdfa: Any, newstate: int, context: Any) -> None:
        dfa, state, node = self.stack[-1]
        newnode: Tuple[int, None, Any, List[Any]] = (type, None, context, [])
        self.stack[-1] = (dfa, newstate, node)
        self.stack.append((newdfa, 0, newnode))

    def pop(self) -> None:
        popdfa, popstate, popnode = self.stack.pop()
        newnode = self.convert(self.grammar, popnode)
        if newnode is not None:
            if self.stack:
                dfa, state, node = self.stack[-1]
                node[-1].append(newnode)
            else:
                self.rootnode = newnode
                self.rootnode.used_names = self.used_names