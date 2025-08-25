from . import token
from typing import List, Tuple, Optional, Callable, Set

class ParseError(Exception):
    def __init__(self, msg: str, type: int, value: str, context: Tuple[int, int]):
        self.msg: str = msg
        self.type: int = type
        self.value: str = value
        self.context: Tuple[int, int] = context

    def __reduce__(self) -> Tuple[type, Tuple[str, int, str, Tuple[int, int]]]:
        return (type(self), (self.msg, self.type, self.value, self.context))

class Parser:
    def __init__(self, grammar, convert: Optional[Callable[[Any, Tuple[int, str, Any, List]], Any]] = None):
        self.grammar = grammar
        self.convert = (convert or (lambda grammar, node: node))

    def setup(self, start: Optional[int] = None) -> None:
        if (start is None):
            start = self.grammar.start
        newnode: Tuple[int, Optional[str], Optional[Any], List] = (start, None, None, [])
        stackentry: Tuple[Any, int, Tuple[int, Optional[str], Optional[Any], List]] = (self.grammar.dfas[start], 0, newnode)
        self.stack: List[Tuple[Any, int, Tuple[int, Optional[str], Optional[Any], List]]] = [stackentry]
        self.rootnode: Optional[Any] = None
        self.used_names: Set[str] = set()

    def addtoken(self, type: int, value: str, context: Tuple[int, int]) -> bool:
        ilabel: int = self.classify(type, value, context)
        while True:
            (dfa, state, node) = self.stack[(- 1)]
            (states, first) = dfa
            arcs = states[state]
            for (i, newstate) in arcs:
                (t, v) = self.grammar.labels[i]
                if (ilabel == i):
                    assert (t < 256)
                    self.shift(type, value, newstate, context)
                    state = newstate
                    while (states[state] == [(0, state)]):
                        self.pop()
                        if (not self.stack):
                            return True
                        (dfa, state, node) = self.stack[(- 1)]
                        (states, first) = dfa
                    return False
                elif (t >= 256):
                    itsdfa = self.grammar.dfas[t]
                    (itsstates, itsfirst) = itsdfa
                    if (ilabel in itsfirst):
                        self.push(t, self.grammar.dfas[t], newstate, context)
                        break
            else:
                if ((0, state) in arcs):
                    self.pop()
                    if (not self.stack):
                        raise ParseError('too much input', type, value, context)
                else:
                    raise ParseError('bad input', type, value, context)

    def classify(self, type: int, value: str, context: Tuple[int, int]) -> int:
        if (type == token.NAME):
            self.used_names.add(value)
            ilabel = self.grammar.keywords.get(value)
            if (ilabel is not None):
                return ilabel
        ilabel = self.grammar.tokens.get(type)
        if (ilabel is None):
            raise ParseError('bad token', type, value, context)
        return ilabel

    def shift(self, type: int, value: str, newstate: int, context: Tuple[int, int]) -> None:
        (dfa, state, node) = self.stack[(- 1)]
        newnode: Tuple[int, str, Tuple[int, int], List] = (type, value, context, None)
        newnode = self.convert(self.grammar, newnode)
        if (newnode is not None):
            node[(- 1)].append(newnode)
        self.stack[(- 1)] = (dfa, newstate, node)

    def push(self, type: int, newdfa: Any, newstate: int, context: Tuple[int, int]) -> None:
        (dfa, state, node) = self.stack[(- 1)]
        newnode: Tuple[int, Optional[str], Tuple[int, int], List] = (type, None, context, [])
        self.stack[(- 1)] = (dfa, newstate, node)
        self.stack.append((newdfa, 0, newnode))

    def pop(self) -> None:
        (popdfa, popstate, popnode) = self.stack.pop()
        newnode = self.convert(self.grammar, popnode)
        if (newnode is not None):
            if self.stack:
                (dfa, state, node) = self.stack[(- 1)]
                node[(- 1)].append(newnode)
            else:
                self.rootnode = newnode
                self.rootnode.used_names = self.used_names
