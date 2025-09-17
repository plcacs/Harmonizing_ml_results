from typing import Any, Callable, List, Optional, Set, Tuple
from . import token

class ParseError(Exception):
    """Exception to signal the parser is stuck."""

    def __init__(self, msg: str, type: Any, value: Any, context: Any) -> None:
        Exception.__init__(self, '%s: type=%r, value=%r, context=%r' % (msg, type, value, context))
        self.msg: str = msg
        self.type: Any = type
        self.value: Any = value
        self.context: Any = context

# Type aliases for clarity.
# A Node is a tuple (type, value, context, nodes) where nodes is either a list or None.
Node = Tuple[Any, Any, Any, Optional[List[Any]]]
# A DFA is represented as a tuple (states, first)
DFA = Tuple[Any, Any]
# A stack entry is a tuple of (dfa, state, node)
StackEntry = Tuple[Any, int, Node]

class Parser(object):
    """Parser engine.

    The proper usage sequence is:

    p = Parser(grammar, [converter])  # create instance
    p.setup([start])                  # prepare for parsing
    <for each input token>:
        if p.addtoken(...):           # parse a token; may raise ParseError
            break
    root = p.rootnode                 # root of abstract syntax tree

    A Parser instance may be reused by calling setup() repeatedly.

    A Parser instance contains state pertaining to the current token
    sequence, and should not be used concurrently by different threads
    to parse separate token sequences.
    """

    def __init__(self, grammar: Any, convert: Optional[Callable[[Any, Node], Any]] = None) -> None:
        """Constructor.

        The grammar argument is a grammar.Grammar instance.
        The optional convert argument is a function mapping concrete
        syntax tree nodes to abstract syntax tree nodes.
        """
        self.grammar: Any = grammar
        self.convert: Callable[[Any, Node], Any] = convert or (lambda grammar, node: node)
        self.stack: List[StackEntry] = []
        self.rootnode: Optional[Any] = None
        self.used_names: Set[str] = set()

    def setup(self, start: Optional[int] = None) -> None:
        """Prepare for parsing.

        The optional argument is an alternative start symbol; it
        defaults to the grammar's start symbol.
        """
        if start is None:
            start = self.grammar.start
        newnode: Node = (start, None, None, [])
        stackentry: StackEntry = (self.grammar.dfas[start], 0, newnode)
        self.stack = [stackentry]
        self.rootnode = None
        self.used_names = set()

    def addtoken(self, type: int, value: Any, context: Any) -> bool:
        """Add a token; return True iff this is the end of the program."""
        ilabel: int = self.classify(type, value, context)
        while True:
            dfa, state, node = self.stack[-1]
            states, first = dfa
            arcs = states[state]
            for i, newstate in arcs:
                t, v = self.grammar.labels[i]
                if ilabel == i:
                    # t is less than 256 in this arc
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
        """Turn a token into a label.  (Internal)"""
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
        """Shift a token.  (Internal)"""
        dfa, state, node = self.stack[-1]
        newnode: Node = (type, value, context, None)
        newnode = self.convert(self.grammar, newnode)
        if newnode is not None:
            node[-1].append(newnode)
        self.stack[-1] = (dfa, newstate, node)

    def push(self, type: int, newdfa: Any, newstate: int, context: Any) -> None:
        """Push a nonterminal.  (Internal)"""
        dfa, state, node = self.stack[-1]
        newnode: Node = (type, None, context, [])
        self.stack[-1] = (dfa, newstate, node)
        self.stack.append((newdfa, 0, newnode))

    def pop(self) -> None:
        """Pop a nonterminal.  (Internal)"""
        popdfa, popstate, popnode = self.stack.pop()
        newnode = self.convert(self.grammar, popnode)
        if newnode is not None:
            if self.stack:
                dfa, state, node = self.stack[-1]
                node[-1].append(newnode)
            else:
                self.rootnode = newnode
                self.rootnode.used_names = self.used_names