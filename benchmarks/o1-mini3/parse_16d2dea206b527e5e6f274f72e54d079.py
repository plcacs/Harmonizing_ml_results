from __future__ import annotations
from typing import Optional, Callable, Any, Tuple, List, Set
from . import token

class ParseError(Exception):
    'Exception to signal the parser is stuck.'

    def __init__(self, msg: str, type: int, value: Any, context: Any) -> None:
        super().__init__(f'{msg}: type={type!r}, value={value!r}, context={context!r}')
        self.msg: str = msg
        self.type: int = type
        self.value: Any = value
        self.context: Any = context

    def __reduce__(self) -> Tuple[Any, Tuple[str, int, Any, Any]]:
        return (type(self), (self.msg, self.type, self.value, self.context))

class Parser:
    'Parser engine.\n\n    The proper usage sequence is:\n\n    p = Parser(grammar, [converter])  # create instance\n    p.setup([start])                  # prepare for parsing\n    <for each input token>:\n        if p.addtoken(...):           # parse a token; may raise ParseError\n            break\n    root = p.rootnode                 # root of abstract syntax tree\n\n    A Parser instance may be reused by calling setup() repeatedly.\n\n    A Parser instance contains state pertaining to the current token\n    sequence, and should not be used concurrently by different threads\n    to parse separate token sequences.\n\n    See driver.py for how to get input tokens by tokenizing a file or\n    string.\n\n    Parsing is complete when addtoken() returns True; the root of the\n    abstract syntax tree can then be retrieved from the rootnode\n    instance variable.  When a syntax error occurs, addtoken() raises\n    the ParseError exception.  There is no error recovery; the parser\n    cannot be used after a syntax error was reported (but it can be\n    reinitialized by calling setup()).\n\n    '

    def __init__(self, grammar: Any, convert: Optional[Callable[[Any, Tuple[Any, ...]], Any]] = None) -> None:
        'Constructor.\n\n        The grammar argument is a grammar.Grammar instance; see the\n        grammar module for more information.\n\n        The parser is not ready yet for parsing; you must call the\n        setup() method to get it started.\n\n        The optional convert argument is a function mapping concrete\n        syntax tree nodes to abstract syntax tree nodes.  If not\n        given, no conversion is done and the syntax tree produced is\n        the concrete syntax tree.  If given, it must be a function of\n        two arguments, the first being the grammar (a grammar.Grammar\n        instance), and the second being the concrete syntax tree node\n        to be converted.  The syntax tree is converted from the bottom\n        up.\n\n        A concrete syntax tree node is a (type, value, context, nodes)\n        tuple, where type is the node type (a token or symbol number),\n        value is None for symbols and a string for tokens, context is\n        None or an opaque value used for error reporting (typically a\n        (lineno, offset) pair), and nodes is a list of children for\n        symbols, and None for tokens.\n\n        An abstract syntax tree node may be anything; this is entirely\n        up to the converter function.\n\n        '
        self.grammar: Any = grammar
        self.convert: Callable[[Any, Tuple[Any, ...]], Any] = convert or (lambda grammar, node: node)

    def setup(self, start: Optional[int] = None) -> None:
        "Prepare for parsing.\n\n        This *must* be called before starting to parse.\n\n        The optional argument is an alternative start symbol; it\n        defaults to the grammar's start symbol.\n\n        You can use a Parser instance to parse any number of programs;\n        each time you call setup() the parser is reset to an initial\n        state determined by the (implicit or explicit) start symbol.\n\n        "
        if start is None:
            start = self.grammar.start
        newnode: Tuple[Any, Any, Any, List[Any]] = (start, None, None, [])
        stackentry: Tuple[Any, int, Tuple[Any, Any, Any, List[Any]]] = (self.grammar.dfas[start], 0, newnode)
        self.stack: List[Tuple[Any, int, Tuple[Any, Any, Any, List[Any]]]] = [stackentry]
        self.rootnode: Optional[Any] = None
        self.used_names: Set[str] = set()

    def addtoken(self, type: int, value: Any, context: Any) -> bool:
        'Add a token; return True iff this is the end of the program.'
        ilabel: int = self.classify(type, value, context)
        while True:
            dfa, state, node = self.stack[-1]
            states, first = dfa
            arcs = states[state]
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
        'Turn a token into a label.  (Internal)'
        if type == token.NAME:
            self.used_names.add(value)
            ilabel = self.grammar.keywords.get(value)
            if ilabel is not None:
                return ilabel
        ilabel = self.grammar.tokens.get(type)
        if ilabel is None:
            raise ParseError('bad token', type, value, context)
        return ilabel

    def shift(self, type: int, value: Any, newstate: int, context: Any) -> None:
        'Shift a token.  (Internal)'
        dfa, state, node = self.stack[-1]
        newnode: Tuple[Any, Any, Any, Any] = (type, value, context, None)
        newnode = self.convert(self.grammar, newnode)
        if newnode is not None:
            node[-1].append(newnode)
        self.stack[-1] = (dfa, newstate, node)

    def push(self, type: int, newdfa: Any, newstate: int, context: Any) -> None:
        'Push a nonterminal.  (Internal)'
        dfa, state, node = self.stack[-1]
        newnode: Tuple[Any, Any, Any, List[Any]] = (type, None, context, [])
        self.stack[-1] = (dfa, newstate, node)
        self.stack.append((newdfa, 0, newnode))

    def pop(self) -> None:
        'Pop a nonterminal.  (Internal)'
        popdfa, popstate, popnode = self.stack.pop()
        newnode = self.convert(self.grammar, popnode)
        if newnode is not None:
            if self.stack:
                dfa, state, node = self.stack[-1]
                node[-1].append(newnode)
            else:
                self.rootnode = newnode
                self.rootnode.used_names = self.used_names
