from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Optional, Union, cast, List, Tuple, Set
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pytree import NL, Context, Leaf, Node, RawNode, convert
from . import grammar, token, tokenize
if TYPE_CHECKING:
    from blib2to3.pgen2.driver import TokenProxy

Results = dict[str, NL]
Convert = Callable[[Grammar, RawNode], Union[Node, Leaf]]
DFA = List[List[Tuple[int, int]]]
DFAS = Tuple[DFA, dict[int, int]]

def lam_sub(grammar: Grammar, node: RawNode) -> Node:
    assert node[3] is not None
    return Node(type=node[0], children=node[3], context=node[2])

DUMMY_NODE: Tuple[int, None, None, None] = (-1, None, None, None)

def stack_copy(stack: List[Tuple[DFA, int, Any]]) -> List[Tuple[DFA, int, Any]]:
    """Nodeless stack copy."""
    return [(dfa, label, DUMMY_NODE) for dfa, label, _ in stack]

class Recorder:
    def __init__(self, parser: "Parser", ilabels: List[int], context: Context) -> None:
        self.parser: Parser = parser
        self._ilabels: List[int] = ilabels
        self.context: Context = context
        self._dead_ilabels: Set[int] = set()
        self._start_point: List[Tuple[DFA, int, Any]] = self.parser.stack
        self._points: dict[int, List[Tuple[DFA, int, Any]]] = {
            ilabel: stack_copy(self._start_point) for ilabel in ilabels
        }

    @property
    def ilabels(self) -> Set[int]:
        return self._dead_ilabels.symmetric_difference(self._ilabels)

    @contextmanager
    def switch_to(self, ilabel: int) -> Iterator[None]:
        with self.backtrack():
            self.parser.stack = self._points[ilabel]
            try:
                yield
            except ParseError:
                self._dead_ilabels.add(ilabel)
            finally:
                self.parser.stack = self._start_point

    @contextmanager
    def backtrack(self) -> Iterator[None]:
        """
        Use the node-level invariant ones for basic parsing operations (push/pop/shift).
        These still will operate on the stack; but they won't create any new nodes, or
        modify the contents of any other existing nodes.

        This saves us a ton of time when we are backtracking, since we
        want to restore to the initial state as quick as possible, which
        can only be done by having as little mutatations as possible.
        """
        is_backtracking: bool = self.parser.is_backtracking
        try:
            self.parser.is_backtracking = True
            yield
        finally:
            self.parser.is_backtracking = is_backtracking

    def add_token(self, tok_type: int, tok_val: Any, raw: bool = False) -> None:
        if raw:
            func = self.parser._addtoken
        else:
            func = self.parser.addtoken
        for ilabel in self.ilabels:
            with self.switch_to(ilabel):
                args: List[Any] = [tok_type, tok_val, self.context]
                if raw:
                    args.insert(0, ilabel)
                func(*args)

    def determine_route(self, value: Optional[Any] = None, force: bool = False) -> Optional[int]:
        alive_ilabels: Set[int] = self.ilabels
        if len(alive_ilabels) == 0:
            *_, most_successful_ilabel = self._dead_ilabels
            raise ParseError('bad input', most_successful_ilabel, value, self.context)
        ilabel, *rest = alive_ilabels
        if force or not rest:
            return ilabel
        else:
            return None

class ParseError(Exception):
    """Exception to signal the parser is stuck."""
    def __init__(self, msg: str, type: int, value: Any, context: Context) -> None:
        super().__init__(f'{msg}: type={type!r}, value={value!r}, context={context!r}')
        self.msg: str = msg
        self.type: int = type
        self.value: Any = value
        self.context: Context = context

class Parser:
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

    See driver.py for how to get input tokens by tokenizing a file or
    string.

    Parsing is complete when addtoken() returns True; the root of the
    abstract syntax tree can then be retrieved from the rootnode
    instance variable.  When a syntax error occurs, addtoken() raises
    the ParseError exception.  There is no error recovery; the parser
    cannot be used after a syntax error was reported (but it can be
    reinitialized by calling setup()).

    """
    def __init__(self, grammar: Grammar, convert: Optional[Convert] = None) -> None:
        """Constructor.
        """
        self.grammar: Grammar = grammar
        self.convert: Convert = convert or lam_sub
        self.is_backtracking: bool = False
        self.last_token: Optional[int] = None

    def setup(self, proxy: "TokenProxy", start: Optional[int] = None) -> None:
        """Prepare for parsing.
        """
        if start is None:
            start = self.grammar.start
        newnode: RawNode = (start, None, None, [])
        stackentry: Tuple[DFA, int, RawNode] = (self.grammar.dfas[start], 0, newnode)
        self.stack: List[Tuple[DFA, int, RawNode]] = [stackentry]
        self.rootnode: Optional[Node] = None
        self.used_names: Set[str] = set()
        self.proxy: TokenProxy = proxy
        self.last_token = None

    def addtoken(self, type: int, value: Any, context: Context) -> bool:
        """Add a token; return True iff this is the end of the program."""
        ilabels: List[int] = self.classify(type, value, context)
        assert len(ilabels) >= 1
        if len(ilabels) == 1:
            [ilabel] = ilabels
            return self._addtoken(ilabel, type, value, context)
        with self.proxy.release() as proxy:
            counter: int = 0
            force: bool = False
            recorder: Recorder = Recorder(self, ilabels, context)
            recorder.add_token(type, value, raw=True)
            next_token_value: Any = value
            while recorder.determine_route(next_token_value) is None:
                if not proxy.can_advance(counter):
                    force = True
                    break
                next_token_type, next_token_value, *rest = proxy.eat(counter)
                if next_token_type in (tokenize.COMMENT, tokenize.NL):
                    counter += 1
                    continue
                if next_token_type == tokenize.OP:
                    next_token_type = grammar.opmap[next_token_value]
                recorder.add_token(next_token_type, next_token_value)
                counter += 1
            ilabel = cast(int, recorder.determine_route(next_token_value, force=force))
            assert ilabel is not None
        return self._addtoken(ilabel, type, value, context)

    def _addtoken(self, ilabel: int, type: int, value: Any, context: Context) -> bool:
        while True:
            dfa, state, node = self.stack[-1]
            states: List[List[Tuple[int, int]]] = dfa
            arcs: List[Tuple[int, int]] = states[state]
            for i, newstate in arcs:
                t: int = self.grammar.labels[i][0]
                if t >= 256:
                    itsdfa: DFA = self.grammar.dfas[t]
                    itsstates: List[List[Tuple[int, int]]] = itsdfa
                    itsfirst: Any = None  # type: ignore
                    itsfirst = self.grammar.dfas[t][1]  # Assuming second element is first set
                    if ilabel in itsfirst:
                        self.push(t, itsdfa, newstate, context)
                        break
                elif ilabel == i:
                    self.shift(type, value, newstate, context)
                    state = newstate
                    while states[state] == [(0, state)]:
                        self.pop()
                        if not self.stack:
                            return True
                        dfa, state, node = self.stack[-1]
                        states = dfa
                    self.last_token = type
                    return False
            else:
                if (0, state) in arcs:
                    self.pop()
                    if not self.stack:
                        raise ParseError('too much input', type, value, context)
                else:
                    raise ParseError('bad input', type, value, context)

    def classify(self, type: int, value: Any, context: Context) -> List[int]:
        """Turn a token into a label.  (Internal)
        """
        if type == token.NAME:
            self.used_names.add(value)
            if value in self.grammar.keywords:
                return [self.grammar.keywords[value]]
            elif value in self.grammar.soft_keywords:
                assert type in self.grammar.tokens
                if self.last_token not in (None, token.INDENT, token.DEDENT, token.NEWLINE, token.SEMI, token.COLON):
                    return [self.grammar.tokens[type]]
                return [self.grammar.tokens[type], self.grammar.soft_keywords[value]]
        ilabel: Optional[int] = self.grammar.tokens.get(type)
        if ilabel is None:
            raise ParseError('bad token', type, value, context)
        return [ilabel]

    def shift(self, type: int, value: Any, newstate: int, context: Context) -> None:
        """Shift a token.  (Internal)"""
        if self.is_backtracking:
            dfa, state, _ = self.stack[-1]
            self.stack[-1] = (dfa, newstate, DUMMY_NODE)
        else:
            dfa, state, node = self.stack[-1]
            rawnode: RawNode = (type, value, context, None)
            newnode: Union[Node, Leaf] = convert(self.grammar, rawnode)
            assert node[-1] is not None
            node[-1].append(newnode)
            self.stack[-1] = (dfa, newstate, node)

    def push(self, type: int, newdfa: DFA, newstate: int, context: Context) -> None:
        """Push a nonterminal.  (Internal)"""
        if self.is_backtracking:
            dfa, state, _ = self.stack[-1]
            self.stack[-1] = (dfa, newstate, DUMMY_NODE)
            self.stack.append((newdfa, 0, DUMMY_NODE))
        else:
            dfa, state, node = self.stack[-1]
            newnode: RawNode = (type, None, context, [])
            self.stack[-1] = (dfa, newstate, node)
            self.stack.append((newdfa, 0, newnode))

    def pop(self) -> None:
        """Pop a nonterminal.  (Internal)"""
        if self.is_backtracking:
            self.stack.pop()
        else:
            popdfa, popstate, popnode = self.stack.pop()
            newnode: Union[Node, Leaf] = convert(self.grammar, popnode)
            if self.stack:
                dfa, state, node = self.stack[-1]
                assert node[-1] is not None
                node[-1].append(newnode)
            else:
                self.rootnode = newnode
                self.rootnode.used_names = self.used_names  # type: ignore