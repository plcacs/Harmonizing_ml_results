from typing import TYPE_CHECKING, Any, Callable, Iterator, Union

Results = dict[str, NL]
Convert = Callable[[Grammar, RawNode], Union[Node, Leaf]]
DFA = list[list[tuple[int, int]]]
DFAS = tuple[DFA, dict[int, int]]

def lam_sub(grammar: Grammar, node: RawNode) -> Node:
    assert node[3] is not None
    return Node(type=node[0], children=node[3], context=node[2])

DUMMY_NODE: tuple[int, None, None, None] = (-1, None, None, None)

def stack_copy(stack: list[tuple[DFA, int, tuple[int, Optional[str], Any, list[Any]]]]) -> list[tuple[DFA, int, tuple[int, None, None, None]]]:
    return [(dfa, label, DUMMY_NODE) for dfa, label, _ in stack]

class Recorder:
    def __init__(self, parser: Parser, ilabels: set[int], context: Any):
        self.parser = parser
        self._ilabels = ilabels
        self.context = context
        self._dead_ilabels = set()
        self._start_point = self.parser.stack
        self._points = {ilabel: stack_copy(self._start_point) for ilabel in ilabels}

    @property
    def ilabels(self) -> set[int]:
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
        is_backtracking = self.parser.is_backtracking
        try:
            self.parser.is_backtracking = True
            yield
        finally:
            self.parser.is_backtracking = is_backtracking

    def add_token(self, tok_type: int, tok_val: str, raw: bool = False) -> None:
        if raw:
            func = self.parser._addtoken
        else:
            func = self.parser.addtoken
        for ilabel in self.ilabels:
            with self.switch_to(ilabel):
                args = [tok_type, tok_val, self.context]
                if raw:
                    args.insert(0, ilabel)
                func(*args)

    def determine_route(self, value: Any = None, force: bool = False) -> Optional[int]:
        alive_ilabels = self.ilabels
        if len(alive_ilabels) == 0:
            *_, most_successful_ilabel = self._dead_ilabels
            raise ParseError('bad input', most_successful_ilabel, value, self.context)
        ilabel, *rest = alive_ilabels
        if force or not rest:
            return ilabel
        else:
            return None

class ParseError(Exception):
    def __init__(self, msg: str, type: int, value: Any, context: Any):
        Exception.__init__(self, f'{msg}: type={type!r}, value={value!r}, context={context!r}')
        self.msg = msg
        self.type = type
        self.value = value
        self.context = context

class Parser:
    def __init__(self, grammar: Grammar, convert: Convert = None):
        self.grammar = grammar
        self.convert = convert or lam_sub
        self.is_backtracking = False
        self.last_token = None

    def setup(self, proxy: 'TokenProxy', start: Optional[int] = None) -> None:
        if start is None:
            start = self.grammar.start
        newnode: tuple[int, None, None, list[Any]] = (start, None, None, [])
        stackentry: tuple[DFA, int, tuple[int, None, None, list[Any]]] = (self.grammar.dfas[start], 0, newnode)
        self.stack = [stackentry]
        self.rootnode = None
        self.used_names = set()
        self.proxy = proxy
        self.last_token = None

    def addtoken(self, type: int, value: Any, context: Any) -> bool:
        ilabels = self.classify(type, value, context)
        assert len(ilabels) >= 1
        if len(ilabels) == 1:
            [ilabel] = ilabels
            return self._addtoken(ilabel, type, value, context)
        with self.proxy.release() as proxy:
            counter, force = (0, False)
            recorder = Recorder(self, ilabels, context)
            recorder.add_token(type, value, raw=True)
            next_token_value = value
            while recorder.determine_route(next_token_value) is None:
                if not proxy.can_advance(counter):
                    force = True
                    break
                next_token_type, next_token_value, *_ = proxy.eat(counter)
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

    def _addtoken(self, ilabel: int, type: int, value: Any, context: Any) -> bool:
        while True:
            dfa, state, node = self.stack[-1]
            states, first = dfa
            arcs = states[state]
            for i, newstate in arcs:
                t = self.grammar.labels[i][0]
                if t >= 256:
                    itsdfa = self.grammar.dfas[t]
                    itsstates, itsfirst = itsdfa
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
                        states, first = dfa
                    self.last_token = type
                    return False
            else:
                if (0, state) in arcs:
                    self.pop()
                    if not self.stack:
                        raise ParseError('too much input', type, value, context)
                else:
                    raise ParseError('bad input', type, value, context)

    def classify(self, type: int, value: Any, context: Any) -> list[int]:
        if type == token.NAME:
            self.used_names.add(value)
            if value in self.grammar.keywords:
                return [self.grammar.keywords[value]]
            elif value in self.grammar.soft_keywords:
                assert type in self.grammar.tokens
                if self.last_token not in (None, token.INDENT, token.DEDENT, token.NEWLINE, token.SEMI, token.COLON):
                    return [self.grammar.tokens[type]]
                return [self.grammar.tokens[type], self.grammar.soft_keywords[value]]
        ilabel = self.grammar.tokens.get(type)
        if ilabel is None:
            raise ParseError('bad token', type, value, context)
        return [ilabel]

    def shift(self, type: int, value: Any, newstate: int, context: Any) -> None:
        if self.is_backtracking:
            dfa, state, _ = self.stack[-1]
            self.stack[-1] = (dfa, newstate, DUMMY_NODE)
        else:
            dfa, state, node = self.stack[-1]
            rawnode = (type, value, context, None)
            newnode = convert(self.grammar, rawnode)
            assert node[-1] is not None
            node[-1].append(newnode)
            self.stack[-1] = (dfa, newstate, node)

    def push(self, type: int, newdfa: DFA, newstate: int, context: Any) -> None:
        if self.is_backtracking:
            dfa, state, _ = self.stack[-1]
            self.stack[-1] = (dfa, newstate, DUMMY_NODE)
            self.stack.append((newdfa, 0, DUMMY_NODE))
        else:
            dfa, state, node = self.stack[-1]
            newnode = (type, None, context, [])
            self.stack[-1] = (dfa, newstate, node)
            self.stack.append((newdfa, 0, newnode))

    def pop(self) -> None:
        if self.is_backtracking:
            self.stack.pop()
        else:
            popdfa, popstate, popnode = self.stack.pop()
            newnode = convert(self.grammar, popnode)
            if self.stack:
                dfa, state, node = self.stack[-1]
                assert node[-1] is not None
                node[-1].append(newnode)
            else:
                self.rootnode = newnode
                self.rootnode.used_names = self.used_names
