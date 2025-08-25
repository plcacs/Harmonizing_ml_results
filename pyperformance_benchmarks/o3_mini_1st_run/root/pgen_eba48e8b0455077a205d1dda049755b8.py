#!/usr/bin/env python3
from typing import Optional, List, Dict, Tuple, Any, TextIO, Iterator
from . import grammar, token, tokenize

class PgenGrammar(grammar.Grammar):
    pass

class ParserGenerator(object):
    filename: str
    stream: TextIO
    generator: Iterator[Tuple[int, str, Tuple[int, int], Tuple[int, int], str]]
    type: int
    value: str
    begin: Tuple[int, int]
    end: Tuple[int, int]
    line: str
    dfas: Dict[str, List["DFAState"]]
    startsymbol: str
    first: Dict[str, Dict[str, int]]

    def __init__(self, filename: str, stream: Optional[TextIO] = None) -> None:
        close_stream: Optional[Any] = None
        if stream is None:
            stream = open(filename, encoding='utf-8')
            close_stream = stream.close
        self.filename = filename
        self.stream = stream
        self.generator = tokenize.generate_tokens(stream.readline)
        self.gettoken()
        (self.dfas, self.startsymbol) = self.parse()
        if close_stream is not None:
            close_stream()
        self.first = {}
        self.addfirstsets()

    def make_grammar(self) -> PgenGrammar:
        c: PgenGrammar = PgenGrammar()
        names: List[str] = list(self.dfas.keys())
        names.sort()
        names.remove(self.startsymbol)
        names.insert(0, self.startsymbol)
        for name in names:
            i: int = 256 + len(c.symbol2number)
            c.symbol2number[name] = i
            c.number2symbol[i] = name
        for name in names:
            dfa: List[DFAState] = self.dfas[name]
            states: List[List[Tuple[int, int]]] = []
            for state in dfa:
                arcs: List[Tuple[int, int]] = []
                for (label, next_state) in sorted(state.arcs.items()):
                    arcs.append((self.make_label(c, label), dfa.index(next_state)))
                if state.isfinal:
                    arcs.append((0, dfa.index(state)))
                states.append(arcs)
            c.states.append(states)
            c.dfas[c.symbol2number[name]] = (states, self.make_first(c, name))
        c.start = c.symbol2number[self.startsymbol]
        return c

    def make_first(self, c: PgenGrammar, name: str) -> Dict[int, int]:
        rawfirst: Dict[str, int] = self.first[name]
        first: Dict[int, int] = {}
        for label in sorted(rawfirst):
            ilabel: int = self.make_label(c, label)
            first[ilabel] = 1
        return first

    def make_label(self, c: PgenGrammar, label: str) -> int:
        ilabel: int = len(c.labels)
        if label[0].isalpha():
            if label in c.symbol2number:
                if label in c.symbol2label:
                    return c.symbol2label[label]
                else:
                    c.labels.append((c.symbol2number[label], None))
                    c.symbol2label[label] = ilabel
                    return ilabel
            else:
                itoken: Optional[int] = getattr(token, label, None)
                assert isinstance(itoken, int), label
                assert itoken in token.tok_name, label
                if itoken in c.tokens:
                    return c.tokens[itoken]
                else:
                    c.labels.append((itoken, None))
                    c.tokens[itoken] = ilabel
                    return ilabel
        else:
            assert label[0] in ('"', "'"), label
            value = eval(label)
            if value[0].isalpha():
                if value in c.keywords:
                    return c.keywords[value]
                else:
                    c.labels.append((token.NAME, value))
                    c.keywords[value] = ilabel
                    return ilabel
            else:
                itoken = grammar.opmap[value]
                if itoken in c.tokens:
                    return c.tokens[itoken]
                else:
                    c.labels.append((itoken, None))
                    c.tokens[itoken] = ilabel
                    return ilabel

    def addfirstsets(self) -> None:
        names: List[str] = list(self.dfas.keys())
        names.sort()
        for name in names:
            if name not in self.first:
                self.calcfirst(name)

    def calcfirst(self, name: str) -> None:
        dfa: List[DFAState] = self.dfas[name]
        self.first[name] = None  # type: ignore
        state: NFAState = dfa[0].nfaset.keys().__iter__().__next__()
        totalset: Dict[str, int] = {}
        overlapcheck: Dict[str, Dict[str, int]] = {}
        # In this context, we need an NFAState representative from the DFA's start, so using dfa[0] arcs.
        for (label, next_state) in state.arcs:  # type: ignore
            if label in self.dfas:
                if label in self.first:
                    fset: Optional[Dict[str, int]] = self.first[label]
                    if fset is None:
                        raise ValueError("recursion for rule %r" % name)
                else:
                    self.calcfirst(label)
                    fset = self.first[label]
                totalset.update(fset)  # type: ignore
                overlapcheck[label] = fset  # type: ignore
            else:
                totalset[label] = 1
                overlapcheck[label] = {label: 1}
        inverse: Dict[str, str] = {}
        for (lab, itsfirst) in overlapcheck.items():
            for symbol in itsfirst:
                if symbol in inverse:
                    raise ValueError("rule %s is ambiguous; %s is in the first sets of %s as well as %s" % (name, symbol, lab, inverse[symbol]))
                inverse[symbol] = lab
        self.first[name] = totalset

    def parse(self) -> Tuple[Dict[str, List["DFAState"]], str]:
        dfas: Dict[str, List[DFAState]] = {}
        startsymbol: Optional[str] = None
        while self.type != token.ENDMARKER:
            while self.type == token.NEWLINE:
                self.gettoken()
            name: str = self.expect(token.NAME)
            self.expect(token.OP, ':')
            (a, z) = self.parse_rhs()
            self.expect(token.NEWLINE)
            dfa: List[DFAState] = self.make_dfa(a, z)
            oldlen: int = len(dfa)
            self.simplify_dfa(dfa)
            newlen: int = len(dfa)
            dfas[name] = dfa
            if startsymbol is None:
                startsymbol = name
        assert startsymbol is not None
        return (dfas, startsymbol)

    def make_dfa(self, start: "NFAState", finish: "NFAState") -> List["DFAState"]:
        assert isinstance(start, NFAState)
        assert isinstance(finish, NFAState)

        def closure(state: NFAState) -> Dict[NFAState, int]:
            base: Dict[NFAState, int] = {}
            addclosure(state, base)
            return base

        def addclosure(state: NFAState, base: Dict[NFAState, int]) -> None:
            assert isinstance(state, NFAState)
            if state in base:
                return
            base[state] = 1
            for (label, next_state) in state.arcs:
                if label is None:
                    addclosure(next_state, base)

        states: List[DFAState] = [DFAState(closure(start), finish)]
        for state in states:
            arcs: Dict[str, Dict[NFAState, int]] = {}
            for nfastate in state.nfaset:  # type: ignore
                for (label, next_state) in nfastate.arcs:
                    if label is not None:
                        arcs.setdefault(label, {})
                        addclosure(next_state, arcs[label])
            for (label, nfaset) in sorted(arcs.items()):
                for st in states:
                    if st.nfaset == nfaset:
                        break
                else:
                    st = DFAState(nfaset, finish)
                    states.append(st)
                state.addarc(st, label)
        return states

    def dump_nfa(self, name: str, start: "NFAState", finish: "NFAState") -> None:
        print('Dump of NFA for', name)
        todo: List[NFAState] = [start]
        for i, state in enumerate(todo):
            print('  State', i, ('(final)' if (state is finish) else ''))
            for (label, next_state) in state.arcs:
                if next_state in todo:
                    j = todo.index(next_state)
                else:
                    j = len(todo)
                    todo.append(next_state)
                if label is None:
                    print("    -> %d" % j)
                else:
                    print("    %s -> %d" % (label, j))

    def dump_dfa(self, name: str, dfa: List["DFAState"]) -> None:
        print('Dump of DFA for', name)
        for i, state in enumerate(dfa):
            print('  State', i, ('(final)' if state.isfinal else ''))
            for (label, next_state) in sorted(state.arcs.items()):
                print("    %s -> %d" % (label, dfa.index(next_state)))

    def simplify_dfa(self, dfa: List["DFAState"]) -> None:
        changes: bool = True
        while changes:
            changes = False
            for i, state_i in enumerate(dfa):
                j: int = i + 1
                while j < len(dfa):
                    state_j: DFAState = dfa[j]
                    if state_i == state_j:
                        del dfa[j]
                        for state in dfa:
                            state.unifystate(state_j, state_i)
                        changes = True
                    else:
                        j += 1

    def parse_rhs(self) -> Tuple["NFAState", "NFAState"]:
        (a, z) = self.parse_alt()
        if self.value != '|':
            return (a, z)
        else:
            aa: NFAState = NFAState()
            zz: NFAState = NFAState()
            aa.addarc(a)
            z.addarc(zz)
            while self.value == '|':
                self.gettoken()
                (a, z) = self.parse_alt()
                aa.addarc(a)
                z.addarc(zz)
            return (aa, zz)

    def parse_alt(self) -> Tuple["NFAState", "NFAState"]:
        (a, b) = self.parse_item()
        while (self.value in ('(', '[')) or (self.type in (token.NAME, token.STRING)):
            (c, d) = self.parse_item()
            b.addarc(c)
            b = d
        return (a, b)

    def parse_item(self) -> Tuple["NFAState", "NFAState"]:
        if self.value == '[':
            self.gettoken()
            (a, z) = self.parse_rhs()
            self.expect(token.OP, ']')
            a.addarc(z)
            return (a, z)
        else:
            (a, z) = self.parse_atom()
            value = self.value
            if value not in ('+', '*'):
                return (a, z)
            self.gettoken()
            z.addarc(a)
            if value == '+':
                return (a, z)
            else:
                return (a, a)

    def parse_atom(self) -> Tuple["NFAState", "NFAState"]:
        if self.value == '(':
            self.gettoken()
            (a, z) = self.parse_rhs()
            self.expect(token.OP, ')')
            return (a, z)
        elif self.type in (token.NAME, token.STRING):
            a: NFAState = NFAState()
            z: NFAState = NFAState()
            a.addarc(z, self.value)
            self.gettoken()
            return (a, z)
        else:
            self.raise_error('expected (...) or NAME or STRING, got %s/%s', self.type, self.value)
            raise AssertionError  # for type checker

    def expect(self, type_: int, value: Optional[str] = None) -> str:
        if (self.type != type_) or (value is not None and self.value != value):
            self.raise_error('expected %s/%s, got %s/%s', type_, value, self.type, self.value)
        result: str = self.value
        self.gettoken()
        return result

    def gettoken(self) -> None:
        tup: Tuple[int, str, Tuple[int, int], Tuple[int, int], str] = next(self.generator)
        while tup[0] in (tokenize.COMMENT, tokenize.NL):
            tup = next(self.generator)
        (self.type, self.value, self.begin, self.end, self.line) = tup

    def raise_error(self, msg: str, *args: Any) -> None:
        if args:
            try:
                msg = msg % args
            except Exception:
                msg = ' '.join([msg] + list(map(str, args)))
        raise SyntaxError(msg, (self.filename, self.end[0], self.end[1], self.line))

class NFAState(object):
    arcs: List[Tuple[Optional[str], "NFAState"]]

    def __init__(self) -> None:
        self.arcs = []

    def addarc(self, next_state: "NFAState", label: Optional[str] = None) -> None:
        assert label is None or isinstance(label, str)
        assert isinstance(next_state, NFAState)
        self.arcs.append((label, next_state))

class DFAState(object):
    nfaset: Dict[NFAState, Any]
    isfinal: bool
    arcs: Dict[str, "DFAState"]

    def __init__(self, nfaset: Dict[NFAState, Any], final: NFAState) -> None:
        assert isinstance(nfaset, dict)
        # We assume nfaset is non-empty; get an arbitrary element to check.
        some_state = next(iter(nfaset))
        assert isinstance(some_state, NFAState)
        assert isinstance(final, NFAState)
        self.nfaset = nfaset
        self.isfinal = (final in nfaset)
        self.arcs = {}

    def addarc(self, next_state: "DFAState", label: str) -> None:
        assert isinstance(label, str)
        assert label not in self.arcs
        assert isinstance(next_state, DFAState)
        self.arcs[label] = next_state

    def unifystate(self, old: "DFAState", new: "DFAState") -> None:
        for label, next_state in self.arcs.items():
            if next_state is old:
                self.arcs[label] = new

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DFAState):
            return False
        if self.isfinal != other.isfinal:
            return False
        if len(self.arcs) != len(other.arcs):
            return False
        for label, next_state in self.arcs.items():
            if next_state is not other.arcs.get(label):
                return False
        return True

    __hash__ = None

def generate_grammar(filename: str = 'Grammar.txt') -> PgenGrammar:
    p: ParserGenerator = ParserGenerator(filename)
    return p.make_grammar()