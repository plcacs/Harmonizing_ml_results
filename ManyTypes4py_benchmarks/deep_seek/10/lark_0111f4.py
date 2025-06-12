from inspect import signature
from typing import Optional, Dict, List, Set, Tuple, Any, Callable, Union
import lark
from lark.grammar import NonTerminal, Rule, Symbol, Terminal
from lark.lark import Lark
from lark.lexer import TerminalDef
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.conjecture.utils import calc_label_from_name
from hypothesis.internal.validation import check_type
from hypothesis.strategies._internal.regex import IncompatibleWithAlphabet
from hypothesis.strategies._internal.utils import cacheable, defines_strategy
from hypothesis.strategies._internal.core import SearchStrategy
from hypothesis.strategies._internal.strategies import T

__all__ = ['from_lark']

def get_terminal_names(
    terminals: List[TerminalDef],
    rules: List[Rule],
    ignore_names: Set[str]
) -> Set[str]:
    names = {t.name for t in terminals} | set(ignore_names)
    for rule in rules:
        names |= {t.name for t in rule.expansion if isinstance(t, Terminal)}
    return names

class LarkStrategy(SearchStrategy[str]):
    def __init__(
        self,
        grammar: Lark,
        start: Optional[Union[str, List[str]]],
        explicit: Dict[str, SearchStrategy[str]],
        alphabet: SearchStrategy[str]
    ) -> None:
        assert isinstance(grammar, lark.lark.Lark)
        start = grammar.options.start if start is None else [start] if isinstance(start, str) else start
        compile_args = signature(grammar.grammar.compile).parameters
        if 'terminals_to_keep' in compile_args:
            terminals, rules, ignore_names = grammar.grammar.compile(start, ())
        elif 'start' in compile_args:
            terminals, rules, ignore_names = grammar.grammar.compile(start)
        else:
            terminals, rules, ignore_names = grammar.grammar.compile()
        self.names_to_symbols: Dict[str, Symbol] = {}
        for r in rules:
            self.names_to_symbols[r.origin.name] = r.origin
        disallowed = set()
        self.terminal_strategies: Dict[str, SearchStrategy[str]] = {}
        for t in terminals:
            self.names_to_symbols[t.name] = Terminal(t.name)
            s = st.from_regex(t.pattern.to_regexp(), fullmatch=True, alphabet=alphabet)
            try:
                s.validate()
            except IncompatibleWithAlphabet:
                disallowed.add(t.name)
            else:
                self.terminal_strategies[t.name] = s
        self.ignored_symbols = tuple((self.names_to_symbols[n] for n in ignore_names))
        all_terminals = get_terminal_names(terminals, rules, ignore_names)
        if (unknown_explicit := sorted(set(explicit) - all_terminals)):
            raise InvalidArgument(f'The following arguments were passed as explicit_strategies, but there is no {unknown_explicit} terminal production in this grammar.')
        if (missing_declared := sorted(all_terminals - {t.name for t in terminals} - set(explicit))):
            raise InvalidArgument(f'Undefined terminal{'s' * (len(missing_declared) > 1)} {sorted(missing_declared)!r}. Generation does not currently support use of %declare unless you pass `explicit`, a dict of names-to-strategies, such as `{{{missing_declared[0]!r}: st.just("")}}}}`')
        self.terminal_strategies.update(explicit)
        nonterminals: Dict[str, List[Tuple[Symbol, ...]]] = {}
        for rule in rules:
            if disallowed.isdisjoint((r.name for r in rule.expansion)):
                nonterminals.setdefault(rule.origin.name, []).append(tuple(rule.expansion))
        allowed_rules = {*self.terminal_strategies, *nonterminals}
        while dict(nonterminals) != (nonterminals := {k: clean for k, v in nonterminals.items() if (clean := [x for x in v if all((r.name in allowed_rules for r in x))])}):
            allowed_rules = {*self.terminal_strategies, *nonterminals}
        if set(start).isdisjoint(allowed_rules):
            raise InvalidArgument(f'No start rule {tuple(start)} is allowed by alphabet={alphabet!r}')
        self.start = st.sampled_from([self.names_to_symbols[s] for s in start if s in allowed_rules])
        self.nonterminal_strategies = {k: st.sampled_from(sorted(v, key=len)) for k, v in nonterminals.items()}
        self.__rule_labels: Dict[str, int] = {}

    def do_draw(self, data: ConjectureData) -> str:
        state = []
        start = data.draw(self.start)
        self.draw_symbol(data, start, state)
        return ''.join(state)

    def rule_label(self, name: str) -> int:
        try:
            return self.__rule_labels[name]
        except KeyError:
            return self.__rule_labels.setdefault(name, calc_label_from_name(f'LARK:{name}'))

    def draw_symbol(self, data: ConjectureData, symbol: Symbol, draw_state: List[str]) -> None:
        if isinstance(symbol, Terminal):
            strategy = self.terminal_strategies[symbol.name]
            draw_state.append(data.draw(strategy))
        else:
            assert isinstance(symbol, NonTerminal)
            data.start_example(self.rule_label(symbol.name))
            expansion = data.draw(self.nonterminal_strategies[symbol.name])
            for e in expansion:
                self.draw_symbol(data, e, draw_state)
                self.gen_ignore(data, draw_state)
            data.stop_example()

    def gen_ignore(self, data: ConjectureData, draw_state: List[str]) -> None:
        if self.ignored_symbols and data.draw_boolean(1 / 4):
            emit = data.draw(st.sampled_from(self.ignored_symbols))
            self.draw_symbol(data, emit, draw_state)

    def calc_has_reusable_values(self, recur: Callable[[Any], bool]) -> bool:
        return True

def check_explicit(name: str) -> Callable[[str], str]:
    def inner(value: str) -> str:
        check_type(str, value, 'value drawn from ' + name)
        return value
    return inner

@cacheable
@defines_strategy(force_reusable_values=True)
def from_lark(
    grammar: Lark,
    *,
    start: Optional[Union[str, List[str]]] = None,
    explicit: Optional[Dict[str, SearchStrategy[str]]] = None,
    alphabet: SearchStrategy[str] = st.characters(codec='utf-8')
) -> SearchStrategy[str]:
    check_type(lark.lark.Lark, grammar, 'grammar')
    if explicit is None:
        explicit = {}
    else:
        check_type(dict, explicit, 'explicit')
        explicit = {k: v.map(check_explicit(f'explicit[{k!r}]={v!r}')) for k, v in explicit.items()}
    return LarkStrategy(grammar, start, explicit, alphabet)
