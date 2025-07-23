"""
----------------
hypothesis[lark]
----------------

This extra can be used to generate strings matching any context-free grammar,
using the `Lark parser library <https://github.com/lark-parser/lark>`_.

It currently only supports Lark's native EBNF syntax, but we plan to extend
this to support other common syntaxes such as ANTLR and :rfc:`5234` ABNF.
Lark already `supports loading grammars
<https://lark-parser.readthedocs.io/en/stable/tools.html#importing-grammars-from-nearley-js>`_
from `nearley.js <https://nearley.js.org/>`_, so you may not have to write
your own at all.
"""
from inspect import signature
from typing import Optional, Dict, Set, List, Iterable, Any, Callable
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
__all__ = ['from_lark']

def get_terminal_names(
    terminals: List[TerminalDef],
    rules: List[Rule],
    ignore_names: Iterable[str]
) -> Set[str]:
    """Get names of all terminals in the grammar.

    The arguments are the results of calling ``Lark.grammar.compile()``,
    so you would think that the ``terminals`` and ``ignore_names`` would
    have it all... but they omit terminals created with ``@declare``,
    which appear only in the expansion(s) of nonterminals.
    """
    names: Set[str] = {t.name for t in terminals} | set(ignore_names)
    for rule in rules:
        names |= {t.name for t in rule.expansion if isinstance(t, Terminal)}
    return names

class LarkStrategy(st.SearchStrategy[str]):
    """Low-level strategy implementation wrapping a Lark grammar.

    See ``from_lark`` for details.
    """

    def __init__(
        self,
        grammar: Lark,
        start: Optional[List[str]],
        explicit: Dict[str, st.SearchStrategy[Any]],
        alphabet: st.SearchStrategy[str]
    ) -> None:
        assert isinstance(grammar, lark.lark.Lark)
        start = grammar.options.start if start is None else [start]
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
        disallowed: Set[str] = set()
        self.terminal_strategies: Dict[str, st.SearchStrategy[Any]] = {}
        for t in terminals:
            self.names_to_symbols[t.name] = Terminal(t.name)
            s = st.from_regex(t.pattern.to_regexp(), fullmatch=True, alphabet=alphabet)
            try:
                s.validate()
            except IncompatibleWithAlphabet:
                disallowed.add(t.name)
            else:
                self.terminal_strategies[t.name] = s
        self.ignored_symbols: tuple = tuple((self.names_to_symbols[n] for n in ignore_names))
        all_terminals: Set[str] = get_terminal_names(terminals, rules, ignore_names)
        unknown_explicit = sorted(set(explicit) - all_terminals)
        if unknown_explicit:
            raise InvalidArgument(
                f'The following arguments were passed as explicit_strategies, but there is no {unknown_explicit} terminal production in this grammar.'
            )
        missing_declared = sorted(all_terminals - {t.name for t in terminals} - set(explicit))
        if missing_declared:
            raise InvalidArgument(
                f'Undefined terminal{"s" * (len(missing_declared) > 1)} {sorted(missing_declared)!r}. Generation does not currently support use of %declare unless you pass `explicit`, a dict of names-to-strategies, such as `{{{missing_declared[0]!r}: st.just("")}}}}`'
            )
        self.terminal_strategies.update(explicit)
        nonterminals: Dict[str, List[tuple]] = {}
        for rule in rules:
            if disallowed.isdisjoint((r.name for r in rule.expansion)):
                nonterminals.setdefault(rule.origin.name, []).append(tuple(rule.expansion))
        allowed_rules: Set[str] = {*self.terminal_strategies, *nonterminals}
        while dict(nonterminals) != (nonterminals := {
            k: clean for k, v in nonterminals.items() if (clean := [x for x in v if all((r.name in allowed_rules for r in x))])
        }):
            allowed_rules = {*self.terminal_strategies, *nonterminals}
        if set(start).isdisjoint(allowed_rules):
            raise InvalidArgument(
                f'No start rule {tuple(start)} is allowed by alphabet={alphabet!r}'
            )
        self.start: st.SearchStrategy[Any] = st.sampled_from([
            self.names_to_symbols[s] for s in start if s in allowed_rules
        ])
        self.nonterminal_strategies: Dict[str, st.SearchStrategy[tuple]] = {
            k: st.sampled_from(sorted(v, key=len)) for k, v in nonterminals.items()
        }
        self.__rule_labels: Dict[str, int] = {}

    def do_draw(self, data: ConjectureData) -> str:
        state: List[str] = []
        start_symbol: Symbol = data.draw(self.start)
        self.draw_symbol(data, start_symbol, state)
        return ''.join(state)

    def rule_label(self, name: str) -> int:
        try:
            return self.__rule_labels[name]
        except KeyError:
            return self.__rule_labels.setdefault(name, calc_label_from_name(f'LARK:{name}'))

    def draw_symbol(
        self,
        data: ConjectureData,
        symbol: Symbol,
        draw_state: List[str]
    ) -> None:
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

    def gen_ignore(
        self,
        data: ConjectureData,
        draw_state: List[str]
    ) -> None:
        if self.ignored_symbols and data.draw_boolean(1 / 4):
            emit: Symbol = data.draw(st.sampled_from(self.ignored_symbols))
            self.draw_symbol(data, emit, draw_state)

    def calc_has_reusable_values(self, recur: bool) -> bool:
        return True

def check_explicit(name: str) -> Callable[[Any], Any]:
    def inner(value: Any) -> Any:
        check_type(str, value, 'value drawn from ' + name)
        return value
    return inner

@cacheable
@defines_strategy(force_reusable_values=True)
def from_lark(
    grammar: Lark,
    *,
    start: Optional[str] = None,
    explicit: Optional[Dict[str, st.SearchStrategy[Any]]] = None,
    alphabet: st.SearchStrategy[str] = st.characters(codec='utf-8')
) -> st.SearchStrategy[str]:
    """A strategy for strings accepted by the given context-free grammar.

    ``grammar`` must be a ``Lark`` object, which wraps an EBNF specification.
    The Lark EBNF grammar reference can be found
    `here <https://lark-parser.readthedocs.io/en/latest/grammar.html>`_.

    ``from_lark`` will automatically generate strings matching the
    nonterminal ``start`` symbol in the grammar, which was supplied as an
    argument to the Lark class.  To generate strings matching a different
    symbol, including terminals, you can override this by passing the
    ``start`` argument to ``from_lark``.  Note that Lark may remove unreachable
    productions when the grammar is compiled, so you should probably pass the
    same value for ``start`` to both.

    Currently ``from_lark`` does not support grammars that need custom lexing.
    Any lexers will be ignored, and any undefined terminals from the use of
    ``%declare`` will result in generation errors.  To define strategies for
    such terminals, pass a dictionary mapping their name to a corresponding
    strategy as the ``explicit`` argument.

    The :pypi:`hypothesmith` project includes a strategy for Python source,
    based on a grammar and careful post-processing.
    """
    check_type(Lark, grammar, 'grammar')
    if explicit is None:
        explicit_dict: Dict[str, st.SearchStrategy[Any]] = {}
    else:
        check_type(dict, explicit, 'explicit')
        explicit_dict = {
            k: v.map(check_explicit(f'explicit[{k!r}]={v!r}')) for k, v in explicit.items()
        }
    return LarkStrategy(grammar, start, explicit_dict, alphabet)
