from __future__ import annotations
from typing import Dict, Any, List, Tuple

class PgenGrammar(grammar.Grammar):
    pass

class ParserGenerator:
    filename: str
    stream: Any
    generator: Any
    first: Dict[str, Any]
    dfs: Dict[str, Any]
    startsymbol: str

    def __init__(self, filename: str, stream: Any = None) -> None:
        ...

    def make_grammar(self) -> PgenGrammar:
        ...

    def make_first(self, c: PgenGrammar, name: str) -> Dict[int, int]:
        ...

    def make_label(self, c: PgenGrammar, label: str) -> int:
        ...

    def addfirstsets(self) -> None:
        ...

    def calcfirst(self, name: str) -> None:
        ...

    def parse(self) -> Tuple[Dict[str, Any], str]:
        ...

    def make_dfa(self, start: NFAState, finish: NFAState) -> List[DFAState]:
        ...

    def dump_nfa(self, name: str, start: NFAState, finish: NFAState) -> None:
        ...

    def dump_dfa(self, name: str, dfa: List[DFAState]) -> None:
        ...

    def simplify_dfa(self, dfa: List[DFAState]) -> None:
        ...

    def parse_rhs(self) -> Tuple[NFAState, NFAState]:
        ...

    def parse_alt(self) -> Tuple[NFAState, NFAState]:
        ...

    def parse_item(self) -> Tuple[NFAState, NFAState]:
        ...

    def parse_atom(self) -> Tuple[NFAState, NFAState]:
        ...

    def expect(self, type: int, value: str = None) -> str:
        ...

    def gettoken(self) -> None:
        ...

    def raise_error(self, msg: str, *args: Any) -> None:
        ...

class NFAState:
    arcs: List[Tuple[str, NFAState]]

    def __init__(self) -> None:
        ...

    def addarc(self, next: NFAState, label: str = None) -> None:
        ...

class DFAState:
    nfaset: Dict[NFAState, int]
    isfinal: bool
    arcs: Dict[str, DFAState]

    def __init__(self, nfaset: Dict[NFAState, int], final: NFAState) -> None:
        ...

    def addarc(self, next: DFAState, label: str) -> None:
        ...

    def unifystate(self, old: DFAState, new: DFAState) -> None:
        ...

    def __eq__(self, other: DFAState) -> bool:
        ...
