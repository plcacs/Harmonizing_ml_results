from typing import Optional
import lark
from hypothesis import strategies as st

def get_terminal_names(terminals: set, rules: list, ignore_names: set) -> set:
    ...

class LarkStrategy(st.SearchStrategy):
    def __init__(self, grammar: lark.lark.Lark, start: Optional[str], explicit: Optional[dict], alphabet: st.SearchStrategy) -> None:
        ...
    
    def do_draw(self, data: ConjectureData) -> str:
        ...
    
    def rule_label(self, name: str) -> str:
        ...
    
    def draw_symbol(self, data: ConjectureData, symbol: lark.grammar.Symbol, draw_state: list) -> None:
        ...
    
    def gen_ignore(self, data: ConjectureData, draw_state: list) -> None:
        ...
    
    def calc_has_reusable_values(self, recur: bool) -> bool:
        ...

def check_explicit(name: str) -> callable:
    ...

def from_lark(grammar: lark.lark.Lark, *, start: Optional[str] = None, explicit: Optional[dict] = None, alphabet: st.SearchStrategy) -> LarkStrategy:
    ...
