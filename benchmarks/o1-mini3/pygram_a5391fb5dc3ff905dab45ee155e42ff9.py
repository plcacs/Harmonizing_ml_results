"""Export the Python grammar and symbols."""
import os
from .pgen2 import token
from .pgen2 import driver
from . import pytree
from typing import Any

_GRAMMAR_FILE: str = os.path.join(os.path.dirname(__file__), 'Grammar.txt')
_PATTERN_GRAMMAR_FILE: str = os.path.join(os.path.dirname(__file__), 'PatternGrammar.txt')

class Symbols:
    def __init__(self, grammar: Any) -> None:
        """
        Initializer.

        Creates an attribute for each grammar symbol (nonterminal),
        whose value is the symbol's type (an int >= 256).
        """
        for name, symbol in grammar.symbol2number.items():
            setattr(self, name, symbol)

python_grammar: Any = driver.load_packaged_grammar('lib2to3', _GRAMMAR_FILE)
python_symbols: Symbols = Symbols(python_grammar)
python_grammar_no_print_statement: Any = python_grammar.copy()
del python_grammar_no_print_statement.keywords['print']
python_grammar_no_print_and_exec_statement: Any = python_grammar_no_print_statement.copy()
del python_grammar_no_print_and_exec_statement.keywords['exec']
pattern_grammar: Any = driver.load_packaged_grammar('lib2to3', _PATTERN_GRAMMAR_FILE)
pattern_symbols: Symbols = Symbols(pattern_grammar)
