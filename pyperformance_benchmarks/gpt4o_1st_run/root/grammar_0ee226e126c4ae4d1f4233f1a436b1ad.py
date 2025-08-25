import pickle
from . import token
from typing import Dict, List, Tuple, Optional

class Grammar(object):
    def __init__(self) -> None:
        self.symbol2number: Dict[str, int] = {}
        self.number2symbol: Dict[int, str] = {}
        self.states: List[List[Tuple[int, int]]] = []
        self.dfas: Dict[int, Tuple[List[Tuple[int, int]], Dict[int, int]]] = {}
        self.labels: List[Tuple[int, Optional[str]]] = [(0, 'EMPTY')]
        self.keywords: Dict[str, int] = {}
        self.tokens: Dict[int, int] = {}
        self.symbol2label: Dict[str, int] = {}
        self.start: int = 256

    def dump(self, filename: str) -> None:
        'Dump the grammar tables to a pickle file.'
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename: str) -> None:
        'Load the grammar tables from a pickle file.'
        with open(filename, 'rb') as f:
            d = pickle.load(f)
        self.__dict__.update(d)

    def loads(self, pkl: bytes) -> None:
        'Load the grammar tables from a pickle bytes object.'
        self.__dict__.update(pickle.loads(pkl))

    def copy(self) -> 'Grammar':
        '\n        Copy the grammar.\n        '
        new = self.__class__()
        for dict_attr in ('symbol2number', 'number2symbol', 'dfas', 'keywords', 'tokens', 'symbol2label'):
            setattr(new, dict_attr, getattr(self, dict_attr).copy())
        new.labels = self.labels[:]
        new.states = self.states[:]
        new.start = self.start
        return new

    def report(self) -> None:
        'Dump the grammar tables to standard output, for debugging.'
        from pprint import pprint
        print('s2n')
        pprint(self.symbol2number)
        print('n2s')
        pprint(self.number2symbol)
        print('states')
        pprint(self.states)
        print('dfas')
        pprint(self.dfas)
        print('labels')
        pprint(self.labels)
        print('start', self.start)

opmap_raw = '\n( LPAR\n) RPAR\n[ LSQB\n] RSQB\n: COLON\n, COMMA\n; SEMI\n+ PLUS\n- MINUS\n* STAR\n/ SLASH\n| VBAR\n& AMPER\n< LESS\n> GREATER\n= EQUAL\n. DOT\n% PERCENT\n` BACKQUOTE\n{ LBRACE\n} RBRACE\n@ AT\n@= ATEQUAL\n== EQEQUAL\n!= NOTEQUAL\n<> NOTEQUAL\n<= LESSEQUAL\n>= GREATEREQUAL\n~ TILDE\n^ CIRCUMFLEX\n<< LEFTSHIFT\n>> RIGHTSHIFT\n** DOUBLESTAR\n+= PLUSEQUAL\n-= MINEQUAL\n*= STAREQUAL\n/= SLASHEQUAL\n%= PERCENTEQUAL\n&= AMPEREQUAL\n|= VBAREQUAL\n^= CIRCUMFLEXEQUAL\n<<= LEFTSHIFTEQUAL\n>>= RIGHTSHIFTEQUAL\n**= DOUBLESTAREQUAL\n// DOUBLESLASH\n//= DOUBLESLASHEQUAL\n-> RARROW\n:= COLONEQUAL\n'
opmap: Dict[str, int] = {}
for line in opmap_raw.splitlines():
    if line:
        (op, name) = line.split()
        opmap[op] = getattr(token, name)
del line, op, name
