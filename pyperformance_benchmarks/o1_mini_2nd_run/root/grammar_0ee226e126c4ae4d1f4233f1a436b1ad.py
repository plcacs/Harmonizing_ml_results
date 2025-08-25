import pickle
from typing import Dict, List, Tuple, Optional
from . import token


class Grammar(object):
    """
    Pgen parsing tables conversion class.

    Once initialized, this class supplies the grammar tables for the
    parsing engine implemented by parse.py.  The parsing engine
    accesses the instance variables directly.  The class here does not
    provide initialization of the tables; several subclasses exist to
    do this (see the conv and pgen modules).

    The load() method reads the tables from a pickle file, which is
    much faster than the other ways offered by subclasses.  The pickle
    file is written by calling dump() (after loading the grammar
    tables using a subclass).  The report() method prints a readable
    representation of the tables to stdout, for debugging.

    The instance variables are as follows:

    symbol2number -- a dict mapping symbol names to numbers.  Symbol
                     numbers are always 256 or higher, to distinguish
                     them from token numbers, which are between 0 and
                     255 (inclusive).

    number2symbol -- a dict mapping numbers to symbol names;
                     these two are each other's inverse.

    states        -- a list of DFAs, where each DFA is a list of
                     states, each state is a list of arcs, and each
                     arc is a (i, j) pair where i is a label and j is
                     a state number.  The DFA number is the index into
                     this list.  (This name is slightly confusing.)
                     Final states are represented by a special arc of
                     the form (0, j) where j is its own state number.

    dfas          -- a dict mapping symbol numbers to (DFA, first)
                     pairs, where DFA is an item from the states list
                     above, and first is a set of tokens that can
                     begin this grammar rule (represented by a dict
                     whose values are always 1).

    labels        -- a list of (x, y) pairs where x is either a token
                     number or a symbol number, and y is either None
                     or a string; the strings are keywords.  The label
                     number is the index in this list; label numbers
                     are used to mark state transitions (arcs) in the
                     DFAs.

    start         -- the number of the grammar's start symbol.

    keywords      -- a dict mapping keyword strings to arc labels.

    tokens        -- a dict mapping token numbers to arc labels.
    """

    symbol2number: Dict[str, int]
    number2symbol: Dict[int, str]
    states: List[List[List[Tuple[int, int]]]]
    dfas: Dict[int, Tuple[List[List[Tuple[int, int]]], Dict[int, int]]]
    labels: List[Tuple[int, Optional[str]]]
    keywords: Dict[str, int]
    tokens: Dict[int, int]
    symbol2label: Dict[str, int]
    start: int

    def __init__(self) -> None:
        self.symbol2number = {}
        self.number2symbol = {}
        self.states = []
        self.dfas = {}
        self.labels = [(0, 'EMPTY')]
        self.keywords = {}
        self.tokens = {}
        self.symbol2label = {}
        self.start = 256

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
        '''
        Copy the grammar.
        '''
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


opmap_raw: str = '\n( LPAR\n) RPAR\n[ LSQB\n] RSQB\n: COLON\n, COMMA\n; SEMI\n+ PLUS\n- MINUS\n* STAR\n/ SLASH\n| VBAR\n& AMPER\n< LESS\n> GREATER\n= EQUAL\n. DOT\n% PERCENT\n` BACKQUOTE\n{ LBRACE\n} RBRACE\n@ AT\n@= ATEQUAL\n== EQEQUAL\n!= NOTEQUAL\n<> NOTEQUAL\n<= LESSEQUAL\n>= GREATEREQUAL\n~ TILDE\n^ CIRCUMFLEX\n<< LEFTSHIFT\n>> RIGHTSHIFT\n** DOUBLESTAR\n+= PLUSEQUAL\n-= MINEQUAL\n*= STAREQUAL\n/= SLASHEQUAL\n%= PERCENTEQUAL\n&= AMPEREQUAL\n|= VBAREQUAL\n^= CIRCUMFLEXEQUAL\n<<= LEFTSHIFTEQUAL\n>>= RIGHTSHIFTEQUAL\n**= DOUBLESTAREQUAL\n// DOUBLESLASH\n//= DOUBLESLASHEQUAL\n-> RARROW\n:= COLONEQUAL\n'
opmap: Dict[str, int] = {}
for line in opmap_raw.splitlines():
    if line:
        op, name = line.split()
        opmap[op] = getattr(token, name)
del line, op, name
