import re
from pgen2 import grammar, token
from typing import Dict, List, Tuple, Optional

class Converter(grammar.Grammar):
    'Grammar subclass that reads classic pgen output files.\n\n    The run() method reads the tables as produced by the pgen parser\n    generator, typically contained in two C files, graminit.h and\n    graminit.c.  The other methods are for internal use only.\n\n    See the base class for more documentation.\n\n    '

    def run(self, graminit_h: str, graminit_c: str) -> None:
        'Load the grammar tables from the text files written by pgen.'
        self.parse_graminit_h(graminit_h)
        self.parse_graminit_c(graminit_c)
        self.finish_off()

    def parse_graminit_h(self, filename: str) -> bool:
        'Parse the .h file written by pgen.  (Internal)\n\n        This file is a sequence of #define statements defining the\n        nonterminals of the grammar as numbers.  We build two tables\n        mapping the numbers to names and back.\n\n        '
        try:
            f = open(filename)
        except OSError as err:
            print(("Can't open %s: %s" % (filename, err)))
            return False
        self.symbol2number: Dict[str, int] = {}
        self.number2symbol: Dict[int, str] = {}
        lineno = 0
        for line in f:
            lineno += 1
            mo = re.match('^#define\\s+(\\w+)\\s+(\\d+)$', line)
            if ((not mo) and line.strip()):
                print(("%s(%s): can't parse %s" % (filename, lineno, line.strip())))
            else:
                (symbol, number) = mo.groups()
                number = int(number)
                assert (symbol not in self.symbol2number)
                assert (number not in self.number2symbol)
                self.symbol2number[symbol] = number
                self.number2symbol[number] = symbol
        return True

    def parse_graminit_c(self, filename: str) -> bool:
        'Parse the .c file written by pgen.  (Internal)\n\n        The file looks as follows.  The first two lines are always this:\n\n        #include "pgenheaders.h"\n        #include "grammar.h"\n\n        After that come four blocks:\n\n        1) one or more state definitions\n        2) a table defining dfas\n        3) a table defining labels\n        4) a struct defining the grammar\n\n        A state definition has the following form:\n        - one or more arc arrays, each of the form:\n          static arc arcs_<n>_<m>[<k>] = {\n                  {<i>, <j>},\n                  ...\n          };\n        - followed by a state array, of the form:\n          static state states_<s>[<t>] = {\n                  {<k>, arcs_<n>_<m>},\n                  ...\n          };\n\n        '
        try:
            f = open(filename)
        except OSError as err:
            print(("Can't open %s: %s" % (filename, err)))
            return False
        lineno = 0
        (lineno, line) = ((lineno + 1), next(f))
        assert (line == '#include "pgenheaders.h"\n'), (lineno, line)
        (lineno, line) = ((lineno + 1), next(f))
        assert (line == '#include "grammar.h"\n'), (lineno, line)
        (lineno, line) = ((lineno + 1), next(f))
        allarcs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        states: List[List[List[Tuple[int, int]]]] = []
        while line.startswith('static arc '):
            while line.startswith('static arc '):
                mo = re.match('static arc arcs_(\\d+)_(\\d+)\\[(\\d+)\\] = {$', line)
                assert mo, (lineno, line)
                (n, m, k) = list(map(int, mo.groups()))
                arcs: List[Tuple[int, int]] = []
                for _ in range(k):
                    (lineno, line) = ((lineno + 1), next(f))
                    mo = re.match('\\s+{(\\d+), (\\d+)},$', line)
                    assert mo, (lineno, line)
                    (i, j) = list(map(int, mo.groups()))
                    arcs.append((i, j))
                (lineno, line) = ((lineno + 1), next(f))
                assert (line == '};\n'), (lineno, line)
                allarcs[(n, m)] = arcs
                (lineno, line) = ((lineno + 1), next(f))
            mo = re.match('static state states_(\\d+)\\[(\\d+)\\] = {$', line)
            assert mo, (lineno, line)
            (s, t) = list(map(int, mo.groups()))
            assert (s == len(states)), (lineno, line)
            state: List[List[Tuple[int, int]]] = []
            for _ in range(t):
                (lineno, line) = ((lineno + 1), next(f))
                mo = re.match('\\s+{(\\d+), arcs_(\\d+)_(\\d+)},$', line)
                assert mo, (lineno, line)
                (k, n, m) = list(map(int, mo.groups()))
                arcs = allarcs[(n, m)]
                assert (k == len(arcs)), (lineno, line)
                state.append(arcs)
            states.append(state)
            (lineno, line) = ((lineno + 1), next(f))
            assert (line == '};\n'), (lineno, line)
            (lineno, line) = ((lineno + 1), next(f))
        self.states = states
        dfas: Dict[int, Tuple[List[List[Tuple[int, int]]], Dict[int, int]]] = {}
        mo = re.match('static dfa dfas\\[(\\d+)\\] = {$', line)
        assert mo, (lineno, line)
        ndfas = int(mo.group(1))
        for i in range(ndfas):
            (lineno, line) = ((lineno + 1), next(f))
            mo = re.match('\\s+{(\\d+), "(\\w+)", (\\d+), (\\d+), states_(\\d+),$', line)
            assert mo, (lineno, line)
            symbol = mo.group(2)
            (number, x, y, z) = list(map(int, mo.group(1, 3, 4, 5)))
            assert (self.symbol2number[symbol] == number), (lineno, line)
            assert (self.number2symbol[number] == symbol), (lineno, line)
            assert (x == 0), (lineno, line)
            state = states[z]
            assert (y == len(state)), (lineno, line)
            (lineno, line) = ((lineno + 1), next(f))
            mo = re.match('\\s+("(?:\\\\\\d\\d\\d)*")},$', line)
            assert mo, (lineno, line)
            first: Dict[int, int] = {}
            rawbitset = eval(mo.group(1))
            for (i, c) in enumerate(rawbitset):
                byte = ord(c)
                for j in range(8):
                    if (byte & (1 << j)):
                        first[((i * 8) + j)] = 1
            dfas[number] = (state, first)
        (lineno, line) = ((lineno + 1), next(f))
        assert (line == '};\n'), (lineno, line)
        self.dfas = dfas
        labels: List[Tuple[int, Optional[str]]] = []
        (lineno, line) = ((lineno + 1), next(f))
        mo = re.match('static label labels\\[(\\d+)\\] = {$', line)
        assert mo, (lineno, line)
        nlabels = int(mo.group(1))
        for i in range(nlabels):
            (lineno, line) = ((lineno + 1), next(f))
            mo = re.match('\\s+{(\\d+), (0|"\\w+")},$', line)
            assert mo, (lineno, line)
            (x, y) = mo.groups()
            x = int(x)
            if (y == '0'):
                y = None
            else:
                y = eval(y)
            labels.append((x, y))
        (lineno, line) = ((lineno + 1), next(f))
        assert (line == '};\n'), (lineno, line)
        self.labels = labels
        (lineno, line) = ((lineno + 1), next(f))
        assert (line == 'grammar _PyParser_Grammar = {\n'), (lineno, line)
        (lineno, line) = ((lineno + 1), next(f))
        mo = re.match('\\s+(\\d+),$', line)
        assert mo, (lineno, line)
        ndfas = int(mo.group(1))
        assert (ndfas == len(self.dfas))
        (lineno, line) = ((lineno + 1), next(f))
        assert (line == '\tdfas,\n'), (lineno, line)
        (lineno, line) = ((lineno + 1), next(f))
        mo = re.match('\\s+{(\\d+), labels},$', line)
        assert mo, (lineno, line)
        nlabels = int(mo.group(1))
        assert (nlabels == len(self.labels)), (lineno, line)
        (lineno, line) = ((lineno + 1), next(f))
        mo = re.match('\\s+(\\d+)$', line)
        assert mo, (lineno, line)
        start = int(mo.group(1))
        assert (start in self.number2symbol), (lineno, line)
        self.start = start
        (lineno, line) = ((lineno + 1), next(f))
        assert (line == '};\n'), (lineno, line)
        try:
            (lineno, line) = ((lineno + 1), next(f))
        except StopIteration:
            pass
        else:
            assert 0, (lineno, line)

    def finish_off(self) -> None:
        'Create additional useful structures.  (Internal).'
        self.keywords: Dict[str, int] = {}
        self.tokens: Dict[int, int] = {}
        for (ilabel, (type, value)) in enumerate(self.labels):
            if ((type == token.NAME) and (value is not None)):
                self.keywords[value] = ilabel
            elif (value is None):
                self.tokens[type] = ilabel
