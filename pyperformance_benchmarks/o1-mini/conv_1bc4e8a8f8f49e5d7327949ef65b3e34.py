import re
from typing import Dict, List, Optional, Tuple
from pgen2 import grammar, token


class Converter(grammar.Grammar):
    'Grammar subclass that reads classic pgen output files.\n\n    The run() method reads the tables as produced by the pgen parser\n    generator, typically contained in two C files, graminit.h and\n    graminit.c.  The other methods are for internal use only.\n\n    See the base class for more documentation.\n\n    '

    symbol2number: Dict[str, int]
    number2symbol: Dict[int, str]
    states: List[List[List[Tuple[int, int]]]]
    dfas: Dict[int, Tuple[List[List[Tuple[int, int]]], Dict[int, int]]]
    labels: List[Tuple[int, Optional[str]]]
    keywords: Dict[str, int]
    tokens_dict: Dict[int, int]

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
            print(f"Can't open {filename}: {err}")
            return False
        self.symbol2number = {}
        self.number2symbol = {}
        lineno = 0
        for line in f:
            lineno += 1
            mo = re.match(r'^#define\s+(\w+)\s+(\d+)$', line)
            if not mo and line.strip():
                print(f"{filename}({lineno}): can't parse {line.strip()}")
            else:
                symbol, number_str = mo.groups()
                number = int(number_str)
                assert symbol not in self.symbol2number
                assert number not in self.number2symbol
                self.symbol2number[symbol] = number
                self.number2symbol[number] = symbol
        f.close()
        return True

    def parse_graminit_c(self, filename: str) -> bool:
        'Parse the .c file written by pgen.  (Internal)\n\n        The file looks as follows.  The first two lines are always this:\n\n        #include "pgenheaders.h"\n        #include "grammar.h"\n\n        After that come four blocks:\n\n        1) one or more state definitions\n        2) a table defining dfas\n        3) a table defining labels\n        4) a struct defining the grammar\n\n        A state definition has the following form:\n        - one or more arc arrays, each of the form:\n          static arc arcs_<n>_<m>[<k>] = {\n                  {<i>, <j>},\n                  ...\n          };\n        - followed by a state array, of the form:\n          static state states_<s>[<t>] = {\n                  {<k>, arcs_<n>_<m>},\n                  ...\n          };\n\n        '
        try:
            f = open(filename)
        except OSError as err:
            print(f"Can't open {filename}: {err}")
            return False
        lineno = 0
        try:
            line = next(f)
            lineno += 1
            assert line == '#include "pgenheaders.h"\n', (lineno, line)
            line = next(f)
            lineno += 1
            assert line == '#include "grammar.h"\n', (lineno, line)
            line = next(f)
            lineno += 1
            allarcs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
            states: List[List[List[Tuple[int, int]]]] = []
            while line.startswith('static arc '):
                while line.startswith('static arc '):
                    mo = re.match(r'static arc arcs_(\d+)_(\d+)\[(\d+)\] = {$', line)
                    assert mo, (lineno, line)
                    n, m, k = map(int, mo.groups())
                    arcs: List[Tuple[int, int]] = []
                    for _ in range(k):
                        line = next(f)
                        lineno += 1
                        mo = re.match(r'\s+{(\d+), (\d+)},$', line)
                        assert mo, (lineno, line)
                        i, j = map(int, mo.groups())
                        arcs.append((i, j))
                    line = next(f)
                    lineno += 1
                    assert line == '};\n', (lineno, line)
                    allarcs[(n, m)] = arcs
                    line = next(f)
                    lineno += 1
                mo = re.match(r'static state states_(\d+)\[(\d+)\] = {$', line)
                assert mo, (lineno, line)
                s, t = map(int, mo.groups())
                assert s == len(states), (lineno, line)
                state: List[List[Tuple[int, int]]] = []
                for _ in range(t):
                    line = next(f)
                    lineno += 1
                    mo = re.match(r'\s+{(\d+), arcs_(\d+)_(\d+)},$', line)
                    assert mo, (lineno, line)
                    k, n, m = map(int, mo.groups())
                    arcs = allarcs[(n, m)]
                    assert k == len(arcs), (lineno, line)
                    state.append(arcs)
                states.append(state)
                line = next(f)
                lineno += 1
                assert line == '};\n', (lineno, line)
                line = next(f)
                lineno += 1
            self.states = states
            dfas: Dict[int, Tuple[List[List[Tuple[int, int]]], Dict[int, int]]] = {}
            mo = re.match(r'static dfa dfas\[(\d+)\] = {$', line)
            assert mo, (lineno, line)
            ndfas = int(mo.group(1))
            for _ in range(ndfas):
                line = next(f)
                lineno += 1
                mo = re.match(r'\s+{(\d+), "(\w+)", (\d+), (\d+), states_(\d+),$', line)
                assert mo, (lineno, line)
                number_str, symbol, x, y, z = mo.groups()
                number, x, y, z = int(number_str), int(x), int(y), int(z)
                assert self.symbol2number[symbol] == number, (lineno, line)
                assert self.number2symbol[number] == symbol, (lineno, line)
                assert x == 0, (lineno, line)
                state = states[z]
                assert y == len(state), (lineno, line)
                line = next(f)
                lineno += 1
                mo = re.match(r'\s+("(?:\\\d\d\d)*")},$', line)
                assert mo, (lineno, line)
                first: Dict[int, int] = {}
                rawbitset = eval(mo.group(1))
                for i, c in enumerate(rawbitset):
                    byte = ord(c)
                    for j in range(8):
                        if byte & (1 << j):
                            first[(i * 8) + j] = 1
                dfas[number] = (state, first)
            line = next(f)
            lineno += 1
            assert line == '};\n', (lineno, line)
            self.dfas = dfas
            labels: List[Tuple[int, Optional[str]]] = []
            line = next(f)
            lineno += 1
            mo = re.match(r'static label labels\[(\d+)\] = {$', line)
            assert mo, (lineno, line)
            nlabels = int(mo.group(1))
            for _ in range(nlabels):
                line = next(f)
                lineno += 1
                mo = re.match(r'\s+{(\d+), (0|"\\w+")},$', line)
                assert mo, (lineno, line)
                x, y = mo.groups()
                x = int(x)
                if y == '0':
                    y_val: Optional[str] = None
                else:
                    y_val = eval(y)
                labels.append((x, y_val))
            line = next(f)
            lineno += 1
            assert line == '};\n', (lineno, line)
            self.labels = labels
            line = next(f)
            lineno += 1
            assert line == 'grammar _PyParser_Grammar = {\n', (lineno, line)
            line = next(f)
            lineno += 1
            mo = re.match(r'\s+(\d+),$', line)
            assert mo, (lineno, line)
            ndfas_check = int(mo.group(1))
            assert ndfas_check == len(self.dfas)
            line = next(f)
            lineno += 1
            assert line == '\tdfas,\n', (lineno, line)
            line = next(f)
            lineno += 1
            mo = re.match(r'\s+{(\d+), labels},$', line)
            assert mo, (lineno, line)
            nlabels_check = int(mo.group(1))
            assert nlabels_check == len(self.labels)
            line = next(f)
            lineno += 1
            mo = re.match(r'\s+(\d+)$', line)
            assert mo, (lineno, line)
            start = int(mo.group(1))
            assert start in self.number2symbol, (lineno, line)
            self.start = start
            line = next(f)
            lineno += 1
            assert line == '};\n', (lineno, line)
            try:
                line = next(f)
                lineno += 1
                assert False, (lineno, line)
            except StopIteration:
                pass
        except StopIteration:
            print(f"Unexpected end of file {filename} at line {lineno}")
            return False
        finally:
            f.close()
        return True

    def finish_off(self) -> None:
        'Create additional useful structures.  (Internal).'
        self.keywords = {}
        self.tokens_dict = {}
        for ilabel, (type_, value) in enumerate(self.labels):
            if type_ == token.NAME and value is not None:
                self.keywords[value] = ilabel
            elif value is None:
                self.tokens_dict[type_] = ilabel
