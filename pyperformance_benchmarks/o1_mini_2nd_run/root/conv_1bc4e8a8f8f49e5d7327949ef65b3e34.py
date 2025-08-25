import re
from typing import Dict, List, Tuple, Optional
from pgen2 import grammar, token


class Converter(grammar.Grammar):
    """Grammar subclass that reads classic pgen output files.

    The run() method reads the tables as produced by the pgen parser
    generator, typically contained in two C files, graminit.h and
    graminit.c. The other methods are for internal use only.

    See the base class for more documentation.
    """

    symbol2number: Dict[str, int]
    number2symbol: Dict[int, str]
    states: List[List[List[Tuple[int, int]]]]
    dfas: Dict[int, Tuple[List[List[Tuple[int, int]]], Dict[int, int]]]
    labels: List[Tuple[int, Optional[str]]]
    keywords: Dict[str, int]
    tokens: Dict[int, int]
    start: int

    def run(self, graminit_h: str, graminit_c: str) -> None:
        """Load the grammar tables from the text files written by pgen."""
        self.parse_graminit_h(graminit_h)
        self.parse_graminit_c(graminit_c)
        self.finish_off()

    def parse_graminit_h(self, filename: str) -> bool:
        """Parse the .h file written by pgen.  (Internal)

        This file is a sequence of #define statements defining the
        nonterminals of the grammar as numbers. We build two tables
        mapping the numbers to names and back.
        """
        try:
            f = open(filename, 'r')
        except OSError as err:
            print(f"Can't open {filename}: {err}")
            return False
        self.symbol2number = {}
        self.number2symbol = {}
        lineno: int = 0
        for line in f:
            lineno += 1
            mo = re.match(r'^#define\s+(\w+)\s+(\d+)$', line)
            if not mo and line.strip():
                print(f"{filename}({lineno}): can't parse {line.strip()}")
            else:
                symbol, number = mo.groups()
                number = int(number)
                assert symbol not in self.symbol2number
                assert number not in self.number2symbol
                self.symbol2number[symbol] = number
                self.number2symbol[number] = symbol
        f.close()
        return True

    def parse_graminit_c(self, filename: str) -> bool:
        """Parse the .c file written by pgen.  (Internal)

        The file looks as follows. The first two lines are always this:

        #include "pgenheaders.h"
        #include "grammar.h"

        After that come four blocks:

        1) one or more state definitions
        2) a table defining dfas
        3) a table defining labels
        4) a struct defining the grammar

        A state definition has the following form:
        - one or more arc arrays, each of the form:
          static arc arcs_<n>_<m>[<k>] = {
                  {<i>, <j>},
                  ...
          };
        - followed by a state array, of the form:
          static state states_<s>[<t>] = {
                  {<k>, arcs_<n>_<m>},
                  ...
          };
        """
        try:
            f = open(filename, 'r')
        except OSError as err:
            print(f"Can't open {filename}: {err}")
            return False
        lineno: int = 0
        try:
            line = next(f)
            lineno += 1
            assert line == '#include "pgenheaders.h"\n', (lineno, line)
            line = next(f)
            lineno += 1
            assert line == '#include "grammar.h"\n', (lineno, line)
            line = next(f)
            lineno += 1
        except StopIteration:
            f.close()
            return False

        allarcs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        states: List[List[List[Tuple[int, int]]]] = []

        while line.startswith('static arc '):
            while line.startswith('static arc '):
                mo = re.match(r'static arc arcs_(\d+)_(\d+)\[(\d+)\] = {$', line)
                assert mo, (lineno, line)
                n, m, k = map(int, mo.groups())
                arcs: List[Tuple[int, int]] = []
                for _ in range(k):
                    try:
                        line = next(f)
                        lineno += 1
                    except StopIteration:
                        f.close()
                        raise AssertionError((lineno, "Unexpected end of file"))
                    mo = re.match(r'\s+{(\d+), (\d+)},$', line)
                    assert mo, (lineno, line)
                    i, j = map(int, mo.groups())
                    arcs.append((i, j))
                try:
                    line = next(f)
                    lineno += 1
                except StopIteration:
                    f.close()
                    raise AssertionError((lineno, "Unexpected end of file"))
                assert line == '};\n', (lineno, line)
                allarcs[(n, m)] = arcs
                try:
                    line = next(f)
                    lineno += 1
                except StopIteration:
                    f.close()
                    break

            mo = re.match(r'static state states_(\d+)\[(\d+)\] = {$', line)
            assert mo, (lineno, line)
            s, t = map(int, mo.groups())
            assert s == len(states), (lineno, line)
            state: List[List[Tuple[int, int]]] = []
            for _ in range(t):
                try:
                    line = next(f)
                    lineno += 1
                except StopIteration:
                    f.close()
                    raise AssertionError((lineno, "Unexpected end of file"))
                mo = re.match(r'\s+{(\d+), arcs_(\d+)_(\d+)},$', line)
                assert mo, (lineno, line)
                k, n, m = map(int, mo.groups())
                arcs = allarcs[(n, m)]
                assert k == len(arcs), (lineno, line)
                state.append(arcs)
            states.append(state)
            try:
                line = next(f)
                lineno += 1
            except StopIteration:
                f.close()
                raise AssertionError((lineno, "Unexpected end of file"))
            assert line == '};\n', (lineno, line)
            try:
                line = next(f)
                lineno += 1
            except StopIteration:
                f.close()
                break

        self.states = states
        dfas: Dict[int, Tuple[List[List[Tuple[int, int]]], Dict[int, int]]] = {}
        mo = re.match(r'static dfa dfas\[(\d+)\] = {$', line)
        assert mo, (lineno, line)
        ndfas = int(mo.group(1))
        for _ in range(ndfas):
            try:
                line = next(f)
                lineno += 1
            except StopIteration:
                f.close()
                raise AssertionError((lineno, "Unexpected end of file"))
            mo = re.match(r'\s+{(\d+), "(\w+)", (\d+), (\d+), states_(\d+),$', line)
            assert mo, (lineno, line)
            number = int(mo.group(1))
            symbol = mo.group(2)
            x = int(mo.group(3))
            y = int(mo.group(4))
            z = int(mo.group(5))
            assert self.symbol2number[symbol] == number, (lineno, line)
            assert self.number2symbol[number] == symbol, (lineno, line)
            assert x == 0, (lineno, line)
            state = states[z]
            assert y == len(state), (lineno, line)
            try:
                line = next(f)
                lineno += 1
            except StopIteration:
                f.close()
                raise AssertionError((lineno, "Unexpected end of file"))
            mo = re.match(r'\s+("(?:\\\d\d\d)*")},$', line)
            assert mo, (lineno, line)
            rawbitset = eval(mo.group(1))  # type: ignore
            first: Dict[int, int] = {}
            for i, c in enumerate(rawbitset):
                byte = ord(c)
                for j in range(8):
                    if byte & (1 << j):
                        first[(i * 8) + j] = 1
            dfas[number] = (state, first)
        try:
            line = next(f)
            lineno += 1
        except StopIteration:
            f.close()
            raise AssertionError((lineno, "Unexpected end of file"))
        assert line == '};\n', (lineno, line)
        self.dfas = dfas

        labels: List[Tuple[int, Optional[str]]] = []
        try:
            line = next(f)
            lineno += 1
        except StopIteration:
            f.close()
            raise AssertionError((lineno, "Unexpected end of file"))
        mo = re.match(r'static label labels\[(\d+)\] = {$', line)
        assert mo, (lineno, line)
        nlabels = int(mo.group(1))
        for _ in range(nlabels):
            try:
                line = next(f)
                lineno += 1
            except StopIteration:
                f.close()
                raise AssertionError((lineno, "Unexpected end of file"))
            mo = re.match(r'\s+{(\d+), (0|"\\w+")},$', line)
            assert mo, (lineno, line)
            x, y = mo.groups()
            x = int(x)
            y = None if y == '0' else eval(y)  # type: ignore
            labels.append((x, y))
        try:
            line = next(f)
            lineno += 1
        except StopIteration:
            f.close()
            raise AssertionError((lineno, "Unexpected end of file"))
        assert line == '};\n', (lineno, line)
        self.labels = labels

        try:
            line = next(f)
            lineno += 1
        except StopIteration:
            f.close()
            raise AssertionError((lineno, "Unexpected end of file"))
        assert line == 'grammar _PyParser_Grammar = {\n', (lineno, line)

        try:
            line = next(f)
            lineno += 1
        except StopIteration:
            f.close()
            raise AssertionError((lineno, "Unexpected end of file"))
        mo = re.match(r'\s+(\d+),$', line)
        assert mo, (lineno, line)
        ndfas = int(mo.group(1))
        assert ndfas == len(self.dfas)
        
        try:
            line = next(f)
            lineno += 1
        except StopIteration:
            f.close()
            raise AssertionError((lineno, "Unexpected end of file"))
        assert line == '\tdfas,\n', (lineno, line)
        
        try:
            line = next(f)
            lineno += 1
        except StopIteration:
            f.close()
            raise AssertionError((lineno, "Unexpected end of file"))
        mo = re.match(r'\s+{(\d+), labels},$', line)
        assert mo, (lineno, line)
        nlabels = int(mo.group(1))
        assert nlabels == len(self.labels), (lineno, line)
        
        try:
            line = next(f)
            lineno += 1
        except StopIteration:
            f.close()
            raise AssertionError((lineno, "Unexpected end of file"))
        mo = re.match(r'\s+(\d+)$', line)
        assert mo, (lineno, line)
        start = int(mo.group(1))
        assert start in self.number2symbol, (lineno, line)
        self.start = start
        
        try:
            line = next(f)
            lineno += 1
        except StopIteration:
            f.close()
            raise AssertionError((lineno, "Unexpected end of file"))
        assert line == '};\n', (lineno, line)
        
        try:
            next(f)
            f.close()
            raise AssertionError((lineno, "Extra data after grammar definition"))
        except StopIteration:
            f.close()

        return True

    def finish_off(self) -> None:
        """Create additional useful structures.  (Internal)."""
        self.keywords: Dict[str, int] = {}
        self.tokens: Dict[int, int] = {}
        for ilabel, (typ, value) in enumerate(self.labels):
            if typ == token.NAME and value is not None:
                self.keywords[value] = ilabel
            elif value is None:
                self.tokens[typ] = ilabel
