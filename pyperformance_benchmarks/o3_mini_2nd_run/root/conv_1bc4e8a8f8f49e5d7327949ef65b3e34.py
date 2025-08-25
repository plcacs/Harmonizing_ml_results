import re
from typing import Tuple, List, Dict, Optional, IO
from pgen2 import grammar, token

class Converter(grammar.Grammar):
    def run(self, graminit_h: str, graminit_c: str) -> None:
        self.parse_graminit_h(graminit_h)
        self.parse_graminit_c(graminit_c)
        self.finish_off()

    def parse_graminit_h(self, filename: str) -> bool:
        try:
            f: IO[str] = open(filename)
        except OSError as err:
            print(f"Can't open {filename}: {err}")
            return False
        self.symbol2number: Dict[str, int] = {}
        self.number2symbol: Dict[int, str] = {}
        lineno: int = 0
        for line in f:
            lineno += 1
            mo = re.match(r'^#define\s+(\w+)\s+(\d+)$', line)
            if (not mo) and line.strip():
                print(f"{filename}({lineno}): can't parse {line.strip()}")
            else:
                if mo:
                    symbol, number = mo.groups()
                    number_int: int = int(number)
                    assert symbol not in self.symbol2number
                    assert number_int not in self.number2symbol
                    self.symbol2number[symbol] = number_int
                    self.number2symbol[number_int] = symbol
        f.close()
        return True

    def parse_graminit_c(self, filename: str) -> None:
        try:
            f: IO[str] = open(filename)
        except OSError as err:
            print(f"Can't open {filename}: {err}")
            return
        lineno: int = 0
        lineno += 1
        line: str = next(f)
        assert line == '#include "pgenheaders.h"\n', f"{lineno}, {line}"
        lineno += 1
        line = next(f)
        assert line == '#include "grammar.h"\n', f"{lineno}, {line}"
        lineno += 1
        line = next(f)
        allarcs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        states: List[List[List[Tuple[int, int]]]] = []
        while line.startswith('static arc '):
            while line.startswith('static arc '):
                mo = re.match(r'static arc arcs_(\d+)_(\d+)\[(\d+)\] = {$', line)
                assert mo, f"{lineno}, {line}"
                n, m, k = list(map(int, mo.groups()))
                arcs: List[Tuple[int, int]] = []
                for _ in range(k):
                    lineno += 1
                    line = next(f)
                    mo = re.match(r'\s+{(\d+), (\d+)},$', line)
                    assert mo, f"{lineno}, {line}"
                    i, j = list(map(int, mo.groups()))
                    arcs.append((i, j))
                lineno += 1
                line = next(f)
                assert line == '};\n', f"{lineno}, {line}"
                allarcs[(n, m)] = arcs
                lineno += 1
                line = next(f)
            mo = re.match(r'static state states_(\d+)\[(\d+)\] = {$', line)
            assert mo, f"{lineno}, {line}"
            s, t = list(map(int, mo.groups()))
            assert s == len(states), f"{lineno}, {line}"
            state: List[List[Tuple[int, int]]] = []
            for _ in range(t):
                lineno += 1
                line = next(f)
                mo = re.match(r'\s+{(\d+), arcs_(\d+)_(\d+)},$', line)
                assert mo, f"{lineno}, {line}"
                k_val, n_val, m_val = list(map(int, mo.groups()))
                arcs = allarcs[(n_val, m_val)]
                assert k_val == len(arcs), f"{lineno}, {line}"
                state.append(arcs)
            states.append(state)
            lineno += 1
            line = next(f)
            assert line == '};\n', f"{lineno}, {line}"
            lineno += 1
            line = next(f)
        self.states = states
        dfas: Dict[int, Tuple[List[List[Tuple[int, int]]], Dict[int, int]]] = {}
        mo = re.match(r'static dfa dfas\[(\d+)\] = {$', line)
        assert mo, f"{lineno}, {line}"
        ndfas: int = int(mo.group(1))
        for i in range(ndfas):
            lineno += 1
            line = next(f)
            mo = re.match(r'\s+{(\d+), "(\w+)", (\d+), (\d+), states_(\d+),$', line)
            assert mo, f"{lineno}, {line}"
            symbol = mo.group(2)
            number, x, y, z = list(map(int, (mo.group(1), mo.group(3), mo.group(4), mo.group(5))))
            assert self.symbol2number[symbol] == number, f"{lineno}, {line}"
            assert self.number2symbol[number] == symbol, f"{lineno}, {line}"
            assert x == 0, f"{lineno}, {line}"
            state = states[z]
            assert y == len(state), f"{lineno}, {line}"
            lineno += 1
            line = next(f)
            mo = re.match(r'\s+("(?:\\\d\d\d)*")},$', line)
            assert mo, f"{lineno}, {line}"
            first: Dict[int, int] = {}
            rawbitset: str = eval(mo.group(1))
            for i_bit, c in enumerate(rawbitset):
                byte = ord(c)
                for j_bit in range(8):
                    if byte & (1 << j_bit):
                        first[(i_bit * 8) + j_bit] = 1
            dfas[number] = (state, first)
        lineno += 1
        line = next(f)
        assert line == '};\n', f"{lineno}, {line}"
        self.dfas = dfas
        labels: List[Tuple[int, Optional[str]]] = []
        lineno += 1
        line = next(f)
        mo = re.match(r'static label labels\[(\d+)\] = {$', line)
        assert mo, f"{lineno}, {line}"
        nlabels: int = int(mo.group(1))
        for i in range(nlabels):
            lineno += 1
            line = next(f)
            mo = re.match(r'\s+{(\d+), (0|"\w+")},$', line)
            assert mo, f"{lineno}, {line}"
            x_val, y_val = mo.groups()
            x_val_int: int = int(x_val)
            if y_val == '0':
                y_parsed: Optional[str] = None
            else:
                y_parsed = eval(y_val)
            labels.append((x_val_int, y_parsed))
        lineno += 1
        line = next(f)
        assert line == '};\n', f"{lineno}, {line}"
        self.labels = labels
        lineno += 1
        line = next(f)
        assert line == 'grammar _PyParser_Grammar = {\n', f"{lineno}, {line}"
        lineno += 1
        line = next(f)
        mo = re.match(r'\s+(\d+),$', line)
        assert mo, f"{lineno}, {line}"
        ndfas = int(mo.group(1))
        assert ndfas == len(self.dfas)
        lineno += 1
        line = next(f)
        assert line == '\tdfas,\n', f"{lineno}, {line}"
        lineno += 1
        line = next(f)
        mo = re.match(r'\s+{(\d+), labels},$', line)
        assert mo, f"{lineno}, {line}"
        nlabels = int(mo.group(1))
        assert nlabels == len(self.labels), f"{lineno}, {line}"
        lineno += 1
        line = next(f)
        mo = re.match(r'\s+(\d+)$', line)
        assert mo, f"{lineno}, {line}"
        start: int = int(mo.group(1))
        assert start in self.number2symbol, f"{lineno}, {line}"
        self.start = start
        lineno += 1
        line = next(f)
        assert line == '};\n', f"{lineno}, {line}"
        try:
            lineno += 1
            line = next(f)
        except StopIteration:
            pass
        else:
            assert 0, f"{lineno}, {line}"
        f.close()

    def finish_off(self) -> None:
        self.keywords: Dict[str, int] = {}
        self.tokens: Dict[int, int] = {}
        for ilabel, (itype, value) in enumerate(self.labels):
            if itype == token.NAME and value is not None:
                self.keywords[value] = ilabel
            elif value is None:
                self.tokens[itype] = ilabel