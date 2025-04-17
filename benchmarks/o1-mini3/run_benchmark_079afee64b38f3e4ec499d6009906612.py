from decimal import ROUND_HALF_EVEN, ROUND_DOWN, Decimal, getcontext, Context
import io
import os
from struct import unpack
import pyperf
from typing import Tuple


def rel_path(*path: str) -> str:
    return os.path.join(os.path.dirname(__file__), *path)


def bench_telco(loops: int, filename: str) -> float:
    getcontext().rounding = ROUND_DOWN
    rates: list[Decimal] = list(map(Decimal, ('0.0013', '0.00894')))
    twodig: Decimal = Decimal('0.01')
    Banker: Context = Context(rounding=ROUND_HALF_EVEN)
    basictax: Decimal = Decimal('0.0675')
    disttax: Decimal = Decimal('0.0341')
    with open(filename, 'rb') as infil_file:
        data: bytes = infil_file.read()
    infil: io.BytesIO = io.BytesIO(data)
    outfil: io.StringIO = io.StringIO()
    start: float = pyperf.perf_counter()
    for _ in range(loops):
        infil.seek(0)
        sumT: Decimal = Decimal('0')
        sumB: Decimal = Decimal('0')
        sumD: Decimal = Decimal('0')
        for i in range(5000):
            datum: bytes = infil.read(8)
            if datum == b'':
                break
            (n,) = unpack('>Q', datum)
            calltype: int = n & 1
            r: Decimal = rates[calltype]
            p: Decimal = Banker.quantize(r * Decimal(n), twodig)
            b: Decimal = (p * basictax).quantize(twodig)
            sumB += b
            t: Decimal = p + b
            if calltype:
                d: Decimal = (p * disttax).quantize(twodig)
                sumD += d
                t += d
            sumT += t
            print(t, file=outfil)
        outfil.seek(0)
        outfil.truncate()
    end: float = pyperf.perf_counter()
    return end - start


if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Telco decimal benchmark'
    filename: str = rel_path('data', 'telco-bench.b')
    runner.bench_time_func('telco', bench_telco, filename)
