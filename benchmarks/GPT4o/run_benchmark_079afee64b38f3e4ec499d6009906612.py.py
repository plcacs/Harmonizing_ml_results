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
    rates = list(map(Decimal, ('0.0013', '0.00894')))
    twodig = Decimal('0.01')
    Banker = Context(rounding=ROUND_HALF_EVEN)
    basictax = Decimal('0.0675')
    disttax = Decimal('0.0341')
    with open(filename, 'rb') as infil:
        data = infil.read()
    infil = io.BytesIO(data)
    outfil = io.StringIO()
    start = pyperf.perf_counter()
    for _ in range(loops):
        infil.seek(0)
        sumT = Decimal('0')
        sumB = Decimal('0')
        sumD = Decimal('0')
        for i in range(5000):
            datum = infil.read(8)
            if (datum == b''):
                break
            (n,) = unpack('>Q', datum)
            calltype = (n & 1)
            r = rates[calltype]
            p = Banker.quantize((r * n), twodig)
            b = (p * basictax)
            b = b.quantize(twodig)
            sumB += b
            t = (p + b)
            if calltype:
                d = (p * disttax)
                d = d.quantize(twodig)
                sumD += d
                t += d
            sumT += t
            print(t, file=outfil)
        outfil.seek(0)
        outfil.truncate()
    return (pyperf.perf_counter() - start)

if (__name__ == '__main__'):
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Telco decimal benchmark'
    filename = rel_path('data', 'telco-bench.b')
    runner.bench_time_func('telco', bench_telco, filename)
