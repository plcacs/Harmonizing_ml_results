' Telco Benchmark for measuring the performance of decimal calculations\n\n- http://speleotrove.com/decimal/telco.html\n- http://speleotrove.com/decimal/telcoSpec.html\n\nA call type indicator, c, is set from the bottom (least significant) bit of the duration (hence c is 0 or 1).\nA rate, r, is determined from the call type. Those calls with c=0 have a low r: 0.0013; the remainder (‘distance calls’) have a ‘premium’ r: 0.00894. (The rates are, very roughly, in Euros or dollarates per second.)\nA price, p, for the call is then calculated (p=r*n). This is rounded to exactly 2 fractional digits using round-half-even (Banker’s round to nearest).\nA basic tax, b, is calculated: b=p*0.0675 (6.75%). This is truncated to exactly 2 fractional digits (round-down), and the total basic tax variable is then incremented (sumB=sumB+b).\nFor distance calls: a distance tax, d, is calculated: d=p*0.0341 (3.41%). This is truncated to exactly 2 fractional digits (round-down), and then the total distance tax variable is incremented (sumD=sumD+d).\nThe total price, t, is calculated (t=p+b, and, if a distance call, t=t+d).\nThe total prices variable is incremented (sumT=sumT+t).\nThe total price, t, is converted to a string, s.\n\n'
from decimal import ROUND_HALF_EVEN, ROUND_DOWN, Decimal, getcontext, Context
import io
import os
from struct import unpack
import pyperf
from typing import Any

def rel_path(*path: str) -> str:
    return os.path.join(os.path.dirname(__file__), *path)

def bench_telco(loops: int, filename: str) -> float:
    getcontext().rounding = ROUND_DOWN
    rates: list[Decimal] = list(map(Decimal, ('0.0013', '0.00894')))
    twodig: Decimal = Decimal('0.01')
    Banker: Context = Context(rounding=ROUND_HALF_EVEN)
    basictax: Decimal = Decimal('0.0675')
    disttax: Decimal = Decimal('0.0341')
    with open(filename, 'rb') as infil:
        data: bytes = infil.read()
    infil = io.BytesIO(data)
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
            calltype: int = (n & 1)
            r: Decimal = rates[calltype]
            p: Decimal = Banker.quantize((r * n), twodig)
            b: Decimal = (p * basictax)
            b = b.quantize(twodig)
            sumB += b
            t: Decimal = (p + b)
            if calltype:
                d: Decimal = (p * disttax)
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
    filename: str = rel_path('data', 'telco-bench.b')
    runner.bench_time_func('telco', bench_telco, filename)
