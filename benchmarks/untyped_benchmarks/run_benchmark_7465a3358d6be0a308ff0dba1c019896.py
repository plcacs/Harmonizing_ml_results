
'\nCalculate `factorial` using the decimal module.\n\n- 2024-06-14: Michael Droettboom copied this from\n  Modules/_decimal/tests/bench.py in the CPython source and adapted to use\n  pyperf.\n'
import decimal
import pyperf

def factorial(n, m):
    if (n > m):
        return factorial(m, n)
    elif (m == 0):
        return 1
    elif (n == m):
        return n
    else:
        return (factorial(n, ((n + m) // 2)) * factorial((((n + m) // 2) + 1), m))

def bench_decimal_factorial():
    c = decimal.getcontext()
    c.prec = decimal.MAX_PREC
    c.Emax = decimal.MAX_EMAX
    c.Emin = decimal.MIN_EMIN
    for n in [10000, 100000]:
        _ = factorial(decimal.Decimal(n), 0)
if (__name__ == '__main__'):
    runner = pyperf.Runner()
    runner.metadata['description'] = 'decimal_factorial benchmark'
    args = runner.parse_args()
    runner.bench_func('decimal_factorial', bench_decimal_factorial)
