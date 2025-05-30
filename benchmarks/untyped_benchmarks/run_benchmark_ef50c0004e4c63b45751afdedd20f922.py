
'\nCalculating some of the digits of π.\n\nThis benchmark stresses big integer arithmetic.\n\nAdapted from code on:\nhttp://benchmarksgame.alioth.debian.org/\n'
import itertools
import pyperf
DEFAULT_DIGITS = 2000
icount = itertools.count
islice = itertools.islice

def gen_x():
    return map((lambda k: (k, ((4 * k) + 2), 0, ((2 * k) + 1))), icount(1))

def compose(a, b):
    (aq, ar, as_, at) = a
    (bq, br, bs, bt) = b
    return ((aq * bq), ((aq * br) + (ar * bt)), ((as_ * bq) + (at * bs)), ((as_ * br) + (at * bt)))

def extract(z, j):
    (q, r, s, t) = z
    return (((q * j) + r) // ((s * j) + t))

def gen_pi_digits():
    z = (1, 0, 0, 1)
    x = gen_x()
    while 1:
        y = extract(z, 3)
        while (y != extract(z, 4)):
            z = compose(z, next(x))
            y = extract(z, 3)
        z = compose((10, ((- 10) * y), 0, 1), z)
        (yield y)

def calc_ndigits(n):
    return list(islice(gen_pi_digits(), n))

def add_cmdline_args(cmd, args):
    cmd.extend(('--digits', str(args.digits)))
if (__name__ == '__main__'):
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    cmd = runner.argparser
    cmd.add_argument('--digits', type=int, default=DEFAULT_DIGITS, help=('Number of computed pi digits (default: %s)' % DEFAULT_DIGITS))
    args = runner.parse_args()
    runner.metadata['description'] = 'Compute digits of pi.'
    runner.metadata['pidigits_ndigit'] = args.digits
    runner.bench_func('pidigits', calc_ndigits, args.digits)
