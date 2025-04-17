
'\nBenchmark for concurrent model communication.\n'
import pyperf
from multiprocessing.pool import Pool, ThreadPool

def f(x):
    return x

def bench_mp_pool(p, n, chunk):
    with Pool(p) as pool:
        for _ in pool.imap(f, range(n), chunk):
            pass

def bench_thread_pool(c, n, chunk):
    with ThreadPool(c) as pool:
        for _ in pool.imap(f, range(n), chunk):
            pass
if (__name__ == '__main__'):
    runner = pyperf.Runner()
    runner.metadata['description'] = 'concurrent model communication benchmark'
    count = 1000
    chunk = 10
    num_core = 2
    runner.bench_func('bench_mp_pool', bench_mp_pool, num_core, count, chunk)
    runner.bench_func('bench_thread_pool', bench_thread_pool, num_core, count, chunk)
