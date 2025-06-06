'''
Benchmark the Dask scheduler running a large number of simple jobs.

Author: Matt Rocklin, Michael Droettboom
'''
from typing import Iterable, List

from dask.distributed import Client, Scheduler, Worker, Future, wait
import pyperf


def inc(x: int) -> int:
    return x + 1


async def benchmark() -> None:
    async with Scheduler() as scheduler:
        async with Worker(scheduler.address):
            async with Client(scheduler.address, asynchronous=True) as client:
                futures: List[Future] = client.map(inc, range(100))
                for _ in range(10):
                    futures = client.map(inc, futures)
                await wait(futures)


if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Benchmark dask'
    runner.bench_async_func('dask', benchmark)
