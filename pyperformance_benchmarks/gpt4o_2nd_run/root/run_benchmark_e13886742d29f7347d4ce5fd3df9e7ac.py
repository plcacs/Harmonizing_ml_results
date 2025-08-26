from dask.distributed import Client, Worker, Scheduler, wait
from dask import distributed
import pyperf
from typing import List

def inc(x: int) -> int:
    return (x + 1)

async def benchmark() -> None:
    async with Scheduler() as scheduler:
        async with Worker(scheduler.address):
            async with Client(scheduler.address, asynchronous=True) as client:
                futures = client.map(inc, range(100))  # type: List[distributed.Future]
                for _ in range(10):
                    futures = client.map(inc, futures)
                await wait(futures)

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Benchmark dask'
    runner.bench_async_func('dask', benchmark)
