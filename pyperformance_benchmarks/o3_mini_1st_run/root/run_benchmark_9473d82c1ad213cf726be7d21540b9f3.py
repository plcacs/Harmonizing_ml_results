import asyncio
import math
import random
import pyperf
import argparse
from typing import Any, Dict, List, Type

NUM_RECURSE_LEVELS: int = 6
NUM_RECURSE_BRANCHES: int = 6
RANDOM_SEED: int = 0
IO_SLEEP_TIME: float = 0.05
MEMOIZABLE_PERCENTAGE: int = 90
CPU_PROBABILITY: float = 0.5
FACTORIAL_N: int = 500

class AsyncTree:
    cache: Dict[int, bool]

    def __init__(self, use_task_groups: bool) -> None:
        self.cache = {}
        self.use_task_groups: bool = use_task_groups
        random.seed(RANDOM_SEED)

    async def mock_io_call(self) -> None:
        await asyncio.sleep(IO_SLEEP_TIME)

    async def workload_func(self) -> Any:
        raise NotImplementedError("To be implemented by each variant's derived class.")

    async def recurse_with_gather(self, recurse_level: int) -> None:
        if recurse_level == 0:
            await self.workload_func()
            return
        await asyncio.gather(*[self.recurse_with_gather(recurse_level - 1) for _ in range(NUM_RECURSE_BRANCHES)])

    async def recurse_with_task_group(self, recurse_level: int) -> None:
        if recurse_level == 0:
            await self.workload_func()
            return
        async with asyncio.TaskGroup() as tg:
            for _ in range(NUM_RECURSE_BRANCHES):
                tg.create_task(self.recurse_with_task_group(recurse_level - 1))

    async def run(self) -> None:
        if self.use_task_groups:
            await self.recurse_with_task_group(NUM_RECURSE_LEVELS)
        else:
            await self.recurse_with_gather(NUM_RECURSE_LEVELS)

class EagerMixin:
    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        if hasattr(asyncio, 'eager_task_factory'):
            loop.set_task_factory(asyncio.eager_task_factory)
        await super().run()

class NoneAsyncTree(AsyncTree):
    async def workload_func(self) -> None:
        return

class EagerAsyncTree(EagerMixin, NoneAsyncTree):
    pass

class IOAsyncTree(AsyncTree):
    async def workload_func(self) -> None:
        await self.mock_io_call()

class EagerIOAsyncTree(EagerMixin, IOAsyncTree):
    pass

class MemoizationAsyncTree(AsyncTree):
    async def workload_func(self) -> int:
        data: int = random.randint(1, 100)
        if data <= MEMOIZABLE_PERCENTAGE:
            if self.cache.get(data):
                return data
            self.cache[data] = True
        await self.mock_io_call()
        return data

class EagerMemoizationAsyncTree(EagerMixin, MemoizationAsyncTree):
    pass

class CpuIoMixedAsyncTree(MemoizationAsyncTree):
    async def workload_func(self) -> int:
        if random.random() < CPU_PROBABILITY:
            return math.factorial(FACTORIAL_N)
        else:
            return await MemoizationAsyncTree.workload_func(self)

class EagerCpuIoMixedAsyncTree(EagerMixin, CpuIoMixedAsyncTree):
    pass

def add_metadata(runner: pyperf.Runner) -> None:
    runner.metadata['description'] = 'Async tree workloads.'
    runner.metadata['async_tree_recurse_levels'] = NUM_RECURSE_LEVELS
    runner.metadata['async_tree_recurse_branches'] = NUM_RECURSE_BRANCHES
    runner.metadata['async_tree_random_seed'] = RANDOM_SEED
    runner.metadata['async_tree_io_sleep_time'] = IO_SLEEP_TIME
    runner.metadata['async_tree_memoizable_percentage'] = MEMOIZABLE_PERCENTAGE
    runner.metadata['async_tree_cpu_probability'] = CPU_PROBABILITY
    runner.metadata['async_tree_factorial_n'] = FACTORIAL_N

def add_cmdline_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd.append(args.benchmark)
    if args.task_groups:
        cmd.append('--task-groups')

def add_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        'benchmark',
        choices=['none', 'eager', 'io', 'eager_io', 'memoization', 'eager_memoization', 'cpu_io_mixed', 'eager_cpu_io_mixed'],
        help=('Determines which benchmark to run. Options:\n'
              '1) "none": No actual async work in the async tree.\n'
              '2) "io": All leaf nodes simulate async IO workload (async sleep 50ms).\n'
              '3) "memoization": All leaf nodes simulate async IO workload with 90%% of\n'
              '                  the data memoized\n'
              '4) "cpu_io_mixed": Half of the leaf nodes simulate CPU-bound workload and\n'
              '                   the other half simulate the same workload as the\n'
              '                   "memoization" variant.\n')
    )
    parser.add_argument('--task-groups', action='store_true', default=False, help='Use TaskGroups instead of gather.')

BENCHMARKS: Dict[str, Type[AsyncTree]] = {
    'none': NoneAsyncTree,
    'eager': EagerAsyncTree,
    'io': IOAsyncTree,
    'eager_io': EagerIOAsyncTree,
    'memoization': MemoizationAsyncTree,
    'eager_memoization': EagerMemoizationAsyncTree,
    'cpu_io_mixed': CpuIoMixedAsyncTree,
    'eager_cpu_io_mixed': EagerCpuIoMixedAsyncTree
}

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    add_metadata(runner)
    add_parser_args(runner.argparser)
    args: argparse.Namespace = runner.parse_args()
    benchmark: str = args.benchmark
    async_tree_class: Type[AsyncTree] = BENCHMARKS[benchmark]
    async_tree: AsyncTree = async_tree_class(use_task_groups=args.task_groups)
    bench_name: str = f'async_tree_{benchmark}'
    if args.task_groups:
        bench_name += '_tg'
    runner.bench_async_func(bench_name, async_tree.run)