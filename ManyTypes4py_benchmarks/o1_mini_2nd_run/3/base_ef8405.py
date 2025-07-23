import asyncio
import os
import random
import sys
from itertools import count
from time import monotonic
from faust.cli import option
from typing import Any, Callable, Iterator, List, Optional, Tuple
import faust
from faust.types import AppT, EventT, StreamT, TopicT

__all__ = ['Benchmark']
TIME_EVERY: int = 10000
BENCH_TYPE: str = os.environ.get('F_BENCH', 'worker')
ACKS: int = int(os.environ.get('F_ACKS') or 0)
BSIZE: int = int(os.environ.get('F_BSIZE', 16384))
LINGER: int = int(os.environ.get('F_LINGER', 0))

class Benchmark:
    _agent: Optional[faust.Agent] = None
    _produce: Optional[Callable[..., Any]] = None
    produce_options: List[Any] = [
        option(
            '--max-latency',
            type=float,
            default=0.0,
            envvar='PRODUCE_LATENCY',
            help='Add delay of (at most) n seconds between publishing.'
        ),
        option(
            '--max-messages',
            type=int,
            default=None,
            help='Send at most N messages or 0 for infinity.'
        )
    ]

    def __init__(
        self, 
        app: AppT, 
        topic: TopicT, 
        *, 
        n: int = TIME_EVERY, 
        consume_topic: Optional[TopicT] = None
    ) -> None:
        self.app: AppT = app
        self.topic: TopicT = topic
        if consume_topic is None:
            consume_topic = topic
        self.consume_topic: TopicT = consume_topic
        self.n: int = n
        self.app.finalize()
        self.app.conf.producer_acks = ACKS
        self.app.conf.producer_max_batch_size = BSIZE
        self.app.conf.producer_linger = LINGER

    def install(self, main_name: str) -> None:
        self.create_benchmark_agent()
        self.create_produce_command()
        if main_name == '__main__':
            bench_args: dict = {'worker': ['worker', '-l', 'info'], 'produce': ['produce']}
            if len(sys.argv) < 2:
                sys.argv.extend(bench_args[BENCH_TYPE])
            self.app.main()

    def create_benchmark_agent(self) -> None:
        self._agent = self.app.agent(self.consume_topic)(self.process)

    async def process(self, stream: StreamT[EventT, Any]) -> None:
        time_last: Optional[float] = None
        async for i, value in stream.enumerate():
            if not i:
                time_last = monotonic()
            await self.process_value(value)
            if i and (i % self.n == 0):
                now: float = monotonic()
                assert time_last is not None
                runtime: float = now - time_last
                time_last = now
                print(f'RECV {i} in {runtime}s')

    async def process_value(self, value: Any) -> None:
        ...

    def create_produce_command(self) -> None:
        self._produce = self.app.command(*self.produce_options)(self.produce)

    async def produce(
        self, 
        max_latency: float, 
        max_messages: Optional[int], 
        **kwargs: Any
    ) -> None:
        i: int = 0
        time_start: Optional[float] = None
        app: AppT = self.app
        topic: TopicT = self.topic
        for i, (key, value) in enumerate(self.generate_values(max_messages)):
            callback: Optional[Callable[[faust.types.Metadata], None]] = None
            if i == 0:
                time_start = monotonic()
                time_1st: float = monotonic()

                def on_published(meta: faust.types.Metadata) -> None:
                    print(f'1ST OK: {meta} AFTER {monotonic() - time_1st}s')

                callback = on_published
            await topic.send(key=key, value=value, callback=callback)
            if i and (i % self.n == 0):
                assert time_start is not None
                print(f'+SEND {i} in {monotonic() - time_start}s')
                time_start = monotonic()
            if max_latency:
                await asyncio.sleep(random.uniform(0, max_latency))
        await asyncio.sleep(10)
        await app.producer.stop()
        print(f'Time spent on {i} messages: {monotonic() - time_start}s')

    def generate_values(self, max_messages: Optional[int]) -> Iterator[Tuple[Optional[Any], bytes]]:
        return ((None, str(i).encode()) for i in (range(max_messages) if max_messages and max_messages > 0 else count()))
