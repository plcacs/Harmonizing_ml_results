import asyncio
import os
import random
import sys
from itertools import count
from time import monotonic
from typing import Any, AsyncIterable, Callable, Iterator, Optional, Tuple

from faust.cli import option

__all__ = ['Benchmark']
TIME_EVERY: int = 10000
BENCH_TYPE: str = os.environ.get('F_BENCH', 'worker')
ACKS: int = int(os.environ.get('F_ACKS') or 0)
BSIZE: int = int(os.environ.get('F_BSIZE', 16384))
LINGER: int = int(os.environ.get('F_LINGER', 0))


class Benchmark:
    _agent: Optional[Any] = None
    _produce: Optional[Callable[..., Any]] = None
    produce_options: list = [
        option('--max-latency', type=float, default=0.0, envvar='PRODUCE_LATENCY',
               help='Add delay of (at most) n seconds between publishing.'),
        option('--max-messages', type=int, default=None,
               help='Send at most N messages or 0 for infinity.')
    ]

    def __init__(self, app: Any, topic: Any, *, n: int = TIME_EVERY, consume_topic: Optional[Any] = None) -> None:
        self.app: Any = app
        self.topic: Any = topic
        if consume_topic is None:
            consume_topic = topic
        self.consume_topic: Any = consume_topic
        self.n: int = n
        self.app.finalize()
        self.app.conf.producer_acks = ACKS
        self.app.conf.producer_max_batch_size = BSIZE
        self.app.conf.producer_linger = LINGER

    def install(self, main_name: str) -> None:
        self.create_benchmark_agent()
        self.create_produce_command()
        if main_name == '__main__':
            bench_args: dict[str, list[str]] = {
                'worker': ['worker', '-l', 'info'],
                'produce': ['produce']
            }
            if len(sys.argv) < 2:
                sys.argv.extend(bench_args[BENCH_TYPE])
            self.app.main()

    def create_benchmark_agent(self) -> None:
        self._agent = self.app.agent(self.consume_topic)(self.process)

    async def process(self, stream: AsyncIterable[Any]) -> None:
        time_last: Optional[float] = None
        async for i, value in stream.enumerate():
            if not i:
                time_last = monotonic()
            await self.process_value(value)
            if i and (not i % self.n) and time_last is not None:
                now: float = monotonic()
                runtime: float = now - time_last
                time_last = now
                print(f'RECV {i} in {runtime}s')

    async def process_value(self, value: Any) -> None:
        ...

    def create_produce_command(self) -> None:
        self._produce = self.app.command(*self.produce_options)(self.produce)

    async def produce(self, max_latency: float, max_messages: Optional[int], **kwargs: Any) -> None:
        i: int = 0
        time_start: Optional[float] = None
        app: Any = self.app
        topic: Any = self.topic
        for i, (key, value) in enumerate(self.generate_values(max_messages)):
            callback: Optional[Callable[[Any], None]] = None
            if not i:
                time_start = monotonic()
                time_1st: float = monotonic()

                def on_published(meta: Any) -> None:
                    print(f'1ST OK: {meta} AFTER {monotonic() - time_1st}s')
                callback = on_published
            await topic.send(key=key, value=value, callback=callback)
            if i and (not i % self.n) and time_start is not None:
                elapsed: float = monotonic() - time_start
                print(f'+SEND {i} in {elapsed}s')
                time_start = monotonic()
            if max_latency:
                await asyncio.sleep(random.uniform(0, max_latency))
        await asyncio.sleep(10)
        await app.producer.stop()
        final_time: float = monotonic() - (time_start if time_start is not None else monotonic())
        print(f'Time spent on {i} messages: {final_time}s')

    def generate_values(self, max_messages: Optional[int]) -> Iterator[Tuple[None, bytes]]:
        return ((None, str(i).encode()) for i in (range(max_messages) if max_messages else count()))
