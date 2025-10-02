import collections
import os
import pickle
import sys
import threading
import time
from types import FrameType
from typing import IO, Any, Callable, Dict, List, NewType, Optional
import greenlet
import objgraph
import psutil
from .constants import INTERVAL_SECONDS, MEGA
from .timer import TIMER, TIMER_SIGNAL, Timer

FlameStack = NewType('FlameStack', str)
FlameGraph = Dict[FlameStack, int]

def frame_format(frame: FrameType) -> str:
    block_name: str = frame.f_code.co_name
    module_name: Optional[str] = frame.f_globals.get('__name__')
    return '{}({})'.format(block_name, module_name)

def collect_frames(frame: FrameType) -> List[str]:
    callstack: List[str] = []
    optional_frame: Optional[FrameType] = frame
    while optional_frame is not None:
        callstack.append(frame_format(optional_frame))
        optional_frame = optional_frame.f_back
    callstack.reverse()
    return callstack

def flamegraph_format(stack_count: FlameGraph) -> str:
    return '\n'.join(('%s %d' % (key, value) for key, value in sorted(stack_count.items())))

def sample_stack(stack_count: FlameGraph, frame: FrameType, timespent: float) -> None:
    callstack: List[str] = collect_frames(frame)
    formatted_stack: FlameStack = FlameStack(';'.join(callstack))
    stack_count[formatted_stack] += int(timespent)

def process_memory_mb(pid: int) -> float:
    process: psutil.Process = psutil.Process(pid)
    memory: int = process.memory_info().rss
    for child in process.children(recursive=True):
        memory += child.memory_info().rss
    return memory / MEGA

def sample_memory(timestamp: float, pid: int, stream: IO[str]) -> None:
    memory: float = process_memory_mb(pid)
    stream.write('{timestamp:.6f} {memory:.4f}\n'.format(timestamp=timestamp, memory=memory))

def sample_objects(timestamp: float, stream: IO[bytes]) -> None:
    count_per_type: Dict[str, int] = objgraph.typestats()
    data: List[Any] = [timestamp, count_per_type]
    data_pickled: bytes = pickle.dumps(data)
    stream.write(data_pickled)

class FlameGraphCollector:
    def __init__(self, stack_stream: IO[str]) -> None:
        self.stack_stream: IO[str] = stack_stream
        self.stack_count: FlameGraph = collections.defaultdict(int)
        self.last_timestamp: Optional[float] = None

    def collect(self, frame: Optional[FrameType], timestamp: float) -> None:
        if self.last_timestamp is not None and frame is not None:
            sample_stack(self.stack_count, frame, timestamp - self.last_timestamp)
        self.last_timestamp = timestamp

    def stop(self) -> None:
        stack_data: str = flamegraph_format(self.stack_count)
        self.stack_stream.write(stack_data)
        self.stack_stream.close()
        del self.stack_stream
        del self.last_timestamp

class MemoryCollector:
    def __init__(self, memory_stream: IO[str]) -> None:
        self.memory_stream: IO[str] = memory_stream

    def collect(self, _frame: Optional[FrameType], timestamp: float) -> None:
        sample_memory(timestamp, os.getpid(), self.memory_stream)
        self.memory_stream.flush()

    def stop(self) -> None:
        self.memory_stream.close()
        del self.memory_stream

class ObjectCollector:
    def __init__(self, objects_stream: IO[bytes]) -> None:
        self.objects_stream: IO[bytes] = objects_stream

    def collect(self, _frame: Optional[FrameType], timestamp: float) -> None:
        sample_objects(timestamp, self.objects_stream)
        self.objects_stream.flush()

    def stop(self) -> None:
        self.objects_stream.close()
        del self.objects_stream

class TraceSampler:
    def __init__(self, collector: Any, sample_interval: float = 0.1) -> None:
        self.collector: Any = collector
        self.sample_interval: float = sample_interval
        self.last_timestamp: float = time.time()
        self.old_frame: Optional[FrameType] = None
        self.previous_callback: Optional[Callable[..., Any]] = greenlet.gettrace()
        greenlet.settrace(self._greenlet_profiler)
        sys.setprofile(self._thread_profiler)

    def _should_sample(self, timestamp: float) -> bool:
        if timestamp - self.last_timestamp >= self.sample_interval:
            self.last_timestamp = timestamp
            return True
        return False

    def _greenlet_profiler(self, event: str, args: Any) -> Optional[Callable[..., Any]]:
        timestamp: float = time.time()
        try:
            frame: FrameType = sys._getframe(1)
        except ValueError:
            frame = sys._getframe(0)
        if self._should_sample(timestamp):
            self.collector.collect(self.old_frame, timestamp)
        self.old_frame = frame
        if self.previous_callback is not None:
            return self.previous_callback(event, args)
        return None

    def _thread_profiler(self, frame: FrameType, _event: str, _arg: Any) -> None:
        timestamp: float = time.time()
        if self._should_sample(timestamp):
            self.collector.collect(self.old_frame, timestamp)
        self.old_frame = frame

    def stop(self) -> None:
        sys.setprofile(None)
        threading.setprofile(None)
        greenlet.settrace(self.previous_callback)
        self.collector.stop()
        self.collector = None

class SignalSampler:
    """Signal based sampler."""

    def __init__(
        self,
        collector: Any,
        timer: Timer = TIMER,
        interval: float = INTERVAL_SECONDS,
        timer_signal: int = TIMER_SIGNAL
    ) -> None:
        self.collector: Any = collector
        self.timer: Timer = Timer(callback=self._timer_callback, timer=timer, interval=interval, timer_signal=timer_signal)

    def _timer_callback(self, signum: int, frame: Optional[FrameType]) -> None:
        if self.collector is not None:
            self.collector.collect(frame, time.time())

    def stop(self) -> None:
        timer: Timer = self.timer
        collector: Any = self.collector
        del self.timer
        del self.collector
        timer.stop()
        collector.stop()
