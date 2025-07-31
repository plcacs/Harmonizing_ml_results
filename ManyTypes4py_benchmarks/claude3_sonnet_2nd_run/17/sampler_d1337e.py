import collections
import os
import pickle
import sys
import threading
import time
from types import FrameType
from typing import IO, Any, Dict, List, NewType, Optional, Callable, Tuple, DefaultDict, Union
import greenlet
import objgraph
import psutil
from .constants import INTERVAL_SECONDS, MEGA
from .timer import TIMER, TIMER_SIGNAL, Timer

FlameStack = NewType('FlameStack', str)
FlameGraph = Dict[FlameStack, int]

def frame_format(frame: FrameType) -> str:
    block_name = frame.f_code.co_name
    module_name = frame.f_globals.get('__name__')
    return '{}({})'.format(block_name, module_name)

def collect_frames(frame: FrameType) -> List[str]:
    callstack: List[str] = []
    optional_frame: Optional[FrameType] = frame
    while optional_frame is not None:
        callstack.append(frame_format(optional_frame))
        optional_frame = optional_frame.f_back
    callstack.reverse()
    return callstack

def flamegraph_format(stack_count: Dict[FlameStack, int]) -> str:
    return '\n'.join(('%s %d' % (key, value) for key, value in sorted(stack_count.items())))

def sample_stack(stack_count: DefaultDict[FlameStack, int], frame: FrameType, timespent: float) -> None:
    callstack = collect_frames(frame)
    formatted_stack = FlameStack(';'.join(callstack))
    stack_count[formatted_stack] += timespent

def process_memory_mb(pid: int) -> float:
    process = psutil.Process(pid)
    memory = process.memory_info()[0]
    for child in process.children(recursive=True):
        memory += child.memory_info()[0]
    return memory / MEGA

def sample_memory(timestamp: float, pid: int, stream: IO[str]) -> None:
    memory = process_memory_mb(pid)
    stream.write('{timestamp:.6f} {memory:.4f}\n'.format(timestamp=timestamp, memory=memory))

def sample_objects(timestamp: float, stream: IO[bytes]) -> None:
    count_per_type = objgraph.typestats()
    data: List[Any] = [timestamp, count_per_type]
    data_pickled = pickle.dumps(data)
    stream.write(data_pickled)

class FlameGraphCollector:
    def __init__(self, stack_stream: IO[str]) -> None:
        self.stack_stream = stack_stream
        self.stack_count: DefaultDict[FlameStack, int] = collections.defaultdict(int)
        self.last_timestamp: Optional[float] = None

    def collect(self, frame: FrameType, timestamp: float) -> None:
        if self.last_timestamp is not None:
            sample_stack(self.stack_count, frame, timestamp)
        self.last_timestamp = timestamp

    def stop(self) -> None:
        stack_data = flamegraph_format(self.stack_count)
        self.stack_stream.write(stack_data)
        self.stack_stream.close()
        del self.stack_stream
        del self.last_timestamp

class MemoryCollector:
    def __init__(self, memory_stream: IO[str]) -> None:
        self.memory_stream = memory_stream

    def collect(self, _frame: FrameType, timestamp: float) -> None:
        sample_memory(timestamp, os.getpid(), self.memory_stream)
        self.memory_stream.flush()

    def stop(self) -> None:
        self.memory_stream.close()
        del self.memory_stream

class ObjectCollector:
    def __init__(self, objects_stream: IO[bytes]) -> None:
        self.objects_stream = objects_stream

    def collect(self, _frame: FrameType, timestamp: float) -> None:
        sample_objects(timestamp, self.objects_stream)
        self.objects_stream.flush()

    def stop(self) -> None:
        self.objects_stream.close()
        del self.objects_stream

class TraceSampler:
    def __init__(self, collector: Union[FlameGraphCollector, MemoryCollector, ObjectCollector], sample_interval: float = 0.1) -> None:
        self.collector = collector
        self.sample_interval = sample_interval
        self.last_timestamp = time.time()
        self.old_frame: Optional[FrameType] = None
        self.previous_callback: Optional[Callable] = greenlet.gettrace()
        greenlet.settrace(self._greenlet_profiler)
        sys.setprofile(self._thread_profiler)

    def _should_sample(self, timestamp: float) -> bool:
        if timestamp - self.last_timestamp >= self.sample_interval:
            self.last_timestamp = timestamp
            return True
        return False

    def _greenlet_profiler(self, event: str, args: Tuple) -> Any:
        timestamp = time.time()
        try:
            frame = sys._getframe(1)
        except ValueError:
            frame = sys._getframe(0)
        if self._should_sample(timestamp):
            self.collector.collect(self.old_frame, timestamp)
        self.old_frame = frame
        if self.previous_callback is not None:
            return self.previous_callback(event, args)
        return None

    def _thread_profiler(self, frame: FrameType, _event: str, _arg: Any) -> None:
        timestamp = time.time()
        if self._should_sample(timestamp):
            self.collector.collect(self.old_frame, timestamp)
        self.old_frame = frame

    def stop(self) -> None:
        sys.setprofile(None)
        threading.setprofile(None)
        greenlet.settrace(self.previous_callback)
        self.collector.stop()
        self.collector = None  # type: ignore

class SignalSampler:
    """Signal based sampler."""

    def __init__(self, collector: Union[FlameGraphCollector, MemoryCollector, ObjectCollector], timer: Any = TIMER, interval: float = INTERVAL_SECONDS, timer_signal: int = TIMER_SIGNAL) -> None:
        self.collector = collector
        self.timer = Timer(callback=self._timer_callback, timer=timer, interval=interval, timer_signal=timer_signal)

    def _timer_callback(self, signum: int, frame: FrameType) -> None:
        if self.collector is not None:
            self.collector.collect(frame, time.time())

    def stop(self) -> None:
        timer = self.timer
        collector = self.collector
        del self.timer
        del self.collector
        timer.stop()
        collector.stop()
