import collections
import os
import pickle
import sys
import threading
import time
from types import FrameType
from typing import IO, Any, Dict, List, NewType, Optional
import greenlet
import objgraph
import psutil
from .constants import INTERVAL_SECONDS, MEGA
from .timer import TIMER, TIMER_SIGNAL, Timer
FlameStack = NewType('FlameStack', str)
FlameGraph = Dict[FlameStack, int]

def frame_format(frame):
    block_name = frame.f_code.co_name
    module_name = frame.f_globals.get('__name__')
    return '{}({})'.format(block_name, module_name)

def collect_frames(frame):
    callstack = []
    optional_frame = frame
    while optional_frame is not None:
        callstack.append(frame_format(optional_frame))
        optional_frame = optional_frame.f_back
    callstack.reverse()
    return callstack

def flamegraph_format(stack_count):
    return '\n'.join(('%s %d' % (key, value) for key, value in sorted(stack_count.items())))

def sample_stack(stack_count, frame, timespent):
    callstack = collect_frames(frame)
    formatted_stack = FlameStack(';'.join(callstack))
    stack_count[formatted_stack] += timespent

def process_memory_mb(pid):
    process = psutil.Process(pid)
    memory = process.memory_info()[0]
    for child in process.children(recursive=True):
        memory += child.memory_info()[0]
    return memory / MEGA

def sample_memory(timestamp, pid, stream):
    memory = process_memory_mb(pid)
    stream.write('{timestamp:.6f} {memory:.4f}\n'.format(timestamp=timestamp, memory=memory))

def sample_objects(timestamp, stream):
    count_per_type = objgraph.typestats()
    data = [timestamp, count_per_type]
    data_pickled = pickle.dumps(data)
    stream.write(data_pickled)

class FlameGraphCollector:

    def __init__(self, stack_stream):
        self.stack_stream = stack_stream
        self.stack_count = collections.defaultdict(int)
        self.last_timestamp = None

    def collect(self, frame, timestamp):
        if self.last_timestamp is not None:
            sample_stack(self.stack_count, frame, timestamp)
        self.last_timestamp = timestamp

    def stop(self):
        stack_data = flamegraph_format(self.stack_count)
        self.stack_stream.write(stack_data)
        self.stack_stream.close()
        del self.stack_stream
        del self.last_timestamp

class MemoryCollector:

    def __init__(self, memory_stream):
        self.memory_stream = memory_stream

    def collect(self, _frame, timestamp):
        sample_memory(timestamp, os.getpid(), self.memory_stream)
        self.memory_stream.flush()

    def stop(self):
        self.memory_stream.close()
        del self.memory_stream

class ObjectCollector:

    def __init__(self, objects_stream):
        self.objects_stream = objects_stream

    def collect(self, _frame, timestamp):
        sample_objects(timestamp, self.objects_stream)
        self.objects_stream.flush()

    def stop(self):
        self.objects_stream.close()
        del self.objects_stream

class TraceSampler:

    def __init__(self, collector, sample_interval=0.1):
        self.collector = collector
        self.sample_interval = sample_interval
        self.last_timestamp = time.time()
        self.old_frame = None
        self.previous_callback = greenlet.gettrace()
        greenlet.settrace(self._greenlet_profiler)
        sys.setprofile(self._thread_profiler)

    def _should_sample(self, timestamp):
        if timestamp - self.last_timestamp >= self.sample_interval:
            self.last_timestamp = timestamp
            return True
        return False

    def _greenlet_profiler(self, event, args):
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

    def _thread_profiler(self, frame, _event, _arg):
        timestamp = time.time()
        if self._should_sample(timestamp):
            self.collector.collect(self.old_frame, timestamp)
        self.old_frame = frame

    def stop(self):
        sys.setprofile(None)
        threading.setprofile(None)
        greenlet.settrace(self.previous_callback)
        self.collector.stop()
        self.collector = None

class SignalSampler:
    """Signal based sampler."""

    def __init__(self, collector, timer=TIMER, interval=INTERVAL_SECONDS, timer_signal=TIMER_SIGNAL):
        self.collector = collector
        self.timer = Timer(callback=self._timer_callback, timer=timer, interval=interval, timer_signal=timer_signal)

    def _timer_callback(self, signum, frame):
        if self.collector is not None:
            self.collector.collect(frame, time.time())

    def stop(self):
        timer = self.timer
        collector = self.collector
        del self.timer
        del self.collector
        timer.stop()
        collector.stop()