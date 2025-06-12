"""
Script for testing the performance of logging simple messages.

Rationale for logging_silent by Antoine Pitrou:

"The performance of silent logging calls is actually important for all
applications which have debug() calls in their critical paths.  This is
quite common in network and/or distributed programming where you want to
allow logging many events for diagnosis of unexpected runtime issues
(because many unexpected conditions can appear), but with those logs
disabled by default for performance and readability reasons."

https://mail.python.org/pipermail/speed/2017-May/000576.html
"""
import io
import logging
import pyperf
from typing import Dict, Callable, Any, List, TextIO, Optional, Tuple

FORMAT: str = 'important: %s'
MESSAGE: str = 'some important information to be logged'

def truncate_stream(stream: TextIO) -> None:
    stream.seek(0)
    stream.truncate()

def bench_silent(loops: int, logger: logging.Logger, stream: TextIO) -> float:
    truncate_stream(stream)
    m: str = MESSAGE
    range_it = range(loops)
    t0: float = pyperf.perf_counter()
    for _ in range_it:
        logger.debug(m)
        logger.debug(m)
        logger.debug(m)
        logger.debug(m)
        logger.debug(m)
        logger.debug(m)
        logger.debug(m)
        logger.debug(m)
        logger.debug(m)
        logger.debug(m)
    dt: float = (pyperf.perf_counter() - t0)
    if (len(stream.getvalue()) != 0):
        raise ValueError('stream is expected to be empty')
    return dt

def bench_simple_output(loops: int, logger: logging.Logger, stream: TextIO) -> float:
    truncate_stream(stream)
    m: str = MESSAGE
    range_it = range(loops)
    t0: float = pyperf.perf_counter()
    for _ in range_it:
        logger.warning(m)
        logger.warning(m)
        logger.warning(m)
        logger.warning(m)
        logger.warning(m)
        logger.warning(m)
        logger.warning(m)
        logger.warning(m)
        logger.warning(m)
        logger.warning(m)
    dt: float = (pyperf.perf_counter() - t0)
    lines: List[str] = stream.getvalue().splitlines()
    if (len(lines) != (loops * 10)):
        raise ValueError('wrong number of lines')
    return dt

def bench_formatted_output(loops: int, logger: logging.Logger, stream: TextIO) -> float:
    truncate_stream(stream)
    fmt: str = FORMAT
    msg: str = MESSAGE
    range_it = range(loops)
    t0: float = pyperf.perf_counter()
    for _ in range_it:
        logger.warning(fmt, msg)
        logger.warning(fmt, msg)
        logger.warning(fmt, msg)
        logger.warning(fmt, msg)
        logger.warning(fmt, msg)
        logger.warning(fmt, msg)
        logger.warning(fmt, msg)
        logger.warning(fmt, msg)
        logger.warning(fmt, msg)
        logger.warning(fmt, msg)
    dt: float = (pyperf.perf_counter() - t0)
    lines: List[str] = stream.getvalue().splitlines()
    if (len(lines) != (loops * 10)):
        raise ValueError('wrong number of lines')
    return dt

def add_cmdline_args(cmd: List[str], args: Any) -> None:
    if args.benchmark:
        cmd.append(args.benchmark)

BENCHMARKS: Dict[str, Callable[[int, logging.Logger, TextIO], float]] = {
    'silent': bench_silent,
    'simple': bench_simple_output,
    'format': bench_formatted_output
}

if (__name__ == '__main__'):
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'Test the performance of logging.'
    parser = runner.argparser
    parser.add_argument('benchmark', nargs='?', choices=sorted(BENCHMARKS))
    options = runner.parse_args()
    stream: TextIO = io.StringIO()
    handler: logging.StreamHandler = logging.StreamHandler(stream=stream)
    logger: logging.Logger = logging.getLogger('benchlogger')
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    benchmarks: Tuple[str, ...]
    if options.benchmark:
        benchmarks = (options.benchmark,)
    else:
        benchmarks = tuple(sorted(BENCHMARKS))
    for bench in benchmarks:
        name: str = ('logging_%s' % bench)
        bench_func: Callable[[int, logging.Logger, TextIO], float] = BENCHMARKS[bench]
        runner.bench_time_func(name, bench_func, logger, stream, inner_loops=10)
