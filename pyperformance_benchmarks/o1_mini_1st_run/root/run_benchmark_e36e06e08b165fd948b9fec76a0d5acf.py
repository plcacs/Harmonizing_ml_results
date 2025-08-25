import io
import logging
import pyperf
from typing import Callable, Dict, Optional, Tuple


FORMAT: str = 'important: %s'
MESSAGE: str = 'some important information to be logged'


def truncate_stream(stream: io.StringIO) -> None:
    stream.seek(0)
    stream.truncate()


def bench_silent(loops: int, logger: logging.Logger, stream: io.StringIO) -> float:
    truncate_stream(stream)
    m: str = MESSAGE
    range_it = range(loops)
    t0 = pyperf.perf_counter()
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
    dt = pyperf.perf_counter() - t0
    if len(stream.getvalue()) != 0:
        raise ValueError('stream is expected to be empty')
    return dt


def bench_simple_output(loops: int, logger: logging.Logger, stream: io.StringIO) -> float:
    truncate_stream(stream)
    m: str = MESSAGE
    range_it = range(loops)
    t0 = pyperf.perf_counter()
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
    dt = pyperf.perf_counter() - t0
    lines = stream.getvalue().splitlines()
    if len(lines) != (loops * 10):
        raise ValueError('wrong number of lines')
    return dt


def bench_formatted_output(loops: int, logger: logging.Logger, stream: io.StringIO) -> float:
    truncate_stream(stream)
    fmt: str = FORMAT
    msg: str = MESSAGE
    range_it = range(loops)
    t0 = pyperf.perf_counter()
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
    dt = pyperf.perf_counter() - t0
    lines = stream.getvalue().splitlines()
    if len(lines) != (loops * 10):
        raise ValueError('wrong number of lines')
    return dt


def add_cmdline_args(cmd: list[str], args: Optional[pyperf.Args]) -> None:
    if args and args.benchmark:
        cmd.append(args.benchmark)


BENCHMARKS: Dict[str, Callable[[int, logging.Logger, io.StringIO], float]] = {
    'silent': bench_silent,
    'simple': bench_simple_output,
    'format': bench_formatted_output
}


if __name__ == '__main__':
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'Test the performance of logging.'
    parser = runner.argparser
    parser.add_argument('benchmark', nargs='?', choices=sorted(BENCHMARKS))
    options = runner.parse_args()
    stream: io.StringIO = io.StringIO()
    handler: logging.StreamHandler = logging.StreamHandler(stream=stream)
    logger: logging.Logger = logging.getLogger('benchlogger')
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    if options.benchmark:
        benchmarks: Tuple[str, ...] = (options.benchmark,)
    else:
        benchmarks = tuple(sorted(BENCHMARKS))
    for bench in benchmarks:
        name: str = f'logging_{bench}'
        bench_func: Callable[[int, logging.Logger, io.StringIO], float] = BENCHMARKS[bench]
        runner.bench_time_func(name, bench_func, logger, stream, inner_loops=10)
