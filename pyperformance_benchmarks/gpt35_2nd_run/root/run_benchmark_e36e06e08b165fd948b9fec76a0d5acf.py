from typing import IO, Any, Dict, List, Tuple
import logging
import pyperf

FORMAT: str = 'important: %s'
MESSAGE: str = 'some important information to be logged'

def truncate_stream(stream: IO[str]) -> None:
    stream.seek(0)
    stream.truncate()

def bench_silent(loops: int, logger: logging.Logger, stream: IO[str]) -> float:
    truncate_stream(stream)
    m: str = MESSAGE
    range_it: range = range(loops)
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

def bench_simple_output(loops: int, logger: logging.Logger, stream: IO[str]) -> float:
    truncate_stream(stream)
    m: str = MESSAGE
    range_it: range = range(loops)
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

def bench_formatted_output(loops: int, logger: logging.Logger, stream: IO[str]) -> float:
    truncate_stream(stream)
    fmt: str = FORMAT
    msg: str = MESSAGE
    range_it: range = range(loops)
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

BENCHMARKS: Dict[str, Any] = {'silent': bench_silent, 'simple': bench_simple_output, 'format': bench_formatted_output}

def main() -> None:
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'Test the performance of logging.'
    parser: Any = runner.argparser
    parser.add_argument('benchmark', nargs='?', choices=sorted(BENCHMARKS))
    options: Any = runner.parse_args()
    stream: IO[str] = io.StringIO()
    handler: logging.StreamHandler = logging.StreamHandler(stream=stream)
    logger: logging.Logger = logging.getLogger('benchlogger')
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    if options.benchmark:
        benchmarks: Tuple[str] = (options.benchmark,)
    else:
        benchmarks: List[str] = sorted(BENCHMARKS)
    for bench in benchmarks:
        name: str = ('logging_%s' % bench)
        bench_func: Any = BENCHMARKS[bench]
        runner.bench_time_func(name, bench_func, logger, stream, inner_loops=10)

if (__name__ == '__main__'):
    main()
