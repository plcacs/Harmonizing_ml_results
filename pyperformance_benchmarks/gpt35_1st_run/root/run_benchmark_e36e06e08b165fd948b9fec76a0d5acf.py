from typing import IO, Any, Dict, List, Tuple
import logging
import pyperf

FORMAT: str = 'important: %s'
MESSAGE: str = 'some important information to be logged'

def truncate_stream(stream: IO[str]) -> None:
    stream.seek(0)
    stream.truncate()

def bench_silent(loops: int, logger: logging.Logger, stream: IO[str]) -> float:
    ...

def bench_simple_output(loops: int, logger: logging.Logger, stream: IO[str]) -> float:
    ...

def bench_formatted_output(loops: int, logger: logging.Logger, stream: IO[str]) -> float:
    ...

def add_cmdline_args(cmd: List[str], args: Any) -> None:
    ...

BENCHMARKS: Dict[str, Any] = {'silent': bench_silent, 'simple': bench_simple_output, 'format': bench_formatted_output}

if (__name__ == '__main__'):
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
