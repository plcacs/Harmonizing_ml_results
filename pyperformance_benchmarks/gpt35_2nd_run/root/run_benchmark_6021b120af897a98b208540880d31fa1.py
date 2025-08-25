from typing import Dict, List

def bench_parse(loops: int) -> float:
    elapsed: float = 0
    for _ in range(loops):
        t0: float = pyperf.perf_counter()
        parse_one(SQL)
        elapsed += (pyperf.perf_counter() - t0)
    return elapsed

def bench_transpile(loops: int) -> float:
    elapsed: float = 0
    for _ in range(loops):
        t0: float = pyperf.perf_counter()
        transpile(SQL, write='spark')
        elapsed += (pyperf.perf_counter() - t0)
    return elapsed

def bench_optimize(loops: int) -> float:
    elapsed: float = 0
    for _ in range(loops):
        t0: float = pyperf.perf_counter()
        optimize(parse_one(SQL), TPCH_SCHEMA)
        elapsed += (pyperf.perf_counter() - t0)
    return elapsed

def bench_normalize(loops: int) -> float:
    elapsed: float = 0
    for _ in range(loops):
        conjunction = parse_one('(A AND B) OR (C AND D) OR (E AND F) OR (G AND H)')
        t0: float = pyperf.perf_counter()
        normalize.normalize(conjunction)
        elapsed += (pyperf.perf_counter() - t0)
    return elapsed

def add_cmdline_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd.append(args.benchmark)

def add_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('benchmark', choices=BENCHMARKS, help='Which benchmark to run.')

def main() -> None:
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'SQLGlot V2 benchmark'
    add_parser_args(runner.argparser)
    args: argparse.Namespace = runner.parse_args()
    benchmark: str = args.benchmark
    runner.bench_time_func(f'sqlglot_v2_{benchmark}', BENCHMARKS[benchmark])

if __name__ == '__main__':
    main()
