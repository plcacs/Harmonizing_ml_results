import pyperf
from sqlglot import parse_one, transpile
from sqlglot.optimizer import optimize, normalize
from typing import Dict, Any, Callable, List
import argparse

SQL: str = "\nselect\n        supp_nation,\n        cust_nation,\n        l_year,\n        sum(volume) as revenue\nfrom\n        (\n                select\n                        n1.n_name as supp_nation,\n                        n2.n_name as cust_nation,\n                        extract(year from l_shipdate) as l_year,\n                        l_extendedprice * (1 - l_discount) as volume\n                from\n                        supplier,\n                        lineitem,\n                        orders,\n                        customer,\n                        nation n1,\n                        nation n2\n                where\n                        s_suppkey = l_suppkey\n                        and o_orderkey = l_orderkey\n                        and c_custkey = o_custkey\n                        and s_nationkey = n1.n_nationkey\n                        and c_nationkey = n2.n_nationkey\n                        and (\n                                (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')\n                                or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE')\n                        )\n                        and l_shipdate between date '1995-01-01' and date '1996-12-31'\n        ) as shipping\ngroup by\n        supp_nation,\n        cust_nation,\n        l_year\norder by\n        supp_nation,\n        cust_nation,\n        l_year;\n"
TPCH_SCHEMA: Dict[str, Dict[str, str]] = {'lineitem': {'l_orderkey': 'uint64', 'l_partkey': 'uint64', 'l_suppkey': 'uint64', 'l_linenumber': 'uint64', 'l_quantity': 'float64', 'l_extendedprice': 'float64', 'l_discount': 'float64', 'l_tax': 'float64', 'l_returnflag': 'string', 'l_linestatus': 'string', 'l_shipdate': 'date32', 'l_commitdate': 'date32', 'l_receiptdate': 'date32', 'l_shipinstruct': 'string', 'l_shipmode': 'string', 'l_comment': 'string'}, 'orders': {'o_orderkey': 'uint64', 'o_custkey': 'uint64', 'o_orderstatus': 'string', 'o_totalprice': 'float64', 'o_orderdate': 'date32', 'o_orderpriority': 'string', 'o_clerk': 'string', 'o_shippriority': 'int32', 'o_comment': 'string'}, 'customer': {'c_custkey': 'uint64', 'c_name': 'string', 'c_address': 'string', 'c_nationkey': 'uint64', 'c_phone': 'string', 'c_acctbal': 'float64', 'c_mktsegment': 'string', 'c_comment': 'string'}, 'part': {'p_partkey': 'uint64', 'p_name': 'string', 'p_mfgr': 'string', 'p_brand': 'string', 'p_type': 'string', 'p_size': 'int32', 'p_container': 'string', 'p_retailprice': 'float64', 'p_comment': 'string'}, 'supplier': {'s_suppkey': 'uint64', 's_name': 'string', 's_address': 'string', 's_nationkey': 'uint64', 's_phone': 'string', 's_acctbal': 'float64', 's_comment': 'string'}, 'partsupp': {'ps_partkey': 'uint64', 'ps_suppkey': 'uint64', 'ps_availqty': 'int32', 'ps_supplycost': 'float64', 'ps_comment': 'string'}, 'nation': {'n_nationkey': 'uint64', 'n_name': 'string', 'n_regionkey': 'uint64', 'n_comment': 'string'}, 'region': {'r_regionkey': 'uint64', 'r_name': 'string', 'r_comment': 'string'}}

def bench_parse(loops: int) -> float:
    elapsed: float = 0.0
    for _ in range(loops):
        t0: float = pyperf.perf_counter()
        parse_one(SQL)
        elapsed += (pyperf.perf_counter() - t0)
    return elapsed

def bench_transpile(loops: int) -> float:
    elapsed: float = 0.0
    for _ in range(loops):
        t0: float = pyperf.perf_counter()
        transpile(SQL, write='spark')
        elapsed += (pyperf.perf_counter() - t0)
    return elapsed

def bench_optimize(loops: int) -> float:
    elapsed: float = 0.0
    for _ in range(loops):
        t0: float = pyperf.perf_counter()
        optimize(parse_one(SQL), TPCH_SCHEMA)
        elapsed += (pyperf.perf_counter() - t0)
    return elapsed

def bench_normalize(loops: int) -> float:
    elapsed: float = 0.0
    for _ in range(loops):
        conjunction = parse_one('(A AND B) OR (C AND D) OR (E AND F) OR (G AND H)')
        t0: float = pyperf.perf_counter()
        normalize.normalize(conjunction)
        elapsed += (pyperf.perf_counter() - t0)
    return elapsed

BENCHMARKS: Dict[str, Callable[[int], float]] = {'parse': bench_parse, 'transpile': bench_transpile, 'optimize': bench_optimize, 'normalize': bench_normalize}

def add_cmdline_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd.append(args.benchmark)

def add_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('benchmark', choices=BENCHMARKS, help='Which benchmark to run.')

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'SQLGlot V2 benchmark'
    add_parser_args(runner.argparser)
    args: argparse.Namespace = runner.parse_args()
    benchmark: str = args.benchmark
    runner.bench_time_func(f'sqlglot_v2_{benchmark}', BENCHMARKS[benchmark])
