from typing import List, Tuple

DEFAULT_INIT_LEN: int = 100000
DEFAULT_RNG_SEED: int = 42
ALU: str = 'GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGACCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAATACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCAGCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGGAGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCCAGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA'
IUB: List[Tuple[str, List[float]]] = list(zip('acgtBDHKMNRSVWY', ([0.27, 0.12, 0.12, 0.27] + ([0.02] * 11)))
HOMOSAPIENS: List[Tuple[str, float]] = [('a', 0.302954942668), ('c', 0.1979883004921), ('g', 0.1975473066391), ('t', 0.3015094502008)]

def make_cumulative(table: List[Tuple[str, float]]) -> Tuple[List[float], List[int]]:
    P: List[float] = []
    C: List[int] = []
    prob: float = 0.0
    for (char, p) in table:
        prob += p
        P += [prob]
        C += [ord(char)]
    return (P, C)

def repeat_fasta(src: str, n: int, nprint) -> None:
    width: int = 60
    is_trailing_line: bool = False
    count_modifier: float = 0.0
    len_of_src: int = len(src)
    ss: str = ((src + src) + src[:(n % len_of_src)])
    s: bytearray = bytearray(ss, encoding='utf8')
    if (n % width):
        is_trailing_line = True
        count_modifier = 1.0
    count: float = 0
    end: float = ((n / float(width)) - count_modifier)
    while (count < end):
        i: int = ((count * 60) % len_of_src)
        nprint((s[i:(i + 60)] + b'\n'))
        count += 1
    if is_trailing_line:
        nprint((s[(- (n % width)):] + b'\n'))

def random_fasta(table: List[Tuple[str, List[float]]], n: int, seed: int, nprint) -> int:
    width: int = 60
    r: range = range(width)
    bb = bisect.bisect
    is_trailing_line: bool = False
    count_modifier: float = 0.0
    line: bytearray = bytearray((width + 1))
    (probs, chars) = make_cumulative(table)
    im: float = 139968.0
    seed: float = float(seed)
    if (n % width):
        is_trailing_line = True
        count_modifier = 1.0
    count: float = 0.0
    end: float = ((n / float(width)) - count_modifier)
    while (count < end):
        for i in r:
            seed = (((seed * 3877.0) + 29573.0) % 139968.0)
            line[i] = chars[bb(probs, (seed / im))]
        line[60] = 10
        nprint(line)
        count += 1.0
    if is_trailing_line:
        for i in range((n % width)):
            seed = (((seed * 3877.0) + 29573.0) % 139968.0)
            line[i] = chars[bb(probs, (seed / im))]
        nprint((line[:(i + 1)] + b'\n'))
    return seed

def init_benchmarks(n: int, rng_seed: int) -> bytes:
    result: bytearray = bytearray()
    nprint = result.extend
    nprint(b'>ONE Homo sapiens alu\n')
    repeat_fasta(ALU, (n * 2), nprint=nprint)
    nprint(b'>TWO IUB ambiguity codes\n')
    seed: int = random_fasta(IUB, (n * 3), seed=rng_seed, nprint=nprint)
    nprint(b'>THREE Homo sapiens frequency\n')
    random_fasta(HOMOSAPIENS, (n * 5), seed, nprint=nprint)
    return bytes(result)

VARIANTS: List[bytes] = (b'agggtaaa|tttaccct', b'[cgt]gggtaaa|tttaccc[acg]', b'a[act]ggtaaa|tttacc[agt]t', b'ag[act]gtaaa|tttac[agt]ct', b'agg[act]taaa|ttta[agt]cct', b'aggg[acg]aaa|ttt[cgt]ccct', b'agggt[cgt]aa|tt[acg]accct', b'agggta[cgt]a|t[acg]taccct', b'agggtaa[cgt]|[acg]ttaccct')
SUBST: List[Tuple[bytes, bytes]] = ((b'B', b'(c|g|t)'), (b'D', b'(a|g|t)'), (b'H', b'(a|c|t)'), (b'K', b'(g|t)'), (b'M', b'(a|c)'), (b'N', b'(a|c|g|t)'), (b'R', b'(a|g)'), (b'S', b'(c|g)'), (b'V', b'(a|c|g)'), (b'W', b'(a|t)'), (b'Y', b'(c|t)'))

def run_benchmarks(seq: bytes) -> Tuple[List[int], int, int, int]:
    ilen: int = len(seq)
    seq: bytes = re.sub(b'>.*\n|\n', b'', seq)
    clen: int = len(seq)
    results: List[int] = []
    for f in VARIANTS:
        results.append(len(re.findall(f, seq)))
    for (f, r) in SUBST:
        seq = re.sub(f, r, seq)
    return (results, ilen, clen, len(seq))

def bench_regex_dna(loops: int, seq: bytes, expected_res: Tuple[List[int], int, int, int]) -> float:
    range_it: range = range(loops)
    t0: float = pyperf.perf_counter()
    for i in range_it:
        res: Tuple[List[int], int, int, int] = run_benchmarks(seq)
    dt: float = (pyperf.perf_counter() - t0)
    if ((expected_res is not None) and (res != expected_res)):
        raise Exception('run_benchmarks() error')
    return dt

def add_cmdline_args(cmd: List[str], args) -> None:
    cmd.extend(('--fasta-length', str(args.fasta_length), '--rng-seed', str(args.rng_seed))

if (__name__ == '__main__'):
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'Test the performance of regexps using benchmarks from The Computer Language Benchmarks Game.'
    cmd: argparse.ArgumentParser = runner.argparser
    cmd.add_argument('--fasta-length', type=int, default=DEFAULT_INIT_LEN, help=('Length of the fasta sequence (default: %s)' % DEFAULT_INIT_LEN))
    cmd.add_argument('--rng-seed', type=int, default=DEFAULT_RNG_SEED, help=('Seed of the random number generator (default: %s)' % DEFAULT_RNG_SEED))
    args: argparse.Namespace = runner.parse_args()
    if (args.fasta_length == 100000):
        expected_len: int = 1016745
        expected_res: Tuple[List[int], int, int, int] = ([6, 26, 86, 58, 113, 31, 31, 32, 43], 1016745, 1000000, 1336326)
    else:
        expected_len: int = None
        expected_res: Tuple[List[int], int, int, int] = None
    runner.metadata['regex_dna_fasta_len'] = args.fasta_length
    runner.metadata['regex_dna_rng_seed'] = args.rng_seed
    seq: bytes = init_benchmarks(args.fasta_length, args.rng_seed)
    if ((expected_len is not None) and (len(seq) != expected_len)):
        raise Exception('init_benchmarks() error')
    runner.bench_time_func('regex_dna', bench_regex_dna, seq, expected_res)
