from array import array
import math
import pyperf
from typing import Optional, List, Tuple, Iterator, Union, Callable, Dict

class Array2D:
    def __init__(self, w: int, h: int, data: Optional[List[List[float]]] = None) -> None:
        self.width: int = w
        self.height: int = h
        self.data: array = array('d', [0.0] * (w * h))
        if data is not None:
            self.setup(data)

    def _idx(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return y * self.width + x
        raise IndexError

    def __getitem__(self, x_y: Tuple[int, int]) -> float:
        x, y = x_y
        return self.data[self._idx(x, y)]

    def __setitem__(self, x_y: Tuple[int, int], val: float) -> None:
        x, y = x_y
        self.data[self._idx(x, y)] = val

    def setup(self, data: List[List[float]]) -> 'Array2D':
        for y in range(self.height):
            for x in range(self.width):
                self[(x, y)] = data[y][x]
        return self

    def indexes(self) -> Iterator[Tuple[int, int]]:
        for y in range(self.height):
            for x in range(self.width):
                yield (x, y)

    def copy_data_from(self, other: 'Array2D') -> None:
        self.data[:] = other.data[:]

class Random:
    MDIG: int = 32
    ONE: int = 1
    m1: int = (ONE << (MDIG - 2)) + ((ONE << (MDIG - 2)) - ONE)
    m2: int = ONE << (MDIG // 2)
    dm1: float = 1.0 / float(m1)

    def __init__(self, seed: int) -> None:
        self.initialize(seed)
        self.left: float = 0.0
        self.right: float = 1.0
        self.width: float = 1.0
        self.haveRange: bool = False

    def initialize(self, seed: int) -> None:
        self.seed: int = seed
        seed = abs(seed)
        jseed: int = min(seed, self.m1)
        if jseed % 2 == 0:
            jseed -= 1
        k0: int = 9069 % self.m2
        k1: float = 9069 / self.m2
        j0: int = jseed % self.m2
        j1: float = jseed / self.m2
        self.m: array = array('d', [0.0] * 17)
        for iloop in range(17):
            jseed = j0 * k0
            j1 = ((jseed / self.m2) + (j0 * k1) + (j1 * k0)) % (self.m2 / 2)
            j0 = jseed % self.m2
            self.m[iloop] = j0 + (self.m2 * j1)
        self.i: int = 4
        self.j: int = 16

    def nextDouble(self) -> float:
        I: int
        J: int
        m: array
        I, J, m = self.i, self.j, self.m
        k: float = m[I] - m[J]
        if k < 0:
            k += self.m1
        m[J] = k
        if I == 0:
            I = 16
        else:
            I -= 1
        self.i = I
        if J == 0:
            J = 16
        else:
            J -= 1
        self.j = J
        if self.haveRange:
            return self.left + (self.dm1 * float(k) * self.width)
        else:
            return self.dm1 * float(k)

    def RandomMatrix(self, a: 'Array2D') -> 'Array2D':
        for x, y in a.indexes():
            a[(x, y)] = self.nextDouble()
        return a

    def RandomVector(self, n: int) -> array:
        return array('d', [self.nextDouble() for _ in range(n)])

def copy_vector(vec: array) -> array:
    vec2: array = array('d')
    vec2[:] = vec[:]
    return vec2

class ArrayList(Array2D):

    def __init__(self, w: int, h: int, data: Optional[List[List[float]]] = None) -> None:
        self.width: int = w
        self.height: int = h
        self.data: List[array] = [array('d', [0.0] * w) for _ in range(h)]
        if data is not None:
            self.setup(data)

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Union[array, float]:
        if isinstance(idx, tuple):
            return self.data[idx[1]][idx[0]]
        else:
            return self.data[idx]

    def __setitem__(self, idx: Union[int, Tuple[int, int]], val: float) -> None:
        if isinstance(idx, tuple):
            self.data[idx[1]][idx[0]] = val
        else:
            self.data[idx] = val

    def copy_data_from(self, other: 'ArrayList') -> None:
        for l1, l2 in zip(self.data, other.data):
            l1[:] = l2

def SOR_execute(omega: float, G: Array2D, cycles: int, Array: Callable[..., Array2D]) -> None:
    for _ in range(cycles):
        for y in range(1, G.height - 1):
            for x in range(1, G.width - 1):
                G[(x, y)] = (omega * 0.25 * (
                    G[(x, y - 1)] +
                    G[(x, y + 1)] +
                    G[(x - 1, y)] +
                    G[(x + 1, y)]
                ) + (1.0 - omega) * G[(x, y)])

def bench_SOR(loops: int, n: int, cycles: int, Array: Callable[..., Array2D]) -> float:
    t0: float = pyperf.perf_counter()
    for _ in range(loops):
        G: Array2D = Array(n, n)
        SOR_execute(1.25, G, cycles, Array)
    return pyperf.perf_counter() - t0

def SparseCompRow_matmult(
    M: int,
    y: array,
    val: array,
    row: array,
    col: array,
    x: array,
    num_iterations: int
) -> float:
    t0: float = pyperf.perf_counter()
    for _ in range(num_iterations):
        for r in range(M):
            sa: float = 0.0
            for i in range(row[r], row[r + 1]):
                sa += x[col[i]] * val[i]
            y[r] = sa
    return pyperf.perf_counter() - t0

def bench_SparseMatMult(cycles: int, N: int, nz: int) -> float:
    x: array = array('d', [0.0] * N)
    y: array = array('d', [0.0] * N)
    nr: int = nz // N
    anz: int = nr * N
    val: array = array('d', [0.0] * anz)
    col: array = array('i', [0] * nz)
    row: array = array('i', [0] * (N + 1))
    row[0] = 0
    for r in range(N):
        rowr: int = row[r]
        step: int = r // nr
        row[r + 1] = rowr + nr
        if step < 1:
            step = 1
        for i in range(nr):
            col[rowr + i] = i * step
    return SparseCompRow_matmult(M=N, y=y, val=val, row=row, col=col, x=x, num_iterations=cycles)

def MonteCarlo(Num_samples: int) -> float:
    rnd: Random = Random(113)
    under_curve: int = 0
    for _ in range(Num_samples):
        x: float = rnd.nextDouble()
        y: float = rnd.nextDouble()
        if (x * x + y * y) <= 1.0:
            under_curve += 1
    return (float(under_curve) / Num_samples) * 4.0

def bench_MonteCarlo(loops: int, Num_samples: int) -> float:
    t0: float = pyperf.perf_counter()
    for _ in range(loops):
        MonteCarlo(Num_samples)
    return pyperf.perf_counter() - t0

def LU_factor(A: Array2D, pivot: array) -> None:
    M: int = A.height
    N: int = A.width
    minMN: int = min(M, N)
    for j in range(minMN):
        jp: int = j
        t: float = abs(A[j, j])
        for i in range(j + 1, M):
            ab: float = abs(A[i, j])
            if ab > t:
                jp = i
                t = ab
        pivot[j] = jp
        if A[jp, j] == 0:
            raise Exception('factorization failed because of zero pivot')
        if jp != j:
            A.data[j], A.data[jp] = A.data[jp], A.data[j]
        if j < M - 1:
            recp: float = 1.0 / A[j, j]
            for k in range(j + 1, M):
                A[k, j] *= recp
        if j < minMN - 1:
            for ii in range(j + 1, M):
                for jj in range(j + 1, N):
                    A[ii, jj] -= A[ii, j] * A[j, jj]

def LU(lu: Array2D, A: Array2D, pivot: array) -> None:
    lu.copy_data_from(A)
    LU_factor(lu, pivot)

def bench_LU(cycles: int, N: int) -> float:
    rnd: Random = Random(7)
    A: ArrayList = rnd.RandomMatrix(ArrayList(N, N))
    lu: ArrayList = ArrayList(N, N)
    pivot: array = array('i', [0] * N)
    t0: float = pyperf.perf_counter()
    for _ in range(cycles):
        LU(lu, A, pivot)
    return pyperf.perf_counter() - t0

def int_log2(n: int) -> int:
    k: int = 1
    log: int = 0
    while k < n:
        k *= 2
        log += 1
    if n != (1 << log):
        raise Exception(f'FFT: Data length is not a power of 2: {n}')
    return log

def FFT_num_flops(N: int) -> float:
    return ((5.0 * N - 2) * int_log2(N) + 2 * (N + 1))

def FFT_transform_internal(N: int, data: array, direction: int) -> None:
    n: int = N // 2
    bit: int = 0
    dual: int = 1
    if n == 1:
        return
    logn: int = int_log2(n)
    if N == 0:
        return
    FFT_bitreverse(N, data)
    bit = 0
    while bit < logn:
        w_real: float = 1.0
        w_imag: float = 0.0
        theta: float = (2.0 * direction * math.pi) / (2.0 * float(dual))
        s: float = math.sin(theta)
        t: float = math.sin(theta / 2.0)
        s2: float = 2.0 * t * t
        for b in range(0, n, 2 * dual):
            i: int = 2 * b
            j: int = 2 * (b + dual)
            wd_real: float = data[j]
            wd_imag: float = data[j + 1]
            data[j] = data[i] - wd_real
            data[j + 1] = data[i + 1] - wd_imag
            data[i] += wd_real
            data[i + 1] += wd_imag
        for a in range(1, dual):
            tmp_real: float = w_real - s * w_imag - s2 * w_real
            tmp_imag: float = w_imag + s * w_real - s2 * w_imag
            w_real = tmp_real
            w_imag = tmp_imag
            for b in range(0, n, 2 * dual):
                i: int = 2 * (b + a)
                j: int = 2 * ((b + a) + dual)
                z1_real: float = data[j]
                z1_imag: float = data[j + 1]
                wd_real: float = w_real * z1_real - w_imag * z1_imag
                wd_imag: float = w_real * z1_imag + w_imag * z1_real
                data[j] = data[i] - wd_real
                data[j + 1] = data[i + 1] - wd_imag
                data[i] += wd_real
                data[i + 1] += wd_imag
        bit += 1
        dual *= 2

def FFT_bitreverse(N: int, data: array) -> None:
    n: int = N // 2
    nm1: int = n - 1
    j: int = 0
    for i in range(nm1):
        ii: int = i << 1
        jj: int = j << 1
        k: int = n >> 1
        if i < j:
            tmp_real: float = data[ii]
            tmp_imag: float = data[ii + 1]
            data[ii] = data[jj]
            data[ii + 1] = data[jj + 1]
            data[jj] = tmp_real
            data[jj + 1] = tmp_imag
        while k <= j:
            j -= k
            k >>= 1
        j += k

def FFT_transform(N: int, data: array) -> None:
    FFT_transform_internal(N, data, -1)

def FFT_inverse(N: int, data: array) -> None:
    FFT_transform_internal(N, data, +1)
    n: float = N / 2
    norm: float = 1.0 / n
    for i in range(N):
        data[i] *= norm

def bench_FFT(loops: int, N: int, cycles: int) -> float:
    twoN: int = 2 * N
    init_vec: array = Random(7).RandomVector(twoN)
    t0: float = pyperf.perf_counter()
    for _ in range(loops):
        x: array = copy_vector(init_vec)
        for _ in range(cycles):
            FFT_transform(twoN, x)
            FFT_inverse(twoN, x)
    return pyperf.perf_counter() - t0

def add_cmdline_args(cmd: List[str], args: 'argparse.Namespace') -> None:
    if getattr(args, 'benchmark', None):
        cmd.append(args.benchmark)

BENCHMARKS: Dict[str, Tuple[Callable, int, Union[int, Tuple[int, ...]], Optional[Callable]]] = {
    'sor': (bench_SOR, 100, 10, Array2D),
    'sparse_mat_mult': (bench_SparseMatMult, 1000, 50 * 1000, None),
    'monte_carlo': (bench_MonteCarlo, 100 * 1000, None, None),
    'lu': (bench_LU, 100, None, None),
    'fft': (bench_FFT, 1024, 50, None)
}

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.argparser.add_argument('benchmark', nargs='?', choices=sorted(BENCHMARKS))
    parsed_args = runner.parse_args()
    if parsed_args.benchmark:
        benchmarks: Tuple[str, ...] = (parsed_args.benchmark,)
    else:
        benchmarks = tuple(sorted(BENCHMARKS))
    for bench in benchmarks:
        name: str = f'scimark_{bench}'
        bench_args = BENCHMARKS[bench]
        # Filter out None values for arguments
        filtered_args = tuple(arg for arg in bench_args if arg is not None)
        runner.bench_time_func(name, *filtered_args)
