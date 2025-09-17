from collections import defaultdict
from itertools import product, permutations
from typing import List, Tuple, Optional, Sequence, Dict, Callable, Any
import copy
import itertools
import multiprocessing
import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import scipy.stats
import time
from joblib import Parallel, delayed
try:
    from joblib import parallel_config
except Exception:
    print('Some stuff might fail: issue in joblib')
import nevergrad as ng

default_budget: int = 300
default_steps: int = 100
default_order: float = 2
default_stepsize: float = 10
methods: Dict[str, Callable[..., Any]] = {}
metrics: Dict[str, Callable[..., float]] = {}

def normalize(x: List[np.ndarray]) -> List[np.ndarray]:
    for i in range(len(x)):
        normalization: float = np.sqrt(np.sum(x[i] ** 2.0))
        if normalization > 0.0:
            x[i] = x[i] / normalization
        else:
            x[i] = np.random.randn(np.prod(x[i].shape)).reshape(x[i].shape)
            x[i] = x[i] / np.sqrt(np.sum(x[i] ** 2.0))
    return x

def convo(x: np.ndarray, k: Optional[Sequence[float]]) -> np.ndarray:
    if k is None:
        return x
    return scipy.ndimage.gaussian_filter(x, sigma=list(k) + [0.0] * (len(x.shape) - len(k)))

def convo_mult(x: np.ndarray, k: Optional[Sequence[float]]) -> np.ndarray:
    if k is None:
        return x
    return scipy.ndimage.gaussian_filter(x, sigma=[0] + list(k) + [0.0] * (len(x.shape) - len(k) - 1))

def pure_random(n: int, shape: Tuple[int, ...], conv: Optional[Any] = None) -> List[np.ndarray]:
    return normalize([np.random.randn(*shape) for i in range(n)])

def antithetic_pm(n: int, shape: Tuple[int, ...], conv: Optional[Any] = None) -> np.ndarray:
    m: int = n // 2
    x: List[np.ndarray] = [np.random.randn(*shape) for i in range(m)]
    x = normalize(x)
    x = x + [-xi for xi in x]
    if len(x) < n:
        x = x + [np.random.randn(*shape)]
    return np.array(x)

def antithetic_order(n: int, shape: Tuple[int, ...], axis: int = -1, also_sym: bool = False, conv: Optional[Any] = None) -> List[np.ndarray]:
    x: List[np.ndarray] = []
    s: int = shape[axis]
    indices: List[Any] = [slice(0, s, 1) for s in shape]
    indices_sym: List[Any] = [slice(0, s, 1) for s in shape]
    while len(x) < n:
        icx: np.ndarray = normalize([np.random.randn(*shape)])[0]
        for p in permutations(range(s)):
            if len(x) < n:
                indices[axis] = p
                cx: np.ndarray = copy.deepcopy(icx)[tuple(indices)]
                x = x + [cx]
                if also_sym:
                    order: List[Tuple[bool, ...]] = list(itertools.product((False, True), repeat=s))
                    np.random.shuffle(order)
                    for ordering in order:
                        if any(ordering) and len(x) < n:
                            scx: np.ndarray = copy.deepcopy(cx)
                            for o in [i for i, o in enumerate(ordering) if o]:
                                indices_sym[axis] = o
                                scx[tuple(indices_sym)] = -scx[tuple(indices_sym)]
                            x = x + [scx]
    return x

def antithetic_order_and_sign(n: int, shape: Tuple[int, ...], axis: int = -1, conv: Optional[Any] = None) -> List[np.ndarray]:
    return antithetic_order(n, shape, axis, also_sym=True)

def manual_avg_pool3d(arr: np.ndarray, kernel_size: Tuple[int, int, int]) -> np.ndarray:
    output_shape: Tuple[int, int, int] = (arr.shape[0] // kernel_size[0],
                                           arr.shape[1] // kernel_size[1],
                                           arr.shape[2] // kernel_size[2])
    result: np.ndarray = np.zeros(output_shape)
    for z in range(output_shape[0]):
        for y in range(output_shape[1]):
            for x_idx in range(output_shape[2]):
                result[z, y, x_idx] = np.mean(arr[z * kernel_size[0]:(z + 1) * kernel_size[0],
                                                   y * kernel_size[1]:(y + 1) * kernel_size[1],
                                                   x_idx * kernel_size[2]:(x_idx + 1) * kernel_size[2]])
    return result

def max_pooling(n: int, shape: Tuple[int, ...], budget: int = default_budget, conv: Optional[Any] = None) -> np.ndarray:
    pooling: Tuple[int, ...] = tuple([max(1, s // 8) for s in shape])
    if conv is not None:
        pooling = (1, *conv)
    old_latents: List[np.ndarray] = []
    x: List[np.ndarray] = []
    for i in range(n):
        latents: np.ndarray = np.random.randn(*shape)
        latents_pooling: np.ndarray = manual_avg_pool3d(latents, pooling)
        if old_latents:
            dist: float = min([np.linalg.norm(latents_pooling - old) for old in old_latents])
            max_dist: float = dist
            t0: float = time.time()
            while time.time() - t0 < 0.01 * budget / n:
                latents_new: np.ndarray = np.random.randn(*shape)
                latents_pooling_new: np.ndarray = manual_avg_pool3d(latents_new, pooling)
                dist_new: float = min([np.linalg.norm(latents_pooling_new - old) for old in old_latents])
                if dist_new > max_dist:
                    latents = latents_new
                    max_dist = dist_new
                    latents_pooling = latents_pooling_new
        x.append(latents)
        old_latents.append(latents_pooling)
    x = np.stack(x)
    x = normalize(list(x))
    return x

def max_without_pooling(n: int, shape: Tuple[int, ...], budget: int = default_budget, conv: Optional[List[int]] = None) -> np.ndarray:
    return max_pooling(n, shape, budget, conv)

def max_small_pooling(n: int, shape: Tuple[int, ...], budget: int = default_budget, conv: Optional[List[int]] = [8, 8]) -> np.ndarray:
    return max_pooling(n, shape, budget, conv)

def greedy_dispersion(n: int, shape: Tuple[int, ...], budget: int = default_budget, conv: Optional[List[int]] = None) -> List[np.ndarray]:
    x: List[np.ndarray] = normalize([np.random.randn(*shape)])
    for i in range(n - 1):
        bigdist: float = -1
        t0: float = time.time()
        newy: np.ndarray = x[0]
        while time.time() < t0 + 0.01 * budget / n:
            def rand_and_dist(i: int) -> Tuple[np.ndarray, float]:
                y: np.ndarray = normalize([np.random.randn(*shape)])[0]
                dist: float = min((np.linalg.norm(convo(y, conv) - convo(x_elem, conv)) for x_elem in x))
                return (y, dist)
            with parallel_config(backend='threading'):
                r: List[Tuple[np.ndarray, float]] = Parallel(n_jobs=-1)(
                    (delayed(rand_and_dist)(i) for i in range(multiprocessing.cpu_count())))
            dists: List[float] = [r_elem[1] for r_elem in r]
            index: int = dists.index(max(dists))
            newy = r[index][0]
        x += [newy]
    return x

def dispersion(n: int, shape: Tuple[int, ...], budget: int = default_budget, conv: Optional[List[int]] = None) -> List[np.ndarray]:
    x: List[np.ndarray] = greedy_dispersion(n, shape, budget // 2, conv=conv)
    t0: float = time.time()
    num: int = n
    num_iterations: int = 0
    while time.time() < t0 + 0.01 * budget // 2:
        num += 1
        for j in range(len(x)):
            bigdist: float = -1
            def rand_and_dist(idx: int) -> Tuple[np.ndarray, float]:
                if idx > 0:
                    y: np.ndarray = normalize([np.random.randn(*shape)])[0]
                else:
                    y = x[j]
                convoy: np.ndarray = convo(y, conv)
                dist: float = min((np.linalg.norm(convoy - convo(x[i], conv)) for i in range(len(x)) if i != j))
                return (y, dist)
            with parallel_config(backend='threading'):
                num_jobs: int = max(2 * num, multiprocessing.cpu_count())
                r = Parallel(n_jobs=multiprocessing.cpu_count())((delayed(rand_and_dist)(i) for i in range(num_jobs)))
                num_iterations += num_jobs
            dists: List[float] = [r_elem[1] for r_elem in r]
            index: int = dists.index(max(dists))
            x[j] = r[index][0]
            if time.time() > t0 + 0.01 * budget // 2:
                break
        if time.time() > t0 + 0.01 * budget // 2:
            break
    score: float = metrics['metric_pack_big_conv'](x)
    return x

def dispersion_with_conv(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return dispersion(n, shape, budget=budget, conv=[8, 8])

def greedy_dispersion_with_conv(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return greedy_dispersion(n, shape, budget=budget, conv=[8, 8])

def dispersion_with_big_conv(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return dispersion(n, shape, budget=budget, conv=[24, 24])

def greedy_dispersion_with_big_conv(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return greedy_dispersion(n, shape, budget=budget, conv=[24, 24])

def dispersion_with_mini_conv(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return dispersion(n, shape, budget=budget, conv=[2, 2])

def greedy_dispersion_with_mini_conv(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return greedy_dispersion(n, shape, budget=budget, conv=[2, 2])

def Riesz_blurred_gradient(n: int, shape: Tuple[int, ...], budget: int = default_budget, order: float = default_order, step_size: float = default_stepsize, conv: Optional[List[int]] = None) -> np.ndarray:
    t_shape: Tuple[int, ...] = (n,) + tuple(shape)
    x: np.ndarray = np.random.randn(*t_shape)
    x = np.array(normalize(list(x)))
    t0: float = time.time()
    for steps in range(int(1000000000.0 * budget)):
        Temp: np.ndarray = np.zeros(t_shape)
        Blurred: np.ndarray = convo_mult(x, conv)
        for i in range(n):
            for j in range(n):
                if j != i:
                    T: np.ndarray = np.subtract(Blurred[i], Blurred[j])
                    factor: float = 1 / (np.sqrt(np.sum(T ** 2.0)) ** (order + 2))
                    Temp[i] = np.add(Temp[i], T * factor)
            Temp[i] = Temp[i] * step_size
        x = np.add(x, Temp)
        x = np.array(normalize(list(x)))
        if time.time() > t0 + 0.01 * budget:
            break
    return x

def Riesz_blursum_gradient(n: int, shape: Tuple[int, ...], budget: int = default_budget, order: float = default_order, step_size: float = default_stepsize, conv: Optional[List[int]] = None) -> np.ndarray:
    t_shape: Tuple[int, ...] = (n,) + tuple(shape)
    x: np.ndarray = np.random.randn(*t_shape)
    x = np.array(normalize(list(x)))
    t0: float = time.time()
    for steps in range(int(1000000000.0 * budget)):
        Blurred: np.ndarray = np.zeros(t_shape)
        for i in range(n):
            for j in range(n):
                if j != i:
                    T: np.ndarray = np.subtract(x[i], x[j])
                    factor: float = 1 / (np.sqrt(np.sum(T ** 2.0)) ** (order + 2))
                    Blurred[i] = np.add(Blurred[i], T * factor)
        Blurred = convo_mult(Blurred, conv)
        x = np.add(x, Blurred)
        x = np.array(normalize(list(x)))
        if time.time() > t0 + 0.01 * budget:
            break
    return x

def Riesz_noblur_gradient(n: int, shape: Tuple[int, ...], budget: int = default_budget, order: float = default_order, step_size: float = default_stepsize, conv: Optional[Any] = None) -> np.ndarray:
    t_shape: Tuple[int, ...] = (n,) + tuple(shape)
    x: np.ndarray = np.random.randn(*t_shape)
    x = np.array(normalize(list(x)))
    t0: float = time.time()
    for steps in range(int(1000000000.0 * budget)):
        Temp: np.ndarray = np.zeros(t_shape)
        for i in range(n):
            for j in range(n):
                if j != i:
                    T: np.ndarray = np.subtract(x[i], x[j])
                    factor: float = 1 / (np.sqrt(np.sum(T ** 2.0)) ** (order + 2))
                    Temp[i] = np.add(Temp[i], T * factor)
        x = np.add(x, Temp)
        x = np.array(normalize(list(x)))
        if time.time() > t0 + 0.01 * budget:
            break
    return x

def Riesz_noblur_lowconv_loworder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_noblur_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[2, 2])

def Riesz_noblur_lowconv_midorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_noblur_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[2, 2])

def Riesz_noblur_lowconv_highorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_noblur_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[2, 2])

def Riesz_blursum_lowconv_hugeorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[2, 2])

def Riesz_blursum_medconv_hugeorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[8, 8])

def Riesz_blursum_highconv_hugeorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[24, 24])

def Riesz_blursum_lowconv_tinyorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[2, 2])

def Riesz_blursum_medconv_tinyorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[8, 8])

def Riesz_blursum_highconv_tinyorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[24, 24])

def Riesz_blurred_lowconv_hugeorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[2, 2])

def Riesz_blurred_medconv_hugeorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[8, 8])

def Riesz_blurred_highconv_hugeorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[24, 24])

def Riesz_blurred_lowconv_tinyorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[2, 2])

def Riesz_blurred_medconv_tinyorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[8, 8])

def Riesz_blurred_highconv_tinyorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[24, 24])

def Riesz_blursum_bigconv_loworder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[24, 24])

def Riesz_blursum_bigconv_midorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[24, 24])

def Riesz_blursum_bigconv_highorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[24, 24])

def Riesz_blursum_medconv_loworder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[8, 8])

def Riesz_blursum_medconv_midorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[8, 8])

def Riesz_blursum_medconv_highorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[8, 8])

def Riesz_blursum_lowconv_loworder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[2, 2])

def Riesz_blursum_lowconv_midorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[2, 2])

def Riesz_blursum_lowconv_highorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[2, 2])

def Riesz_blurred_bigconv_loworder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[24, 24])

def Riesz_blurred_bigconv_midorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[24, 24])

def Riesz_blurred_bigconv_highorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[24, 24])

def Riesz_blurred_medconv_loworder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[8, 8])

def Riesz_blurred_medconv_midorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[8, 8])

def Riesz_blurred_medconv_highorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[8, 8])

def Riesz_blurred_lowconv_loworder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[2, 2])

def Riesz_blurred_lowconv_midorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[2, 2])

def Riesz_blurred_lowconv_highorder(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[2, 2])

def block_symmetry(n: int, shape: Tuple[int, ...], num_blocks: Optional[List[int]] = None) -> List[np.ndarray]:
    x: List[np.ndarray] = []
    if num_blocks is None:
        num_blocks = [4, 4]
    for pindex in range(n):
        newx: np.ndarray = normalize([np.random.randn(*shape)])[0]
        s: int = np.prod(num_blocks)
        num_blocks_extended: List[int] = num_blocks + [1] * (len(shape) - len(num_blocks))
        order: List[Tuple[bool, ...]] = list(itertools.product((False, True), repeat=s))
        np.random.shuffle(order)
        ranges: List[List[int]] = [list(range(nb)) for nb in num_blocks_extended]
        for o in order:
            tentativex: np.ndarray = copy.deepcopy(newx)
            for i, multi_index in enumerate(itertools.product(*ranges)):
                if o[i]:
                    slices: List[Any] = [None] * len(shape)
                    for c, p in enumerate(multi_index):
                        assert p >= 0
                        assert p < num_blocks_extended[c]
                        a: int = p * shape[c] // num_blocks_extended[c]
                        b: int = min((p + 1) * shape[c] // num_blocks_extended[c], shape[c])
                        slices[c] = slice(a, b)
                    slices_tuple = tuple(slices)
                    tentativex[slices_tuple] = -tentativex[slices_tuple]
            if len(x) >= n:
                return x
            x += [tentativex]
    return x

def big_block_symmetry(n: int, shape: Tuple[int, ...]) -> List[np.ndarray]:
    return block_symmetry(n, shape, num_blocks=[2, 2])

def covering(n: int, shape: Tuple[int, ...], budget: int = default_budget, conv: Optional[List[int]] = None) -> List[np.ndarray]:
    x: List[np.ndarray] = greedy_dispersion(n, shape, budget // 2, conv)
    mindists: List[float] = []
    c: float = 0.01
    previous_score: float = float('inf')
    num: int = 0
    t0: float = time.time()
    while time.time() < t0 + 0.01 * budget // 2:
        num += 1
        t_rand: np.ndarray = normalize([np.random.randn(*shape)])[0]
        convt: np.ndarray = convo(t_rand, conv)
        mindist: float = float('inf')
        index: int = 0
        for k in range(len(x)):
            dist: float = np.linalg.norm(convt - convo(x[k], conv))
            if dist < mindist:
                mindist = dist
                index = k
        mindists += [mindist]
        if len(mindists) % 2000 == 0:
            score: float = np.sum(mindists[-2000:]) / len(mindists[-2000:])
            c *= 2 if score < previous_score else 0.5
            previous_score = score
        x[index] = normalize([x[index] + c / (35 + n + np.sqrt(num)) * (t_rand - x[index])])[0]
    return x

def covering_conv(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return covering(n, shape, budget, conv=[8, 8])

def covering_mini_conv(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return covering(n, shape, budget, conv=[2, 2])

def get_class(x: np.ndarray, num_blocks: List[int], just_max: bool) -> int:
    shape = x.shape
    split_volume: int = len(num_blocks)
    num_blocks_extended: List[int] = num_blocks + [1] * (len(shape) - len(num_blocks))
    ranges: List[List[int]] = [list(range(nb)) for nb in num_blocks_extended]
    result: List[Any] = []
    for _i, multi_index in enumerate(itertools.product(*ranges)):
        slices: List[Any] = [None] * len(shape)
        for c, p in enumerate(multi_index):
            assert p >= 0
            assert p < num_blocks_extended[c]
            a: int = p * shape[c] // num_blocks_extended[c]
            b: int = min((p + 1) * shape[c] // num_blocks_extended[c], shape[c])
            slices[c] = slice(a, b)
        slices_tuple = tuple(slices)
        if just_max:
            result = result + [list(np.argsort(np.sum(x[slices_tuple], tuple(range(split_volume))).flatten()))[-1]]
        else:
            result = result + list(np.argsort(np.sum(x[slices_tuple], tuple(range(split_volume))).flatten()))
    return hash(str(result))

def jittered(n: int, shape: Tuple[int, ...], num_blocks: Optional[List[int]] = None, just_max: bool = False) -> List[np.ndarray]:
    if num_blocks is None:
        num_blocks = [2, 2]
    hash_to_set: Dict[Any, List[np.ndarray]] = defaultdict(list)
    for i in range(int(np.sqrt(n)) * n):
        x: np.ndarray = normalize([np.random.randn(*shape)])[0]
        hash_to_set[get_class(x, num_blocks, just_max)] += [x]
    min_num: int = 10000000
    max_num: int = -1
    while True:
        for k in hash_to_set.keys():
            min_num = min(min_num, len(hash_to_set[k]))
            max_num = max(max_num, len(hash_to_set[k]))
        if min_num < n / len(hash_to_set.keys()):
            x: np.ndarray = normalize([np.random.randn(*shape)])[0]
            hash_to_set[get_class(x, num_blocks, just_max)] += [x]
        else:
            break
    x_list: List[np.ndarray] = []
    while len(x_list) < n:
        num_take: int = max(1, (n - len(x_list)) // len(hash_to_set))
        for k in hash_to_set.keys():
            if len(x_list) < n:
                x_list += hash_to_set[k][:num_take]
            hash_to_set[k] = hash_to_set[k][num_take:]
    assert len(x_list) == n
    return x_list

def reduced_jittered(n: int, shape: Tuple[int, ...]) -> List[np.ndarray]:
    return jittered(n, shape, [2, 2], just_max=True)

def covering_big_conv(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return covering(n, shape, budget, [24, 24])

def lhs(n: int, shape: Tuple[int, ...]) -> List[np.ndarray]:
    num: int = np.prod(shape)
    x: np.ndarray = np.zeros([n, num])
    for i in range(num):
        xb: np.ndarray = 1.0 / n * np.random.rand(n)
        xplus: np.ndarray = np.linspace(0, n - 1, n) / n
        np.random.shuffle(xplus)
        x[:, i] = scipy.stats.norm.ppf(xb + xplus)
    thex: List[np.ndarray] = []
    for i in range(n):
        thex += normalize([x[i].reshape(*shape)])
    assert len(thex) == n
    assert thex[0].shape == tuple(shape), f' we get {x[0].shape} instead of {tuple(shape)}'
    return thex

def metric_half(x: List[np.ndarray], budget: int = default_budget, conv: Optional[List[int]] = None) -> float:
    shape: Tuple[int, ...] = x[0].shape
    t0: float = time.time()
    xconv: np.ndarray = np.array([convo(x_, conv).flatten() for x_ in x])
    scores: List[float] = []
    while time.time() < t0 + 0.01 * budget:
        y: np.ndarray = convo(normalize([np.random.randn(*shape)])[0], conv).flatten()
        scores += [np.average(np.matmul(xconv, y) > 0.0)]
    return float(np.average((np.array(scores) - 0.5) ** 2))

def metric_half_conv(x: List[np.ndarray], budget: int = default_budget) -> float:
    return metric_half(x, budget, conv=[8, 8])

def metric_cap(x: List[np.ndarray], budget: int = default_budget, conv: Optional[List[int]] = None) -> float:
    shape: Tuple[int, ...] = x[0].shape
    t0: float = time.time()
    c: float = 1.0 / np.sqrt(len(x[0].flatten()))
    xconv: np.ndarray = np.array(normalize([convo(x_, conv).flatten() for x_ in x]))
    scores: List[float] = []
    while time.time() < t0 + 0.01 * budget:
        y: np.ndarray = convo(normalize([np.random.randn(*shape)])[0], conv).flatten()
        scores += [np.average(np.matmul(xconv, y) > c)]
        scores += [np.average(np.matmul(xconv, y) < -c)]
    return float(np.std(np.array(scores)))

def metric_cap_conv(x: List[np.ndarray], budget: int = default_budget) -> float:
    return metric_cap(x, budget, conv=[8, 8])

def metric_pack_absavg(x: List[np.ndarray], budget: int = default_budget, conv: Optional[List[int]] = None) -> float:
    shape: Tuple[int, ...] = x[0].shape
    xconv: np.ndarray = np.array(normalize([convo(x_, conv).flatten() for x_ in x]))
    scores: np.ndarray = np.matmul(xconv, xconv.transpose())
    for i in range(len(scores)):
        assert 0.99 < scores[i, i] < 1.01
        scores[i, i] = 0
    scores_flat: np.ndarray = scores.flatten()
    assert len(scores_flat) == len(x) ** 2
    return float(np.average(np.abs(scores_flat)))

def metric_pack_absavg_conv(x: List[np.ndarray], budget: int = default_budget) -> float:
    return metric_pack_absavg(x, budget=default_budget, conv=[8, 8])

def metric_riesz_avg(x: List[np.ndarray], budget: int = default_budget, conv: Optional[List[int]] = None, r: float = 1.0) -> float:
    shape: Tuple[int, ...] = x[0].shape
    xconv: np.ndarray = np.array(normalize([convo(x_, conv).flatten() for x_ in x]))
    scores: List[float] = []
    for i in range(len(xconv)):
        for j in range(i):
            scores += [np.linalg.norm(xconv[i] - xconv[j]) ** (-r)]
    return float(np.average(scores))

def metric_riesz_avg2(x: List[np.ndarray], budget: int = default_budget, conv: Optional[List[int]] = None, r: float = 2.0) -> float:
    return metric_riesz_avg(x, budget=budget, conv=conv, r=2.0)

def metric_riesz_avg05(x: List[np.ndarray], budget: int = default_budget, conv: Optional[List[int]] = None, r: float = 0.5) -> float:
    return metric_riesz_avg(x, budget=budget, conv=conv, r=0.5)

def metric_riesz_avg_conv(x: List[np.ndarray], budget: int = default_budget, conv: List[int] = [8, 8], r: float = 1.0) -> float:
    return metric_riesz_avg(x, budget=default_budget, conv=conv, r=r)

def metric_riesz_avg_conv2(x: List[np.ndarray], budget: int = default_budget, conv: List[int] = [8, 8], r: float = 2.0) -> float:
    return metric_riesz_avg(x, budget=default_budget, conv=conv, r=r)

def metric_riesz_avg_conv05(x: List[np.ndarray], budget: int = default_budget, conv: List[int] = [8, 8], r: float = 0.5) -> float:
    return metric_riesz_avg(x, budget=default_budget, conv=conv, r=r)

def metric_pack_avg(x: List[np.ndarray], budget: int = default_budget, conv: Optional[List[int]] = None) -> float:
    shape: Tuple[int, ...] = x[0].shape
    xconv: np.ndarray = np.array(normalize([convo(x_, conv).flatten() for x_ in x]))
    scores: np.ndarray = np.matmul(xconv, xconv.transpose())
    for i in range(len(scores)):
        assert 0.99 < scores[i, i] < 1.01
        scores[i, i] = 0
    scores_flat: np.ndarray = scores.flatten()
    assert len(scores_flat) == len(x) ** 2
    return float(np.average(scores_flat))

def metric_pack_avg_conv(x: List[np.ndarray], budget: int = default_budget) -> float:
    return metric_pack_avg(x, budget=default_budget, conv=[8, 8])

def metric_pack(x: List[np.ndarray], budget: int = default_budget, conv: Optional[List[int]] = None) -> float:
    shape: Tuple[int, ...] = x[0].shape
    xconv: np.ndarray = np.array(normalize([convo(x_, conv).flatten() for x_ in x]))
    scores: np.ndarray = np.matmul(xconv, xconv.transpose())
    for i in range(len(scores)):
        assert 0.99 < scores[i, i] < 1.01, 'we get score ' + str(scores[i, i])
        scores[i, i] = 0
    scores_flat: np.ndarray = scores.flatten()
    assert len(scores_flat) == len(x) ** 2
    return float(max(scores_flat))

def metric_pack_conv(x: List[np.ndarray], budget: int = default_budget) -> float:
    return metric_pack(x, budget=default_budget, conv=[8, 8])

def metric_pack_big_conv(x: List[np.ndarray], budget: int = default_budget) -> float:
    return metric_pack(x, budget=default_budget, conv=[24, 24])

list_of_methods: List[str] = ['ng_TwoPointsDE', 'ng_DE', 'ng_PSO', 'ng_OnePlusOne', 'ng_DiagonalCMA', 'lhs', 'reduced_jittered', 'jittered', 'big_block_symmetry', 'block_symmetry', 'greedy_dispersion', 'dispersion', 'pure_random', 'antithetic_pm', 'dispersion', 'antithetic_order', 'antithetic_order_and_sign', 'dispersion_with_conv', 'dispersion_with_big_conv', 'greedy_dispersion_with_big_conv', 'dispersion_with_mini_conv', 'greedy_dispersion_with_mini_conv', 'covering', 'covering_conv', 'covering_mini_conv', 'rs', 'rs_mhc', 'rs_pack', 'rs_pa', 'rs_pc', 'rs_pac', 'rs_cap', 'rs_cc', 'rs_all', 'rs_ra', 'rs_ra2', 'rs_ra05', 'rs_rac', 'rs_rac2', 'rs_rac05', 'Riesz_blurred_bigconv_loworder', 'Riesz_blurred_bigconv_midorder', 'Riesz_blurred_bigconv_highorder', 'Riesz_blurred_medconv_loworder', 'Riesz_blurred_medconv_midorder', 'Riesz_blurred_medconv_highorder', 'Riesz_blurred_lowconv_loworder', 'Riesz_blurred_lowconv_midorder', 'Riesz_blurred_lowconv_highorder', 'Riesz_blursum_lowconv_hugeorder', 'Riesz_blursum_medconv_hugeorder', 'Riesz_blursum_highconv_hugeorder', 'Riesz_blursum_lowconv_tinyorder', 'Riesz_blursum_medconv_tinyorder', 'Riesz_blursum_highconv_tinyorder', 'Riesz_blurred_lowconv_hugeorder', 'Riesz_blurred_medconv_hugeorder', 'Riesz_blurred_highconv_hugeorder', 'Riesz_blurred_lowconv_tinyorder', 'Riesz_blurred_medconv_tinyorder', 'Riesz_blurred_highconv_tinyorder', 'Riesz_noblur_lowconv_loworder', 'Riesz_noblur_lowconv_midorder', 'Riesz_noblur_lowconv_highorder', 'Riesz_blursum_bigconv_loworder', 'Riesz_blursum_bigconv_midorder', 'Riesz_blursum_bigconv_highorder', 'Riesz_blursum_medconv_loworder', 'Riesz_blursum_medconv_midorder', 'Riesz_blursum_medconv_highorder', 'Riesz_blursum_lowconv_loworder', 'Riesz_blursum_lowconv_midorder', 'Riesz_blursum_lowconv_highorder', 'max_pooling', 'max_without_pooling', 'max_small_pooling']
list_metrics: List[str] = ['metric_half', 'metric_half_conv', 'metric_pack', 'metric_pack_conv', 'metric_pack_big_conv', 'metric_pack_avg', 'metric_pack_avg_conv', 'metric_pack_absavg', 'metric_pack_absavg_conv', 'metric_cap', 'metric_cap_conv', 'metric_riesz_avg', 'metric_riesz_avg2', 'metric_riesz_avg05', 'metric_riesz_avg_conv', 'metric_riesz_avg_conv2', 'metric_riesz_avg_conv05']
for u in list_metrics:
    metrics[u] = eval(u)

def rs(n: int, shape: Tuple[int, ...], budget: int = default_budget, k: str = 'metric_half', ngtool: Optional[str] = None) -> List[np.ndarray]:
    bestm: float = float('inf')
    if ngtool is not None:
        opt = ng.optimizers.registry[ngtool](ng.p.Array(shape=tuple([n] + list(shape))), budget=10000000000000)
    t0: float = time.time()
    bestx: Optional[List[np.ndarray]] = None
    while time.time() < t0 + 0.01 * budget or bestx is None:
        if ngtool is None:
            x: List[np.ndarray] = pure_random(n, shape)
        else:
            candidate = opt.ask()
            x = list(candidate.value)
            assert len(x) == n
            x = normalize(x)
        if k == 'all':
            m: float = np.sum([metrics[k2](x, budget / len(list_metrics) / max(10, np.sqrt(budget / 100))) for k2 in list_metrics])
        else:
            m = metrics[k](x, int(budget / max(10, np.sqrt(budget / 100))))
        if ngtool is not None:
            print('ng gets ', m)
            opt.tell(candidate, m)
        if m < bestm:
            bestm = m
            bestx = x
    return bestx  # type: ignore

def rs_mhc(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='metric_half_conv')

def rs_cap(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='metric_cap')

def rs_cc(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='metric_cap_conv')

def rs_pack(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='metric_pack')

def rs_ra(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='metric_riesz_avg')

def rs_ra2(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='metric_riesz_avg2')

def rs_ra05(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='metric_riesz_avg05')

def rs_rac(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='metric_riesz_avg_conv')

def rs_rac2(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='metric_riesz_avg_conv2')

def rs_rac05(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='metric_riesz_avg_conv05')

def rs_pa(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='metric_pack_avg')

def rs_pc(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='metric_pack_conv')

def rs_pac(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='metric_pack_avg_conv')

def rs_all(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='all')

def ng_TwoPointsDE(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='all', ngtool='TwoPointsDE')

def ng_DE(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='all', ngtool='DE')

def ng_PSO(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='all', ngtool='PSO')

def ng_OnePlusOne(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='all', ngtool='OnePlusOne')

def ng_DiagonalCMA(n: int, shape: Tuple[int, ...], budget: int = default_budget) -> List[np.ndarray]:
    return rs(n, shape, budget, k='all', ngtool='DiagonalCMA')

data: defaultdict = defaultdict(lambda: defaultdict(list))

def do_plot(tit: str, values: Dict[str, List[float]]) -> None:
    import matplotlib.pyplot as plt
    plt.clf()
    plt.title(tit.replace('_', ' '))
    x_vals: np.ndarray = np.cos(np.linspace(0.0, 2 * 3.14159, 20))
    y_vals: np.ndarray = np.sin(np.linspace(0.0, 2 * 3.14159, 20))
    sorted_keys = sorted(values.keys(), key=lambda k: np.average(values[k]))
    for i, v in enumerate(sorted_keys):
        print(f'context {tit}, {v} ==> {values[v]}')
        plt.plot([i + r for r in x_vals], [np.average(values[v]) + r * np.std(values[v]) for r in y_vals])
        plt.text(i, np.average(values[v]) + np.std(values[v]), f'{v}', rotation=30)
        if i > 0:
            plt.savefig(f'comparison_{tit}_time{default_budget}.png'.replace(' ', '_').replace(']', '_').replace('[', '_'))
    plt.clf()

def heatmap(y: List[str], x: List[str], table: np.ndarray, name: str) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    for cn in ['viridis', 'plasma', 'inferno', 'magma', 'cividis']:
        print(f'Creating a heatmap with name {name}: {table}')
        plt.clf()
        fig, ax = plt.subplots()
        tab = copy.deepcopy(table)
        for j in range(len(tab[0, :])):
            for i in range(len(tab)):
                tab[i, j] = np.average(table[:, j] < table[i, j])
        print(tab)
        im = ax.imshow(tab, aspect='auto', cmap=mpl.colormaps[cn])
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x)
        ax.set_yticks(np.arange(len(y)))
        ax.set_yticklabels(y)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        for i in range(len(y)):
            for j in range(len(x)):
                ax.text(j, i, str(tab[i, j])[:4], ha='center', va='center', color='k')
        fig.tight_layout()
        plt.savefig(f'TIME{default_budget}_cmap{cn}' + name)
        plt.clf()

def parallel_create_statistics(n: int, shape: Tuple[int, ...], list_of_methods: List[str], list_of_metrics: List[str], num: int = 1) -> None:
    shape_int: List[int] = [int(s) for s in list(shape)]
    for _ in range(num):
        def deal_with_method(method: str) -> List[float]:
            print(f'{method}')
            x: List[np.ndarray] = methods[method](n, tuple(shape_int))
            np.array(x).tofile(f'pointset_{n}_{shape_int}_{method}_{default_budget}_{np.random.randint(50000)}.dat'.replace(' ', '_').replace('[', ' ').replace(']', ' '))
            print(f'{method}({n}, {shape_int}) created in time {default_budget}')
            metrics_values: List[float] = []
            for k in list_of_metrics:
                m: float = metrics[k](x)
                metrics_values += [m]
                print(f'{method}({n}, {shape_int}) evaluated for {k}')
            return metrics_values
        results: List[List[float]] = Parallel(n_jobs=70)(
            (delayed(deal_with_method)(method) for method in list_of_methods))
        for i, method in enumerate(list_of_methods):
            for j, k in enumerate(list_of_metrics):
                data[k][method] += [results[i][j]]
        for k in list_of_metrics:
            if len(data[k]) > 1:
                do_plot(k + f'_number{n}_shape{shape_int}', data[k])
        tab: np.ndarray = np.array([[np.average(data[k][method]) for k in list_of_metrics] for method in list_of_methods])
        print('we have ', tab)
        heatmap(list_of_methods, list_of_metrics, tab, str(shape_int) + '_' + str(n) + '_' + str(np.random.randint(50000)) + '_bigartifcompa.png')

def bigcheck() -> None:
    n: int = 20
    shape: Tuple[int, int, int] = (8, 8, 3)
    for k in ['lhs', 'reduced_jittered', 'jittered', 'big_block_symmetry', 'block_symmetry', 'greedy_dispersion', 'dispersion', 'pure_random', 'antithetic_pm', 'dispersion', 'antithetic_order', 'antithetic_order_and_sign', 'dispersion_with_conv', 'dispersion_with_big_conv', 'greedy_dispersion_with_big_conv', 'dispersion_with_mini_conv', 'greedy_dispersion_with_mini_conv', 'covering', 'covering_conv', 'covering_mini_conv', 'Riesz_blurred_bigconv_highorder', 'Riesz_blursum_bigconv_highorder']:
        print('Starting to play with ', k)
        eval(f'{k}(n, shape)')
        print(f' {k} has been used for generating a batch of {n} points with shape {shape}')

def get_a_point_set(n: int, shape: Tuple[int, ...], method: Optional[str] = None) -> Tuple[str, List[np.ndarray]]:
    k: str = np.random.choice(list_of_methods)
    if method is not None:
        assert method in list_of_methods, f'{method} is unknown.'
        k = method
    print('Working with ', k)
    x: List[np.ndarray] = eval(f'{k}(n, shape)')
    for i in range(len(x)):
        assert 0.999 < np.linalg.norm(x[i]) < 1.001, 'we have norm ' + str(np.linalg.norm(x[i]))
    return k, x

def quasi_randomize(pointset: List[np.ndarray], method: Optional[str] = None) -> List[np.ndarray]:
    n: int = len(pointset)
    shape: Tuple[int, ...] = tuple(int(i) for i in list(pointset[0].shape))
    norms: List[float] = [np.linalg.norm(pointset[i]) for i in range(n)]
    if method is None or method == 'none':
        method = 'dispersion_with_big_conv' if len(shape) > 1 and shape[0] > 1 else 'covering'
    _k, x = get_a_point_set(n, shape, method)
    x = normalize(x)
    for i in range(n):
        x[i] = norms[i] * x[i]
    return x