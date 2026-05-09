import multiprocessing
import scipy
import scipy.signal
import scipy.stats
import time
import copy
import numpy as np
import itertools
from joblib import Parallel, delayed
try:
    from joblib import parallel_config
except:
    print('Some stuff might fail: issue in joblib')
from collections import defaultdict
import nevergrad as ng
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import os
import sys
import matplotlib
matplotlib.use('Agg')
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

def normalize(x: np.ndarray) -> np.ndarray:
    for i in range(len(x)):
        normalization = np.sqrt(np.sum(x[i] ** 2.0))
        if normalization > 0.0:
            x[i] = x[i] / normalization
        else:
            x[i] = np.random.randn(np.prod(x[i].shape)).reshape(x[i].shape)
            x[i] = x[i] / np.sqrt(np.sum(x[i] ** 2.0))
    return x

def convo(x: np.ndarray, k: list | None) -> np.ndarray:
    if k is None:
        return x
    return scipy.ndimage.gaussian_filter(x, sigma=list(k) + [0.0] * (len(x.shape) - len(k)))

def convo_mult(x: np.ndarray, k: list | None) -> np.ndarray:
    if k is None:
        return x
    return scipy.ndimage.gaussian_filter(x, sigma=[0] + list(k) + [0.0] * (len(x.shape) - len(k) - 1))

def pure_random(n: int, shape: tuple, conv: list | None = None) -> np.ndarray:
    return normalize([np.random.randn(*shape) for i in range(n)])

def antithetic_pm(n: int, shape: tuple, conv: list | None = None) -> np.ndarray:
    m = n // 2
    x = [np.random.randn(*shape) for i in range(m)]
    x = normalize(x)
    x = x + [-xi for xi in x]
    if len(x) < n:
        x = x + [np.random.randn(*shape)]
    return np.array(x)

def antithetic_order(n: int, shape: tuple, axis: int = -1, also_sym: bool = False, conv: list | None = None) -> np.ndarray:
    x = []
    s = shape[axis]
    indices = [slice(0, s, 1) for s in shape]
    indices_sym = [slice(0, s, 1) for s in shape]
    while len(x) < n:
        icx = normalize([np.random.randn(*shape)])[0]
        for p in itertools.permutations(range(s)):
            if len(x) < n:
                indices[axis] = p
                cx = copy.deepcopy(icx)[tuple(indices)]
                x = x + [cx]
                if also_sym:
                    order = list(itertools.product((False, True), repeat=s))
                    np.random.shuffle(order)
                    for ordering in order:
                        if any(ordering) and len(x) < n:
                            scx = copy.deepcopy(cx)
                            for o in [i for i, o in enumerate(ordering) if o]:
                                indices_sym[axis] = o
                                scx[tuple(indices_sym)] = -scx[tuple(indices_sym)]
                            x = x + [scx]
    return x

def antithetic_order_and_sign(n: int, shape: tuple, axis: int = -1, conv: list | None = None) -> np.ndarray:
    return antithetic_order(n, shape, axis, also_sym=True, conv=conv)

def manual_avg_pool3d(arr: np.ndarray, kernel_size: tuple) -> np.ndarray:
    output_shape = (arr.shape[0] // kernel_size[0], arr.shape[1] // kernel_size[1], arr.shape[2] // kernel_size[2])
    result = np.zeros(output_shape)
    for z in range(output_shape[0]):
        for y in range(output_shape[1]):
            for x in range(output_shape[2]):
                result[z, y, x] = np.mean(arr[z * kernel_size[0]:(z + 1) * kernel_size[0], y * kernel_size[1]:(y + 1) * kernel_size[1], x * kernel_size[2]:(x + 1) * kernel_size[2]])
    return result

def max_pooling(n: int, shape: tuple, budget: int = 300, conv: list | None = None) -> np.ndarray:
    pooling = tuple([max(1, s // 8) for s in shape])
    if conv != None:
        pooling = (1, *conv)
    old_latents = []
    x = []
    for i in range(n):
        latents = np.random.randn(*shape)
        latents_pooling = manual_avg_pool3d(latents, pooling)
        if old_latents:
            dist = min([np.linalg.norm(latents_pooling - old) for old in old_latents])
            max_dist = dist
            t0 = time.time()
            while time.time() - t0 < 0.01 * budget / n:
                latents_new = np.random.randn(*shape)
                latents_pooling_new = manual_avg_pool3d(latents_new, pooling)
                dist_new = min([np.linalg.norm(latents_pooling_new - old) for old in old_latents])
                if dist_new > max_dist:
                    latents = latents_new
                    max_dist = dist_new
                    latents_pooling = latents_pooling_new
        x.append(latents)
        old_latents.append(latents_pooling)
    x = np.stack(x)
    x = normalize(x)
    return x

def max_without_pooling(n: int, shape: tuple, budget: int = 300, conv: list | None = [1, 1]) -> np.ndarray:
    return max_pooling(n, shape, budget, conv)

def max_small_pooling(n: int, shape: tuple, budget: int = 300, conv: list | None = [8, 8]) -> np.ndarray:
    return max_pooling(n, shape, budget, conv)

def greedy_dispersion(n: int, shape: tuple, budget: int = 300, conv: list | None = None) -> np.ndarray:
    x = normalize([np.random.randn(*shape)])
    for i in range(n - 1):
        bigdist = -1
        t0 = time.time()
        while time.time() < t0 + 0.01 * budget / n:

            def rand_and_dist(i: int):
                y = normalize([np.random.randn(*shape)])[0]
                dist = min((np.linalg.norm(convo(y, conv) - convo(x[i], conv)) for i in range(len(x))))
                return (y, dist)
            with parallel_config(backend='threading'):
                r = Parallel(n_jobs=-1)((delayed(rand_and_dist)(i) for i in range(multiprocessing.cpu_count())))
            dist = [r[i][1] for i in range(len(r))]
            index = dist.index(max(dist))
            newy = r[index][0]
        x += [newy]
    return x

def dispersion(n: int, shape: tuple, budget: int = 300, conv: list | None = None) -> np.ndarray:
    x = greedy_dispersion(n, shape, budget / 2, conv=conv)
    t0 = time.time()
    num = n
    num_iterations = 0
    while time.time() < t0 + 0.01 * budget / 2:
        num = num + 1
        for j in range(len(x)):
            bigdist = -1

            def rand_and_dist(idx: int):
                if idx > 0:
                    y = normalize([np.random.randn(*shape)])[0]
                else:
                    y = x[j]
                convoy = convo(y, conv)
                dist = min((np.linalg.norm(convoy - convo(x[i], conv)) for i in range(len(x)) if i != j))
                return (y, dist)
            with parallel_config(backend='threading'):
                num_jobs = max(2 * num, multiprocessing.cpu_count())
                r = Parallel(n_jobs=multiprocessing.cpu_count())((delayed(rand_and_dist)(i) for i in range(num_jobs)))
                num_iterations += num_jobs
            dist = [r[i][1] for i in range(len(r))]
            index = dist.index(max(dist))
            x[j] = r[index][0]
            if time.time() > t0 + 0.01 * budget / 2:
                break
        if time.time() > t0 + 0.01 * budget / 2:
            break
    score = metrics['metric_pack_big_conv'](x)
    return x

def dispersion_with_conv(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return dispersion(n, shape, budget=budget, conv=[8, 8])

def greedy_dispersion_with_conv(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return greedy_dispersion(n, shape, budget=budget, conv=[8, 8])

def dispersion_with_big_conv(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return dispersion(n, shape, budget=budget, conv=[24, 24])

def greedy_dispersion_with_big_conv(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return greedy_dispersion(n, shape, budget=budget, conv=[24, 24])

def dispersion_with_mini_conv(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return dispersion(n, shape, budget=budget, conv=[2, 2])

def greedy_dispersion_with_mini_conv(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return greedy_dispersion(n, shape, budget=budget, conv=[2, 2])

def Riesz_blurred_gradient(n: int, shape: tuple, budget: int = 300, order: int = 2, step_size: float = 10, conv: list | None = None) -> np.ndarray:
    t = (n,) + tuple(shape)
    x = np.random.randn(*t)
    x = normalize(x)
    t0 = time.time()
    for steps in range(int(1000000000.0 * budget)):
        Temp = np.zeros(t)
        Blurred = convo_mult(x, conv)
        for i in range(n):
            for j in range(n):
                if j != i:
                    T = np.add(Blurred[i], -Blurred[j])
                    Temp[i] = np.add(Temp[i], np.multiply(T, 1 / np.sqrt(np.sum(T ** 2.0)) ** (order + 2)))
            Temp[i] = np.multiply(Temp[i], step_size)
        x = np.add(x, Temp)
        x = normalize(x)
        if time.time() > t0 + 0.01 * budget:
            break
    return x

def Riesz_blursum_gradient(n: int, shape: tuple, budget: int = 300, order: int = 2, step_size: float = 10, conv: list | None = None) -> np.ndarray:
    t = (n,) + tuple(shape)
    x = np.random.randn(*t)
    x = normalize(x)
    t0 = time.time()
    for steps in range(int(1000000000.0 * budget)):
        Blurred = np.zeros(t)
        for i in range(n):
            for j in range(n):
                if j != i:
                    T = np.add(x[i], -x[j])
                    Blurred[i] = np.add(np.multiply(T, 1 / np.sqrt(np.sum(T ** 2.0)) ** (order + 2)), Blurred[i])
        Blurred = convo_mult(Blurred, conv)
        x = np.add(x, Blurred)
        x = normalize(x)
        if time.time() > t0 + 0.01 * budget:
            break
    return x

def Riesz_noblur_gradient(n: int, shape: tuple, budget: int = 300, order: int = 2, step_size: float = 10, conv: list | None = None) -> np.ndarray:
    t = (n,) + tuple(shape)
    x = np.random.randn(*t)
    x = normalize(x)
    t0 = time.time()
    for steps in range(int(1000000000.0 * budget)):
        Temp = np.zeros(t)
        for i in range(n):
            for j in range(n):
                if j != i:
                    T = np.add(x[i], -x[j])
                    Temp[i] = np.add(Temp[i], np.multiply(T, 1 / np.sqrt(np.sum(T ** 2.0)) ** (order + 2)))
        x = np.add(x, Temp)
        x = normalize(x)
        if time.time() > t0 + 0.01 * budget:
            break
    return x

def Riesz_noblur_lowconv_loworder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_noblur_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[2, 2])

def Riesz_noblur_lowconv_midorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_noblur_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[2, 2])

def Riesz_noblur_lowconv_highorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_noblur_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[2, 2])

def Riesz_blursum_lowconv_hugeorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[2, 2])

def Riesz_blursum_medconv_hugeorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[8, 8])

def Riesz_blursum_highconv_hugeorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[24, 24])

def Riesz_blursum_lowconv_tinyorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[2, 2])

def Riesz_blursum_medconv_tinyorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[8, 8])

def Riesz_blursum_highconv_tinyorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[24, 24])

def Riesz_blurred_lowconv_hugeorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[2, 2])

def Riesz_blurred_medconv_hugeorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[8, 8])

def Riesz_blurred_highconv_hugeorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[24, 24])

def Riesz_blurred_lowconv_tinyorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[2, 2])

def Riesz_blurred_medconv_tinyorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[8, 8])

def Riesz_blurred_highconv_tinyorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[24, 24])

def Riesz_blursum_bigconv_loworder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[24, 24])

def Riesz_blursum_bigconv_midorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[24, 24])

def Riesz_blursum_bigconv_highorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[24, 24])

def Riesz_blursum_medconv_loworder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[8, 8])

def Riesz_blursum_medconv_midorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[8, 8])

def Riesz_blursum_medconv_highorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[8, 8])

def Riesz_blursum_lowconv_loworder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[2, 2])

def Riesz_blursum_lowconv_midorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[2, 2])

def Riesz_blursum_lowconv_highorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[2, 2])

def Riesz_blurred_bigconv_loworder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[24, 24])

def Riesz_blurred_bigconv_midorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[24, 24])

def Riesz_blurred_bigconv_highorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[24, 24])

def Riesz_blurred_medconv_loworder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[8, 8])

def Riesz_blurred_medconv_midorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[8, 8])

def Riesz_blurred_medconv_highorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[8, 8])

def Riesz_blurred_lowconv_loworder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[2, 2])

def Riesz_blurred_lowconv_midorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[2, 2])

def Riesz_blurred_lowconv_highorder(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[2, 2])

def block_symmetry(n: int, shape: tuple, num_blocks: list | None = None) -> np.ndarray:
    x = []
    if num_blocks is None:
        num_blocks = [4, 4]
    for pindex in range(n):
        newx = normalize([np.random.randn(*shape)])[0]
        s = np.prod(num_blocks)
        num_blocks = num_blocks + [1] * (len(shape) - len(num_blocks))
        order = list(itertools.product((False, True), repeat=s))
        np.random.shuffle(order)
        ranges = [list(range(n)) for n in num_blocks]
        for o in order:
            tentativex = copy.deepcopy(newx)
            for i, multi_index in enumerate(itertools.product(*ranges)):
                if o[i]:
                    slices = [[]] * len(shape)
                    for c, p in enumerate(multi_index):
                        assert p >= 0
                        assert p < num_blocks[c]
                        a = p * shape[c] // num_blocks[c]
                        b = min((p + 1) * shape[c] // num_blocks[c], shape[c])
                        slices[c] = slice(a, b)
                    slices = tuple(slices)
                    tentativex[slices] = -tentativex[slices]
            if len(x) >= n:
                return x
            x += [tentativex]

def big_block_symmetry(n: int, shape: tuple) -> np.ndarray:
    return block_symmetry(n, shape, num_blocks=[2, 2])

def covering(n: int, shape: tuple, budget: int = 300, conv: list | None = None) -> np.ndarray:
    x = greedy_dispersion(n, shape, budget / 2, conv)
    mindists = []
    c = 0.01
    previous_score = float('inf')
    num = 0
    t0 = time.time()
    while time.time() < t0 + 0.01 * budget / 2:
        num = num + 1
        t = normalize([np.random.randn(*shape)])[0]
        convt = convo(t, conv)
        mindist = float('inf')
        for k in range(len(x)):
            dist = np.linalg.norm(convt - convo(x[k], conv))
            if dist < mindist:
                mindist = dist
                index = k
        mindists += [mindist]
        if len(mindists) % 2000 == 0:
            score = np.sum(mindists[-2000:]) / len(mindists[-2000:])
            c *= 2 if score < previous_score else 0.5
            previous_score = score
        x[index] = normalize([x[index] + c / (35 + n + np.sqrt(num)) * (t - x[index])])[0]
    return x

def covering_conv(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return covering(n, shape, budget, conv=[8, 8])

def covering_mini_conv(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return covering(n, shape, budget, conv=[2, 2])

def get_class(x: np.ndarray, num_blocks: list, just_max: bool = False) -> int:
    shape = x.shape
    split_volume = len(num_blocks)
    num_blocks = num_blocks + [1] * (len(shape) - len(num_blocks))
    ranges = [list(range(n)) for n in num_blocks]
    result = []
    for _, multi_index in enumerate(itertools.product(*ranges)):
        slices = [[]] * len(shape)
        for c, p in enumerate(multi_index):
            assert p >= 0
            assert p < num_blocks[c]
            a = p * shape[c] // num_blocks[c]
            b = min((p + 1) * shape[c] // num_blocks[c], shape[c])
            slices[c] = slice(a, b)
        slices = tuple(slices)
        if just_max:
            result = result + [list(np.argsort(np.sum(x[slices], tuple(range(split_volume))).flatten()))[-1]]
        else:
            result = result + list(np.argsort(np.sum(x[slices], tuple(range(split_volume))).flatten()))
    return hash(str(result))

def jittered(n: int, shape: tuple, num_blocks: list | None = None, just_max: bool = False) -> np.ndarray:
    if num_blocks is None:
        num_blocs = [2, 2]
    hash_to_set = defaultdict(list)
    for i in range(int(np.sqrt(n)) * n):
        x = normalize([np.random.randn(*shape)])[0]
        hash_to_set[get_class(x, num_blocks, just_max)] += [x]
    min_num = 10000000
    max_num = -1
    while True:
        for k in hash_to_set.keys():
            min_num = min(min_num, len(hash_to_set[k]))
            max_num = max(max_num, len(hash_to_set[k]))
        if min_num < n / len(hash_to_set.keys()):
            x = normalize([np.random.randn(*shape)])[0]
            hash_to_set[get_class(x, num_blocks, just_max)] += [x]
        else:
            break
    x = []
    while len(x) < n:
        num = max(1, (n - len(x)) // len(hash_to_set))
        for k in hash_to_set.keys():
            if len(x) < n:
                x += hash_to_set[k][:num]
            hash_to_set[k] = hash_to_set[k][num:]
    assert len(x) == n
    return x

def reduced_jittered(n: int, shape: tuple) -> np.ndarray:
    return jittered(n, shape, [2, 2], just_max=True)

def covering_big_conv(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return covering(n, shape, budget, [24, 24])

def lhs(n: int, shape: tuple) -> np.ndarray:
    num = np.prod(shape)
    x = np.zeros([n, num])
    for i in range(num):
        xb = 1.0 / n * np.random.rand(n)
        xplus = np.linspace(0, n - 1, n) / n
        np.random.shuffle(xplus)
        x[:, i] = scipy.stats.norm.ppf(xb + xplus)
    thex = []
    for i in range(n):
        thex += normalize([x[i].reshape(*shape)])
    assert len(thex) == n
    assert thex[0].shape == tuple(shape), f' we get {x[0].shape} instead of {tuple(shape)}'
    return thex

def metric_half(x: np.ndarray, budget: int = 300, conv: list | None = None) -> float:
    shape = x[0].shape
    t0 = time.time()
    xconv = np.array([convo(x_, conv).flatten() for x_ in x])
    scores = []
    while time.time() < t0 + 0.01 * budget:
        y = convo(normalize([np.random.randn(*shape)])[0], conv).flatten()
        scores += [np.average(np.matmul(xconv, y) > 0.0)]
    return np.average((np.array(scores) - 0.5) ** 2)

def metric_half_conv(x: np.ndarray, budget: int = 300) -> float:
    return metric_half(x, budget, conv=[8, 8])

def metric_cap(x: np.ndarray, budget: int = 300, conv: list | None = None) -> float:
    shape = x[0].shape
    t0 = time.time()
    c = 1.0 / np.sqrt(len(x[0].flatten()))
    xconv = np.array(normalize([convo(x_, conv).flatten() for x_ in x]))
    scores = []
    while time.time() < t0 + 0.01 * budget:
        y = convo(normalize([np.random.randn(*shape)])[0], conv).flatten()
        scores += [np.average(np.matmul(xconv, y) > c)]
        scores += [np.average(np.matmul(xconv, y) < -c)]
    return np.std(np.array(scores))

def metric_cap_conv(x: np.ndarray, budget: int = 300) -> float:
    return metric_cap(x, budget, conv=[8, 8])

def metric_pack_absavg(x: np.ndarray, budget: int = 300, conv: list | None = None) -> float:
    shape = x[0].shape
    xconv = np.array(normalize([convo(x_, conv).flatten() for x_ in x]))
    scores = np.matmul(xconv, xconv.transpose())
    for i in range(len(scores)):
        assert 0.99 < scores[i, i] < 1.01
        scores[i, i] = 0
    scores = scores.flatten()
    assert len(scores) == len(x) ** 2
    return np.average(np.abs(scores))

def metric_pack_absavg_conv(x: np.ndarray, budget: int = 300) -> float:
    return metric_pack_absavg(x, budget, conv=[8, 8])

def metric_riesz_avg(x: np.ndarray, budget: int = 300, conv: list | None = None, r: float = 1.0) -> float:
    shape = x[0].shape
    xconv = np.array(normalize([convo(x_, conv).flatten() for x_ in x]))
    scores = []
    for i in range(len(xconv)):
        for j in range(i):
            scores += [np.linalg.norm(xconv[i] - xconv[j]) ** (-r)]
    return np.average(scores)

def metric_riesz_avg2(x: np.ndarray, budget: int = 300, conv: list | None = None, r: float = 2.0) -> float:
    return metric_riesz_avg(x, budget=budget, conv=conv, r=2.0)

def metric_riesz_avg05(x: np.ndarray, budget: int = 300, conv: list | None = None, r: float = 0.5) -> float:
    return metric_riesz_avg(x, budget=budget, conv=conv, r=0.5)

def metric_riesz_avg_conv(x: np.ndarray, budget: int = 300, conv: list | None = [8, 8], r: float = 1.0) -> float:
    return metric_riesz_avg(x, budget=default_budget, conv=conv, r=r)

def metric_riesz_avg_conv2(x: np.ndarray, budget: int = 300, conv: list | None = [8, 8], r: float = 2.0) -> float:
    return metric_riesz_avg(x, budget=default_budget, conv=conv, r=r)

def metric_riesz_avg_conv05(x: np.ndarray, budget: int = 300, conv: list | None = [8, 8], r: float = 0.5) -> float:
    return metric_riesz_avg(x, budget=default_budget, conv=conv, r=r)

def metric_pack_avg(x: np.ndarray, budget: int = 300, conv: list | None = None) -> float:
    shape = x[0].shape
    xconv = np.array(normalize([convo(x_, conv).flatten() for x_ in x]))
    scores = np.matmul(xconv, xconv.transpose())
    for i in range(len(scores)):
        assert 0.99 < scores[i, i] < 1.01
        scores[i, i] = 0
    scores = scores.flatten()
    assert len(scores) == len(x) ** 2
    return np.average(scores)

def metric_pack_avg_conv(x: np.ndarray, budget: int = 300) -> float:
    return metric_pack_avg(x, budget=default_budget, conv=[8, 8])

def metric_pack(x: np.ndarray, budget: int = 300, conv: list | None = None) -> float:
    shape = x[0].shape
    xconv = np.array(normalize([convo(x_, conv).flatten() for x_ in x]))
    scores = np.matmul(xconv, xconv.transpose())
    for i in range(len(scores)):
        assert 0.99 < scores[i, i] < 1.01
        scores[i, i] = 0
    scores = scores.flatten()
    assert len(scores) == len(x) ** 2
    return max(scores)

def metric_pack_conv(x: np.ndarray, budget: int = 300) -> float:
    return metric_pack(x, budget=default_budget, conv=[8, 8])

def metric_pack_big_conv(x: np.ndarray, budget: int = 300) -> float:
    return metric_pack(x, budget=default_budget, conv=[24, 24])

list_of_methods = ['ng_TwoPointsDE', 'ng_DE', 'ng_PSO', 'ng_OnePlusOne', 'ng_DiagonalCMA', 'lhs', 'reduced_jittered', 'jittered', 'big_block_symmetry', 'block_symmetry', 'greedy_dispersion', 'dispersion', 'pure_random', 'antithetic_pm', 'dispersion', 'antithetic_order', 'antithetic_order_and_sign', 'dispersion_with_conv', 'dispersion_with_big_conv', 'greedy_dispersion_with_big_conv', 'dispersion_with_mini_conv', 'greedy_dispersion_with_mini_conv', 'covering', 'covering_conv', 'covering_mini_conv', 'rs', 'rs_mhc', 'rs_pack', 'rs_pa', 'rs_pc', 'rs_pac', 'rs_cap', 'rs_cc', 'rs_all', 'rs_ra', 'rs_ra2', 'rs_ra05', 'rs_rac', 'rs_rac2', 'rs_rac05', 'Riesz_blurred_bigconv_loworder', 'Riesz_blurred_bigconv_midorder', 'Riesz_blurred_bigconv_highorder', 'Riesz_blurred_medconv_loworder', 'Riesz_blurred_medconv_midorder', 'Riesz_blurred_medconv_highorder', 'Riesz_blurred_lowconv_loworder', 'Riesz_blurred_lowconv_midorder', 'Riesz_blurred_lowconv_highorder', 'Riesz_blursum_lowconv_hugeorder', 'Riesz_blursum_medconv_hugeorder', 'Riesz_blursum_highconv_hugeorder', 'Riesz_blursum_lowconv_tinyorder', 'Riesz_blursum_medconv_tinyorder', 'Riesz_blursum_highconv_tinyorder', 'Riesz_blurred_lowconv_hugeorder', 'Riesz_blurred_medconv_hugeorder', 'Riesz_blurred_highconv_hugeorder', 'Riesz_blurred_lowconv_tinyorder', 'Riesz_blurred_medconv_tinyorder', 'Riesz_blurred_highconv_tinyorder', 'Riesz_noblur_lowconv_loworder', 'Riesz_noblur_lowconv_midorder', 'Riesz_noblur_lowconv_highorder', 'Riesz_blursum_bigconv_loworder', 'Riesz_blursum_bigconv_midorder', 'Riesz_blursum_bigconv_highorder', 'Riesz_blursum_medconv_loworder', 'Riesz_blursum_medconv_midorder', 'Riesz_blursum_medconv_highorder', 'Riesz_blursum_lowconv_loworder', 'Riesz_blursum_lowconv_midorder', 'Riesz_blursum_lowconv_highorder', 'max_pooling', 'max_without_pooling', 'max_small_pooling']
list_metrics = ['metric_half', 'metric_half_conv', 'metric_pack', 'metric_pack_conv', 'metric_pack_big_conv', 'metric_pack_avg', 'metric_pack_avg_conv', 'metric_pack_absavg', 'metric_pack_absavg_conv', 'metric_cap', 'metric_cap_conv', 'metric_riesz_avg', 'metric_riesz_avg2', 'metric_riesz_avg05', 'metric_riesz_avg_conv', 'metric_riesz_avg_conv2', 'metric_riesz_avg_conv05']
for u in list_metrics:
    metrics[u] = eval(u)

def rs(n: int, shape: tuple, budget: int = 300, k: str | None = None, ngtool: str | None = None) -> np.ndarray:
    bestm = float('inf')
    if ngtool is not None:
        opt = ng.optimizers.registry[ngtool](ng.p.Array(shape=tuple([n] + list(shape))), budget=10000000000000)
    t0 = time.time()
    bestx = None
    while time.time() < t0 + 0.01 * budget or bestx is None:
        if ngtool is None:
            x = pure_random(n, shape)
        else:
            candidate = opt.ask()
            x = list(candidate.value)
            assert len(x) == n
            x = normalize(x)
        if k == 'all':
            m = np.sum([metrics[k2](x, budget / len(list_metrics) / max(10, np.sqrt(budget / 100))) for k2 in list_metrics])
        else:
            m = metrics[k](x, budget / max(10, np.sqrt(budget / 100)))
        if ngtool is not None:
            print('ng gets ', m)
            opt.tell(candidate, m)
        if m < bestm:
            bestm = m
            bestx = x
    return bestx

def rs_mhc(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='metric_half_conv')

def rs_cap(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='metric_cap')

def rs_cc(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='metric_cap_conv')

def rs_pack(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='metric_pack')

def rs_ra(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='metric_riesz_avg')

def rs_ra2(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='metric_riesz_avg2')

def rs_ra05(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='metric_riesz_avg05')

def rs_rac(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='metric_riesz_avg_conv')

def rs_rac2(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='metric_riesz_avg_conv2')

def rs_rac05(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='metric_riesz_avg_conv05')

def rs_pa(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='metric_pack_avg')

def rs_pc(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='metric_pack_conv')

def rs_pac(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='metric_pack_avg_conv')

def rs_all(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='all')

def ng_TwoPointsDE(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='all', ngtool='TwoPointsDE')

def ng_DE(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='all', ngtool='DE')

def ng_PSO(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='all', ngtool='PSO')

def ng_OnePlusOne(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='all', ngtool='OnePlusOne')

def ng_DiagonalCMA(n: int, shape: tuple, budget: int = 300) -> np.ndarray:
    return rs(n, shape, budget, k='all', ngtool='DiagonalCMA')

data = defaultdict(lambda: defaultdict(list))

def do_plot(tit: str, values: dict) -> None:
    plt.clf()
    plt.title(tit.replace('_', ' '))
    x = np.cos(np.linspace(0.0, 2 * 3.14159, 20))
    y = np.sin(np.linspace(0.0, 2 * 3.14159, 20))
    for i, v in enumerate(sorted(values.keys(), key=lambda k: np.average(values[k]))):
        print(f'context {tit}, {v} ==> {values[v]}')
        plt.plot([i + r for r in x], [np.average(values[v]) + r * np.std(values[v]) for r in y])
        plt.text(i, np.average(values[v]) + np.std(values[v]), f'{v}', rotation=30)
        if i > 0:
            plt.savefig(f'comparison_{tit}_time{default_budget}.png'.replace(' ', '_').replace(']', '_').replace('[', '_'))

def heatmap(y: list, x: list, table: np.ndarray, name: str) -> None:
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
        ax.set_xticks(np.arange(len(x)), labels=x)
        ax.set_yticks(np.arange(len(y)), labels=y)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        for i in range(len(y)):
            for j in range(len(x)):
                text = ax.text(j, i, str(tab[i, j])[:4], ha='center', va='center', color='k')
        fig.tight_layout()
        plt.savefig(f'TIME{default_budget}_cmap{cn}' + name)

for u in list_of_methods:
    methods[u] = eval(u)

def parallel_create_statistics(n: int, shape: tuple, list_of_methods: list, list_of_metrics: list, num: int = 1) -> None:
    shape = [int(s) for s in list(shape)]
    for _ in range(num):
        def deal_with_method(method: str) -> list:
            print(f'{method}')
            x = methods[method](n, shape)
            np.array(x).tofile(f'pointset_{n}_{shape}_{method}_{default_budget}_{np.random.randint(50000)}.dat'.replace(' ', '_').replace('[', ' ').replace(']', ' '))
            print(f'{method}({n}, {shape}) created in time {default_budget}')
            metrics_values = []
            for k in list_of_metrics:
                m = metrics[k](x)
                metrics_values += [m]
                print(f'{method}({n}, {shape}) evaluated for {k}')
            return metrics_values
        results = Parallel(n_jobs=70)((delayed(deal_with_method)(method) for method in list_of_methods))
        for i, method in enumerate(list_of_methods):
            for j, k in enumerate(list_of_metrics):
                data[k][method] += [results[i][j]]
        for k in list_of_metrics:
            if len(data[k]) > 1:
                do_plot(k + f'_number{n}_shape{shape}', data[k])
        tab = np.array([[np.average(data[k][method]) for k in list_of_metrics] for method in list_of_methods])
        print('we have ', tab)
        heatmap(list_of_methods, list_of_metrics, tab, str(shape) + '_' + str(n) + '_' + str(np.random.randint(50000)) + '_bigartifcompa.png')

def bigcheck() -> None:
    n = 20
    shape = (8, 8, 3)
    for k in ['lhs', 'reduced_jittered', 'jittered', 'big_block_symmetry', 'block_symmetry', 'greedy_dispersion', 'dispersion', 'pure_random', 'antithetic_pm', 'dispersion', 'antithetic_order', 'antithetic_order_and_sign', 'dispersion_with_conv', 'dispersion_with_big_conv', 'greedy_dispersion_with_big_conv', 'dispersion_with_mini_conv', 'greedy_dispersion_with_mini_conv', 'covering', 'covering_conv', 'covering_mini_conv', 'Riesz_blurred_bigconv_highorder', 'Riesz_blursum_bigconv_highorder']:
        print('Starting to play with ', k)
        eval(f'{k}(n, shape)')
        print(f' {k} has been used for generating a batch of {n} points with shape {shape}')

def get_a_point_set(n: int, shape: tuple, method: str | None = None) -> tuple:
    k = np.random.choice(list_of_methods)
    if method is not None:
        assert method in list_of_methods, f'{method} is unknown.'
        k = method
    print('Working with ', k)
    x = eval(f'{k}({n}, {shape})')
    for i in range(len(x)):
        assert 0.999 < np.linalg.norm(x[i]) < 1.001, 'we have norm ' + str(np.linalg.norm(x[i]))
    return (k, x)

def quasi_randomize(pointset: np.ndarray, method: str | None = None) -> np.ndarray:
    n = len(pointset)
    shape = [int(i) for i in list(pointset[0].shape)]
    norms = [np.linalg.norm(pointset[i]) for i in range(n)]
    if method is None or method == 'none':
        method = 'dispersion_with_big_conv' if len(shape) > 1 and shape[0] > 1 else 'covering'
    x = get_a_point_set(n, shape, method)[1]
    x = normalize(x)
    for i in range(n):
        x[i] = norms[i] * x[i]
    return x
