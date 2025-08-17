# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, DefaultDict, Sequence
import scipy
import scipy.signal
import scipy.stats
import time
import copy
import numpy as np
import itertools
from joblib import Parallel, delayed  # type: ignore
from collections import defaultdict
import nevergrad as ng

try:
    from joblib import parallel_config
except ImportError:
    print("Some stuff might fail: issue in joblib")

num_cores: int = multiprocessing.cpu_count()

default_budget: int = 300  # centiseconds
default_steps: int = 100  # nb of steps grad descent
default_order: int = 2  # Riesz energy order
default_stepsize: int = 10  # step size for grad descent
methods: Dict[str, Callable[..., Any]] = {}
metrics: Dict[str, Callable[..., Any]] = {}

def normalize(x: List[np.ndarray]) -> List[np.ndarray]:
    for i in range(len(x)):
        normalization = np.sqrt(np.sum(x[i] ** 2.0))
        if normalization > 0.0:
            x[i] = x[i] / normalization
        else:
            x[i] = np.random.randn(np.prod(x[i].shape)).reshape(x[i].shape)
            x[i] = x[i] / np.sqrt(np.sum(x[i] ** 2.0))
    return x

def convo(x: np.ndarray, k: Optional[Sequence[int]]) -> np.ndarray:
    if k is None:
        return x
    return scipy.ndimage.gaussian_filter(x, sigma=list(k) + [0.0] * (len(x.shape) - len(k)))

def convo_mult(x: np.ndarray, k: Optional[Sequence[int]]) -> np.ndarray:
    if k is None:
        return x
    return scipy.ndimage.gaussian_filter(x, sigma=[0] + list(k) + [0.0] * (len(x.shape) - len(k) - 1))

def pure_random(n: int, shape: Sequence[int], conv: Optional[Sequence[int]] = None) -> List[np.ndarray]:
    return normalize([np.random.randn(*shape) for i in range(n)])

def antithetic_pm(n: int, shape: Sequence[int], conv: Optional[Sequence[int]] = None) -> np.ndarray:
    m = n // 2
    x = [np.random.randn(*shape) for i in range(m)]
    x = normalize(x)
    x = x + [-xi for xi in x]
    if len(x) < n:
        x = x + [np.random.randn(*shape)]
    return np.array(x)

def antithetic_order(n: int, shape: Sequence[int], axis: int = -1, also_sym: bool = False, conv: Optional[Sequence[int]] = None) -> List[np.ndarray]:
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

def antithetic_order_and_sign(n: int, shape: Sequence[int], axis: int = -1, conv: Optional[Sequence[int]] = None) -> List[np.ndarray]:
    return antithetic_order(n, shape, axis, also_sym=True)

def manual_avg_pool3d(arr: np.ndarray, kernel_size: Sequence[int]) -> np.ndarray:
    output_shape = (
        arr.shape[0] // kernel_size[0],
        arr.shape[1] // kernel_size[1],
        arr.shape[2] // kernel_size[2],
    )
    result = np.zeros(output_shape)
    for z in range(output_shape[0]):
        for y in range(output_shape[1]):
            for x in range(output_shape[2]):
                result[z, y, x] = np.mean(
                    arr[
                        z * kernel_size[0] : (z + 1) * kernel_size[0],
                        y * kernel_size[1] : (y + 1) * kernel_size[1],
                        x * kernel_size[2] : (x + 1) * kernel_size[2],
                    ]
                )
    return result

def max_pooling(n: int, shape: Sequence[int], budget: int = default_budget, conv: Optional[Sequence[int]] = None) -> np.ndarray:
    pooling = tuple([max(1, s // 8) for s in shape])

    if conv is not None:
        pooling = (1, *conv)

    old_latents: List[np.ndarray] = []
    x = []
    for i in range(n):
        latents = np.random.randn(*shape)
        latents_pooling = manual_avg_pool3d(latents, pooling)
        if old_latents:
            dist = min([np.linalg.norm(latents_pooling - old) for old in old_latents])
            max_dist = dist
            t0 = time.time()
            while (time.time() - t0) < 0.01 * budget / n:
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

def max_without_pooling(n: int, shape: Sequence[int], budget: int = default_budget, conv: List[int] = [1, 1]) -> np.ndarray:
    return max_pooling(n, shape, budget, conv)

def max_small_pooling(n: int, shape: Sequence[int], budget: int = default_budget, conv: List[int] = [8, 8]) -> np.ndarray:
    return max_pooling(n, shape, budget, conv)

def greedy_dispersion(n: int, shape: Sequence[int], budget: int = default_budget, conv: Optional[Sequence[int]] = None) -> List[np.ndarray]:
    x = normalize([np.random.randn(*shape)])
    for i in range(n - 1):
        bigdist = -1
        t0 = time.time()
        while time.time() < t0 + 0.01 * budget / n:
            def rand_and_dist(i: int) -> Tuple[np.ndarray, float]:
                y = normalize([np.random.randn(*shape)])[0]
                dist = min(np.linalg.norm(convo(y, conv) - convo(x[i], conv)) for i in range(len(x)))
                return (y, dist)

            with parallel_config(backend="threading"):
                r = Parallel(n_jobs=-1)(delayed(rand_and_dist)(i) for i in range(num_cores))
            dist = [r[i][1] for i in range(len(r))]
            index = dist.index(max(dist))
            newy = r[index][0]
        x += [newy]
    return x

def dispersion(n: int, shape: Sequence[int], budget: int = default_budget, conv: Optional[Sequence[int]] = None) -> List[np.ndarray]:
    x = greedy_dispersion(n, shape, budget / 2, conv=conv)
    t0 = time.time()
    num = n
    num_iterations = 0
    while time.time() < t0 + 0.01 * budget / 2:
        num = num + 1
        for j in range(len(x)):
            bigdist = -1

            def rand_and_dist(idx: int) -> Tuple[np.ndarray, float]:
                if idx > 0:
                    y = normalize([np.random.randn(*shape)])[0]
                else:
                    y = x[j]
                convoy = convo(y, conv)
                dist = min(np.linalg.norm(convoy - convo(x[i], conv)) for i in range(len(x)) if i != j)
                return (y, dist)

            with parallel_config(backend="threading"):
                num_jobs = max(2 * num, num_cores)
                r = Parallel(n_jobs=num_cores)(delayed(rand_and_dist)(i) for i in range(num_jobs))
                num_iterations += num_jobs
            dist = [r[i][1] for i in range(len(r))]
            index = dist.index(max(dist))
            x[j] = r[index][0]
            if time.time() > t0 + 0.01 * budget / 2:
                break
        if time.time() > t0 + 0.01 * budget / 2:
            break
    score = metrics["metric_pack_big_conv"](x)
    return x

def dispersion_with_conv(n: int, shape: Sequence[int], budget: int = default_budget) -> List[np.ndarray]:
    return dispersion(n, shape, budget=budget, conv=[8, 8])

def greedy_dispersion_with_conv(n: int, shape: Sequence[int], budget: int = default_budget) -> List[np.ndarray]:
    return greedy_dispersion(n, shape, budget=budget, conv=[8, 8])

def dispersion_with_big_conv(n: int, shape: Sequence[int], budget: int = default_budget) -> List[np.ndarray]:
    return dispersion(n, shape, budget=budget, conv=[24, 24])

def greedy_dispersion_with_big_conv(n: int, shape: Sequence[int], budget: int = default_budget) -> List[np.ndarray]:
    return greedy_dispersion(n, shape, budget=budget, conv=[24, 24])

def dispersion_with_mini_conv(n: int, shape: Sequence[int], budget: int = default_budget) -> List[np.ndarray]:
    return dispersion(n, shape, budget=budget, conv=[2, 2])

def greedy_dispersion_with_mini_conv(n: int, shape: Sequence[int], budget: int = default_budget) -> List[np.ndarray]:
    return greedy_dispersion(n, shape, budget=budget, conv=[2, 2])

def Riesz_blurred_gradient(
    n: int,
    shape: Sequence[int],
    budget: int = default_budget,
    order: int = default_order,
    step_size: int = default_stepsize,
    conv: Optional[Sequence[int]] = None
) -> np.ndarray:
    t = (n,) + tuple(shape)
    x = np.random.randn(*t)
    x = normalize(x)
    t0 = time.time()
    for steps in range(int(1e9 * budget)):
        Temp = np.zeros(t)
        Blurred = convo_mult(x, conv)
        for i in range(n):
            for j in range(n):
                if j != i:
                    T = np.add(Blurred[i], -Blurred[j])
                    Temp[i] = np.add(Temp[i], np.multiply(T, 1 / (np.sqrt(np.sum(T**2.0))) ** (order + 2)))
            Temp[i] = np.multiply(Temp[i], step_size)
        x = np.add(x, Temp)
        x = normalize(x)
        if time.time() > t0 + 0.01 * budget:
            break
    return x

def Riesz_blursum_gradient(
    n: int,
    shape: Sequence[int],
    budget: int = default_budget,
    order: int = default_order,
    step_size: int = default_stepsize,
    conv: Optional[Sequence[int]] = None
) -> np.ndarray:
    t = (n,) + tuple(shape)
    x = np.random.randn(*t)
    x = normalize(x)
    t0 = time.time()
    for steps in range(int(1e9 * budget)):
        Blurred = np.zeros(t)
        for i in range(n):
            for j in range(n):
                if j != i:
                    T = np.add(x[i], -x[j])
                    Blurred[i] = np.add(
                        np.multiply(T, 1 / (np.sqrt(np.sum(T**2.0))) ** (order + 2)), Blurred[i]
                    )
        Blurred = convo_mult(Blurred, conv)
        x = np.add(x, Blurred)
        x = normalize(x)
        if time.time() > t0 + 0.01 * budget:
            break
    return x

def Riesz_noblur_gradient(
    n: int,
    shape: Sequence[int],
    budget: int = default_budget,
    order: int = default_order,
    step_size: int = default_stepsize,
    conv: Optional[Sequence[int]] = None
) -> np.ndarray:
    t = (n,) + tuple(shape)
    x = np.random.randn(*t)
    x = normalize(x)
    t0 = time.time()
    for steps in range(int(1e9 * budget)):
        Temp = np.zeros(t)
        for i in range(n):
            for j in range(n):
                if j != i:
                    T = np.add(x[i], -x[j])
                    Temp[i] = np.add(Temp[i], np.multiply(T, 1 / (np.sqrt(np.sum(T**2.0))) ** (order + 2)))

        x = np.add(x, Temp)
        x = normalize(x)
        if time.time() > t0 + 0.01 * budget:
            break
    return x

def Riesz_noblur_lowconv_loworder(n: int, shape: Sequence[int], budget: int = default_budget) -> np.ndarray:
    return Riesz_noblur_gradient(n, shape, default_steps, order=0.5, step_size=default_stepsize, conv=[2, 2])

def Riesz_noblur_lowconv_midorder(n: int, shape: Sequence[int], budget: int = default_budget) -> np.ndarray:
    return Riesz_noblur_gradient(n, shape, default_steps, order=1, step_size=default_stepsize, conv=[2, 2])

def Riesz_noblur_lowconv_highorder(n: int, shape: Sequence[int], budget: int = default_budget) -> np.ndarray:
    return Riesz_noblur_gradient(n, shape, default_steps, order=2, step_size=default_stepsize, conv=[2, 2])

def Riesz_blursum_lowconv_hugeorder(n: int, shape: Sequence[int], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[2, 2])

def Riesz_blursum_medconv_hugeorder(n: int, shape: Sequence[int], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[8, 8])

def Riesz_blursum_highconv_hugeorder(n: int, shape: Sequence[int], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[24, 24])

def Riesz_blursum_lowconv_tinyorder(n: int, shape: Sequence[int], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[2, 2])

def Riesz_blursum_medconv_tinyorder(n: int, shape: Sequence[int], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[8, 8])

def Riesz_blursum_highconv_tinyorder(n: int, shape: Sequence[int], budget: int = default_budget) -> np.ndarray:
    return Riesz_blursum_gradient(
        n, shape, default_steps, order=0.3, step_size=default_stepsize, conv=[24, 24]
    )

def Riesz_blurred_lowconv_hugeorder(n: int, shape: Sequence[int], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[2, 2])

def Riesz_blurred_medconv_hugeorder(n: int, shape: Sequence[int], budget: int = default_budget) -> np.ndarray:
    return Riesz_blurred_gradient(n, shape, default_steps, order=5, step_size=default_stepsize, conv=[8, 8])

def Riesz_blurred_highconv_hugeorder(n: int, shape: Sequence[int