#!/usr/bin/env python3
import numpy as np
import sys
from joblib import Parallel, delayed
from collections import defaultdict
import nevergrad as ng
import signal
from typing import Any, Callable, DefaultDict, List

def mean(v: List[float]) -> float:
    s: float = np.sum(v)
    m: float = s / len(v)
    if not s == s or len(v) == 0:  # Check for NaN or empty list.
        m = float(0.0)
    return m

def ucb_score(v: List[float], i: float) -> float:
    m: float = mean(v)
    stat: float = (1 + np.sqrt(np.log(3.0 + i))) / np.sqrt(len(v))
    return m - stat

def lcb_score(v: List[float], i: float = 1.0) -> float:
    m: float = mean(v)
    stat: float = (1 + np.sqrt(np.log(3.0 + i))) / np.sqrt(len(v))
    return m + 0.1 * stat

def timeout_handler(signum: int, frame: Any) -> None:
    raise Exception("end of time")

print("python -m instrumentation_examples 3 17")
print("for launching the i^th job among 17 (0 <= i < 17)")
print("or << python -m instrumentation_examples >> for single job")

if len(sys.argv) == 1:
    job_index: int = 0
    job_number: int = 1
else:
    assert len(sys.argv) == 3
    job_index = int(sys.argv[1])
    job_number = int(sys.argv[2])

task_index: int = 0
for dim in [1, 2, 3]:
    for N in [5, 10, 20, 40, 80]:
        task_index += 1
        if task_index % job_number != job_index:
            print("skip ", task_index)
            continue
        print("working on ", task_index)
        shape: tuple = tuple([N] * dim)
        
        def testing_domain_and_loss_and_single_budget(
            d: Any, 
            l: Callable[..., float], 
            b: int, 
            name: str
        ) -> None:
            score: DefaultDict[str, List[float]] = defaultdict(list)

            def get_score(optim_name: str) -> float:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # seconds at most
                try:
                    recommendation = ng.optimizers.registry[optim_name](d, b).minimize(l)
                    try:
                        if "noisy" not in name:
                            loss_value: float = l(recommendation.value)
                        else:
                            loss_value = np.sum([l(recommendation.value) for _ in range(num_noise)]) / num_noise
                    except Exception:
                        if "noisy" not in name:
                            loss_value = l(*recommendation.args, **recommendation.kwargs)
                        else:
                            loss_value = np.sum([l(*recommendation.args, **recommendation.kwargs) for _ in range(num_noise)]) / num_noise
                except Exception as e:
                    loss_value = float(1e10)
                signal.alarm(0)
                if not loss_value == loss_value:  # Check for NaN.
                    loss_value = float(1e10)
                return loss_value

            for idx in range(17):
                print("Iteration ", idx)
                list_optims: List[str] = list(ng.optimizers.registry.keys())
                list_optims = [o for o in list_optims if "BO" not in o and "AX" not in o]
                optim_names: List[str] = sorted(list_optims, key=lambda o: ucb_score(score[o], idx))[:50]
                results: List[float] = Parallel(n_jobs=len(optim_names))(
                    delayed(get_score)(o) for o in optim_names
                )
                assert len(results) == len(optim_names)
                for optim_name, loss_value in zip(optim_names, results):
                    score[optim_name] += [loss_value]

            print(f"List of best for N={N} and budget={b} and name={name} and dim={dim} and shape={shape}:")
            for i, u in enumerate(sorted(score, key=lambda x: lcb_score(score[x]))):
                avg: float = np.sum(score[u]) / len(score[u])
                print("Rank ", i, ":", u, ",", avg, "(", len(score[u]), ")", score[u])

        def testing_domain_and_loss_and_budget(
            d: Any, 
            l: Callable[..., float], 
            b: int, 
            name: str
        ) -> None:
            testing_domain_and_loss_and_single_budget(d, l, b, name)
            testing_domain_and_loss_and_single_budget(d, l, 10 * b, name)
            testing_domain_and_loss_and_single_budget(d, l, 30 * b, name)

        # Let us create a target of shape NxN.
        target: np.ndarray = np.zeros(shape=shape)
        if dim == 1:
            target[N // 2 : N] = 1.0
        elif dim == 2:
            for i in range(N):
                for j in range(N):
                    if abs(i - j) < N // 3:
                        target[i][j] = 1.0
        elif dim == 3:
            for i in range(N):
                for j in range(N):
                    if abs(i - j) < N // 3:
                        target[i][j][:] = 1.0
        else:
            assert False

        # Case "continuous multidimensional array": optimize a NxN array of real numbers.
        print("First, we optimize a continuous function on a continuous domain")
        domain: Any = ng.p.Array(shape=shape)

        def loss(x: np.ndarray) -> float:
            return float(np.sum(np.abs(x - target)))

        testing_domain_and_loss_and_budget(domain, loss, 5 * np.prod(shape), "c0,2d")

        # Case "discrete multidimensional array": optimize on a discrete domain.
        print("Now, we optimize on a discrete domain.")
        domain = ng.p.Array(shape=shape, upper=1.0, lower=0.0).set_integer_casting()

        testing_domain_and_loss_and_budget(domain, loss, 5 * np.prod(shape), "int,2d")

        # Case "combination":
        print("Now, let us work on a mixed domain continuous/discrete.")
        domain = ng.p.Instrumentation(
            x=ng.p.Array(shape=shape, upper=1.0, lower=0.0),
            y=ng.p.Array(shape=shape, upper=1.0, lower=0.0).set_integer_casting(),
        )

        def complex_loss(x: np.ndarray, y: np.ndarray) -> float:
            return loss(x) + loss(np.transpose(y))

        def correlated_loss(x: np.ndarray, y: np.ndarray) -> float:
            return loss(x) + loss(np.transpose(y))

        testing_domain_and_loss_and_budget(domain, complex_loss, 5 * np.prod(shape), "c0+int, 2d")
        testing_domain_and_loss_and_budget(domain, correlated_loss, 5 * np.prod(shape), "c0+int, 2d, corr")

        # Case "multiple blocks" with continuous arrays.
        domain = ng.p.Instrumentation(
            x1=ng.p.Array(shape=shape, upper=1.0, lower=0.0),
            x2=ng.p.Array(shape=shape, upper=1.0, lower=0.0),
            x3=ng.p.Array(shape=shape, upper=1.0, lower=0.0),
            x4=ng.p.Array(shape=shape, upper=1.0, lower=0.0),
            x5=ng.p.Array(shape=shape, upper=1.0, lower=0.0),
            x6=ng.p.Array(shape=shape, upper=1.0, lower=0.0),
        )

        def six_losses(
            x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray, x5: np.ndarray, x6: np.ndarray
        ) -> float:
            return loss(x1) + loss(x2) + loss(x3) + loss(x4) + loss(x5) + loss(x6)

        def complex_six_losses(
            x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray, x5: np.ndarray, x6: np.ndarray
        ) -> float:
            return loss(x1) + loss(x2.transpose()) + loss(x3) + loss(x4.transpose()) + loss(x5) + loss(x6.transpose())

        def noisy_six_losses(
            x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray, x5: np.ndarray, x6: np.ndarray
        ) -> float:
            return six_losses(x1, x2, x3, x4, x5, x6) + np.random.randn() * 17.0

        testing_domain_and_loss_and_budget(domain, six_losses, 5 * np.prod(shape), "sixlosses, 2d, c0")
        testing_domain_and_loss_and_budget(domain, complex_six_losses, 5 * np.prod(shape), "complexsixlosses, 2d, c0")
        testing_domain_and_loss_and_budget(domain, noisy_six_losses, 5 * np.prod(shape), "noisysixlosses, 2d, c0")

        # Case "multiple blocks" with discrete arrays.
        domain = ng.p.Instrumentation(
            x1=ng.p.Array(shape=shape, upper=1.0, lower=0.0).set_integer_casting(),
            x2=ng.p.Array(shape=shape, upper=1.0, lower=0.0).set_integer_casting(),
            x3=ng.p.Array(shape=shape, upper=1.0, lower=0.0).set_integer_casting(),
            x4=ng.p.Array(shape=shape, upper=1.0, lower=0.0).set_integer_casting(),
            x5=ng.p.Array(shape=shape, upper=1.0, lower=0.0).set_integer_casting(),
            x6=ng.p.Array(shape=shape, upper=1.0, lower=0.0).set_integer_casting(),
        )

        testing_domain_and_loss_and_budget(domain, six_losses, 5 * np.prod(shape), "sixlosses, 2d, int")
        testing_domain_and_loss_and_budget(domain, complex_six_losses, 5 * np.prod(shape), "complexsixlosses, 2d, int")
        testing_domain_and_loss_and_budget(domain, noisy_six_losses, 5 * np.prod(shape), "noisysixlosses, 2d, int")

# Global variable used for noise averaging.
num_noise: int = 90
