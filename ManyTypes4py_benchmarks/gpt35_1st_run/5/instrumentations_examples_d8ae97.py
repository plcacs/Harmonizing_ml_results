import numpy as np
import sys
from joblib import Parallel, delayed
from collections import defaultdict
import nevergrad as ng
import signal
from typing import List, Tuple, Dict, Any

def mean(v: List[float]) -> float:
    s = np.sum(v)
    m = s / len(v)
    if not s == s or len(v) == 0:
        m = float(0.0)
    return m

def ucb_score(v: List[float], i: float) -> float:
    m = mean(v)
    stat = (1 + np.sqrt(np.log(3.0 + i))) / np.sqrt(len(v))
    return m - stat

def lcb_score(v: List[float], i: float = 1.0) -> float:
    m = mean(v)
    stat = (1 + np.sqrt(np.log(3.0 + i))) / np.sqrt(len(v))
    return m + 0.1 * stat

def timeout_handler(signum: int, frame: Any) -> None:
    raise Exception('end of time')

num_noise: int = 90

def testing_domain_and_loss_and_single_budget(d: Any, l: Any, b: float, name: str) -> None:
    score: Dict[str, List[float]] = defaultdict(list)

    def get_score(optim_name: str) -> float:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        try:
            recommendation = ng.optimizers.registry[optim_name](d, b).minimize(l)
            try:
                loss_value = l(recommendation.value) if not 'noisy' in name else np.sum([l(recommendation.value) for _ in range(num_noise)]) / num_noise
            except:
                loss_value = l(*recommendation.args, **recommendation.kwargs) if not 'noisy' in name else np.sum([l(*recommendation.args, **recommendation.kwargs) for _ in range(num_noise)]) / num_noise
        except Exception as e:
            loss_value = float(10000000000.0)
        signal.alarm(0)
        if not loss_value == loss_value:
            loss_value = float(10000000000.0)
        return loss_value

    for idx in range(17):
        print('Iteration ', idx)
        list_optims = list(ng.optimizers.registry.keys())
        list_optims = [o for o in list_optims if 'BO' not in o and 'AX' not in o]
        optim_names = sorted(list_optims, key=lambda o: ucb_score(score[o], idx))[:50]
        results = Parallel(n_jobs=len(optim_names))((delayed(get_score)(o) for o in optim_names))
        assert len(results) == len(optim_names)
        for optim_name, loss_value in zip(optim_names, results):
            score[optim_name] += [loss_value]
    print(f'List of best for N={N} and budget={b} and name={name} and dim={dim} and shape={shape}:')
    for i, u in enumerate(sorted(score, key=lambda x: lcb_score(score[x]))):
        print('Rank ', i, ':', u, ',', np.sum(score[u]) / len(score[u]), '(', len(score[u]), ')', score[u]

def testing_domain_and_loss_and_budget(d: Any, l: Any, b: float, name: str) -> None:
    testing_domain_and_loss_and_single_budget(d, l, b, name)
    testing_domain_and_loss_and_single_budget(d, l, 10 * b, name)
    testing_domain_and_loss_and_single_budget(d, l, 30 * b, name)

target: np.ndarray = np.zeros(shape=shape)

def loss(x: np.ndarray) -> float:
    return np.sum(np.abs(x - target)

def complex_loss(x: np.ndarray, y: np.ndarray) -> float:
    return loss(x) + loss(np.transpose(y))

def correlated_loss(x: np.ndarray, y: np.ndarray) -> float:
    return loss(x) + loss(np.transpose(y))

def six_losses(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray, x5: np.ndarray, x6: np.ndarray) -> float:
    return loss(x1) + loss(x2) + loss(x3) + loss(x4) + loss(x5) + loss(x6)

def complex_six_losses(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray, x5: np.ndarray, x6: np.ndarray) -> float:
    return loss(x1) + loss(x2.transpose()) + loss(x3) + loss(x4.transpose()) + loss(x5) + loss(x6.transpose())

def noisy_six_losses(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray, x5: np.ndarray, x6: np.ndarray) -> float:
    return six_losses(x1, x2, x3, x4, x5, x6) + np.random.randn() * 17.0

