import pytest
import numpy as np
import sys
from unittest import SkipTest
import nevergrad as ng
import nevergrad.common.typing as tp
from nevergrad.common import testing
from . import base
from .optimizerlib import registry
from typing import Callable, Optional

def long_name(s: str) -> bool:
    return len(s.replace('DiscreteOnePlusOne', 'D1+1')) > 10

skip_win_perf = pytest.mark.skipif(sys.platform == 'win32', reason='Slow, and no need to test performance on all platforms')

def suggestable(name: str) -> bool:
    keywords = ['TBPSA', 'BO', 'EMNA', 'EDA', 'BO', 'Stupid', 'Pymoo', 'GOMEA']
    return not any((x in name for x in keywords))

def suggestion_testing(
    name: str,
    instrumentation: tp.ParametrizationOrFunction,  # type from nevergrad common typing
    suggestion: np.ndarray,
    budget: int,
    objective_function: Callable[[np.ndarray], float],
    optimum: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> None:
    optimizer_cls = registry[name]
    optim = optimizer_cls(instrumentation, budget)
    if optimum is None:
        optimum = suggestion
    optim.suggest(suggestion)
    optim.minimize(objective_function)
    if threshold is not None:
        assert objective_function(optim.recommend().value) < threshold, (
            f'{name} proposes {optim.recommend().value} instead of {optimum} (threshold={threshold})'
        )
        return
    assert np.all(optim.recommend().value == optimum), f'{name} proposes {optim.recommend().value} instead of {optimum}'

@skip_win_perf
@pytest.mark.parametrize('name', [r for r in registry if suggestable(r)])
def test_suggest_optimizers(name: str) -> None:
    """Checks that each optimizer is able to converge when optimum is given"""
    if 'SA' in name or 'T' in name:
        return
    if sum([ord(c) for c in name]) % 4 > 0 and name not in ['CMA', 'PSO', 'DE']:
        raise SkipTest('Too expensive: we randomly skip 3/4 of these tests.')
    instrum = ng.p.Array(shape=(100,)).set_bounds(0.0, 1.0)
    instrum.set_integer_casting()
    suggestion = np.asarray([0] * 17 + [1] * 17 + [0] * 66)
    target = lambda x: 0 if np.all(np.asarray(x, dtype=int) == suggestion) else 1
    suggestion_testing(name, instrum, suggestion, 7, target)

def good_at_suggest(name: str) -> bool:
    keywords = ['Noisy', 'Optimistic', 'DiscreteDE', 'Multi', 'Anisotropic', 'BSO', 'GOMEA', 'Sparse', 'Adaptive', 'Doerr', 'Recombining', 'SA', 'Lognormal', 'PortfolioDiscreteOne', 'FastGADiscreteOne']
    return not any((k in name for k in keywords))

@skip_win_perf
@pytest.mark.parametrize(
    'name',
    [
        r
        for r in registry
        if 'iscre' in r and 'Smooth' not in r and good_at_suggest(r)
        and (r != 'DiscreteOnePlusOne') and ('Lengler' not in r or 'LenglerOne' in r)
    ],
)
def test_harder_suggest_optimizers(name: str) -> None:
    if 'SA' in name or 'T' in name:
        return
    'Checks that discrete optimizers are good when a suggestion is nearby.'
    if long_name(name):
        return
    if 'OLN' in name:
        return
    instrum = ng.p.Array(shape=(100,)).set_bounds(0.0, 1.0)
    instrum.set_integer_casting()
    optimum = np.asarray([0] * 17 + [1] * 17 + [0] * 66)
    target = lambda x: min(3, np.sum((np.asarray(x, dtype=int) - optimum) ** 2))
    suggestion = np.asarray([0] * 17 + [1] * 16 + [0] * 67)
    extra_budget = 1000 if 'Lengler' in name else 0
    suggestion_testing(name, instrum, suggestion, 1500 + extra_budget, target, optimum)

@skip_win_perf
def test_harder_continuous_suggest_optimizers() -> None:
    """Checks that somes optimizer can converge when provided with a good suggestion."""
    instrum = ng.p.Array(shape=(100,)).set_bounds(0.0, 1.0)
    optimum = np.asarray([0] * 17 + [1] * 17 + [0] * 66)
    target = lambda x: min(2.0, np.sum((x - optimum) ** 2))
    suggestion = np.asarray([0] * 17 + [1] * 16 + [0] * 67)
    suggestion_testing('NGOpt', instrum, suggestion, 1500, target, optimum, threshold=0.9)

@testing.suppress_nevergrad_warnings()
@pytest.mark.parametrize('name', list(registry.keys()))
def test_optimizers_suggest(name: str) -> None:
    if 'SA' in name or 'T' in name:
        return
    optimizer = registry[name](parametrization=4, budget=2)
    optimizer.suggest(np.array([12.0] * 4))
    candidate = optimizer.ask()
    try:
        optimizer.tell(candidate, 12)
        if name not in ['SPSA', 'TBPSA', 'StupidRandom']:
            np.testing.assert_array_almost_equal(
                optimizer.provide_recommendation().value, [12.0] * 4
            )
    except base.errors.TellNotAskedNotSupportedError:
        pass