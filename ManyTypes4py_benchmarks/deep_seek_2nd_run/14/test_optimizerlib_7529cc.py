import inspect
import logging
import os
import platform
import random
import re
import sys
import tempfile
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast
from unittest import SkipTest
from unittest.mock import patch

import nevergrad as ng
import nevergrad.common.typing as tp
import numpy as np
import pandas as pd
import pytest
from bayes_opt.util import acq_max
from nevergrad.common import testing
from scipy import stats
from scipy.ndimage import gaussian_filter

from . import base, es, experimentalvariants as xpvariants, optimizerlib as optlib
from .optimizerlib import NGOptBase, registry

T = TypeVar('T')
OptCls = TypeVar('OptCls', bound=base.Optimizer)

skip_win_perf = pytest.mark.skipif(
    sys.platform == 'win32', 
    reason='Slow, and no need to test performance on all platforms'
)
np.random.seed(0)
CI = bool(os.environ.get('CIRCLECI', False)) or bool(os.environ.get('CI', False))

def long_name(s: str) -> bool:
    if s[-1] in '0123456789':
        return True
    if np.random.rand() > 0.15:
        return True
    if 'Wiz' in s or 'CSEC' in s or 'NGO' in s:
        return True
    if 'NgIoh' in s:
        return True
    if 'DS' in s or 'AX' in s or 'BO' in s or any((x in s for x in [str(i) for i in range(10)])):
        return True
    return len(s.replace('DiscreteOnePlusOne', 'D1+1').replace('Tuned', '')) > 2 and CI

short_registry: List[str] = [r for r in registry if not long_name(r)]

class Fitness:
    """Simple quadratic fitness function which can be used with dimension up to 4"""

    def __init__(self, x0: np.ndarray) -> None:
        self.x0 = np.array(x0, copy=True)
        self.call_times: List[float] = []

    def __call__(self, x: np.ndarray) -> float:
        assert len(self.x0) == len(x)
        self.call_times.append(time.time())
        return float(np.sum((np.asarray(x) - self.x0) ** 2))

    def get_factors(self) -> Tuple[float, float]:
        logdiffs = np.log(np.maximum(1e-15, np.cumsum(np.diff(self.call_times))))
        nums = np.arange(len(logdiffs))
        slope, intercept = (float(np.exp(x)) for x in stats.linregress(nums, logdiffs)[:2])
        return (slope, intercept)

def check_optimizer(optimizer_cls: Type[base.Optimizer], budget: int = 300, verify_value: bool = True) -> None:
    num_workers = 1 if optimizer_cls.recast or optimizer_cls.no_parallelization else 2
    num_attempts = 1 if not verify_value else 3
    optimum = [0.5, -0.8]
    fitness = Fitness(optimum)
    for k in range(1, num_attempts + 1):
        fitness = Fitness(optimum)
        optimizer = optimizer_cls(parametrization=len(optimum), budget=budget, num_workers=num_workers)
        assert isinstance(optimizer.provide_recommendation(), ng.p.Parameter), 'Recommendation should be available from start'
        with testing.suppress_nevergrad_warnings():
            candidate = optimizer.minimize(fitness)
        raised = False
        if verify_value:
            try:
                np.testing.assert_array_almost_equal(candidate.args[0], optimum, decimal=1)
            except AssertionError as e:
                raised = True
                print(f'Attemp #{k}: failed with best point {tuple(candidate.args[0])}')
                if k == num_attempts:
                    raise e
        if not raised:
            break
    if budget > 100:
        slope, intercept = fitness.get_factors()
        print(f'For your information: slope={slope} and intercept={intercept}')
    archive = optimizer.archive
    assert optimizer.current_bests['pessimistic'].pessimistic_confidence_bound == min((v.pessimistic_confidence_bound for v in archive.values()))
    assert not optimizer._asked, 'All `ask`s  should have been followed by a `tell`'
    try:
        data = np.random.normal(0, 1, size=optimizer.dimension)
        candidate = optimizer.parametrization.spawn_child().set_standardized_data(data)
        optimizer.tell(candidate, 12.0)
    except Exception as e:
        if not isinstance(e, base.errors.TellNotAskedNotSupportedError):
            raise AssertionError('Optimizers should raise base.TellNotAskedNotSupportedError at when telling unasked points if they do not support it') from e
    else:
        assert optimizer.num_tell == budget + 1
        assert optimizer.num_tell_not_asked == 1 or 'Smooth' in str(optimizer_cls)

SLOW: List[str] = ['NoisyDE', 'NoisyBandit', 'Noisy13Splits', 'NoisyInfSplits', 'DiscreteNoisy13Splits', 'DiscreteNoisyInfSplits', 'SPSA', 'NoisyOnePlusOne', 'OptimisticNoisyOnePlusOne', 'ASCMADEthird', 'ASCMA2PDEthird', 'MultiScaleCMA', 'PCEDA', 'EDA', 'MicroCMA', 'ES']
UNSEEDABLE: List[str] = ['CmaFmin2', 'MetaModelFmin2', 'NLOPT_GN_CRS2_LM', 'NLOPT_GN_ISRES', 'NLOPT_GN_ESCH', 'GOMEABlock', 'GOMEA', 'GOMEATree', 'BAR4', 'BAR3', 'NGOpt$', 'CMandAS3']

def buggy_function(x: np.ndarray) -> float:
    if any(x[::2] > 0.0):
        return float('nan')
    if any(x > 0.0):
        return float('inf')
    return np.sum(x ** 2)

@pytest.mark.parametrize('dim', [2, 30, 200])
@pytest.mark.parametrize('budget_multiplier', [1, 1000])
@pytest.mark.parametrize('num_workers', [1, 20])
@pytest.mark.parametrize('bounded', [False, True])
@pytest.mark.parametrize('discrete', [False, True])
def test_ngopt(dim: int, budget_multiplier: int, num_workers: int, bounded: bool, discrete: bool) -> None:
    instrumentation = ng.p.Array(shape=(dim,))
    if np.random.rand() < 0.8:
        return
    if bounded:
        instrumentation.set_bounds(lower=-12.0, upper=15.0)
    if discrete:
        instrumentation.set_integer_casting()
    ngopt = optlib.NGOpt(ng.p.Array(shape=(dim,)), budget=budget_multiplier * dim, num_workers=num_workers)
    ngopt.tell(ngopt.ask(), 42.0)

@skip_win_perf
@pytest.mark.parametrize('name', short_registry)
@testing.suppress_nevergrad_warnings()
def test_infnan(name: str) -> None:
    if any((x in name for x in ['SMAC', 'BO', 'AX'])) and CI:
        raise SkipTest('too slow for CI!')
    if 'Force' in name:
        raise SkipTest('Forced methods not tested for infnan')

    def doint(s: str) -> int:
        return 7 + sum([ord(c) * i for i, c in enumerate(s)])
    if doint(name) % 5 > 0:
        raise SkipTest('too many tests for CircleCI!')
    optim_cls = registry[name]
    optim = optim_cls(parametrization=2, budget=70)
    if not any((x in name for x in ['EDA', 'EMNA', 'Stupid', 'NEWUOA', 'Large', 'Fmin2', 'NLOPT', 'TBPSA', 'SMAC', 'BO', 'Noisy', 'Chain', 'chain'])):
        recom = optim.minimize(buggy_function)
        result = buggy_function(recom.value)
        if result < 2.0:
            return
        assert any((x == name for x in ['WidePSO', 'SPSA', 'NGOptBase', 'Shiwa', 'NGO'])) or isinstance(optim, (optlib.Portfolio, optlib._CMA, optlib.recaster.SequentialRecastOptimizer)) or 'NGOpt' in name or ('HS' in name) or ('Adapti' in name) or ('MetaModelDiagonalCMA' in name)

@skip_win_perf
@pytest.mark.parametrize('name', short_registry)
def test_optimizers(name: str) -> None:
    """Checks that each optimizer is able to converge on a simple test case"""
    if any((x in name for x in ['Chain', 'SMAC', 'BO', 'AX'])) and CI:
        raise SkipTest('too slow for CI!')
    if 'BO' in name or 'Chain' in name or 'Tiny' in name or ('Micro' in name):
        return
    if any((x in name for x in ['Tiny', 'Vast'])):
        raise SkipTest('too specific!')

    def doint(s: str) -> int:
        return 7 + sum([ord(c) * i for i, c in enumerate(s)])
    if doint(name) % 5 > 0:
        raise SkipTest('too many tests for CircleCI!')
    if (sum([ord(c) for c in name]) % 4 > 0 and name not in ['DE', 'CMA', 'OnePlusOne', 'Cobyla', 'DiscreteLenglerOnePlusOne', 'PSO'] or 'Tiny' in name or 'Micro' in name) and CI:
        raise SkipTest('Too expensive: we randomly skip 3/4 of these tests.')
    if name in ['CMAbounded', 'NEWUOA']:
        return
    if 'BO' in name:
        return
    optimizer_cls = registry[name]
    if isinstance(optimizer_cls, base.ConfiguredOptimizer):
        assert any((hasattr(mod, name) for mod in (optlib, xpvariants)))
        assert optimizer_cls.__class__(**optimizer_cls._config) == optimizer_cls, 'Similar configuration are not equal'
    nameparts = ['Many', 'Chain', 'BO', 'Discrete', 'NLOPT'] + ['chain']
    is_ngopt = inspect.isclass(optimizer_cls) and issubclass(optimizer_cls, NGOptBase)
    verify = not optimizer_cls.one_shot and name not in SLOW and (not any((x in name for x in nameparts))) and (not is_ngopt)
    budget = 300 if 'BO' not in name and (not is_ngopt) else 4
    patched = partial(acq_max, n_warmup=10000, n_iter=2)
    with patch('bayes_opt.bayesian_optimization.acq_max', patched):
        check_optimizer(optimizer_cls, budget=budget, verify_value=verify)

@pytest.mark.parametrize('name', short_registry)
def test_optimizers_minimal(name: str) -> None:
    optimizer_cls = registry[name]
    if any((x in name for x in ['SMAC', 'BO', 'AX'])) and CI:
        raise SkipTest('too slow for CI!')
    if optimizer_cls.one_shot or name in ['CM', 'NLOPT_LN_PRAXIS', 'NLOPT_GN_CRS2_LM', 'ES', 'RecMixES', 'MiniDE', 'RecMutDE', 'RecES', 'VastLengler', 'VastDE']:
        return
    if any((x in str(optimizer_cls) for x in ['BO', 'DS', 'BAR', 'Meta', 'Voronoi', 'tuning', 'QrDE', 'BIPOP', 'ECMA', 'CMAstd', '_COBYLA', 'HyperOpt', 'Chain', 'CMAbounded', 'Tiny', 'iscrete', 'GOMEA', 'para', 'SPSA', 'EDA', 'FCMA', 'Noisy', 'HS', 'SQPCMA', 'CMandAS', 'GA', 'EMNA', 'RL', 'Milli', 'Small', 'small', 'Chain', 'Tree', 'Mix', 'Micro', 'Naive', 'Portfo', 'ESCH', 'Multi', 'NGO', 'Discrete', 'MixDet', 'Rotated', 'Iso', 'Bandit', 'TBPSA', 'VLP', 'LPC', 'Choice', 'Log', 'Force', 'Multi', 'SQRT', 'NLOPT_GN_ISRES'])):
        raise SkipTest('Skipped because too intricated for this kind of tests!')

    def f(x: np.ndarray) -> float:
        return sum((x - 1.1) ** 2)

    def mf(x: np.ndarray) -> float:
        return sum((x + 1.1) ** 2)

    def f1(x: float) -> float:
        return (x - 1.1) ** 2

    def f2(x: float) -> float:
        return (x + 1.1) ** 2

    def f1p(x: np.ndarray) -> float:
        return sum((x - 1.1) ** 2)

    def f2p(x: np.ndarray) -> float:
        return sum((x + 1.1) ** 2)
    if 'BAR' in name or 'Cma' in name or 'CMA' in name or ('BIPOP' in name) or ('DS3' in name):
        budget = 800
        if 'BAR' in name or 'DS3' in name:
            budget = 3600
        if any((x in name for x in ['Large', 'Tiny', 'Para', 'Diagonal'])):
            return
        val = optimizer_cls(2, budget).minimize(f).value[0]
        assert 1.04 < val < 1.16 or (val > 1.0 and 'BAR' in name), f'pb with {optimizer_cls} for 1.1: {val}'
        val = optimizer_cls(2, budget).minimize(mf).value[0]
        assert -1.17 < val < -1.04 or (val < -1.04 and 'BAR' in name), f'pb with {optimizer_cls} for -1.1.: {val}'
        v = ng.p.Array(shape=(2,), upper=1.0, lower=0.0)
        val = optimizer_cls(v, budget).minimize(f1p).value[0]
        assert 0.9 < val < 1.01, f'pb with {optimizer_cls} for 1.: {val}'
        v = ng.p.Array(shape=(2,), upper=0.3, lower=-0.3)
        val = optimizer_cls(v, budget).minimize(f2p).value[0]
        assert -0.31 < val < -0.24, f'pb with {optimizer_cls} for -0.3: {val}'
    else:
        budget = 100
        if 'DE' in name or 'PSO' in name or 'Hyper' in name:
            budget = 300
        if any((x in name for x in ['QO', 'SODE'])):
            return
        val = optimizer_cls(1, budget).minimize(f).value
        assert 1.04 < val < 1.16 or (val > 1.0 and 'BAR' in name), f'pb with {optimizer_cls} for 1.1: {val}'
        val = optimizer_cls(1, budget).minimize(mf).value
        assert -1.16 < val < -1.04, f'pb with {optimizer_cls} for -1.1.: {val}'
        v = ng.p.Scalar(upper=1.0, lower=0.0)
        val = optimizer_cls(v, budget).minimize(f1).value
        assert 0.94 < val < 1.01, f'pb with {optimizer_cls} for 1.: {val}'
        v = ng.p.Scalar(upper=0.3, lower=-0.3)
        val = optimizer_cls(v, budget).minimize(f2).value
        assert -0.31 < val < -0.24, f'pb with {optimizer_cls} for -0.3: {val}'

class RecommendationKeeper:

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.recommendations = pd.DataFrame(columns=[f'v{k}' for k in range(16)])
        if filepath.exists():
            self.recommendations = pd.read_csv(filepath, index_col=0)

    def save(self) -> None:
        names = sorted((x for x in self.recommendations.index if x in registry))
        recom = self.recommendations.loc[names]
        recom = recom.round(10)
        recom.to_csv(self.filepath)

@pytest.fixture(scope='module')
def recomkeeper() -> RecommendationKeeper:
    keeper = RecommendationKeeper(filepath=Path(__file__).parent / 'recorded_recommendations.csv')
    yield keeper
    keeper