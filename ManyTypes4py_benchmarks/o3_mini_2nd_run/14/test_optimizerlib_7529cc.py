#!/usr/bin/env python3
from __future__ import annotations
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
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

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

skip_win_perf = pytest.mark.skipif(sys.platform == 'win32', reason='Slow, and no need to test performance on all platforms')
np.random.seed(0)
CI: bool = bool(os.environ.get('CIRCLECI', False)) or bool(os.environ.get('CI', False))


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

    def __init__(self, x0: Union[np.ndarray, List[float]]) -> None:
        self.x0: np.ndarray = np.array(x0, copy=True)
        self.call_times: List[float] = []

    def __call__(self, x: Union[np.ndarray, List[float]]) -> float:
        assert len(self.x0) == len(x)
        self.call_times.append(time.time())
        return float(np.sum((np.asarray(x) - self.x0) ** 2))

    def get_factors(self) -> Tuple[float, float]:
        logdiffs = np.log(np.maximum(1e-15, np.cumsum(np.diff(self.call_times))))
        nums = np.arange(len(logdiffs))
        slope, intercept = (float(np.exp(x)) for x in stats.linregress(nums, logdiffs)[:2])
        return (slope, intercept)


def check_optimizer(optimizer_cls: Any, budget: int = 300, verify_value: bool = True) -> None:
    num_workers: int = 1 if optimizer_cls.recast or optimizer_cls.no_parallelization else 2
    num_attempts: int = 1 if not verify_value else 3
    optimum: List[float] = [0.5, -0.8]
    fitness: Fitness = Fitness(optimum)
    for k in range(1, num_attempts + 1):
        fitness = Fitness(optimum)
        optimizer = optimizer_cls(parametrization=len(optimum), budget=budget, num_workers=num_workers)
        assert isinstance(optimizer.provide_recommendation(), ng.p.Parameter), 'Recommendation should be available from start'
        with testing.suppress_nevergrad_warnings():
            candidate = optimizer.minimize(fitness)
        raised: bool = False
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
    optim_cls: Any = registry[name]
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
    optimizer_cls: Any = registry[name]
    if isinstance(optimizer_cls, base.ConfiguredOptimizer):
        assert any((hasattr(mod, name) for mod in (optlib, xpvariants)))
        assert optimizer_cls.__class__(**optimizer_cls._config) == optimizer_cls, 'Similar configuration are not equal'
    nameparts: List[str] = ['Many', 'Chain', 'BO', 'Discrete', 'NLOPT'] + ['chain']
    is_ngopt: bool = inspect.isclass(optimizer_cls) and issubclass(optimizer_cls, NGOptBase)
    verify: bool = not optimizer_cls.one_shot and name not in SLOW and (not any((x in name for x in nameparts))) and (not is_ngopt)
    budget: int = 300 if 'BO' not in name and (not is_ngopt) else 4
    patched: Callable = partial(acq_max, n_warmup=10000, n_iter=2)
    with patch('bayes_opt.bayesian_optimization.acq_max', patched):
        check_optimizer(optimizer_cls, budget=budget, verify_value=verify)


@pytest.mark.parametrize('name', short_registry)
def test_optimizers_minimal(name: str) -> None:
    optimizer_cls: Any = registry[name]
    if any((x in name for x in ['SMAC', 'BO', 'AX'])) and CI:
        raise SkipTest('too slow for CI!')
    if optimizer_cls.one_shot or name in ['CM', 'NLOPT_LN_PRAXIS', 'NLOPT_GN_CRS2_LM', 'ES', 'RecMixES', 'MiniDE', 'RecMutDE', 'RecES', 'VastLengler', 'VastDE']:
        return
    if any((x in str(optimizer_cls) for x in ['BO', 'DS', 'BAR', 'Meta', 'Voronoi', 'tuning', 'QrDE', 'BIPOP', 'ECMA', 'CMAstd', '_COBYLA', 'HyperOpt', 'Chain', 'CMAbounded', 'Tiny', 'iscrete', 'GOMEA', 'para', 'SPSA', 'EDA', 'FCMA', 'Noisy', 'HS', 'SQPCMA', 'CMandAS', 'GA', 'EMNA', 'RL', 'Milli', 'Small', 'small', 'Chain', 'Tree', 'Mix', 'Micro', 'Naive', 'Portfo', 'ESCH', 'Multi', 'NGO', 'Discrete', 'MixDet', 'Rotated', 'Iso', 'Bandit', 'TBPSA', 'VLP', 'LPC', 'Choice', 'Log', 'Force', 'Multi', 'SQRT', 'NLOPT_GN_ISRES'])):
        raise SkipTest('Skipped because too intricated for this kind of tests!')

    def f(x: Union[np.ndarray, float]) -> float:
        return sum((x - 1.1) ** 2)

    def mf(x: Union[np.ndarray, float]) -> float:
        return sum((x + 1.1) ** 2)

    def f1(x: Union[np.ndarray, float]) -> float:
        return (x - 1.1) ** 2

    def f2(x: Union[np.ndarray, float]) -> float:
        return (x + 1.1) ** 2

    def f1p(x: Union[np.ndarray, float]) -> float:
        return sum((x - 1.1) ** 2)

    def f2p(x: Union[np.ndarray, float]) -> float:
        return sum((x + 1.1) ** 2)
    if 'BAR' in name or 'Cma' in name or 'CMA' in name or ('BIPOP' in name) or ('DS3' in name):
        budget: int = 800
        if 'BAR' in name or 'DS3' in name:
            budget = 3600
        if any((x in name for x in ['Large', 'Tiny', 'Para', 'Diagonal'])):
            return
        val: float = optimizer_cls(2, budget).minimize(f).value[0]
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
        self.filepath: Path = filepath
        self.recommendations: pd.DataFrame = pd.DataFrame(columns=[f'v{k}' for k in range(16)])
        if filepath.exists():
            self.recommendations = pd.read_csv(filepath, index_col=0)

    def save(self) -> None:
        names = sorted((x for x in self.recommendations.index if x in registry))
        recom = self.recommendations.loc[names]
        recom = recom.round(10)
        recom.to_csv(self.filepath)


@pytest.fixture(scope='module')
def recomkeeper() -> Generator[RecommendationKeeper, None, None]:
    keeper = RecommendationKeeper(filepath=Path(__file__).parent / 'recorded_recommendations.csv')
    yield keeper
    keeper.save()


@pytest.mark.parametrize('name', short_registry)
def test_optimizers_recommendation(name: str, recomkeeper: RecommendationKeeper) -> None:
    if any((x in name for x in ['SMAC', 'BO', 'AX'])) and CI:
        raise SkipTest('too slow for CI!')
    if name in UNSEEDABLE or 'BAR' in name or 'AX' in name or ('DS' in name) or ('Carola' in name and any((x in name for x in ['8', '9', '1']))) or (name[0] == 'F' or name[-1] == 'F'):
        raise SkipTest('Not playing nicely with the tests (unseedable)')
    if 'SQP' in name and 'CMA' in name or 'Chain' in name:
        raise SkipTest('No combinations of algorithms here')
    if 'BO' in name or 'EDA' in name:
        raise SkipTest('BO differs from one computer to another')
    if 'SMAC' in name:
        raise SkipTest('SMAC is too slow for the 20s limit')
    if 'Sqrt' in name[:5] or 'Log' in name[:5] or 'Multi' in name[:6] or (name == 'BFGSCMA'):
        raise SkipTest('Let us skip combinations.')
    if len(name) > 8:
        raise SkipTest('Let us check only compact methods.')
    optimizer_cls: Any = registry[name]
    np.random.seed(None)
    if optimizer_cls.recast:
        np.random.seed(12)
        random.seed(12)
    budget: int = {'WidePSO': 100, 'PSO': 200, 'MEDA': 100, 'EDA': 100, 'MPCEDA': 100, 'TBPSA': 100}.get(name, 6)
    if isinstance(optimizer_cls, (optlib.DifferentialEvolution, optlib.EvolutionStrategy)):
        budget = 80
    dimension: int = min(16, max(4, int(np.sqrt(budget))))
    fitness = Fitness([0.5, -0.8, 0, 4] + (5 * np.cos(np.arange(dimension - 4))).tolist())
    with testing.suppress_nevergrad_warnings():
        optim = optimizer_cls(parametrization=dimension, budget=budget, num_workers=1)
        optim.parametrization.random_state.seed(12)
        np.testing.assert_equal(optim.name, name)
        recom = optim.minimize(fitness)
    if name not in recomkeeper.recommendations.index:
        for i in range(len(recom.value)):
            recomkeeper.recommendations.loc[name, f'v{i}'] = recom.value[i]
        raise ValueError(f'Recorded the value {tuple(recom.value)} for optimizer "{name}", please rerun this test locally.')
    decimal: int = 2 if isinstance(optimizer_cls, optlib.ParametrizedBO) or 'BO' in name else 5
    np.testing.assert_array_almost_equal(
        recom.value,
        np.array(recomkeeper.recommendations.loc[name, :][:dimension], float),
        decimal=decimal,
        err_msg=f'Something has changed, if this is normal, delete the following file and rerun to update the values:\n{recomkeeper.filepath}'
    )
    if isinstance(optimizer_cls, optlib.EvolutionStrategy):
        assert recom.loss is not None


@testing.parametrized(de=('DE', 10, 10, 30), de_w=('DE', 50, 40, 40), de1=('OnePointDE', 10, 10, 30), de1_w=('OnePointDE', 50, 40, 40), dim_d=('AlmostRotationInvariantDEAndBigPop', 50, 40, 51), dim=('AlmostRotationInvariantDEAndBigPop', 10, 40, 40), dim_d_rot=('RotationInvariantDE', 50, 40, 51), large=('BPRotationInvariantDE', 10, 40, 70))
def test_differential_evolution_popsize(name: str, dimension: int, num_workers: int, expected: int) -> None:
    if long_name(name):
        raise SkipTest('Too many things in CircleCI')
    optim = registry[name](parametrization=dimension, budget=100, num_workers=num_workers)
    np.testing.assert_equal(optim.llambda, expected)


@testing.suppress_nevergrad_warnings()
def test_portfolio_budget() -> None:
    for k in range(3, 13):
        optimizer = optlib.Portfolio(parametrization=2, budget=k)
        np.testing.assert_equal(optimizer.budget, sum((o.budget for o in optimizer.optims)))


def test_optimizer_families_repr() -> None:
    Cls = optlib.DifferentialEvolution
    np.testing.assert_equal(repr(Cls()), 'DifferentialEvolution()')
    np.testing.assert_equal(repr(Cls(initialization='LHS')), "DifferentialEvolution(initialization='LHS')")
    optim = optlib.RandomSearchMaker(sampler='cauchy')
    np.testing.assert_equal(repr(optim), "RandomSearchMaker(sampler='cauchy')")
    optim = optlib.NonObjectOptimizer(method='COBYLA')
    np.testing.assert_equal(repr(optim), "NonObjectOptimizer(method='COBYLA')")
    assert optim.no_parallelization
    optim = optlib.ParametrizedCMA(diagonal=True)
    np.testing.assert_equal(repr(optim), 'ParametrizedCMA(diagonal=True)')
    optim = optlib.NoisySplit(discrete=True)
    np.testing.assert_equal(repr(optim), 'NoisySplit(discrete=True)')
    assert optim._OptimizerClass.multivariate_optimizer is optlib.OptimisticDiscreteOnePlusOne


@pytest.mark.parametrize('name', ['PSO', 'DE'])
def test_tell_not_asked(name: str) -> None:
    param = ng.p.Scalar()
    with testing.suppress_nevergrad_warnings():
        opt = optlib.registry[name](parametrization=param, budget=2, num_workers=2)
    opt.llambda = 2
    t_10 = opt.parametrization.spawn_child(new_value=10)
    t_100 = opt.parametrization.spawn_child(new_value=100)
    assert not opt.population
    opt.tell(t_10, 90)
    assert len(opt.population) == 1
    asked = opt.ask()
    opt.tell(asked, 88)
    assert len(opt.population) == 2
    opt.tell(t_100, 0)
    asked = opt.ask()
    opt.tell(asked, 89)
    assert len(opt.population) == 2
    assert opt.num_tell == 4, opt.num_tell
    assert opt.num_ask == 2
    assert len(opt.population) == 2
    assert int(opt.recommend().value) == 100
    if isinstance(opt.population, dict):
        assert t_100.uid in opt.population
    for point, value in opt.archive.items_as_arrays():
        assert value.count == 1, f'Error for point {point}'


def test_tbpsa_recom_with_update() -> None:
    budget: int = 20
    fitness = Fitness([0.5, -0.8, 0, 4])
    optim = optlib.TBPSA(parametrization=4, budget=budget, num_workers=1)
    optim.parametrization.random_state.seed(12)
    optim.popsize.llambda = 3
    candidate = optim.minimize(fitness)
    np.testing.assert_almost_equal(candidate.args[0], [0.037964, 0.0433031, -0.4688667, 0.3633273])


def _square(x: np.ndarray, y: int = 12) -> float:
    return float(sum((x - 0.5) ** 2)) + abs(y)


def _smooth_target(x: np.ndarray) -> float:
    result: float = 0.0
    d: int = len(x)
    for h in range(d):
        for v in range(d):
            val: float = x[h][v]
            assert np.abs(val) <= 1.0
            target: float = h / d - v / d
            result += 1.0 if np.abs(target - val) > 0.1 else 0.0
    return result


def test_optimization_doc_parametrization_example() -> None:
    instrum = ng.p.Instrumentation(ng.p.Array(shape=(2,)), y=ng.p.Scalar())
    optimizer = optlib.OnePlusOne(parametrization=instrum, budget=100)
    recom = optimizer.minimize(_square)
    assert len(recom.args) == 1
    testing.assert_set_equal(recom.kwargs, ['y'])
    value = _square(*recom.args, **recom.kwargs)
    assert value < 0.25


def test_optimization_discrete_with_one_sample() -> None:
    optimizer = optlib.PortfolioDiscreteOnePlusOne(parametrization=1, budget=10)
    optimizer.minimize(_square)


@pytest.mark.parametrize('name', short_registry)
def test_optim_pickle(name: str) -> None:
    optim = registry[name](parametrization=12, budget=100, num_workers=2)
    with tempfile.TemporaryDirectory() as folder:
        optim.dump(Path(folder) / 'dump_test.pkl')


def test_bo_init() -> None:
    if platform.system() == 'Windows':
        raise SkipTest('This test fails on Windows, no idea why.')
    arg = ng.p.Scalar(init=4, lower=1, upper=10).set_integer_casting()
    gp_param: dict[str, Any] = {'alpha': 1e-05, 'normalize_y': False, 'n_restarts_optimizer': 1, 'random_state': None}
    my_opt = ng.optimizers.ParametrizedBO(gp_parameters=gp_param, initialization=None)
    try:
        optimizer = my_opt(parametrization=arg, budget=10)
        optimizer.minimize(np.abs)
    except Exception as e:
        print(f'Problem {e} in Bayesian optimization.')


def test_chaining() -> None:
    budgets: List[int] = [7, 19]
    optimizer = optlib.Chaining([optlib.LHSSearch, optlib.HaltonSearch, optlib.OnePlusOne], budgets)(2, 40)
    optimizer.minimize(_square)
    expected: List[Tuple[int, int, int]] = [(7, 7, 0), (19, 19 + 7, 7), (14, 14 + 19 + 7, 19 + 7)]
    for (ex_ask, ex_tell, ex_tell_not_asked), opt in zip(expected, optimizer.optimizers):
        assert opt.num_ask == ex_ask
        assert opt.num_tell == ex_tell
        assert opt.num_tell_not_asked == ex_tell_not_asked
    optimizer.ask()
    assert optimizer.optimizers[-1].num_ask == 15


def test_parametrization_optimizer_reproducibility() -> None:
    parametrization = ng.p.Instrumentation(ng.p.Array(shape=(1,)), y=ng.p.Choice(list(range(100))))
    parametrization.random_state.seed(12)
    optimizer = optlib.RandomSearch(parametrization, budget=20)
    recom = optimizer.minimize(_square)
    np.testing.assert_equal(recom.kwargs['y'], 1)
    data = recom.get_standardized_data(reference=optimizer.parametrization)
    recom = optimizer.parametrization.spawn_child()
    with ng.p.helpers.deterministic_sampling(recom):
        recom.set_standardized_data(data)
    np.testing.assert_equal(recom.kwargs['y'], 1)


@testing.suppress_nevergrad_warnings()
def test_parallel_es() -> None:
    opt = optlib.EvolutionStrategy(popsize=3, offsprings=None)(4, budget=20, num_workers=5)
    for k in range(35):
        cand = opt.ask()
        if not k:
            opt.tell(cand, 1)


class QuadFunction:
    """Quadratic function for testing purposes"""

    def __init__(self, scale: float, ellipse: bool) -> None:
        self.scale: float = scale
        self.ellipse: bool = ellipse

    def __call__(self, x: np.ndarray) -> float:
        y = x - self.scale
        if self.ellipse:
            y *= np.arange(1, x.size + 1) ** 2
        return float(sum(y ** 2))


META_TEST_ARGS: List[str] = 'dimension,num_workers,scale,budget,ellipsoid'.split(',')


def get_metamodel_test_settings(seq: bool = False, special: bool = False) -> List[Tuple[int, int, float, int, bool]]:
    tests_metamodel: List[Tuple[int, int, float, int, bool]] = [(2, 8, 1.0, 120, False), (2, 3, 8.0, 130, True), (5, 1, 1.0, 150, False)]
    if special:
        tests_metamodel += [(8, 27, 8.0, 380, True), (2, 1, 8.0, 120, True), (2, 3, 8.0, 70, False), (1, 1, 1.0, 20, True), (1, 3, 5.0, 20, False), (2, 3, 1.0, 70, True), (2, 1, 8.0, 40, False), (5, 3, 1.0, 225, True), (5, 1, 8.0, 150, False), (5, 3, 8.0, 500, True), (9, 27, 8.0, 700, True), (10, 27, 8.0, 400, False)]
    if seq:
        for i, (d, _, s, b, e) in enumerate(tests_metamodel):
            tests_metamodel[i] = (d, 1, s, b, e)
    return tests_metamodel


@testing.suppress_nevergrad_warnings()
@skip_win_perf
@pytest.mark.parametrize('args', get_metamodel_test_settings())
@pytest.mark.parametrize('baseline', ('CMA', 'ECMA'))
def test_metamodel(baseline: str, args: Tuple[int, int, float, int, bool]) -> None:
    """The test can operate on the sphere or on an elliptic funciton."""
    kwargs = dict(zip(META_TEST_ARGS, args))
    check_metamodel(baseline=baseline, **kwargs)


def check_metamodel(dimension: int, num_workers: int, scale: float, budget: int, ellipsoid: bool, baseline: str, num_trials: int = 1) -> None:
    """This check is called in parametrized tests, with several different parametrization
    (see test_special.py)
    """
    target: Callable[[np.ndarray], float] = QuadFunction(scale=scale, ellipse=ellipsoid)
    contextual_budget: int = budget if ellipsoid else 3 * budget
    contextual_budget *= int(max(1, np.sqrt(scale)))
    successes: int = 0
    for _ in range(num_trials):
        if successes > num_trials // 2:
            break
        recommendations: List[float] = []
        for name in ('MetaModel', baseline if dimension > 1 else 'OnePlusOne'):
            opt = registry[name](dimension, contextual_budget, num_workers=num_workers)
            recommendations.append(opt.minimize(target).value)
        metamodel_recom: float = recommendations[0]
        default_recom: float = recommendations[1]
        if target(np.array([default_recom])) < target(np.array([metamodel_recom])):
            continue
        if budget > 60 * dimension:
            if not target(np.array([default_recom])) > 4.0 * target(np.array([metamodel_recom])):
                continue
        if budget > 60 * dimension and (not ellipsoid):
            if not target(np.array([default_recom])) > 7.0 * target(np.array([metamodel_recom])):
                continue
        successes += 1
        assert successes > num_trials // 2, f'Problem for beating {baseline}.'


@pytest.mark.parametrize('penalization,expected,as_layer', [(False, [1.005573, 0.0003965783], False), (True, [0.0, 0.0], False), (False, [1.000132, -0.0003679], True)])
@testing.suppress_nevergrad_warnings()
def test_constrained_optimization(penalization: bool, expected: List[float], as_layer: bool) -> None:

    def constraint(i: Tuple[Any, dict[str, Any]]) -> Union[bool, float]:
        if penalization:
            return -float(abs(i[1]['x'][0] - 1))
        out: bool = i[1]['x'][0] >= 1
        return out if not as_layer else float(not out)
    parametrization = ng.p.Instrumentation(x=ng.p.Array(shape=(1,)), y=ng.p.Scalar())
    optimizer = optlib.OnePlusOne(parametrization, budget=100)
    optimizer.parametrization.random_state.seed(12)
    if penalization:
        optimizer._constraints_manager.update(max_trials=10, penalty_factor=10)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        optimizer.parametrization.register_cheap_constraint(constraint, as_layer=as_layer)
    recom = optimizer.minimize(_square, verbosity=2)
    np.testing.assert_array_almost_equal([recom.kwargs['x'][0], recom.kwargs['y']], expected)


@pytest.mark.parametrize('name', short_registry)
def test_parametrization_offset(name: str) -> None:
    if long_name(name):
        return
    if any((x in name for x in ['SMAC', 'BO', 'AX'])) and CI:
        raise SkipTest('too slow for CI!')
    if sum([ord(c) for c in name]) % 4 > 0:
        raise SkipTest('Randomly skipping 75% of these tests.')
    if 'PSO' in name or 'BO' in name:
        raise SkipTest('PSO and BO have large initial variance')
    if 'Cobyla' in name and platform.system() == 'Windows':
        raise SkipTest('Cobyla is flaky on Windows for unknown reasons')
    parametrization = ng.p.Instrumentation(ng.p.Array(init=[1000000000000.0, 1000000000000.0]))
    with testing.suppress_nevergrad_warnings():
        optimizer = registry[name](parametrization, budget=100, num_workers=1)
    if optimizer.parametrization.tabu_length > 0:
        return
    for k in range(10 if 'BO' not in name else 2):
        candidate = optimizer.ask()
        assert candidate.args[0][0] > 100, f'Candidate value[0] at iteration #{k} is below 100: {candidate.value}'
        optimizer.tell(candidate, 0)


def test_optimizer_sequence() -> None:
    budget: int = 24
    parametrization = ng.p.Tuple(*(ng.p.Scalar(lower=-12, upper=12) for _ in range(2)))
    optimizer = optlib.LHSSearch(parametrization, budget=budget)
    points: List[np.ndarray] = [np.array(optimizer.ask().value) for _ in range(budget)]
    assert sum((any((abs(x) > budget // 2 - 1 for x in p)) for p in points)) > 0


def test_shiwa_dim1() -> None:
    param = ng.p.Log(lower=1, upper=1000).set_integer_casting()
    init = param.value
    optimizer = optlib.Shiwa(param, budget=40)
    recom = optimizer.minimize(np.abs)
    assert recom.value < init


continuous_cases: List[Tuple[str, int, int, int]] = [('NGOpt', d, b, n, f'#CONTINUOUS') for d in [1, 2, 10, 100] for b in [2 * d, 100 * d] for n in [1, 10 * d]]  # type: ignore


@pytest.mark.parametrize('name,param,budget,num_workers,expected', [('Shiwa', 1, 10, 2, 'OnePlusOne'), ('Shiwa', ng.p.Log(lower=1, upper=1000).set_integer_casting(), 10, 2, 'DoubleFastGADiscreteOnePlusOne')] + continuous_cases)  # type: ignore
@testing.suppress_nevergrad_warnings()
def test_ngopt_selection(name: str, param: Any, budget: int, num_workers: int, expected: str, caplog: Any) -> None:
    with caplog.at_level(logging.DEBUG, logger='nevergrad.optimization.optimizerlib'):
        opt = optlib.registry[name](param, budget=budget, num_workers=num_workers)
        opt.optim
        pattern = f'.*{name} selected (?P<name>\\w+?) optimizer\\.'
        match = re.match(pattern, caplog.text.splitlines()[-1])
        assert match is not None, f'Did not detect selection in logs: {caplog.text}'
        choice: str = match.group('name')
        if expected != '#CONTINUOUS':
            assert choice == expected
        else:
            print(f'Continuous param={param} budget={budget} workers={num_workers} --> {choice}')
            if num_workers >= budget > 600:
                assert choice == 'MetaTuneRecentering'
            if num_workers > 1:
                assert choice not in ['SQP', 'Cobyla']
        if 'CMA' not in choice:
            assert choice == opt._info()['sub-optim']
        else:
            assert choice in opt._info()['sub-optim']


def test_bo_ordering() -> None:
    with testing.suppress_nevergrad_warnings():
        optim = ng.optimizers.ParametrizedBO(initialization='Hammersley')(parametrization=ng.p.Choice(range(12)), budget=10)
    cand = optim.ask()
    optim.tell(cand, 12)
    optim.provide_recommendation()


@skip_win_perf
@pytest.mark.parametrize('name,dimension,num_workers,fake_learning,budget,expected', [
    ('NGOpt8', 3, 1, False, 100, ['OnePlusOne', 'OnePlusOne']),
    ('NGOpt8', 3, 1, False, 200, ['SQP', 'SQP']),
    ('NGOpt8', 3, 1, True, 1000, ['SQP', 'monovariate', 'monovariate']),
    (None, 3, 1, False, 1000, ['CMA', 'OnePlusOne']),
    (None, 3, 20, False, 1000, ['MetaModel', 'OnePlusOne'])
])
def test_ngo_split_optimizer(name: Optional[str], dimension: int, num_workers: int, fake_learning: bool, budget: int, expected: List[str]) -> None:
    if fake_learning:
        param = ng.p.Instrumentation(
            learning_rate=ng.p.Log(lower=0.001, upper=1.0),
            batch_size=ng.p.Scalar(lower=1, upper=12).set_integer_casting(),
            architecture=ng.p.Choice(['conv', 'fc'])
        )
    else:
        param = ng.p.Choice(['const', ng.p.Array(init=list(range(dimension)))])
    opt = xpvariants.MetaNGOpt10 if name is None else optlib.ConfSplitOptimizer(multivariate_optimizer=optlib.registry[name])
    optimizer = opt(param, budget=budget, num_workers=num_workers)
    expected_strs: List[str] = [x if x != 'monovariate' else optimizer._config.monovariate_optimizer.name for x in expected]
    assert optimizer._info()['sub-optim'] == ','.join(expected_strs) or 'CMA' in optimizer._info()['sub-optim']


@skip_win_perf
@pytest.mark.parametrize('budget,with_int', [(150, True), (200, True), (666, True), (2000, True), (66, False), (200, False), (666, False), (2000, False)])
def test_ngopt_on_simple_realistic_scenario(budget: int, with_int: bool) -> None:
    if sum([ord(c) for c in f'{budget}-{with_int}']) % 4 > 0:
        raise SkipTest('Randomly skipping 75% of these tests.')

    def fake_training(learning_rate: float, batch_size: Union[int, float], architecture: str) -> float:
        return (learning_rate - 0.2) ** 2 + (batch_size - 4) ** 2 + (0 if architecture == 'conv' else 10)
    parametrization = ng.p.Instrumentation(
        learning_rate=ng.p.Log(lower=0.001, upper=1.0),
        batch_size=ng.p.Scalar(lower=1, upper=12).set_integer_casting() if with_int else ng.p.Scalar(lower=1, upper=12),
        architecture=ng.p.Choice(['conv', 'fc'])
    )
    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=budget)
    recommendation = optimizer.minimize(fake_training)
    result = fake_training(**recommendation.kwargs)
    assert result < (1.0006 if with_int else 0.005), f'{result} not < {(1.0006 if with_int else 0.005)}'


def _multiobjective(z: Tuple[float, float]) -> Tuple[float, float, float]:
    x, y = z
    return (abs(x - 1), abs(y + 1), abs(x - y))


@pytest.mark.parametrize('name', ['DE', 'ES', 'OnePlusOne'])
@testing.suppress_nevergrad_warnings()
def test_mo_constrained(name: str) -> None:
    optimizer = optlib.registry[name](2, budget=60)
    optimizer.parametrization.random_state.seed(12)

    def constraint(arg: Any) -> bool:
        """Random constraint to mess up with the optimizer"""
        return bool(optimizer.parametrization.random_state.rand() > 0.8)
    optimizer.parametrization.register_cheap_constraint(constraint)
    optimizer.minimize(_multiobjective)
    point = optimizer.parametrization.spawn_child(new_value=np.array([1.0, 1.0]))
    optimizer.tell(point, _multiobjective(point.value))
    if isinstance(optimizer, es._EvolutionStrategy):
        assert optimizer._rank_method is not None


@pytest.mark.parametrize('name', ['DE', 'ES', 'OnePlusOne'])
@testing.suppress_nevergrad_warnings()
def test_mo_with_nan(name: str) -> None:
    param = ng.p.Instrumentation(x=ng.p.Scalar(lower=0, upper=5), y=ng.p.Scalar(lower=0, upper=3))
    optimizer = optlib.registry[name](param, budget=60)
    optimizer.tell(ng.p.MultiobjectiveReference(), [10, 10, 10])
    for _ in range(50):
        cand = optimizer.ask()
        optimizer.tell(cand, [-38, 0, np.nan])


@pytest.mark.parametrize('name', ['LhsDE', 'RandomSearch'])
def test_uniform_sampling(name: str) -> None:
    param = ng.p.Scalar(lower=-100, upper=100).set_mutation(sigma=1)
    opt = optlib.registry[name](param, budget=600, num_workers=100)
    above_50: int = 0
    for _ in range(100):
        above_50 += abs(opt.ask().value) > 50
    assert above_50 > 20


def test_paraportfolio_de() -> None:
    workers: int = 40
    opt = optlib.ParaPortfolio(12, budget=100 * workers, num_workers=workers)
    for _ in range(3):
        cands: List[Any] = [opt.ask() for _ in range(workers)]
        for cand in cands:
            opt.tell(cand, np.random.rand())


def test_cma_logs(capsys: Any) -> None:
    opt = registry['CMA'](2, budget=300, num_workers=4)
    [opt.ask() for _ in range(4)]
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''


def _simple_multiobjective(x: np.ndarray) -> List[float]:
    return [np.sum(x ** 2), np.sum((x - 1) ** 2)]


def test_pymoo_pf() -> None:
    optimizer = ng.optimizers.PymooNSGA2(parametrization=2, budget=300)
    optimizer.parametrization.random_state.seed(12)
    optimizer.minimize(_simple_multiobjective)
    pf = optimizer.pareto_front()
    fixed_points: List[List[float]] = [[0.25, 0.75], [0.75, 0.25]]
    for fixed_point in fixed_points:
        values = _simple_multiobjective(np.array(fixed_point))
        assert any((_simple_multiobjective(x.value)[0] < values[0] and _simple_multiobjective(x.value)[1] < values[1] for x in pf))


def test_pymoo_batched() -> None:
    optimizer = ng.optimizers.PymooBatchNSGA2(parametrization=2, budget=300)
    optimizer.parametrization.random_state.seed(12)
    candidates: List[Any] = []
    losses: List[Any] = []
    optimizer.num_objectives = 2
    for _ in range(3):
        asks_from_batch: int = 0
        while optimizer.num_ask == optimizer.num_tell or asks_from_batch < optimizer.batch_size:
            x = optimizer.ask()
            loss = _simple_multiobjective(*x.args, **x.kwargs)
            candidates.append(x)
            losses.append(loss)
            asks_from_batch += 1
        assert asks_from_batch == 100
        while optimizer.num_ask > optimizer.num_tell:
            x = candidates.pop()
            loss = losses.pop()
            optimizer.tell(x, loss)
    assert len(optimizer._current_batch) == 0


def test_smoother() -> None:
    x = ng.p.Array(shape=(5, 5))
    assert optlib.smooth_copy(x).get_standardized_data(reference=x).shape == x.get_standardized_data(reference=x).shape
    x = ng.p.Array(shape=(5, 5)).set_integer_casting()
    assert optlib.smooth_copy(x).get_standardized_data(reference=x).shape == x.get_standardized_data(reference=x).shape


@pytest.mark.parametrize('n', [5, 10, 15, 25, 40])
@pytest.mark.parametrize('b_per_dim', [10, 20])
def test_voronoide(n: int, b_per_dim: int) -> None:
    if n < 25 or (b_per_dim < 1 and (not CI)):
        raise SkipTest('Only big things outside CI.')
    list_optims: List[str] = ['CMA', 'DE', 'PSO', 'RandomSearch', 'TwoPointsDE', 'OnePlusOne']
    if CI and (n > 10 or n * b_per_dim > 100):
        raise SkipTest('Topology optimization too slow in CI')
    if CI or (n < 10 or b_per_dim < 20):
        list_optims = ['CMA', 'PSO', 'OnePlusOne']
    if n > 20:
        list_optims = ['DE', 'TwoPointsDE']
    fails: dict[str, int] = {}
    for o in list_optims:
        fails[o] = 0
    size: int = n * n
    sqrtsize: int = n
    b: int = b_per_dim * size
    nw: int = 20
    num_tests: int = 20
    array = ng.p.Array(shape=(n, n), lower=-1.0, upper=1.0)
    for idx in range(num_tests):
        xa: int = idx % 3
        xb: int = 2 - xa
        xs = 1.5 * (np.array([float(np.cos(xa * i + xb * j) < 0.0) for i in range(n) for j in range(n)]).reshape(n, n) - 0.5)
        if idx // 3 % 2 > 0:
            xs = np.transpose(xs)
        if idx // 6 % 2 > 0:
            xs = -xs

        def f(x: np.ndarray, xs: np.ndarray = xs) -> float:
            return 5.0 * np.sum(np.abs(x - xs) > 0.3) / size + 13.0 * np.linalg.norm(x - gaussian_filter(x, sigma=3)) / sqrtsize
        VoronoiDE = ng.optimizers.VoronoiDE(array, budget=b, num_workers=nw)
        vde: float = f(VoronoiDE.minimize(f).value)
        for o in list_optims:
            try:
                other = ng.optimizers.registry[o](array, budget=b, num_workers=nw)
                val: float = f(other.minimize(f).value)
            except Exception:
                print(f'crash in {o}')
                val = float(10000000.0)
            if val < vde:
                fails[o] += 1
    ratio = min([(num_tests - fails[o]) / (0.001 + fails[o]) for o in list_optims])
    print(f'VoronoiDE for DO: {ratio}', num_tests, fails, f'({n}-{b_per_dim})')
    for o in list_optims:
        ratio_limit = 3.0 if 'DE' not in o else 2.0
        assert num_tests - fails[o] > ratio_limit * fails[o], f'Failure {o}: {fails[o]} / {num_tests}    ({n}-{b_per_dim})'


def test_weighted_moo_de() -> None:
    for _ in range(1):
        D: int = 2
        N: int = 3
        DE = ng.optimizers.TwoPointsDE(D, budget=500)
        index: int = np.random.choice(range(N))
        w: np.ndarray = np.ones(N)
        w[index] = 30.0
        DE.set_objective_weights(w)
        targ: List[np.ndarray] = [np.array([np.cos(2 * np.pi * i / N), np.sin(2 * np.pi * i / N)]) for i in range(N)]
        DE.minimize(lambda x: [np.linalg.norm(x - xi) for xi in targ])
        x: np.ndarray = np.zeros(N)
        for u in DE.pareto_front():
            x = x + u.losses
        assert index == list(x).index(min(x))
