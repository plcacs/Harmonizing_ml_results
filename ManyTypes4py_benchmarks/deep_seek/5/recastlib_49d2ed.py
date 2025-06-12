import functools
import math
import warnings
import weakref
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import numpy as np
from scipy import optimize as scipyoptimize
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.common import errors
from . import base
from .base import IntOrParameter
from . import recaster

class _NonObjectMinimizeBase(recaster.SequentialRecastOptimizer):

    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
        *,
        method: str = 'Nelder-Mead',
        random_restart: bool = False
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.multirun = 1
        self._normalizer: Optional[p.helpers.Normalizer] = None
        self.initial_guess: Optional[np.ndarray] = None
        self.method = method
        self.random_restart = random_restart
        assert method in ['CmaFmin2', 'gomea', 'gomeablock', 'gomeatree', 'SMAC3', 'BFGS', 'RBFGS', 'LBFGSB', 'L-BFGS-B', 'SMAC', 'AX', 'Lamcts', 'Nelder-Mead', 'COBYLA', 'BOBYQA', 'SLSQP', 'pysot', 'negpysot', 'Powell'] or 'NLOPT' in method or 'DS' in method or ('BFGS' in method), f"Unknown method '{method}'"
        if method == 'CmaFmin2' or 'NLOPT' in method or 'AX' in method or ('BOBYQA' in method) or ('pysot' in method) or ('SMAC' in method):
            normalizer = p.helpers.Normalizer(self.parametrization)
            self._normalizer = normalizer

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        """Called whenever calling "tell" on a candidate that was not "asked".
        Defaults to the standard tell pipeline.
        """

    def get_optimization_function(self) -> tp.Callable[[tp.Callable[[np.ndarray], float]], np.ndarray]:
        return functools.partial(self._optimization_function, weakref.proxy(self))

    @staticmethod
    def _optimization_function(weakself: weakref.ReferenceType['_NonObjectMinimizeBase'], objective_function: tp.Callable[[np.ndarray], float]) -> np.ndarray:
        budget = np.inf if weakself().budget is None else weakself().budget
        best_res = np.inf
        best_x = weakself().current_bests['average'].x
        if weakself().initial_guess is not None:
            best_x = np.array(weakself().initial_guess, copy=True)
        remaining = budget - weakself()._num_ask

        def ax_obj(p: Dict[str, float]) -> float:
            data = [p['x' + str(i)] for i in range(weakself().dimension)]
            if weakself()._normalizer:
                data = weakself()._normalizer.backward(np.asarray(data, dtype=np.float64))
            return objective_function(data)
        while remaining > 0:
            options: Dict[str, Any] = {} if weakself().budget is None else {'maxiter': remaining}
            if weakself().method == 'BOBYQA' or (weakself().method == 'CmaFmin2' and weakself().dimension == 1):
                import pybobyqa
                res = pybobyqa.solve(objective_function, best_x, maxfun=budget, do_logging=False)
                if res.f < best_res:
                    best_res = res.f
                    best_x = res.x
            elif weakself().method[:2] == 'DS':
                import directsearch
                dict_solvers = {'base': directsearch.solve_directsearch, 'proba': directsearch.solve_probabilistic_directsearch, 'subspace': directsearch.solve_subspace_directsearch, '3p': directsearch.solve_stp}
                solve = dict_solvers[weakself().method[2:]]
                best_x = solve(objective_function, x0=best_x, maxevals=budget).x
                if weakself()._normalizer is not None:
                    best_x = weakself()._normalizer.backward(np.asarray(best_x, dtype=np.float32))
            elif weakself().method[:3] == 'PDS':
                import directsearch
                solve = directsearch.solve_probabilistic_directsearch
                DSseed = int(weakself().method[3:])
                best_x = solve(objective_function, x0=best_x, maxevals=budget, gamma_inc=1.0 + np.random.RandomState(DSseed).rand() * 3.0, gamma_dec=np.random.RandomState(DSseed + 42).rand()).x
                if weakself()._normalizer is not None:
                    best_x = weakself()._normalizer.backward(np.asarray(best_x, dtype=np.float32))
            elif weakself().method == 'AX':
                from ax import optimize as axoptimize
                parameters = [{'name': 'x' + str(i), 'type': 'range', 'bounds': [0.0, 1.0]} for i in range(weakself().dimension)]
                best_parameters, _best_values, _experiment, _model = axoptimize(parameters, evaluation_function=ax_obj, minimize=True, total_trials=budget)
                best_x = np.array([float(best_parameters['x' + str(i)]) for i in range(weakself().dimension)])
                best_x = weakself()._normalizer.backward(np.asarray(best_x, dtype=float))
            elif weakself().method[:5] == 'NLOPT':
                import nlopt

                def nlopt_objective_function(*args: Any) -> float:
                    try:
                        data = np.asarray([arg for arg in args if len(arg) > 0])[0]
                    except Exception as e:
                        raise ValueError(f'{e}:\n{args}\n {[arg for arg in args]}')
                    assert len(data) == weakself().dimension, str(data) + ' does not have length ' + str(weakself().dimension)
                    if weakself()._normalizer is not None:
                        data = weakself()._normalizer.backward(np.asarray(data, dtype=np.float32))
                    return objective_function(data)
                nlopt_param = getattr(nlopt, weakself().method[6:]) if len(weakself().method) > 5 else nlopt.LN_SBPLX
                opt = nlopt.opt(nlopt_param, weakself().dimension)
                opt.set_min_objective(nlopt_objective_function)
                opt.set_lower_bounds(np.zeros(weakself().dimension))
                opt.set_upper_bounds(np.ones(weakself().dimension))
                opt.set_maxeval(budget)
                firstguess = 0.5 * np.ones(weakself().dimension)
                best_x = opt.optimize(firstguess)
                if weakself()._normalizer is not None:
                    best_x = weakself()._normalizer.backward(np.asarray(best_x, dtype=np.float32))
            elif 'pysot' in weakself().method:
                from poap.controller import BasicWorkerThread, ThreadController
                from pySOT.experimental_design import SymmetricLatinHypercube
                from pySOT.optimization_problems import OptimizationProblem
                from pySOT.strategy import DYCORSStrategy
                from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant

                class LocalOptimizationProblem(OptimizationProblem):

                    def eval(self, data: np.ndarray) -> float:
                        if weakself()._normalizer is not None:
                            data = weakself()._normalizer.backward(np.asarray(data, dtype=np.float32))
                        val = float(objective_function(data)) if 'negpysot' not in weakself().method else -float(objective_function(data))
                        return val
                dim = weakself().dimension
                opt_prob = LocalOptimizationProblem()
                opt_prob.dim = dim
                opt_prob.lb = np.array([0.0] * dim)
                opt_prob.ub = np.array([1.0] * dim)
                opt_prob.int_var = []
                opt_prob.cont_var = np.array(range(dim))
                rbf = RBFInterpolant(dim=opt_prob.dim, lb=opt_prob.lb, ub=opt_prob.ub, kernel=CubicKernel(), tail=LinearTail(opt_prob.dim))
                slhd = SymmetricLatinHypercube(dim=opt_prob.dim, num_pts=2 * (opt_prob.dim + 1))
                controller = ThreadController()
                controller.strategy = DYCORSStrategy(opt_prob=opt_prob, exp_design=slhd, surrogate=rbf, max_evals=budget, asynchronous=True)
                worker = BasicWorkerThread(controller, opt_prob.eval)
                controller.launch_worker(worker)
                result = controller.run()
                best_res = result.value
                best_x = result.params[0]
            elif weakself().method == 'SMAC3':
                from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter
                from smac import HyperparameterOptimizationFacade, Scenario
                import threading
                import os
                import time
                from pathlib import Path
                the_date = str(time.time()) + '_' + str(np.random.rand())
                tag = str(np.random.rand())
                feed = '/tmp/smac_feed' + the_date + '.txt'
                fed = '/tmp/smac_fed' + the_date + '.txt'

                def dummy_function() -> None:
                    for _ in range(remaining):
                        while not Path(feed).is_file() or os.stat(feed).st_size == 0:
                            time.sleep(0.1)
                        time.sleep(0.1)
                        data = np.loadtxt(feed)
                        os.remove(feed)
                        res = objective_function(data)
                        f = open(fed, 'w')
                        f.write(str(res))
                        f.close()
                    return
                thread = threading.Thread(target=dummy_function)
                thread.start()
                cs = ConfigurationSpace()
                cs.add_hyperparameters([UniformFloatHyperparameter(f'x{tag}{i}', 0.0, 1.0, default_value=0.0) for i in range(weakself().dimension)])

                def smac2_obj(p: Dict[str, float], seed: int = 0) -> float:
                    pdata = [p[f'x{tag}{i}'] for i in range(len(p.keys()))]
                    data = weakself()._normalizer.backward(np.asarray(pdata, dtype=float))
                    if Path(fed).is_file():
                        os.remove(fed)
                    np.savetxt(feed, data)
                    while not Path(fed).is_file() or os.stat(fed).st_size == 0:
                        time.sleep(0.1)
                    time.sleep(0.1)
                    f = open(fed, 'r')
                    res = float(f.read())
                    f.close()
                    return res
                scenario = Scenario(cs, deterministic=True, n_trials=int(remaining))
                smac = HyperparameterOptimizationFacade(scenario, smac2_obj)
                res = smac.optimize()
                best_x = np.array([res[f'x{tag}{k}'] for k in range(len(res.keys()))])
                best_x = weakself()._normalizer.backward(np.asarray(best_x, dtype=float))
                thread.join()
                weakself()._num_ask = budget
            elif weakself().method == 'CmaFmin2' and weakself().dimension > 1:
                import cma

                def cma_objective_function(data: np.ndarray) -> float:
                    if weakself()._normalizer is not None and weakself()._normalizer.fully_bounded:
                        data = weakself()._normalizer.backward(np.asarray(data, dtype=np.float32))
                    return objective_function(data)
                x0 = 0.5 * np.ones(weakself().dimension) if weakself()._normalizer is not None and weakself()._normalizer.fully_bounded else np.zeros(weakself().dimension)
                num_calls = 0
                while budget - num_calls > 0:
                    options = {'maxfevals': budget - num_calls, 'verbose': -9}
                    if weakself()._normalizer is not None and weakself()._normalizer.fully_bounded:
                        options['bounds'] = [0.0, 1.0]
                    res = cma.fmin(cma_objective_function, x0=x0, sigma0=0.2, options=options, restarts=9)
                    x0 = 0.5 + np.random.uniform() * np.random.uniform(low=-0.5, high=0.5, size=weakself().dimension) if weakself()._normalizer is not None and weakself()._normalizer.fully_bounded else np.random.randn(weakself().dimension)
                    if res[1] < best_res:
                        best_res = res[1]
                        best_x = res[0]
                        if weakself()._normalizer is not None:
                            best_x = weakself()._normalizer.backward(np.asarray(best_x, dtype=np.float32))
                    num_calls += res[2]
            else:
                res = scipyoptimize.minimize(objective_function, best_x if not weakself().random_restart else weakself()._rng.normal(0.0, 1.0, weakself().dimension), method=weakself().method, options=options, tol=0)
                if res.fun < best_res:
                    best_res = res.fun
                    best_x = res.x
            remaining = budget - weakself()._num_ask
        assert best_x is not None
        return best_x

class NonObjectOptimizer(base.ConfiguredOptimizer):
    """Wrapper over Scipy optimizer implementations, in standard ask and tell format."""
    recast = True
    no_parallelization = True

    def __init__(self, *, method: str = 'Nelder-Mead', random_restart: bool = False) -> None:
        super().__init__(_NonObjectMinimizeBase, locals())

AX = NonObjectOptimizer(method='AX').set_name('AX', register=True)
BOBYQA = NonObjectOptimizer(method='BOBYQA').set_name('BOBYQA', register=True)
NelderMead = NonObjectOptimizer(method='Nelder-Mead').set_name('NelderMead', register=True)
CmaFmin2 = NonObjectOptimizer(method='CmaFmin2').set_name('CmaFmin2', register=True)
Powell = NonObjectOptimizer(method='Powell').set_name('Powell', register=True)
RPowell = NonObjectOptimizer(method='Powell', random_restart=True).set_name('RPowell', register=True)
BFGS = NonObjectOptimizer(method='BFGS', random_restart=False).set_name('BFGS', register=True)
RBFGS = NonObjectOptimizer(method='BFGS', random_restart=True).set_name('RBFGS', register=True)
LBFGSB = NonObjectOptimizer(method='L-BFGS-B', random_restart=True).set_name('LBFGSB', register=True)
Cobyla = NonObjectOptimizer(method='COBYLA').set_name('Cobyla', register=True)
RCobyla = NonObjectOptimizer(method='COBYLA', random_restart=True).set_name('RCobyla', register=True)
SQP = NonObjectOptimizer(method='SLSQP').set_name('SQP', register=True)
SLSQP = SQP
RSQP = NonObjectOptimizer(method='SLSQP', random_restart=True).set_name('RSQP', register=True)
RSLSQP = RSQP
NLOPT_LN_SBPLX = NonObjectOptimizer(method='NLOPT_LN_SBPLX').set_name('NLOPT_LN_SBPLX', register=True)
NLOPT_LN_PRAXIS = NonObjectOptimizer(method='NLOPT_LN_PRAXIS').set_name('NLOPT_LN_PRAXIS', register=True)
NLOPT_GN_DIRECT = NonObjectOptimizer(method='NLOPT_GN_DIRECT').set_name('NLOPT_GN_DIRECT', register=True)
NLOPT_GN_DIRECT_L = NonObjectOptimizer(method='NLOPT_GN_DIRECT_L').set_name('NLOPT_GN_DIRECT_L', register=True)
NLOPT_GN_CRS2_LM = NonObjectOptimizer(method='NLOPT_GN_CRS2_LM').set_name('NLOPT_GN_CRS2_LM', register=True)
NLOPT_GN_AGS = NonObjectOptimizer(method='NLOPT_GN_AGS').set_name('NLOPT_GN_AGS', register=True)
NLOPT_GN_ISRES = NonObjectOptimizer(method='NLOPT_GN_ISRES').set_name('NLOPT_GN_ISRES', register=True)
NLOPT_GN_ESCH = NonObjectOptimizer(method='NLOPT_GN_ESCH').set_name('NLOPT_GN_ESCH', register=True)
NLOPT_LN_COBYLA = NonObjectOptimizer(method='NLOPT_LN_COBYLA').set_name('NLOPT_LN_COBYLA', register=True)
NLOPT_LN_BOBYQA = NonObjectOptimizer(method='NLOPT_LN_BOBYQA').set_name('NLOPT_LN_BOBYQA', register=True)
NLOPT_LN_NEWUOA_BOUND = NonObjectOptimizer(method='NLOPT_LN_NEWUOA_BOUND').set_name('NLOPT_LN_NEWUOA_BOUND', register=True)
NLOPT_LN_NELDER