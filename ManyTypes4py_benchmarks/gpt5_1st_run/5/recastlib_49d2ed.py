import functools
import math
import warnings
import weakref
from typing import Any, Callable, Dict, Optional, Union, Sequence

import numpy as np
from scipy import optimize as scipyoptimize
import nevergrad.common.typing as tp  # noqa: F401
from nevergrad.parametrization import parameter as p
from nevergrad.common import errors
from . import base
from .base import IntOrParameter  # noqa: F401
from . import recaster


class _NonObjectMinimizeBase(recaster.SequentialRecastOptimizer):

    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
        *,
        method: str = "Nelder-Mead",
        random_restart: bool = False,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.multirun: int = 1
        self._normalizer: Optional[Any] = None
        self.initial_guess: Optional[Sequence[float]] = None
        self.method: str = method
        self.random_restart: bool = random_restart
        assert method in [
            "CmaFmin2",
            "gomea",
            "gomeablock",
            "gomeatree",
            "SMAC3",
            "BFGS",
            "RBFGS",
            "LBFGSB",
            "L-BFGS-B",
            "SMAC",
            "AX",
            "Lamcts",
            "Nelder-Mead",
            "COBYLA",
            "BOBYQA",
            "SLSQP",
            "pysot",
            "negpysot",
            "Powell",
        ] or "NLOPT" in method or "DS" in method or ("BFGS" in method), f"Unknown method '{method}'"
        if method == "CmaFmin2" or "NLOPT" in method or "AX" in method or ("BOBYQA" in method) or ("pysot" in method) or ("SMAC" in method):
            normalizer = p.helpers.Normalizer(self.parametrization)
            self._normalizer = normalizer

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: Union[float, np.ndarray]) -> None:
        """Called whenever calling "tell" on a candidate that was not "asked".
        Defaults to the standard tell pipeline.
        """

    def get_optimization_function(self) -> Callable[[Callable[[np.ndarray], Union[float, np.ndarray]]], np.ndarray]:
        return functools.partial(self._optimization_function, weakref.proxy(self))

    @staticmethod
    def _optimization_function(
        weakself: Any, objective_function: Callable[[np.ndarray], Union[float, np.ndarray]]
    ) -> np.ndarray:
        budget: float = np.inf if weakself.budget is None else weakself.budget  # type: ignore[assignment]
        best_res: float = np.inf
        best_x: np.ndarray = weakself.current_bests["average"].x  # type: ignore[attr-defined]
        if weakself.initial_guess is not None:
            best_x = np.array(weakself.initial_guess, copy=True, dtype=float)
        remaining: float = budget - weakself._num_ask  # type: ignore[operator]

        def ax_obj(param_dict: Dict[str, float]) -> float:
            data = [param_dict["x" + str(i)] for i in range(weakself.dimension)]
            if weakself._normalizer:
                data = weakself._normalizer.backward(np.asarray(data, dtype=np.float64))
            return float(objective_function(np.asarray(data, dtype=float)))

        while remaining > 0:
            options: Dict[str, Union[int, float]] = {} if weakself.budget is None else {"maxiter": remaining}
            if weakself.method == "BOBYQA" or (weakself.method == "CmaFmin2" and weakself.dimension == 1):
                import pybobyqa  # type: ignore[import-not-found]

                res: Any = pybobyqa.solve(objective_function, best_x, maxfun=budget, do_logging=False)  # type: ignore[arg-type]
                if res.f < best_res:
                    best_res = float(res.f)
                    best_x = np.asarray(res.x, dtype=float)
            elif weakself.method[:2] == "DS":
                import directsearch  # type: ignore[import-not-found]

                dict_solvers: Dict[str, Callable[..., Any]] = {
                    "base": directsearch.solve_directsearch,
                    "proba": directsearch.solve_probabilistic_directsearch,
                    "subspace": directsearch.solve_subspace_directsearch,
                    "3p": directsearch.solve_stp,
                }
                solve = dict_solvers[weakself.method[2:]]
                best_x = solve(objective_function, x0=best_x, maxevals=budget).x  # type: ignore[arg-type]
                if weakself._normalizer is not None:
                    best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=np.float32))
            elif weakself.method[:3] == "PDS":
                import directsearch  # type: ignore[import-not-found]

                solve = directsearch.solve_probabilistic_directsearch
                DSseed = int(weakself.method[3:])
                best_x = solve(
                    objective_function,
                    x0=best_x,
                    maxevals=budget,
                    gamma_inc=1.0 + np.random.RandomState(DSseed).rand() * 3.0,
                    gamma_dec=np.random.RandomState(DSseed + 42).rand(),
                ).x  # type: ignore[arg-type]
                if weakself._normalizer is not None:
                    best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=np.float32))
            elif weakself.method == "AX":
                from ax import optimize as axoptimize  # type: ignore[import-not-found]

                parameters = [{"name": "x" + str(i), "type": "range", "bounds": [0.0, 1.0]} for i in range(weakself.dimension)]
                best_parameters, _best_values, _experiment, _model = axoptimize(
                    parameters, evaluation_function=ax_obj, minimize=True, total_trials=budget  # type: ignore[arg-type]
                )
                best_x = np.array([float(best_parameters["x" + str(i)]) for i in range(weakself.dimension)], dtype=float)
                best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=float))
            elif weakself.method[:5] == "NLOPT":
                import nlopt  # type: ignore[import-not-found]

                def nlopt_objective_function(*args: np.ndarray) -> float:
                    try:
                        data = np.asarray([arg for arg in args if len(arg) > 0])[0]
                    except Exception as e:
                        raise ValueError(f"{e}:\n{args}\n {[arg for arg in args]}")
                    assert len(data) == weakself.dimension, str(data) + " does not have length " + str(weakself.dimension)
                    if weakself._normalizer is not None:
                        data = weakself._normalizer.backward(np.asarray(data, dtype=np.float32))
                    return float(objective_function(np.asarray(data, dtype=float)))

                nlopt_param = getattr(nlopt, weakself.method[6:]) if len(weakself.method) > 5 else nlopt.LN_SBPLX  # type: ignore[attr-defined]
                opt = nlopt.opt(nlopt_param, weakself.dimension)  # type: ignore[attr-defined]
                opt.set_min_objective(nlopt_objective_function)
                opt.set_lower_bounds(np.zeros(weakself.dimension))
                opt.set_upper_bounds(np.ones(weakself.dimension))
                opt.set_maxeval(budget)  # type: ignore[arg-type]
                firstguess = 0.5 * np.ones(weakself.dimension)
                best_x = np.asarray(opt.optimize(firstguess), dtype=float)  # type: ignore[attr-defined]
                if weakself._normalizer is not None:
                    best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=np.float32))
            elif "pysot" in weakself.method:
                from poap.controller import BasicWorkerThread, ThreadController  # type: ignore[import-not-found]
                from pySOT.experimental_design import SymmetricLatinHypercube  # type: ignore[import-not-found]
                from pySOT.optimization_problems import OptimizationProblem  # type: ignore[import-not-found]
                from pySOT.strategy import DYCORSStrategy  # type: ignore[import-not-found]
                from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant  # type: ignore[import-not-found]

                class LocalOptimizationProblem(OptimizationProblem):  # type: ignore[misc]
                    def eval(self, data: np.ndarray) -> float:
                        if weakself._normalizer is not None:
                            data = weakself._normalizer.backward(np.asarray(data, dtype=np.float32))
                        val = float(objective_function(np.asarray(data, dtype=float))) if "negpysot" not in weakself.method else -float(
                            objective_function(np.asarray(data, dtype=float))
                        )
                        return val

                dim = weakself.dimension
                opt_prob = LocalOptimizationProblem()
                opt_prob.dim = dim
                opt_prob.lb = np.array([0.0] * dim)
                opt_prob.ub = np.array([1.0] * dim)
                opt_prob.int_var = []
                opt_prob.cont_var = np.array(range(dim))
                rbf = RBFInterpolant(dim=opt_prob.dim, lb=opt_prob.lb, ub=opt_prob.ub, kernel=CubicKernel(), tail=LinearTail(opt_prob.dim))
                slhd = SymmetricLatinHypercube(dim=opt_prob.dim, num_pts=2 * (opt_prob.dim + 1))
                controller = ThreadController()
                controller.strategy = DYCORSStrategy(opt_prob=opt_prob, exp_design=slhd, surrogate=rbf, max_evals=budget, asynchronous=True)  # type: ignore[arg-type]
                worker = BasicWorkerThread(controller, opt_prob.eval)
                controller.launch_worker(worker)
                result = controller.run()
                best_res = float(result.value)
                best_x = np.asarray(result.params[0], dtype=float)
            elif weakself.method == "SMAC3":
                from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter  # type: ignore[import-not-found]  # noqa: F401
                from smac import HyperparameterOptimizationFacade, Scenario  # type: ignore[import-not-found]
                import threading
                import os
                import time
                from pathlib import Path

                the_date = str(time.time()) + "_" + str(np.random.rand())
                tag = str(np.random.rand())
                feed = "/tmp/smac_feed" + the_date + ".txt"
                fed = "/tmp/smac_fed" + the_date + ".txt"

                def dummy_function() -> None:
                    for _ in range(int(remaining)):
                        while not Path(feed).is_file() or os.stat(feed).st_size == 0:
                            time.sleep(0.1)
                        time.sleep(0.1)
                        data = np.loadtxt(feed)
                        os.remove(feed)
                        res = objective_function(np.asarray(data, dtype=float))
                        f = open(fed, "w")
                        f.write(str(res))
                        f.close()
                    return

                thread = threading.Thread(target=dummy_function)
                thread.start()
                cs = ConfigurationSpace()
                cs.add_hyperparameters(
                    [UniformFloatHyperparameter(f"x{tag}{i}", 0.0, 1.0, default_value=0.0) for i in range(weakself.dimension)]
                )

                def smac2_obj(pconf: Dict[str, float], seed: int = 0) -> float:  # noqa: ARG001
                    pdata = [pconf[f"x{tag}{i}"] for i in range(len(pconf.keys()))]
                    data = weakself._normalizer.backward(np.asarray(pdata, dtype=float))
                    if Path(fed).is_file():
                        os.remove(fed)
                    np.savetxt(feed, data)
                    while not Path(fed).is_file() or os.stat(fed).st_size == 0:
                        time.sleep(0.1)
                    time.sleep(0.1)
                    f = open(fed, "r")
                    res = float(f.read())
                    f.close()
                    return res

                scenario = Scenario(cs, deterministic=True, n_trials=int(remaining))
                smac = HyperparameterOptimizationFacade(scenario, smac2_obj)
                res = smac.optimize()
                best_x = np.array([res[f"x{tag}{k}"] for k in range(len(res.keys()))], dtype=float)
                best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=float))
                thread.join()
                weakself._num_ask = budget  # type: ignore[assignment]
            elif weakself.method == "CmaFmin2" and weakself.dimension > 1:
                import cma  # type: ignore[import-not-found]

                def cma_objective_function(data: np.ndarray) -> float:
                    if weakself._normalizer is not None and weakself._normalizer.fully_bounded:
                        data = weakself._normalizer.backward(np.asarray(data, dtype=np.float32))
                    return float(objective_function(np.asarray(data, dtype=float)))

                x0 = (
                    0.5 * np.ones(weakself.dimension)
                    if weakself._normalizer is not None and weakself._normalizer.fully_bounded
                    else np.zeros(weakself.dimension)
                )
                num_calls = 0
                while budget - num_calls > 0:
                    options = {"maxfevals": budget - num_calls, "verbose": -9}
                    if weakself._normalizer is not None and weakself._normalizer.fully_bounded:
                        options["bounds"] = [0.0, 1.0]
                    res = cma.fmin(cma_objective_function, x0=x0, sigma0=0.2, options=options, restarts=9)  # type: ignore[attr-defined]
                    x0 = (
                        0.5
                        + np.random.uniform() * np.random.uniform(low=-0.5, high=0.5, size=weakself.dimension)
                        if weakself._normalizer is not None and weakself._normalizer.fully_bounded
                        else np.random.randn(weakself.dimension)
                    )
                    if res[1] < best_res:
                        best_res = float(res[1])
                        best_x = np.asarray(res[0], dtype=float)
                        if weakself._normalizer is not None:
                            best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=np.float32))
                    num_calls += int(res[2])
            else:
                res: Any = scipyoptimize.minimize(
                    objective_function,
                    best_x if not weakself.random_restart else weakself._rng.normal(0.0, 1.0, weakself.dimension),  # type: ignore[attr-defined]
                    method=weakself.method,
                    options=options,
                    tol=0,
                )
                if res.fun < best_res:
                    best_res = float(res.fun)
                    best_x = np.asarray(res.x, dtype=float)
            remaining = budget - weakself._num_ask  # type: ignore[operator]
        assert best_x is not None
        return best_x


class NonObjectOptimizer(base.ConfiguredOptimizer):
    """Wrapper over Scipy optimizer implementations, in standard ask and tell format.
    This is actually an import from scipy-optimize, including Sequential Quadratic Programming,

    Parameters
    ----------
    method: str
        Name of the method to use among:

        - Nelder-Mead
        - COBYLA
        - SQP (or SLSQP): very powerful e.g. in continuous noisy optimization. It is based on
          approximating the objective function by quadratic models.
        - Powell
        - NLOPT* (https://nlopt.readthedocs.io/en/latest/; by default, uses Sbplx, based on Subplex);
            can be NLOPT,
                NLOPT_LN_SBPLX,
                NLOPT_LN_PRAXIS,
                NLOPT_GN_DIRECT,
                NLOPT_GN_DIRECT_L,
                NLOPT_GN_CRS2_LM,
                NLOPT_GN_AGS,
                NLOPT_GN_ISRES,
                NLOPT_GN_ESCH,
                NLOPT_LN_COBYLA,
                NLOPT_LN_BOBYQA,
                NLOPT_LN_NEWUOA_BOUND,
                NLOPT_LN_NELDERMEAD.
    random_restart: bool
        whether to restart at a random point if the optimizer converged but the budget is not entirely
        spent yet (otherwise, restarts from best point)

    Note
    ----
    These optimizers do not support asking several candidates in a row
    """
    recast: bool = True
    no_parallelization: bool = True

    def __init__(self, *, method: str = "Nelder-Mead", random_restart: bool = False) -> None:
        super().__init__(_NonObjectMinimizeBase, locals())


AX: NonObjectOptimizer = NonObjectOptimizer(method="AX").set_name("AX", register=True)
BOBYQA: NonObjectOptimizer = NonObjectOptimizer(method="BOBYQA").set_name("BOBYQA", register=True)
NelderMead: NonObjectOptimizer = NonObjectOptimizer(method="Nelder-Mead").set_name("NelderMead", register=True)
CmaFmin2: NonObjectOptimizer = NonObjectOptimizer(method="CmaFmin2").set_name("CmaFmin2", register=True)
Powell: NonObjectOptimizer = NonObjectOptimizer(method="Powell").set_name("Powell", register=True)
RPowell: NonObjectOptimizer = NonObjectOptimizer(method="Powell", random_restart=True).set_name("RPowell", register=True)
BFGS: NonObjectOptimizer = NonObjectOptimizer(method="BFGS", random_restart=False).set_name("BFGS", register=True)
RBFGS: NonObjectOptimizer = NonObjectOptimizer(method="BFGS", random_restart=True).set_name("RBFGS", register=True)
LBFGSB: NonObjectOptimizer = NonObjectOptimizer(method="L-BFGS-B", random_restart=True).set_name("LBFGSB", register=True)
Cobyla: NonObjectOptimizer = NonObjectOptimizer(method="COBYLA").set_name("Cobyla", register=True)
RCobyla: NonObjectOptimizer = NonObjectOptimizer(method="COBYLA", random_restart=True).set_name("RCobyla", register=True)
SQP: NonObjectOptimizer = NonObjectOptimizer(method="SLSQP").set_name("SQP", register=True)
SLSQP: NonObjectOptimizer = SQP
RSQP: NonObjectOptimizer = NonObjectOptimizer(method="SLSQP", random_restart=True).set_name("RSQP", register=True)
RSLSQP: NonObjectOptimizer = RSQP
NLOPT_LN_SBPLX: NonObjectOptimizer = NonObjectOptimizer(method="NLOPT_LN_SBPLX").set_name("NLOPT_LN_SBPLX", register=True)
NLOPT_LN_PRAXIS: NonObjectOptimizer = NonObjectOptimizer(method="NLOPT_LN_PRAXIS").set_name("NLOPT_LN_PRAXIS", register=True)
NLOPT_GN_DIRECT: NonObjectOptimizer = NonObjectOptimizer(method="NLOPT_GN_DIRECT").set_name("NLOPT_GN_DIRECT", register=True)
NLOPT_GN_DIRECT_L: NonObjectOptimizer = NonObjectOptimizer(method="NLOPT_GN_DIRECT_L").set_name("NLOPT_GN_DIRECT_L", register=True)
NLOPT_GN_CRS2_LM: NonObjectOptimizer = NonObjectOptimizer(method="NLOPT_GN_CRS2_LM").set_name("NLOPT_GN_CRS2_LM", register=True)
NLOPT_GN_AGS: NonObjectOptimizer = NonObjectOptimizer(method="NLOPT_GN_AGS").set_name("NLOPT_GN_AGS", register=True)
NLOPT_GN_ISRES: NonObjectOptimizer = NonObjectOptimizer(method="NLOPT_GN_ISRES").set_name("NLOPT_GN_ISRES", register=True)
NLOPT_GN_ESCH: NonObjectOptimizer = NonObjectOptimizer(method="NLOPT_GN_ESCH").set_name("NLOPT_GN_ESCH", register=True)
NLOPT_LN_COBYLA: NonObjectOptimizer = NonObjectOptimizer(method="NLOPT_LN_COBYLA").set_name("NLOPT_LN_COBYLA", register=True)
NLOPT_LN_BOBYQA: NonObjectOptimizer = NonObjectOptimizer(method="NLOPT_LN_BOBYQA").set_name("NLOPT_LN_BOBYQA", register=True)
NLOPT_LN_NEWUOA_BOUND: NonObjectOptimizer = NonObjectOptimizer(method="NLOPT_LN_NEWUOA_BOUND").set_name("NLOPT_LN_NEWUOA_BOUND", register=True)
NLOPT_LN_NELDERMEAD: NonObjectOptimizer = NonObjectOptimizer(method="NLOPT_LN_NELDERMEAD").set_name("NLOPT_LN_NELDERMEAD", register=True)
SMAC3: NonObjectOptimizer = NonObjectOptimizer(method="SMAC3").set_name("SMAC3", register=True)


class _PymooMinimizeBase(recaster.SequentialRecastOptimizer):

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1, *, algorithm: str) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.algorithm: str = algorithm
        self._no_hypervolume: bool = True
        self._initial_seed: int = -1

    def get_optimization_function(self) -> Callable[[Callable[[np.ndarray], Union[float, np.ndarray]]], None]:
        if self._initial_seed == -1:
            self._initial_seed = self._rng.randint(2 ** 30)  # type: ignore[attr-defined]
        return functools.partial(self._optimization_function, weakref.proxy(self))

    @staticmethod
    def _optimization_function(weakself: Any, objective_function: Callable[[np.ndarray], Union[float, np.ndarray]]) -> None:
        from pymoo import optimize as pymoooptimize  # type: ignore[import-not-found]
        from pymoo.factory import get_algorithm as get_pymoo_algorithm  # type: ignore[import-not-found]

        problem = _create_pymoo_problem(weakself, objective_function)
        if weakself.algorithm == "CMAES":
            from pymoo.algorithms.soo.nonconvex.cmaes import CMAES  # type: ignore[import-not-found]

            algorithm = CMAES(x0=np.random.random(problem.n_var), maxfevals=weakself.budget)
        elif weakself.algorithm == "BIPOP":
            from pymoo.algorithms.soo.nonconvex.cmaes import CMAES  # type: ignore[import-not-found]

            algorithm = CMAES(
                x0=np.random.random(problem.n_var),
                sigma=0.5,
                restarts=2,
                maxfevals=weakself.budget,
                tolfun=1e-06,
                tolx=1e-06,
                restart_from_best=True,
                bipop=True,
            )
        else:
            algorithm = get_pymoo_algorithm(weakself.algorithm)
        pymoooptimize.minimize(problem, algorithm, seed=weakself._initial_seed)
        return None

    def _internal_ask_candidate(self) -> p.Parameter:
        """
        Special version to make sure that num_objectives has been set before
        the proper _internal_ask_candidate, in our parent class, is called.
        """
        if self.num_objectives == 0:
            warnings.warn(
                "with this optimizer, it is more efficient to set num_objectives before the optimization begins",
                errors.NevergradRuntimeWarning,
            )
            return self.parametrization.spawn_child()
        return super()._internal_ask_candidate()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: Union[float, np.ndarray]) -> None:
        """
        Special version to make sure that we the extra initial evaluation which
        we may have done in order to get num_objectives, is discarded.
        Note that this discarding means that the extra point will not make it into
        replay_archive_tell. Correspondingly, because num_objectives will make it into
        the pickle, __setstate__ will never need a dummy ask.
        """
        if self._messaging_thread is None:
            return
        super()._internal_tell_candidate(candidate, loss)

    def _post_loss(self, candidate: p.Parameter, loss: Union[float, np.ndarray]) -> Any:
        """
        Multi-Objective override for this function.
        """
        return candidate.losses


class Pymoo(base.ConfiguredOptimizer):
    """Wrapper over Pymoo optimizer implementations, in standard ask and tell format.
    This is actually an import from Pymoo Optimize.

    Parameters
    ----------
    algorithm: str

        Use "algorithm-name" with following names to access algorithm classes:
        Single-Objective
        -"de"
        -'ga'
        -"brkga"
        -"nelder-mead"
        -"pattern-search"
        -"cmaes"
        Multi-Objective
        -"nsga2"
        Multi-Objective requiring reference directions, points or lines
        -"rnsga2"
        -"nsga3"
        -"unsga3"
        -"rnsga3"
        -"moead"
        -"ctaea"

    Note
    ----
    These optimizers do not support asking several candidates in a row
    """
    recast: bool = True
    no_parallelization: bool = True

    def __init__(self, *, algorithm: str) -> None:
        super().__init__(_PymooMinimizeBase, locals())


class _PymooBatchMinimizeBase(recaster.BatchRecastOptimizer):

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1, *, algorithm: str) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.algorithm: str = algorithm
        self._no_hypervolume: bool = True
        self._initial_seed: int = -1

    def get_optimization_function(self) -> Callable[[Callable[[np.ndarray], Union[float, np.ndarray]]], None]:
        if self._initial_seed == -1:
            self._initial_seed = self._rng.randint(2 ** 30)  # type: ignore[attr-defined]
        return functools.partial(self._optimization_function, weakref.proxy(self))

    @staticmethod
    def _optimization_function(weakself: Any, objective_function: Callable[[np.ndarray], Union[float, np.ndarray]]) -> None:
        from pymoo import optimize as pymoooptimize  # type: ignore[import-not-found]
        from pymoo.factory import get_algorithm as get_pymoo_algorithm  # type: ignore[import-not-found]

        algorithm = get_pymoo_algorithm(weakself.algorithm)
        problem = _create_pymoo_problem(weakself, objective_function, False)
        pymoooptimize.minimize(problem, algorithm, seed=weakself._initial_seed)
        return None

    def _internal_ask_candidate(self) -> p.Parameter:
        """Reads messages from the thread in which the underlying optimization function is running
        New messages are sent as "ask".
        """
        if self.num_objectives == 0:
            warnings.warn(
                "with this optimizer, it is more efficient to set num_objectives before the optimization begins",
                errors.NevergradRuntimeWarning,
            )
            return self.parametrization.spawn_child()
        return super()._internal_ask_candidate()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: Union[float, np.ndarray]) -> None:
        """Returns value for a point which was "asked"
        (none asked point cannot be "tell")
        """
        if self._messaging_thread is None:
            return
        super()._internal_tell_candidate(candidate, loss)

    def _post_loss(self, candidate: p.Parameter, loss: Union[float, np.ndarray]) -> Any:
        """
        Multi-Objective override for this function.
        """
        return candidate.losses


class PymooBatch(base.ConfiguredOptimizer):
    """Wrapper over Pymoo optimizer implementations, in standard ask and tell format.
    This is actually an import from Pymoo Optimize.

    Parameters
    ----------
    algorithm: str

        Use "algorithm-name" with following names to access algorithm classes:
        Single-Objective
        -"de"
        -'ga'
        -"brkga"
        -"nelder-mead"
        -"pattern-search"
        -"cmaes"
        Multi-Objective
        -"nsga2"
        Multi-Objective requiring reference directions, points or lines
        -"rnsga2"
        -"nsga3"
        -"unsga3"
        -"rnsga3"
        -"moead"
        -"ctaea"

    Note
    ----
    These optimizers do not support asking several candidates in a row
    """
    recast: bool = True

    def __init__(self, *, algorithm: str) -> None:
        super().__init__(_PymooBatchMinimizeBase, locals())


def _create_pymoo_problem(
    optimizer: Any, objective_function: Callable[[np.ndarray], Union[float, np.ndarray]], elementwise: bool = True
) -> Any:
    kwargs: Dict[str, Any] = {}
    try:
        from pymoo.core.problem import ElementwiseProblem, Problem  # type: ignore[import-not-found]

        Base = ElementwiseProblem if elementwise else Problem
    except ImportError:
        from pymoo.model.problem import Problem as Base  # type: ignore[import-not-found]
        kwargs = {"elementwise_evaluation": elementwise}

    class _PymooProblem(Base):  # type: ignore[misc]
        def __init__(self, optimizer: Any, objective_function: Callable[[np.ndarray], Union[float, np.ndarray]]) -> None:
            self.objective_function: Callable[[np.ndarray], Union[float, np.ndarray]] = objective_function
            super().__init__(
                n_var=optimizer.dimension,
                n_obj=optimizer.num_objectives,
                n_constr=0,
                xl=-math.pi * 0.5,
                xu=math.pi * 0.5,
                **kwargs,
            )

        def _evaluate(self, X: np.ndarray, out: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
            out["F"] = self.objective_function(np.tan(X))

    return _PymooProblem(optimizer, objective_function)


PymooCMAES: Pymoo = Pymoo(algorithm="CMAES").set_name("PymooCMAES", register=True)
PymooBIPOP: Pymoo = Pymoo(algorithm="BIPOP").set_name("PymooBIPOP", register=True)
PymooNSGA2: Pymoo = Pymoo(algorithm="nsga2").set_name("PymooNSGA2", register=True)
PymooBatchNSGA2: PymooBatch = PymooBatch(algorithm="nsga2").set_name("PymooBatchNSGA2", register=False)
pysot: NonObjectOptimizer = NonObjectOptimizer(method="pysot").set_name("pysot", register=True)
DSbase: NonObjectOptimizer = NonObjectOptimizer(method="DSbase").set_name("DSbase", register=True)
DS3p: NonObjectOptimizer = NonObjectOptimizer(method="DS3p").set_name("DS3p", register=True)
DSsubspace: NonObjectOptimizer = NonObjectOptimizer(method="DSsubspace").set_name("DSsubspace", register=True)
DSproba: NonObjectOptimizer = NonObjectOptimizer(method="DSproba").set_name("DSproba", register=True)