import sys
import time
import random
import numbers
import warnings
import traceback
from nevergrad.common.typing import Typing as ngtp
from nevergrad.common.typing import Typing as tp
import numpy as np
from nevergrad.parametrization import parameter as p
from nevergrad.common import decorators
from nevergrad.common import errors
try:
    from ..functions.rl.agents import torch
except ModuleNotFoundError:
    pass
from ..functions import base as fbase
from ..optimization import base as obase
from ..optimization.optimizerlib import registry as optimizer_registry
from . import execution
registry = decorators.Registry()

def _assert_singleobjective_callback(optimizer: tp.Any, candidate: tp.Any, loss: tp.Any) -> None:
    if optimizer.num_tell <= 1 and (not isinstance(loss, numbers.Number)):
        raise TypeError(f"Cannot process loss {loss} of type {type(loss)}.\nFor multiobjective functions, did you forget to specify 'func.multiobjective_upper_bounds'?")

class OptimizerSettings:
    def __init__(self, optimizer: tp.Union[str, tp.Any], budget: int, num_workers: int = 1, batch_mode: bool = True) -> None:
        self._setting_names: tp.List[str] = [x for x in locals() if x != 'self']
        if isinstance(optimizer, str):
            assert optimizer in optimizer_registry, f'{optimizer} is not registered'
        self.optimizer = optimizer
        self.budget = budget
        self.num_workers = num_workers
        self.executor = execution.MockedTimedExecutor(batch_mode)

    @property
    def name(self) -> str:
        try:
            try:
                return self.optimizer.name
            except:
                return self.optimizer.__name__
        except:
            return self.optimizer if isinstance(self.optimizer, str) else repr(self.optimizer)

    @property
    def batch_mode(self) -> bool:
        return self.executor.batch_mode

    def __repr__(self) -> str:
        return f'Experiment: {self.name}<budget={self.budget}, num_workers={self.num_workers}, batch_mode={self.batch_mode}>'

    def _get_factory(self) -> tp.Any:
        return optimizer_registry[self.optimizer] if isinstance(self.optimizer, str) else self.optimizer

    @property
    def is_incoherent(self) -> bool:
        return self._get_factory().no_parallelization and bool(self.num_workers > 1)

    def instantiate(self, parametrization: tp.Any) -> tp.Any:
        return self._get_factory()(parametrization=parametrization, budget=self.budget, num_workers=self.num_workers)

    def get_description(self) -> tp.Dict[str, tp.Any]:
        descr = {x: getattr(self, x) for x in self._setting_names if x != 'optimizer'}
        descr['optimizer_name'] = self.name
        return descr

    def __eq__(self, other: tp.Any) -> bool:
        if isinstance(other, self.__class__):
            for attr in self._setting_names:
                x, y = (getattr(settings, attr) for settings in [self, other])
                if x != y:
                    return False
            return True
        return False

def create_seed_generator(seed: tp.Union[int, None]) -> tp.Generator[tp.Union[int, None], None, None]:
    generator = None if seed is None else np.random.RandomState(seed=seed)
    while True:
        yield (None if generator is None else generator.randint(2 ** 32, dtype=np.uint32))

class Experiment:
    def __init__(self, function: fbase.ExperimentFunction, optimizer: tp.Any, budget: int, num_workers: int = 1, batch_mode: bool = True, seed: tp.Union[int, None], constraint_violation: tp.Any = None, penalize_violation_at_test: bool = True, suggestions: tp.Any = None) -> None:
        self.penalize_violation_at_test = penalize_violation_at_test
        self.suggestions = suggestions
        assert isinstance(function, fbase.ExperimentFunction), 'All experiment functions should derive from ng.functions.ExperimentFunction'
        assert function.dimension, 'Nothing to optimize'
        self.function = function
        self.constraint_violation = constraint_violation
        self.seed = seed
        self.optimsettings = OptimizerSettings(optimizer=optimizer, num_workers=num_workers, budget=budget, batch_mode=batch_mode)
        self.result = {'loss': np.nan, 'elapsed_budget': np.nan, 'elapsed_time': np.nan, 'error': ''}
        self._optimizer = None
        self.function.parametrization.random_state

    def __repr__(self) -> str:
        return f'Experiment: {self.optimsettings} (dim={self.function.dimension}, param={self.function.parametrization}) on {self.function} with seed {self.seed}'

    @property
    def is_incoherent(self) -> bool:
        return self.optimsettings.is_incoherent

    def run(self) -> tp.Dict[str, tp.Any]:
        try:
            self._run_with_error()
        except (errors.ExperimentFunctionCopyError, errors.UnsupportedExperiment) as ex:
            raise ex
        except Exception as e:
            self.result['error'] = e.__class__.__name__
            print(f'Error when applying {self}:', file=sys.stderr)
            traceback.print_exc()
            print('\n', file=sys.stderr)
        return self.get_description()

    def _log_results(self, pfunc: tp.Any, t0: float, num_calls: int) -> None:
        self.result['elapsed_time'] = time.time() - t0
        self.result['pseudotime'] = self.optimsettings.executor.time
        opt = self._optimizer
        assert opt is not None
        self.result['num_objectives'] = opt.num_objectives
        self.result['loss'] = pfunc.evaluation_function(*opt.pareto_front())
        if self.constraint_violation and np.max([f(opt.recommend().value) for f in self.constraint_violation]) > 0 or (len(self.function.parametrization._constraint_checkers) > 0 and (not opt.recommend().satisfies_constraints(pfunc.parametrization))):
            if self.penalize_violation_at_test:
                self.result['loss'] += 1000000000.0
        self.result['elapsed_budget'] = num_calls
        if num_calls > self.optimsettings.budget:
            raise RuntimeError(f'Too much elapsed budget {num_calls} for {self.optimsettings.name} on {self.function}')
        self.result.update({f'info/{x}': y for x, y in opt._info().items()})

    def _run_with_error(self, callbacks: tp.Any = None) -> None:
        if self.seed is not None and self._optimizer is None:
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
        pfunc = self.function.copy()
        assert len(pfunc.parametrization._constraint_checkers) == len(self.function.parametrization._constraint_checkers)
        if self._optimizer is None:
            self._optimizer = self.optimsettings.instantiate(parametrization=pfunc.parametrization)
            if pfunc.multiobjective_upper_bounds is not None:
                self._optimizer.tell(p.MultiobjectiveReference(), pfunc.multiobjective_upper_bounds)
            else:
                self._optimizer.register_callback('tell', _assert_singleobjective_callback)
        if callbacks is not None:
            for name, func in callbacks.items():
                self._optimizer.register_callback(name, func)
        assert self._optimizer.budget is not None, 'A budget must be provided'
        t0 = time.time()
        executor = self.optimsettings.executor
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=errors.InefficientSettingsWarning)
            try:
                if self.suggestions is not None:
                    for s in self.suggestions:
                        self._optimizer.suggest(s)
                obase.Optimizer.minimize(self._optimizer, pfunc, batch_mode=executor.batch_mode, executor=executor, constraint_violation=self.constraint_violation, max_time=3600 * 24 * 2.5)
            except Exception as e:
                self._log_results(pfunc, t0, self._optimizer.num_ask)
                raise e
        self._log_results(pfunc, t0, self._optimizer.num_ask)

    def get_description(self) -> tp.Dict[str, tp.Any]:
        summary = dict(self.result, seed=-1 if self.seed is None else self.seed)
        summary.update(self.function.descriptors)
        summary.update(self.optimsettings.get_description())
        return summary

    def __eq__(self, other: tp.Any) -> bool:
        if not isinstance(other, Experiment):
            return False
        same_seed = other.seed is None if self.seed is None else other.seed == self.seed
        return same_seed and self.function.equivalent_to(other.function) and (self.optimsettings == other.optimsettings)
