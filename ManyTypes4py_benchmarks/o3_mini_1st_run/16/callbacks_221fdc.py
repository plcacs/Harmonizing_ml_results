import json
import time
import warnings
import inspect
import datetime
import logging
from pathlib import Path
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import errors
from nevergrad.parametrization import parameter as p
from nevergrad.parametrization import helpers
from . import base
from typing import Any, Callable, Dict, List

global_logger = logging.getLogger(__name__)


class OptimizationPrinter:
    """Printer to register as callback in an optimizer, for printing
    best point regularly.

    Parameters
    ----------
    print_interval_tells: int
        max number of evaluation before performing another print
    print_interval_seconds: float
        max number of seconds before performing another print
    """

    def __init__(self, print_interval_tells: int = 1, print_interval_seconds: float = 60.0) -> None:
        assert print_interval_tells > 0
        assert print_interval_seconds > 0
        self._print_interval_tells: int = int(print_interval_tells)
        self._print_interval_seconds: float = print_interval_seconds
        self._next_tell: int = self._print_interval_tells
        self._next_time: float = time.time() + print_interval_seconds

    def __call__(self, optimizer: Any, *args: Any, **kwargs: Any) -> None:
        if time.time() >= self._next_time or self._next_tell >= optimizer.num_tell:
            self._next_time = time.time() + self._print_interval_seconds
            self._next_tell = optimizer.num_tell + self._print_interval_tells
            x = optimizer.provide_recommendation()
            print(f'After {optimizer.num_tell}, recommendation is {x}')


class OptimizationLogger:
    """Logger to register as callback in an optimizer, for Logging
    best point regularly.

    Parameters
    ----------
    logger:
        given logger that callback will use to log
    log_level:
        log level that logger will write to
    log_interval_tells: int
        max number of evaluation before performing another log
    log_interval_seconds:
        max number of seconds before performing another log
    """

    def __init__(
        self,
        *,
        logger: Any = global_logger,
        log_level: int = logging.INFO,
        log_interval_tells: int = 1,
        log_interval_seconds: float = 60.0,
    ) -> None:
        assert log_interval_tells > 0
        assert log_interval_seconds > 0
        self._logger: Any = logger
        self._log_level: int = log_level
        self._log_interval_tells: int = int(log_interval_tells)
        self._log_interval_seconds: float = log_interval_seconds
        self._next_tell: int = self._log_interval_tells
        self._next_time: float = time.time() + log_interval_seconds

    def __call__(self, optimizer: Any, *args: Any, **kwargs: Any) -> None:
        if time.time() >= self._next_time or self._next_tell >= optimizer.num_tell:
            self._next_time = time.time() + self._log_interval_seconds
            self._next_tell = optimizer.num_tell + self._log_interval_tells
            if optimizer.num_objectives == 1:
                x = optimizer.provide_recommendation()
                self._logger.log(self._log_level, 'After %s, recommendation is %s', optimizer.num_tell, x)
            else:
                losses = optimizer._hypervolume_pareto.get_min_losses()
                self._logger.log(
                    self._log_level,
                    'After %s, the respective minimum loss for each objective in the pareto front is %s',
                    optimizer.num_tell,
                    losses,
                )


class ParametersLogger:
    """Logs parameter and run information throughout into a file during
    optimization.

    Parameters
    ----------
    filepath: str or pathlib.Path
        the path to dump data to
    append: bool
        whether to append the file (otherwise it replaces it)
    order: int
        order of the internal/model parameters to extract

    Example
    -------

    .. code-block:: python

        logger = ParametersLogger(filepath)
        optimizer.register_callback("tell",  logger)
        optimizer.minimize()
        list_of_dict_of_data = logger.load()

    Note
    ----
    Arrays are converted to lists
    """

    def __init__(self, filepath: Any, append: bool = True, order: int = 1) -> None:
        self._session: str = datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S')
        self._filepath: Path = Path(filepath)
        self._order: int = order
        if self._filepath.exists() and (not append):
            self._filepath.unlink()
        self._filepath.parent.mkdir(exist_ok=True, parents=True)

    def __call__(self, optimizer: Any, candidate: Any, loss: Any) -> None:
        data: Dict[str, Any] = {
            '#parametrization': optimizer.parametrization.name,
            '#optimizer': optimizer.name,
            '#session': self._session,
            '#num-ask': optimizer.num_ask,
            '#num-tell': optimizer.num_tell,
            '#num-tell-not-asked': optimizer.num_tell_not_asked,
            '#uid': candidate.uid,
            '#lineage': candidate.heritage['lineage'],
            '#generation': candidate.generation,
            '#parents_uids': [],
            '#loss': loss,
        }
        if optimizer.num_objectives > 1:
            data.update({f'#losses#{k}': val for k, val in enumerate(candidate.losses)})
            data['#pareto-length'] = len(optimizer.pareto_front())
        if hasattr(optimizer, '_configured_optimizer'):
            configopt = optimizer._configured_optimizer
            if isinstance(configopt, base.ConfiguredOptimizer):
                data.update({'#optimizer#' + x: str(y) for x, y in configopt.config().items()})
        if isinstance(candidate._meta.get('sigma'), float):
            data['#meta-sigma'] = candidate._meta['sigma']
        if candidate.generation > 1:
            data['#parents_uids'] = candidate.parents_uids
        for name, param in helpers.flatten(candidate, with_containers=False, order=1):
            val = param.value
            if isinstance(val, (np.float64, np.int_, np.bool_)):
                val = val.item()
            if inspect.ismethod(val):
                val = repr(val.__self__)
            data[name if name else '0'] = val.tolist() if isinstance(val, np.ndarray) else val
            if isinstance(param, p.Data):
                sigma_val = param.sigma.value
                data[(name if name else '0') + '#sigma'] = sigma_val.tolist() if isinstance(sigma_val, np.ndarray) else sigma_val
        try:
            with self._filepath.open('a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            warnings.warn(f'Failing to json data: {e}')

    def load(self) -> List[Dict[str, Any]]:
        """Loads data from the log file"""
        data: List[Dict[str, Any]] = []
        if self._filepath.exists():
            with self._filepath.open('r') as f:
                for line in f.readlines():
                    data.append(json.loads(line))
        return data

    def load_flattened(self, max_list_elements: int = 24) -> List[Dict[str, Any]]:
        """Loads data from the log file, and splits lists (arrays) into multiple arguments

        Parameters
        ----------
        max_list_elements: int
            Maximum number of elements displayed from the array, each element is given a
            unique id of type list_name#i0_i1_...
        """
        data = self.load()
        flat_data: List[Dict[str, Any]] = []
        for element in data:
            list_keys = {key for key, val in element.items() if isinstance(val, list)}
            flat_data.append({key: val for key, val in element.items() if key not in list_keys})
            for key in list_keys:
                for k, (indices, value) in enumerate(np.ndenumerate(element[key])):
                    if k >= max_list_elements:
                        break
                    flat_data[-1][key + '#' + '_'.join((str(i) for i in indices))] = value
        return flat_data

    def to_hiplot_experiment(self, max_list_elements: int = 24) -> Any:
        """Converts the logs into an hiplot experiment for display.

        Parameters
        ----------
        max_list_elements: int
            maximum number of elements of list/arrays to export (only the first elements are extracted)

        Example
        -------
        .. code-block:: python

            exp = logs.to_hiplot_experiment()
            exp.display(force_full_width=True)

        Note
        ----
        - You can easily change the axes of the XY plot:
          :code:`exp.display_data(hip.Displays.XY).update({'axis_x': '0#0', 'axis_y': '0#1'})`
        - For more context about hiplot, check:

          - blogpost: https://ai.facebook.com/blog/hiplot-high-dimensional-interactive-plots-made-easy/
          - github repo: https://github.com/facebookresearch/hiplot
          - documentation: https://facebookresearch.github.io/hiplot/
        """
        try:
            import hiplot as hip
        except ImportError as e:
            raise ImportError(
                f'{self.__class__.__name__} requires hiplot which is not installed by default (pip install hiplot)'
            ) from e
        exp = hip.Experiment()
        for xp in self.load_flattened(max_list_elements=max_list_elements):
            dp = hip.Datapoint(
                from_uid=xp.get('#parents_uids#0'),
                uid=xp['#uid'],
                values={x: y for x, y in xp.items() if not (x.startswith('#') and ('uid' in x or 'ask' in x))}
            )
            exp.datapoints.append(dp)
        exp.display_data(hip.Displays.XY).update({'axis_x': '#num-tell', 'axis_y': '#loss'})
        exp.display_data(hip.Displays.XY).update({'lines_thickness': 1.0, 'lines_opacity': 1.0})
        return exp


class OptimizerDump:
    """Dumps the optimizer to a pickle file at every call.

    Parameters
    ----------
    filepath: str or Path
        path to the pickle file
    """

    def __init__(self, filepath: Any) -> None:
        self._filepath: Any = filepath

    def __call__(self, opt: Any, *args: Any, **kwargs: Any) -> None:
        opt.dump(self._filepath)


class ProgressBar:
    """Progress bar to register as callback in an optimizer"""

    def __init__(self) -> None:
        self._progress_bar: Any = None
        self._current: int = 0

    def __call__(self, optimizer: Any, *args: Any, **kwargs: Any) -> None:
        if self._progress_bar is None:
            try:
                from tqdm import tqdm
            except ImportError as e:
                raise ImportError(
                    f'{self.__class__.__name__} requires tqdm which is not installed by default (pip install tqdm)'
                ) from e
            self._progress_bar = tqdm()
            self._progress_bar.total = optimizer.budget
            self._progress_bar.update(self._current)
        self._progress_bar.update(1)
        self._current += 1

    def __getstate__(self) -> Dict[str, Any]:
        """Used for pickling (tqdm is not picklable)"""
        state: Dict[str, Any] = dict(self.__dict__)
        state['_progress_bar'] = None
        return state


class EarlyStopping:
    """Callback for stopping the :code:`minimize` method before the budget is
    fully used.

    Parameters
    ----------
    stopping_criterion: func(optimizer) -> bool
        function that takes the current optimizer as input and returns True
        if the minimization must be stopped

    Note
    ----
    This callback must be register on the "ask" method only.

    Example
    -------
    In the following code, the :code:`minimize` method will be stopped at the 4th "ask"

    >>> early_stopping = ng.callbacks.EarlyStopping(lambda opt: opt.num_ask > 3)
    >>> optimizer.register_callback("ask", early_stopping)
    >>> optimizer.minimize(_func, verbosity=2)

    A couple other options (equivalent in case of non-noisy optimization) for stopping
    if the loss is below 12:

    >>> early_stopping = ng.callbacks.EarlyStopping(lambda opt: opt.recommend().loss < 12)
    >>> early_stopping = ng.callbacks.EarlyStopping(lambda opt: opt.current_bests["minimum"].mean < 12)
    """

    def __init__(self, stopping_criterion: Callable[[Any], bool]) -> None:
        self.stopping_criterion: Callable[[Any], bool] = stopping_criterion

    def __call__(self, optimizer: Any, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            raise errors.NevergradRuntimeError('EarlyStopping must be registered on ask method')
        if self.stopping_criterion(optimizer):
            raise errors.NevergradEarlyStopping('Early stopping criterion is reached')

    @classmethod
    def timer(cls, max_duration: float) -> "EarlyStopping":
        """Early stop when max_duration seconds has been reached (from the first ask)"""
        return cls(_DurationCriterion(max_duration))

    @classmethod
    def no_improvement_stopper(cls, tolerance_window: int) -> "EarlyStopping":
        """Early stop when loss didn't reduce during tolerance_window asks"""
        return cls(_LossImprovementToleranceCriterion(tolerance_window))


class _DurationCriterion:
    def __init__(self, max_duration: float) -> None:
        self._start: float = float('inf')
        self._max_duration: float = max_duration

    def __call__(self, optimizer: Any) -> bool:
        if np.isinf(self._start):
            self._start = time.time()
        return time.time() > self._start + self._max_duration


class _LossImprovementToleranceCriterion:
    def __init__(self, tolerance_window: int) -> None:
        self._tolerance_window: int = tolerance_window
        self._best_value: Any = None
        self._tolerance_count: int = 0

    def __call__(self, optimizer: Any) -> bool:
        best_param = optimizer.provide_recommendation()
        if best_param is None or (best_param.loss is None and best_param._losses is None):
            return False
        best_last_losses = best_param.losses
        if self._best_value is None:
            self._best_value = best_last_losses
            return False
        if self._best_value <= best_last_losses:
            self._tolerance_count += 1
        else:
            self._tolerance_count = 0
            self._best_value = best_last_losses
        return self._tolerance_count > self._tolerance_window
