import datetime
import warnings
import itertools
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from nevergrad.common.typing import Callable
from nevergrad.optimization.utils import SequentialExecutor
from .experiments import registry
from .experiments import Experiment
from . import utils

def import_additional_module(filepath: str) -> None:
    filepath = Path(filepath)
    spec = importlib.util.spec_from_file_location('nevergrad.additionalimport.' + filepath.with_suffix('').name, str(filepath))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

def save_or_append_to_csv(df: pd.DataFrame, path: Path) -> None:
    if path.exists():
        print('Appending to existing file')
        try:
            predf = pd.read_csv(str(path), on_bad_lines='warn')
        except:
            predf = pd.read_csv(str(path))
        df = pd.concat([predf, df], sort=False)
    df.to_csv(path, index=False)

class Moduler:
    def __init__(self, modulo: int, index: int, total_length: int = None) -> None:
        assert modulo > 0, 'Modulo must be strictly positive'
        assert index < modulo, 'Index must be strictly smaller than modulo'
        self.modulo = modulo
        self.index = index
        self.total_length = total_length

    def split(self, number: int) -> List['Moduler']:
        return [Moduler(self.modulo * number, self.index + k * self.modulo, self.total_length) for k in range(number)]

    def __len__(self) -> int:
        if self.total_length is None:
            raise RuntimeError('Cannot give an expected length if total_length was not provided')
        return self.total_length // self.modulo + (self.index < self.total_length % self.modulo)

    def __call__(self, index: int) -> bool:
        return index % self.modulo == self.index

    def __repr__(self) -> str:
        return f'Moduler({self.index}, {self.modulo}, total_length={self.total_length})'

class BenchmarkChunk:
    def __init__(self, name: str, repetitions: int = 1, seed: int = None, cap_index: int = None) -> None:
        self.name = name
        self.seed = seed
        self.cap_index = None if cap_index is None else max(1, int(cap_index))
        self._moduler = None
        self.repetitions = repetitions
        self.summaries = []
        self._current_experiment = None
        self._id = datetime.datetime.now().strftime('%y-%m-%d_%H%M') + '_' + ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 4)

    @property
    def moduler(self) -> Moduler:
        if self._moduler is None:
            total_length = sum((1 for _ in itertools.islice(registry[self.name](), 0, self.cap_index))) * self.repetitions
            self._moduler = Moduler(1, 0, total_length=total_length)
        return self._moduler

    @property
    def id(self) -> str:
        return f'{self._id}_i{self.moduler.index}m{self.moduler.modulo}'

    def __iter__(self) -> Iterator:
        maker = registry[self.name]
        seeds = (None for _ in range(self.repetitions)) if self.seed is None else range(self.seed, self.seed + self.repetitions)
        generators = [maker() if seed is None else maker(seed=seed) for seed in seeds]
        generators = [itertools.islice(g, 0, self.cap_index) for g in generators]
        enumerated_selection = ((k, s) for k, s in enumerate(itertools.chain.from_iterable(generators)) if self.moduler(k))
        return enumerated_selection

    def split(self, number: int) -> List['BenchmarkChunk']:
        chunks = []
        for submoduler in self.moduler.split(number):
            chunk = BenchmarkChunk(name=self.name, repetitions=self.repetitions, seed=self.seed, cap_index=self.cap_index)
            chunk._moduler = submoduler
            chunk._id = self._id
            chunks.append(chunk)
        return chunks

    def __repr__(self) -> str:
        return f'BenchmarkChunk({self.name}, {self.repetitions}, {self.seed}) with {self.moduler}'

    def __len__(self) -> int:
        return len(self.moduler)

    def compute(self, process_function: Callable = None) -> 'Selector':
        for local_ind, (index, xp) in enumerate(self):
            if local_ind < len(self.summaries):
                continue
            indstr = f'{index} ({local_ind + 1}/{len(self)} of worker)'
            print(f'Starting {indstr}: {xp}', flush=True)
            if self._current_experiment is None:
                self._current_experiment = xp
            elif xp != self._current_experiment:
                warnings.warn(f'Could not resume unfinished xp: {self._current_experiment}')
                self._current_experiment = xp
            else:
                opt = self._current_experiment._optimizer
                if opt is not None:
                    print(f'Resuming existing experiment from iteration {opt.num_ask}.', flush=True)
            self._current_experiment.run()
            summary = self._current_experiment.get_description()
            if process_function is not None:
                process_function(self, self._current_experiment)
            self.summaries.append(summary)
            self._current_experiment = None
            print(f'Finished {indstr}', flush=True)
        return utils.Selector(data=self.summaries)

def _submit_jobs(experiment_name: str, num_workers: int = 1, seed: int = None, executor: 'Executor' = None, print_function: Callable = None, cap_index: int = None) -> List['Job']:
    if executor is None:
        if num_workers > 1:
            raise ValueError('An executor must be provided to run multiple jobs in parallel')
        executor = SequentialExecutor()
    jobs = []
    bench = BenchmarkChunk(name=experiment_name, seed=seed, cap_index=cap_index)
    next(registry[experiment_name]())
    for chunk in bench.split(num_workers):
        jobs.append(executor.submit(chunk.compute, print_function))
    return jobs

def compute(experiment_name: str, num_workers: int = 1, seed: int = None, executor: 'Executor' = None, print_function: Callable = None, cap_index: int = None) -> pd.DataFrame:
    jobs = _submit_jobs(**locals())
    dfs = [j.result() for j in jobs]
    return utils.Selector(pd.concat(dfs))
