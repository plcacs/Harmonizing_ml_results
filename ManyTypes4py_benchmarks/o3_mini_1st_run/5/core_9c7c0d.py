#!/usr/bin/env python3
import datetime
import warnings
import itertools
import importlib.util
from pathlib import Path
from typing import Union, Callable, Optional, Iterator, Tuple, List, Any

import numpy as np
import pandas as pd
import nevergrad.common.typing as tp
from nevergrad.optimization.utils import SequentialExecutor
from .experiments import registry as registry
from .experiments import Experiment as Experiment
from . import utils


def import_additional_module(filepath: Union[str, Path]) -> None:
    """Imports an additional file at runtime

    Parameter
    ---------
    filepath: str or Path
        the file to import
    """
    filepath = Path(filepath)
    spec = importlib.util.spec_from_file_location(
        'nevergrad.additionalimport.' + filepath.with_suffix('').name, str(filepath)
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)


def save_or_append_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Saves a dataframe to a file in append mode"""
    if path.exists():
        print('Appending to existing file')
        try:
            predf = pd.read_csv(str(path), on_bad_lines='warn')
        except Exception:
            predf = pd.read_csv(str(path))
        df = pd.concat([predf, df], sort=False)
    df.to_csv(path, index=False)


class Moduler:
    """Provides a selector of indices based on the modulo
    moduler(number) will be true iff number = modulo * k + index with k an integer

    Parameters
    ----------
    modulo: int
        modulo for number selection
    index: int
        the congruence of the number for the moduler function to evaluate to True
    total_length: int or None
        total length of the sequence the moduler will be applied on. If provided,
        this allows to compute the length of the modulated sequence.
    """

    def __init__(self, modulo: int, index: int, total_length: Optional[int] = None) -> None:
        assert modulo > 0, 'Modulo must be strictly positive'
        assert index < modulo, 'Index must be strictly smaller than modulo'
        self.modulo: int = modulo
        self.index: int = index
        self.total_length: Optional[int] = total_length

    def split(self, number: int) -> List["Moduler"]:
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
    """Splittable chunk of a benchmark

    Parameters
    ----------
    name: str
        Name of the benchmark
    repetitions: int
        Number of repetitions to perform on the benchmark
    seed: int or None
        A seed for the experiment plan (if seedable)
    cap_index: int or None
        index at which the experiment plan must be stopped (convenient for testing if the experiment
        plan holds 10k experiment, we can select the first cap_index=100 for instance)
    """

    def __init__(self, name: str, repetitions: int = 1, seed: Optional[int] = None, cap_index: Optional[int] = None) -> None:
        self.name: str = name
        self.seed: Optional[int] = seed
        self.cap_index: Optional[int] = None if cap_index is None else max(1, int(cap_index))
        self._moduler: Optional[Moduler] = None
        self.repetitions: int = repetitions
        self.summaries: List[Any] = []
        self._current_experiment: Optional[Experiment] = None
        self._id: str = datetime.datetime.now().strftime('%y-%m-%d_%H%M') + '_' + ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 4))

    @property
    def moduler(self) -> Moduler:
        if self._moduler is None:
            total_length = sum((1 for _ in itertools.islice(registry[self.name](), 0, self.cap_index))) * self.repetitions
            self._moduler = Moduler(1, 0, total_length=total_length)
        return self._moduler

    @property
    def id(self) -> str:
        """Unique ID which can be used to print in a file for instance"""
        return f'{self._id}_i{self.moduler.index}m{self.moduler.modulo}'

    def __iter__(self) -> Iterator[Tuple[int, Experiment]]:
        maker = registry[self.name]
        seeds = (None for _ in range(self.repetitions)) if self.seed is None else range(self.seed, self.seed + self.repetitions)
        generators = [maker() if seed is None else maker(seed=seed) for seed in seeds]
        generators = [itertools.islice(g, 0, self.cap_index) for g in generators]
        enumerated_selection = ((k, s) for k, s in enumerate(itertools.chain.from_iterable(generators)) if self.moduler(k))
        return enumerated_selection

    def split(self, number: int) -> List["BenchmarkChunk"]:
        """Create n BenchmarkChunk which split the experiments of the current BenchmarkChunk

        Parameters
        ----------
        number: int
            The number of sub-chunks to create

        Returns
        -------
        list
            A list of new sub-chunks
        """
        chunks: List[BenchmarkChunk] = []
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

    def compute(self, process_function: Optional[Callable[["BenchmarkChunk", Experiment], None]] = None) -> Any:
        """Run all the experiments and returns the result dataframe.

        Parameters
        ----------
        process_function: Callable or None
            a function to print at the end of each experiment (for custom logging)
        """
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


def _submit_jobs(
    experiment_name: str,
    num_workers: int = 1,
    seed: Optional[int] = None,
    executor: Optional[Any] = None,
    print_function: Optional[Callable[[BenchmarkChunk, Experiment], None]] = None,
    cap_index: Optional[int] = None
) -> List[Any]:
    """Submits a job for computation

    Parameters
    ----------
    experiment_name: str
        name of the experiment plan (must be registered in experiments.registry)
    num_workers: int
        number of workers onto which the jobs will be distributed
    seed: int or None
        a seed for the experiment plan (if seedable)
    executor: Executor-like object or None
        an object such as concurrent.futures.ProcessPoolExecutor for running experiments in parallel
    print_function: Callable or None
        a function to print at the end of each experiment (for custom logging)
    cap_index: int or None
        index at which the experiment plan must be stopped (convenient for testing)
    
    Returns
    -------
    list
        A list of jobs corresponding to each of the workers
    """
    if executor is None:
        if num_workers > 1:
            raise ValueError('An executor must be provided to run multiple jobs in parallel')
        executor = SequentialExecutor()
    jobs: List[Any] = []
    bench = BenchmarkChunk(name=experiment_name, seed=seed, cap_index=cap_index)
    next(registry[experiment_name]())
    for chunk in bench.split(num_workers):
        jobs.append(executor.submit(chunk.compute, print_function))
    return jobs


def compute(
    experiment_name: str,
    num_workers: int = 1,
    seed: Optional[int] = None,
    executor: Optional[Any] = None,
    print_function: Optional[Callable[[BenchmarkChunk, Experiment], None]] = None,
    cap_index: Optional[int] = None
) -> Any:
    """Submits a job for computation

    Parameters
    ----------
    experiment_name: str
        name of the experiment plan (must be registered in experiments.registry)
    num_workers: int
        number of workers onto which the jobs will be distributed
    seed: int or None
        a seed for the experiment plan (if seedable)
    executor: Executor-like object or None
        an object such as concurrent.futures.ProcessPoolExecutor for running experiments in parallel
    print_function: Callable or None
        a function to print at the end of each experiment (for custom logging)
    cap_index: int or None
        index at which the experiment plan must be stopped (convenient for testing)

    Returns
    -------
    Any
        The dataframe summarizing all the experiments (each experiment is a line)
    """
    jobs: List[Any] = _submit_jobs(
        experiment_name=experiment_name,
        num_workers=num_workers,
        seed=seed,
        executor=executor,
        print_function=print_function,
        cap_index=cap_index
    )
    dfs = [j.result() for j in jobs]
    return utils.Selector(pd.concat(dfs))
