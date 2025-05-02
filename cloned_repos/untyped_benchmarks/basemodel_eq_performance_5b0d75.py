from __future__ import annotations
import dataclasses
import enum
import gc
import itertools
import operator
import sys
import textwrap
import timeit
from collections.abc import Iterable, Sized
from importlib import metadata
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar
if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import axes, figure
import pydantic
PYTHON_VERSION = '.'.join(map(str, sys.version_info))
PYDANTIC_VERSION = metadata.version('pydantic')

class OldImplementationModel(pydantic.BaseModel, frozen=True):

    def __eq__(self, other):
        if isinstance(other, pydantic.BaseModel):
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
            return self_type == other_type and self.__dict__ == other.__dict__ and (self.__pydantic_private__ == other.__pydantic_private__) and (self.__pydantic_extra__ == other.__pydantic_extra__)
        else:
            return NotImplemented

class DictComprehensionEqModel(pydantic.BaseModel, frozen=True):

    def __eq__(self, other):
        if isinstance(other, pydantic.BaseModel):
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
            field_names = type(self).model_fields.keys()
            return self_type == other_type and {k: self.__dict__[k] for k in field_names} == {k: other.__dict__[k] for k in field_names} and (self.__pydantic_private__ == other.__pydantic_private__) and (self.__pydantic_extra__ == other.__pydantic_extra__)
        else:
            return NotImplemented

class ItemGetterEqModel(pydantic.BaseModel, frozen=True):

    def __eq__(self, other):
        if isinstance(other, pydantic.BaseModel):
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
            model_fields = type(self).model_fields.keys()
            getter = operator.itemgetter(*model_fields) if model_fields else lambda _: None
            return self_type == other_type and getter(self.__dict__) == getter(other.__dict__) and (self.__pydantic_private__ == other.__pydantic_private__) and (self.__pydantic_extra__ == other.__pydantic_extra__)
        else:
            return NotImplemented

class ItemGetterEqModelFastPath(pydantic.BaseModel, frozen=True):

    def __eq__(self, other):
        if isinstance(other, pydantic.BaseModel):
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
            if not (self_type == other_type and self.__pydantic_private__ == other.__pydantic_private__ and (self.__pydantic_extra__ == other.__pydantic_extra__)):
                return False
            if self.__dict__ == other.__dict__:
                return True
            else:
                model_fields = type(self).model_fields.keys()
                getter = operator.itemgetter(*model_fields) if model_fields else lambda _: None
                return getter(self.__dict__) == getter(other.__dict__)
        else:
            return NotImplemented
K = TypeVar('K')
V = TypeVar('V')

class _SentinelType(enum.Enum):
    SENTINEL = enum.auto()
_SENTINEL = _SentinelType.SENTINEL

@dataclasses.dataclass
class _SafeGetItemProxy(Generic[K, V]):
    """Wrapper redirecting `__getitem__` to `get` and a sentinel value

    This makes is safe to use in `operator.itemgetter` when some keys may be missing
    """

    def __getitem__(self, key, /):
        return self.wrapped.get(key, _SENTINEL)

    def __contains__(self, key, /):
        return self.wrapped.__contains__(key)

class SafeItemGetterEqModelFastPath(pydantic.BaseModel, frozen=True):

    def __eq__(self, other):
        if isinstance(other, pydantic.BaseModel):
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
            if not (self_type == other_type and self.__pydantic_private__ == other.__pydantic_private__ and (self.__pydantic_extra__ == other.__pydantic_extra__)):
                return False
            if self.__dict__ == other.__dict__:
                return True
            else:
                model_fields = type(self).model_fields.keys()
                getter = operator.itemgetter(*model_fields) if model_fields else lambda _: None
                return getter(_SafeGetItemProxy(self.__dict__)) == getter(_SafeGetItemProxy(other.__dict__))
        else:
            return NotImplemented

class ItemGetterEqModelFastPathFallback(pydantic.BaseModel, frozen=True):

    def __eq__(self, other):
        if isinstance(other, pydantic.BaseModel):
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
            if not (self_type == other_type and self.__pydantic_private__ == other.__pydantic_private__ and (self.__pydantic_extra__ == other.__pydantic_extra__)):
                return False
            if self.__dict__ == other.__dict__:
                return True
            else:
                model_fields = type(self).model_fields.keys()
                getter = operator.itemgetter(*model_fields) if model_fields else lambda _: None
                try:
                    return getter(self.__dict__) == getter(other.__dict__)
                except KeyError:
                    return getter(_SafeGetItemProxy(self.__dict__)) == getter(_SafeGetItemProxy(other.__dict__))
        else:
            return NotImplemented
IMPLEMENTATIONS = {'itemgetter': ItemGetterEqModel, 'itemgetter+fastpath': ItemGetterEqModelFastPath, 'itemgetter+fastpath+safe-fallback': ItemGetterEqModelFastPathFallback}

def plot_all_benchmark(bases, sizes):
    import matplotlib.pyplot as plt
    n_rows, n_cols = (len(BENCHMARKS), 2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    for row, (name, benchmark) in enumerate(BENCHMARKS.items()):
        for col, mimic_cached_property in enumerate([False, True]):
            plot_benchmark(f'{name}, mimic_cached_property={mimic_cached_property!r}', benchmark, bases=bases, sizes=sizes, mimic_cached_property=mimic_cached_property, ax=axes[row, col])
    for ax in axes.ravel():
        ax.legend()
    fig.suptitle(f'python {PYTHON_VERSION}, pydantic {PYDANTIC_VERSION}')
    return fig

def plot_benchmark(title, benchmark, bases, sizes, mimic_cached_property, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np
    ax = ax or plt.gca()
    arr_sizes = np.asarray(sizes)
    baseline = benchmark(title=f'{title}, baseline', base=OldImplementationModel, sizes=sizes, mimic_cached_property=mimic_cached_property)
    ax.plot(sizes, baseline / baseline, label='baseline')
    for name, base in bases.items():
        times = benchmark(title=f'{title}, {name}', base=base, sizes=sizes, mimic_cached_property=mimic_cached_property)
        mask_valid = ~np.isnan(times)
        ax.plot(arr_sizes[mask_valid], times[mask_valid] / baseline[mask_valid], label=name)
    ax.set_title(title)
    ax.set_xlabel('Number of pydantic fields')
    ax.set_ylabel('Average time relative to baseline')
    return ax

class SizedIterable(Sized, Iterable):
    pass

def run_benchmark_nodiff(title, base, sizes, mimic_cached_property, n_execution=10000, n_repeat=5):
    setup = textwrap.dedent('\n        import pydantic\n\n        Model = pydantic.create_model(\n            "Model",\n            __base__=Base,\n            **{f\'x{i}\': (int, i) for i in range(%(size)d)}\n        )\n        left = Model()\n        right = Model()\n        ')
    if mimic_cached_property:
        setup += textwrap.dedent("\n            object.__setattr__(left, 'cache', None)\n            object.__setattr__(right, 'cache', -1)\n            ")
    statement = 'left == right'
    namespace = {'Base': base}
    return run_benchmark(title, setup=setup, statement=statement, n_execution=n_execution, n_repeat=n_repeat, globals=namespace, params={'size': sizes})

def run_benchmark_first_diff(title, base, sizes, mimic_cached_property, n_execution=10000, n_repeat=5):
    setup = textwrap.dedent('\n        import pydantic\n\n        Model = pydantic.create_model(\n            "Model",\n            __base__=Base,\n            **{f\'x{i}\': (int, i) for i in range(%(size)d)}\n        )\n        left = Model()\n        right = Model(f0=-1) if %(size)d > 0 else Model()\n        ')
    if mimic_cached_property:
        setup += textwrap.dedent("\n            object.__setattr__(left, 'cache', None)\n            object.__setattr__(right, 'cache', -1)\n            ")
    statement = 'left == right'
    namespace = {'Base': base}
    return run_benchmark(title, setup=setup, statement=statement, n_execution=n_execution, n_repeat=n_repeat, globals=namespace, params={'size': sizes})

def run_benchmark_last_diff(title, base, sizes, mimic_cached_property, n_execution=10000, n_repeat=5):
    setup = textwrap.dedent('\n        import pydantic\n\n        Model = pydantic.create_model(\n            "Model",\n            __base__=Base,\n            # shift the range() so that there is a field named size\n            **{f\'x{i}\': (int, i) for i in range(1, %(size)d + 1)}\n        )\n        left = Model()\n        right = Model(f%(size)d=-1) if %(size)d > 0 else Model()\n        ')
    if mimic_cached_property:
        setup += textwrap.dedent("\n            object.__setattr__(left, 'cache', None)\n            object.__setattr__(right, 'cache', -1)\n            ")
    statement = 'left == right'
    namespace = {'Base': base}
    return run_benchmark(title, setup=setup, statement=statement, n_execution=n_execution, n_repeat=n_repeat, globals=namespace, params={'size': sizes})

def run_benchmark_random_unequal(title, base, sizes, mimic_cached_property, n_samples=100, n_execution=1000, n_repeat=5):
    import numpy as np
    setup = textwrap.dedent('\n        import pydantic\n\n        Model = pydantic.create_model(\n            "Model",\n            __base__=Base,\n            **{f\'x{i}\': (int, i) for i in range(%(size)d)}\n        )\n        left = Model()\n        right = Model(f%(field)d=-1)\n        ')
    if mimic_cached_property:
        setup += textwrap.dedent("\n            object.__setattr__(left, 'cache', None)\n            object.__setattr__(right, 'cache', -1)\n            ")
    statement = 'left == right'
    namespace = {'Base': base}
    arr_sizes = np.fromiter(sizes, dtype=int)
    mask_valid_sizes = arr_sizes > 0
    arr_valid_sizes = arr_sizes[mask_valid_sizes]
    rng = np.random.default_rng()
    arr_fields = rng.integers(arr_valid_sizes, size=(n_samples, *arr_valid_sizes.shape))
    arr_size_broadcast, _ = np.meshgrid(arr_valid_sizes, arr_fields[:, 0])
    results = run_benchmark(title, setup=setup, statement=statement, n_execution=n_execution, n_repeat=n_repeat, globals=namespace, params={'size': arr_size_broadcast.ravel(), 'field': arr_fields.ravel()})
    times = np.empty(arr_sizes.shape, dtype=float)
    times[~mask_valid_sizes] = np.nan
    times[mask_valid_sizes] = results.reshape((n_samples, *arr_valid_sizes.shape)).mean(axis=0)
    return times
BENCHMARKS = {'All field equals': run_benchmark_nodiff, 'First field unequal': run_benchmark_first_diff, 'Last field unequal': run_benchmark_last_diff, 'Random unequal field': run_benchmark_random_unequal}

def run_benchmark(title, setup, statement, n_execution=10000, n_repeat=5, globals=None, progress_bar=True, params=None):
    import numpy as np
    import tqdm.auto as tqdm
    namespace = globals or {}
    if not params:
        length = 1
        packed_params = [()]
    else:
        length = len(next(iter(params.values())))
        param_pairs = [zip(itertools.repeat(name), value) for name, value in params.items()]
        packed_params = zip(*param_pairs)
    times = [min(timeit.Timer(setup=setup % dict(local_params), stmt=statement, globals=namespace).repeat(repeat=n_repeat, number=n_execution)) / n_execution for local_params in tqdm.tqdm(packed_params, desc=title, total=length, disable=not progress_bar)]
    gc.collect()
    return np.asarray(times, dtype=float)
if __name__ == '__main__':
    import argparse
    import pathlib
    try:
        import matplotlib
        import numpy
        import tqdm
    except ImportError as err:
        raise ImportError('This benchmark additionally depends on numpy, matplotlib and tqdm. Install those in your environment to run the benchmark.') from err
    parser = argparse.ArgumentParser(description='Test the performance of various BaseModel.__eq__ implementations fixing GH-7444.')
    parser.add_argument('-o', '--output-path', default=None, type=pathlib.Path, help='Output directory or file in which to save the benchmark results. If a directory is passed, a default filename is used.')
    parser.add_argument('--min-n-fields', type=int, default=0, help='Test the performance of BaseModel.__eq__ on models with at least this number of fields. Defaults to 0.')
    parser.add_argument('--max-n-fields', type=int, default=100, help='Test the performance of BaseModel.__eq__ on models with up to this number of fields. Defaults to 100.')
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    sizes = list(range(args.min_n_fields, args.max_n_fields))
    fig = plot_all_benchmark(IMPLEMENTATIONS, sizes=sizes)
    plt.tight_layout()
    if args.output_path is None:
        plt.show()
    else:
        if args.output_path.suffix:
            filepath = args.output_path
        else:
            filepath = args.output_path / f'eq-benchmark_python-{PYTHON_VERSION.replace('.', '-')}.png'
        fig.savefig(filepath, dpi=200, facecolor='white', transparent=False)
        print(f'wrote {filepath!s}', file=sys.stderr)