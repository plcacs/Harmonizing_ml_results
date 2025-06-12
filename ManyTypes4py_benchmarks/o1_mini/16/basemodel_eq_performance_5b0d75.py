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
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, Dict, List, Tuple, Optional, Union, Iterator
if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import axes, figure
import pydantic
from pydantic import BaseModel
from pydantic.fields import ModelField
import matplotlib.pyplot as plt
import numpy as np
import tqdm.auto as tqdm

PYTHON_VERSION: str = '.'.join(map(str, sys.version_info))
PYDANTIC_VERSION: str = metadata.version('pydantic')


class OldImplementationModel(pydantic.BaseModel):
    class Config:
        frozen = True

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, pydantic.BaseModel):
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
            return (
                self_type == other_type and
                self.__dict__ == other.__dict__ and
                self.__pydantic_private__ == other.__pydantic_private__ and
                self.__pydantic_extra__ == other.__pydantic_extra__
            )
        else:
            return NotImplemented


class DictComprehensionEqModel(pydantic.BaseModel):
    class Config:
        frozen = True

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, pydantic.BaseModel):
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
            field_names: Iterable[str] = type(self).model_fields.keys()
            return (
                self_type == other_type and
                {k: self.__dict__[k] for k in field_names} == {k: other.__dict__[k] for k in field_names} and
                self.__pydantic_private__ == other.__pydantic_private__ and
                self.__pydantic_extra__ == other.__pydantic_extra__
            )
        else:
            return NotImplemented


class ItemGetterEqModel(pydantic.BaseModel):
    class Config:
        frozen = True

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, pydantic.BaseModel):
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
            model_fields: Iterable[str] = type(self).model_fields.keys()
            getter: Callable[[Dict[str, Any]], Tuple[Any, ...]] = operator.itemgetter(*model_fields) if model_fields else lambda _: ()
            return (
                self_type == other_type and
                getter(self.__dict__) == getter(other.__dict__) and
                self.__pydantic_private__ == other.__pydantic_private__ and
                self.__pydantic_extra__ == other.__pydantic_extra__
            )
        else:
            return NotImplemented


class ItemGetterEqModelFastPath(pydantic.BaseModel):
    class Config:
        frozen = True

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, pydantic.BaseModel):
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
            if not (
                self_type == other_type and
                self.__pydantic_private__ == other.__pydantic_private__ and
                self.__pydantic_extra__ == other.__pydantic_extra__
            ):
                return False
            if self.__dict__ == other.__dict__:
                return True
            else:
                model_fields: Iterable[str] = type(self).model_fields.keys()
                getter: Callable[[Dict[str, Any]], Tuple[Any, ...]] = operator.itemgetter(*model_fields) if model_fields else lambda _: ()
                return getter(self.__dict__) == getter(other.__dict__)
        else:
            return NotImplemented


K = TypeVar('K')
V = TypeVar('V')


class _SentinelType(enum.Enum):
    SENTINEL = enum.auto()


_SENTINEL: _SentinelType = _SentinelType.SENTINEL


@dataclasses.dataclass
class _SafeGetItemProxy(Generic[K, V]):
    wrapped: Dict[K, V]

    def __getitem__(self, key: K) -> Union[V, _SentinelType]:
        return self.wrapped.get(key, _SENTinel)

    def __contains__(self, key: K) -> bool:
        return key in self.wrapped


class SafeItemGetterEqModelFastPath(pydantic.BaseModel):
    class Config:
        frozen = True

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, pydantic.BaseModel):
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
            if not (
                self_type == other_type and
                self.__pydantic_private__ == other.__pydantic_private__ and
                self.__pydantic_extra__ == other.__pydantic_extra__
            ):
                return False
            if self.__dict__ == other.__dict__:
                return True
            else:
                model_fields: Iterable[str] = type(self).model_fields.keys()
                getter: Callable[[Dict[str, Any]], Tuple[Any, ...]] = operator.itemgetter(*model_fields) if model_fields else lambda _: ()
                return getter(_SafeGetItemProxy(self.__dict__)) == getter(_SafeGetItemProxy(other.__dict__))
        else:
            return NotImplemented


class ItemGetterEqModelFastPathFallback(pydantic.BaseModel):
    class Config:
        frozen = True

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, pydantic.BaseModel):
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
            if not (
                self_type == other_type and
                self.__pydantic_private__ == other.__pydantic_private__ and
                self.__pydantic_extra__ == other.__pydantic_extra__
            ):
                return False
            if self.__dict__ == other.__dict__:
                return True
            else:
                model_fields: Iterable[str] = type(self).model_fields.keys()
                getter: Callable[[Dict[str, Any]], Tuple[Any, ...]] = operator.itemgetter(*model_fields) if model_fields else lambda _: ()
                try:
                    return getter(self.__dict__) == getter(other.__dict__)
                except KeyError:
                    return getter(_SafeGetItemProxy(self.__dict__)) == getter(_SafeGetItemProxy(other.__dict__))
        else:
            return NotImplemented


IMPLEMENTATIONS: Dict[str, type[pydantic.BaseModel]] = {
    'itemgetter': ItemGetterEqModel,
    'itemgetter+fastpath': ItemGetterEqModelFastPath,
    'itemgetter+fastpath+safe-fallback': ItemGetterEqModelFastPathFallback
}


def plot_all_benchmark(
    bases: Dict[str, type[pydantic.BaseModel]],
    sizes: List[int]
) -> figure.Figure:
    import matplotlib.pyplot as plt
    n_rows: int = len(BENCHMARKS)
    n_cols: int = 2
    fig: figure.Figure
    axes: np.ndarray
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    for row, (name, benchmark) in enumerate(BENCHMARKS.items()):
        for col, mimic_cached_property in enumerate([False, True]):
            plot_benchmark(
                title=f'{name}, mimic_cached_property={mimic_cached_property!r}',
                benchmark=benchmark,
                bases=bases,
                sizes=sizes,
                mimic_cached_property=mimic_cached_property,
                ax=axes[row, col]
            )
    for ax in axes.ravel():
        ax.legend()
    fig.suptitle(f'python {PYTHON_VERSION}, pydantic {PYDANTIC_VERSION}')
    return fig


def plot_benchmark(
    title: str,
    benchmark: Callable[..., np.ndarray],
    bases: Dict[str, type[pydantic.BaseModel]],
    sizes: List[int],
    mimic_cached_property: bool,
    ax: Optional[axes.Axes] = None
) -> axes.Axes:
    import matplotlib.pyplot as plt
    import numpy as np
    ax = ax or plt.gca()
    arr_sizes: np.ndarray = np.asarray(sizes)
    baseline: np.ndarray = benchmark(
        title=f'{title}, baseline',
        base=OldImplementationModel,
        sizes=sizes,
        mimic_cached_property=mimic_cached_property
    )
    ax.plot(sizes, baseline / baseline, label='baseline')
    for name, base in bases.items():
        times: np.ndarray = benchmark(
            title=f'{title}, {name}',
            base=base,
            sizes=sizes,
            mimic_cached_property=mimic_cached_property
        )
        mask_valid: np.ndarray = ~np.isnan(times)
        ax.plot(arr_sizes[mask_valid], times[mask_valid] / baseline[mask_valid], label=name)
    ax.set_title(title)
    ax.set_xlabel('Number of pydantic fields')
    ax.set_ylabel('Average time relative to baseline')
    return ax


class SizedIterable(Sized, Iterable):
    pass


def run_benchmark_nodiff(
    title: str,
    base: type[pydantic.BaseModel],
    sizes: List[int],
    mimic_cached_property: bool,
    n_execution: int = 10000,
    n_repeat: int = 5
) -> np.ndarray:
    setup: str = textwrap.dedent("""
        import pydantic

        Model = pydantic.create_model(
            "Model",
            __base__=Base,
            **{f'x{i}': (int, i) for i in range(%(size)d)}
        )
        left = Model()
        right = Model()
    """) % {}
    setup = textwrap.dedent("""
        import pydantic

        Model = pydantic.create_model(
            "Model",
            __base__=Base,
            **{f'x{i}': (int, i) for i in range(%(size)d)}
        )
        left = Model()
        right = Model()
    """) % {'size': 0}  # Placeholder, will be formatted later

    setup = textwrap.dedent("""
        import pydantic

        Model = pydantic.create_model(
            "Model",
            __base__=Base,
            **{f'x{i}': (int, i) for i in range(%(size)d)}
        )
        left = Model()
        right = Model()
    """) % {'size': 0}  # Update later

    setup = textwrap.dedent("""
        import pydantic

        Model = pydantic.create_model(
            "Model",
            __base__=Base,
            **{f'x{i}': (int, i) for i in range(%(size)d)}
        )
        left = Model()
        right = Model()
    """)
    setup = setup % {'size': 0}  # Placeholder

    # The original code updates setup with size from params
    setup_template: str = textwrap.dedent("""
        import pydantic

        Model = pydantic.create_model(
            "Model",
            __base__=Base,
            **{{f'x{{i}}': (int, i) for i in range(%(size)d)}}
        )
        left = Model()
        right = Model()
    """)

    if mimic_cached_property:
        setup_template += textwrap.dedent("""
            object.__setattr__(left, 'cache', None)
            object.__setattr__(right, 'cache', -1)
        """)

    statement: str = 'left == right'
    namespace: Dict[str, Any] = {'Base': base}
    return run_benchmark(
        title=title,
        setup=setup_template,
        statement=statement,
        n_execution=n_execution,
        n_repeat=n_repeat,
        globals=namespace,
        params={'size': sizes}
    )


def run_benchmark_first_diff(
    title: str,
    base: type[pydantic.BaseModel],
    sizes: List[int],
    mimic_cached_property: bool,
    n_execution: int = 10000,
    n_repeat: int = 5
) -> np.ndarray:
    setup_template: str = textwrap.dedent("""
        import pydantic

        Model = pydantic.create_model(
            "Model",
            __base__=Base,
            **{{f'x{{i}}': (int, i) for i in range(%(size)d)}}
        )
        left = Model()
        right = Model(f0=-1) if %(size)d > 0 else Model()
    """)

    if mimic_cached_property:
        setup_template += textwrap.dedent("""
            object.__setattr__(left, 'cache', None)
            object.__setattr__(right, 'cache', -1)
        """)

    statement: str = 'left == right'
    namespace: Dict[str, Any] = {'Base': base}
    return run_benchmark(
        title=title,
        setup=setup_template,
        statement=statement,
        n_execution=n_execution,
        n_repeat=n_repeat,
        globals=namespace,
        params={'size': sizes}
    )


def run_benchmark_last_diff(
    title: str,
    base: type[pydantic.BaseModel],
    sizes: List[int],
    mimic_cached_property: bool,
    n_execution: int = 10000,
    n_repeat: int = 5
) -> np.ndarray:
    setup_template: str = textwrap.dedent("""
        import pydantic

        Model = pydantic.create_model(
            "Model",
            __base__=Base,
            # shift the range() so that there is a field named size
            **{{f'x{{i}}': (int, i) for i in range(1, %(size)d + 1)}}
        )
        left = Model()
        right = Model(f%(size)d=-1) if %(size)d > 0 else Model()
    """)

    if mimic_cached_property:
        setup_template += textwrap.dedent("""
            object.__setattr__(left, 'cache', None)
            object.__setattr__(right, 'cache', -1)
        """)

    statement: str = 'left == right'
    namespace: Dict[str, Any] = {'Base': base}
    return run_benchmark(
        title=title,
        setup=setup_template,
        statement=statement,
        n_execution=n_execution,
        n_repeat=n_repeat,
        globals=namespace,
        params={'size': sizes}
    )


def run_benchmark_random_unequal(
    title: str,
    base: type[pydantic.BaseModel],
    sizes: List[int],
    mimic_cached_property: bool,
    n_samples: int = 100,
    n_execution: int = 1000,
    n_repeat: int = 5
) -> np.ndarray:
    setup_template: str = textwrap.dedent("""
        import pydantic

        Model = pydantic.create_model(
            "Model",
            __base__=Base,
            **{{f'x{{i}}': (int, i) for i in range(%(size)d)}}
        )
        left = Model()
        right = Model(f%(field)d=-1)
    """)

    if mimic_cached_property:
        setup_template += textwrap.dedent("""
            object.__setattr__(left, 'cache', None)
            object.__setattr__(right, 'cache', -1)
        """)

    statement: str = 'left == right'
    namespace: Dict[str, Any] = {'Base': base}

    arr_sizes: np.ndarray = np.fromiter(sizes, dtype=int)
    mask_valid_sizes: np.ndarray = arr_sizes > 0
    arr_valid_sizes: np.ndarray = arr_sizes[mask_valid_sizes]
    rng = np.random.default_rng()
    arr_fields: np.ndarray = rng.integers(
        low=0, high=arr_valid_sizes.max(), size=(n_samples, arr_valid_sizes.size)
    )

    packed_sizes: List[int] = arr_sizes.tolist()
    packed_fields: List[int] = arr_fields.ravel().tolist()

    results: np.ndarray = run_benchmark(
        title=title,
        setup=setup_template,
        statement=statement,
        n_execution=n_execution,
        n_repeat=n_repeat,
        globals=namespace,
        params={'size': packed_sizes, 'field': packed_fields}
    )

    times: np.ndarray = np.empty_like(arr_sizes, dtype=float)
    times[~mask_valid_sizes] = np.nan
    if arr_valid_sizes.size > 0:
        reshaped_results: np.ndarray = results.reshape((n_samples, arr_valid_sizes.size))
        times[mask_valid_sizes] = reshaped_results.mean(axis=0)
    return times


BENCHMARKS: Dict[str, Callable[..., np.ndarray]] = {
    'All field equals': run_benchmark_nodiff,
    'First field unequal': run_benchmark_first_diff,
    'Last field unequal': run_benchmark_last_diff,
    'Random unequal field': run_benchmark_random_unequal
}


def run_benchmark(
    title: str,
    setup: str,
    statement: str,
    n_execution: int = 10000,
    n_repeat: int = 5,
    globals: Optional[Dict[str, Any]] = None,
    progress_bar: bool = True,
    params: Optional[Dict[str, List[int]]] = None
) -> np.ndarray:
    import numpy as np
    import tqdm.auto as tqdm
    namespace: Dict[str, Any] = globals or {}
    if not params:
        length: int = 1
        packed_params: List[Tuple[()]] = [()]
    else:
        length = len(next(iter(params.values())))
        param_pairs: List[Iterator[Tuple[str, int]]] = [zip(itertools.repeat(name), value) for name, value in params.items()]
        packed_params: Iterator[Tuple[Tuple[str, int], ...]] = zip(*param_pairs)
    times: List[float] = []
    for local_params in tqdm.tqdm(
        packed_params, desc=title, total=length, disable=not progress_bar
    ):
        local_params_dict: Dict[str, int] = dict(local_params)
        timer = timeit.Timer(
            stmt=statement,
            setup=setup % local_params_dict,
            globals=namespace
        )
        best_time: float = min(timer.repeat(repeat=n_repeat, number=n_execution))
        times.append(best_time / n_execution)
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
    parser.add_argument(
        '-o',
        '--output-path',
        default=None,
        type=pathlib.Path,
        help='Output directory or file in which to save the benchmark results. If a directory is passed, a default filename is used.'
    )
    parser.add_argument(
        '--min-n-fields',
        type=int,
        default=0,
        help='Test the performance of BaseModel.__eq__ on models with at least this number of fields. Defaults to 0.'
    )
    parser.add_argument(
        '--max-n-fields',
        type=int,
        default=100,
        help='Test the performance of BaseModel.__eq__ on models with up to this number of fields. Defaults to 100.'
    )
    args = parser.parse_args()
    sizes: List[int] = list(range(args.min_n_fields, args.max_n_fields))
    fig: figure.Figure = plot_all_benchmark(IMPLEMENTATIONS, sizes=sizes)
    plt.tight_layout()
    if args.output_path is None:
        plt.show()
    else:
        if args.output_path.suffix:
            filepath: pathlib.Path = args.output_path
        else:
            filepath = args.output_path / f'eq-benchmark_python-{PYTHON_VERSION.replace(".", "-")}.png'
        fig.savefig(filepath, dpi=200, facecolor='white', transparent=False)
        print(f'wrote {filepath!s}', file=sys.stderr)
