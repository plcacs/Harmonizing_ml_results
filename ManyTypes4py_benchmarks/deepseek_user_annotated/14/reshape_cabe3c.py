from __future__ import annotations

import itertools
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
    Any,
    Optional,
    Union,
    List,
    Tuple,
    Set,
    FrozenSet,
    Dict,
    Sequence,
    Iterable,
    Callable,
    TypeVar,
    Generic,
    Mapping,
    Hashable,
)
import warnings

import numpy as np
from numpy.typing import NDArray

from pandas._config.config import get_option

import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    find_common_type,
    maybe_promote,
)
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_1d_only_ea_dtype,
    is_integer,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna

import pandas.core.algorithms as algos
from pandas.core.algorithms import (
    factorize,
    unique,
)
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    default_index,
)
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
    compress_group_index,
    decons_obs_group_ids,
    get_compressed_ids,
    get_group_index,
    get_group_index_sorter,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Level,
        npt,
        Axis,
        Suffixes,
        SuffixesType,
        Scalar,
        Ordered,
        AnyArrayLike,
        Dtype,
        DtypeObj,
        Shape,
        IndexLabel,
        HashableT,
    )

    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.frozen import FrozenList


class _Unstacker:
    """
    Helper class to unstack data / pivot with multi-level index

    Parameters
    ----------
    index : MultiIndex
    level : int or str, default last level
        Level to "unstack". Accepts a name for the level.
    fill_value : scalar, optional
        Default value to fill in missing values if subgroups do not have the
        same set of labels. By default, missing values will be replaced with
        the default fill value for that data type, NaN for float, NaT for
        datetimelike, etc. For integer types, by default data will converted to
        float and missing values will be set to NaN.
    constructor : object
        Pandas ``DataFrame`` or subclass used to create unstacked
        response.  If None, DataFrame will be used.
    """

    def __init__(
        self, index: MultiIndex, level: Level, constructor: Any, sort: bool = True
    ) -> None:
        self.constructor = constructor
        self.sort = sort

        self.index = index.remove_unused_levels()

        self.level = self.index._get_level_number(level)

        # when index includes `nan`, need to lift levels/strides by 1
        self.lift = 1 if -1 in self.index.codes[self.level] else 0

        # Note: the "pop" below alters these in-place.
        self.new_index_levels = list(self.index.levels)
        self.new_index_names = list(self.index.names)

        self.removed_name = self.new_index_names.pop(self.level)
        self.removed_level = self.new_index_levels.pop(self.level)
        self.removed_level_full = index.levels[self.level]
        if not self.sort:
            unique_codes = unique(self.index.codes[self.level])
            self.removed_level = self.removed_level.take(unique_codes)
            self.removed_level_full = self.removed_level_full.take(unique_codes)

        if get_option("performance_warnings"):
            # Bug fix GH 20601
            # If the data frame is too big, the number of unique index combination
            # will cause int32 overflow on windows environments.
            # We want to check and raise an warning before this happens
            num_rows = max(index_level.size for index_level in self.new_index_levels)
            num_columns = self.removed_level.size

            # GH20601: This forces an overflow if the number of cells is too high.
            # GH 26314: Previous ValueError raised was too restrictive for many users.
            num_cells = num_rows * num_columns
            if num_cells > np.iinfo(np.int32).max:
                warnings.warn(
                    f"The following operation may generate {num_cells} cells "
                    f"in the resulting pandas object.",
                    PerformanceWarning,
                    stacklevel=find_stack_level(),
                )

        self._make_selectors()

    @cache_readonly
    def _indexer_and_to_sort(
        self,
    ) -> tuple[
        NDArray[np.intp],
        List[NDArray[Any]],  # each has _some_ signed integer dtype
    ]:
        v = self.level

        codes = list(self.index.codes)
        if not self.sort:
            # Create new codes considering that labels are already sorted
            codes = [factorize(code)[0] for code in codes]
        levs = list(self.index.levels)
        to_sort = codes[:v] + codes[v + 1 :] + [codes[v]]
        sizes = tuple(len(x) for x in levs[:v] + levs[v + 1 :] + [levs[v]])

        comp_index, obs_ids = get_compressed_ids(to_sort, sizes)
        ngroups = len(obs_ids)

        indexer = get_group_index_sorter(comp_index, ngroups)
        return indexer, to_sort

    @cache_readonly
    def sorted_labels(self) -> List[NDArray[Any]]:
        indexer, to_sort = self._indexer_and_to_sort
        if self.sort:
            return [line.take(indexer) for line in to_sort]
        return to_sort

    def _make_sorted_values(self, values: NDArray[Any]) -> NDArray[Any]:
        indexer, _ = self._indexer_and_to_sort
        sorted_values = algos.take_nd(values, indexer, axis=0)
        return sorted_values

    def _make_selectors(self) -> None:
        new_levels = self.new_index_levels

        # make the mask
        remaining_labels = self.sorted_labels[:-1]
        level_sizes = tuple(len(x) for x in new_levels)

        comp_index, obs_ids = get_compressed_ids(remaining_labels, level_sizes)
        ngroups = len(obs_ids)

        comp_index = ensure_platform_int(comp_index)
        stride = self.index.levshape[self.level] + self.lift
        self.full_shape = ngroups, stride

        selector = self.sorted_labels[-1] + stride * comp_index + self.lift
        mask = np.zeros(np.prod(self.full_shape), dtype=bool)
        mask.put(selector, True)

        if mask.sum() < len(self.index):
            raise ValueError("Index contains duplicate entries, cannot reshape")

        self.group_index = comp_index
        self.mask = mask
        if self.sort:
            self.compressor = comp_index.searchsorted(np.arange(ngroups))
        else:
            self.compressor = np.sort(np.unique(comp_index, return_index=True)[1])

    @cache_readonly
    def mask_all(self) -> bool:
        return bool(self.mask.all())

    @cache_readonly
    def arange_result(self) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
        # We cache this for reuse in ExtensionBlock._unstack
        dummy_arr = np.arange(len(self.index), dtype=np.intp)
        new_values, mask = self.get_new_values(dummy_arr, fill_value=-1)
        return new_values, mask.any(0)
        # TODO: in all tests we have mask.any(0).all(); can we rely on that?

    def get_result(self, obj: Any, value_columns: Any, fill_value: Any) -> DataFrame:
        values = obj._values
        if values.ndim == 1:
            values = values[:, np.newaxis]

        if value_columns is None and values.shape[1] != 1:  # pragma: no cover
            raise ValueError("must pass column labels for multi-column data")

        new_values, _ = self.get_new_values(values, fill_value)
        columns = self.get_new_columns(value_columns)
        index = self.new_index

        result = self.constructor(
            new_values, index=index, columns=columns, dtype=new_values.dtype, copy=False
        )
        if isinstance(values, np.ndarray):
            base, new_base = values.base, new_values.base
        elif isinstance(values, NDArrayBackedExtensionArray):
            base, new_base = values._ndarray.base, new_values._ndarray.base
        else:
            base, new_base = 1, 2  # type: ignore[assignment]
        if base is new_base:
            # We can only get here if one of the dimensions is size 1
            result._mgr.add_references(obj._mgr)
        return result

    def get_new_values(self, values: Any, fill_value: Any = None) -> tuple[Any, Any]:
        if values.ndim == 1:
            values = values[:, np.newaxis]

        sorted_values = self._make_sorted_values(values)

        # place the values
        length, width = self.full_shape
        stride = values.shape[1]
        result_width = width * stride
        result_shape = (length, result_width)
        mask = self.mask
        mask_all = self.mask_all

        # we can simply reshape if we don't have a mask
        if mask_all and len(values):
            # TODO: Under what circumstances can we rely on sorted_values
            #  matching values?  When that holds, we can slice instead
            #  of take (in particular for EAs)
            new_values = (
                sorted_values.reshape(length, width, stride)
                .swapaxes(1, 2)
                .reshape(result_shape)
            )
            new_mask = np.ones(result_shape, dtype=bool)
            return new_values, new_mask

        dtype = values.dtype

        if isinstance(dtype, ExtensionDtype):
            # GH#41875
            # We are assuming that fill_value can be held by this dtype,
            #  unlike the non-EA case that promotes.
            cls = dtype.construct_array_type()
            new_values = cls._empty(result_shape, dtype=dtype)
            if not mask_all:
                new_values[:] = fill_value
        else:
            if not mask_all:
                dtype, fill_value = maybe_promote(dtype, fill_value)
            new_values = np.empty(result_shape, dtype=dtype)
            if not mask_all:
                new_values.fill(fill_value)

        name = dtype.name
        new_mask = np.zeros(result_shape, dtype=bool)

        # we need to convert to a basic dtype
        # and possibly coerce an input to our output dtype
        # e.g. ints -> floats
        if needs_i8_conversion(values.dtype):
            sorted_values = sorted_values.view("i8")
            new_values = new_values.view("i8")
        else:
            sorted_values = sorted_values.astype(name, copy=False)

        # fill in our values & mask
        libreshape.unstack(
            sorted_values,
            mask.view("u1"),
            stride,
            length,
            width,
            new_values,
            new_mask.view("u1"),
        )

        # reconstruct dtype if needed
        if needs_i8_conversion(values.dtype):
            # view as datetime64 so we can wrap in DatetimeArray and use
            #  DTA's view method
            new_values = new_values.view("M8[ns]")
            new_values = ensure_wrapped_if_datetimelike(new_values)
            new_values = new_values.view(values.dtype)

        return new_values, new_mask

    def get_new_columns(self, value_columns: Index | None) -> Any:
        if value_columns is None:
            if self.lift == 0:
                return self.removed_level._rename(name=self.removed_name)

            lev = self.removed_level.insert(0, item=self.removed_level._na_value)
            return lev.rename(self.removed_name)

        stride = len(self.removed_level) + self.lift
        width = len(value_columns)
        propagator = np.repeat(np.arange(width), stride)

        new_levels: FrozenList | list[Index]

        if isinstance(value_columns, MultiIndex):
            # error: Cannot determine type of "__add__"  [has-type]
            new_levels = value_columns.levels + (  # type: ignore[has-type]
                self.removed_level_full,
            )
            new_names = value_columns.names + (self.removed_name,)

            new_codes = [lab.take(propagator) for lab in value_columns.codes]
        else:
            new_levels = [
                value_columns,
                self.removed_level_full,
            ]
            new_names = [value_columns.name, self.removed_name]
            new_codes = [propagator]

        repeater = self._repeater

        # The entire level is then just a repetition of the single chunk:
        new_codes.append(np.tile(repeater, width))
        return MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )

    @cache_readonly
    def _repeater(self) -> NDArray[Any]:
        # The two indices differ only if the unstacked level had unused items:
        if len(self.removed_level_full) != len(self.removed_level):
            # In this case, we remap the new codes to the original level:
            repeater = self.removed_level_full.get_indexer(self.removed_level)
            if self.lift:
                repeater = np.insert(repeater, 0, -1)
        else:
            # Otherwise, we just use each level item exactly once:
            stride = len(self.removed_level) + self.lift
            repeater = np.arange(stride) - self.lift

        return repeater

    @cache_readonly
    def new_index(self) -> MultiIndex | Index:
        # Does not depend on values or value_columns
        if self.sort:
            labels = self.sorted_labels[:-1]
        else:
            v = self.level
            codes = list(self.index.codes)
            labels = codes[:v] + codes[v + 1 :]
        result_codes = [lab.take(self.compressor) for lab in labels]

        # construct the new index
        if len(self.new_index_levels) == 1:
            level, level_codes = self.new_index_levels[0], result_codes[0]
            if (level_codes == -1).any():
                level = level.insert(len(level), level._na_value)
            return level.take(level_codes).rename(self.new_index_names[0])

        return MultiIndex(
            levels=self.new_index_levels,
            codes=result_codes,
            names=self.new_index_names,
            verify_integrity=False,
        )


def _unstack_multiple(
    data: Series | DataFrame, clocs: Any, fill_value: Any = None, sort: bool = True
) -> DataFrame:
    if len(clocs) == 0:
        return data

    # NOTE: This doesn't deal with hierarchical columns yet

    index = data.index
    index = cast(MultiIndex, index)  # caller is responsible for checking

    # GH 19966 Make sure if MultiIndexed index has tuple name, they will be
    # recognised as a whole
    if clocs in index.names:
        clocs = [clocs]
    clocs = [index._get_level_number(i) for i in clocs]

    rlocs = [i for i in range(index.nlevels) if i not in clocs]

    clevels = [index.levels[i] for i in clocs]
    ccodes = [index.codes[i] for i in clocs]
    cnames = [index.names[i] for i in clocs]
    rlevels = [index.levels[i] for i in rlocs]
    rcodes = [index.codes[i] for i in rlocs]
    rnames = [index.names[i] for i in rlocs]

    shape = tuple(len(x) for x in clevels)
    group_index = get_group_index(ccodes, shape, sort=False, xnull=False)

    comp_ids, obs_ids = compress_group_index(group_index, sort=False)
    recons_codes = decons_obs_group_ids(comp_ids, obs_ids, shape, ccodes, xnull=False)

    if not rlocs:
        # Everything is in clocs, so the dummy df has a regular index
        dummy_index = Index(obs_ids, name="__placeholder__")
    else:
        dummy_index = MultiIndex(
            levels=rlevels + [obs_ids],
            codes=rcodes + [comp_ids],
            names=rnames + ["__placeholder__"],
            verify_integrity=False,
        )

    if isinstance(data, Series):
        dummy = data.copy(deep=False)
        dummy.index = dummy_index

        unstacked = dummy.unstack("__placeholder__", fill_value=fill_value, sort=sort)
        new_levels = clevels
        new_names = cnames
        new_codes = recons_codes
    else:
        if isinstance(data.columns, MultiIndex):
            result = data
            while clocs:
                val = clocs.pop(0)
                # error: Incompatible types in assignment (expression has type
                # "DataFrame | Series", variable has type "DataFrame")
                result = result.unstack(  # type: ignore[assignment]
                    val, fill_value=fill_value, sort=sort
                )
                clocs = [v if v < val else v - 1 for v in clocs]

            return result

        # GH#42579 deep=False to avoid consolidating
        dummy_df = data.copy(deep