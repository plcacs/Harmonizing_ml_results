"""
SQL-style merge routines
"""
from __future__ import annotations
from collections.abc import Hashable, Sequence
import datetime
from functools import partial
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
    final,
    Optional,
    Union,
    List,
    Tuple,
    Callable,
    Any,
)
import uuid
import warnings
import numpy as np
import pandas.core.algorithms as algos
from pandas import (
    ArrowDtype,
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
from pandas._libs import Timedelta, hashtable as libhashtable, join as libjoin, lib
from pandas._libs.lib import is_range_indexer
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    IndexLabel,
    JoinHow,
    MergeHow,
    Shape,
    Suffixes,
    npt,
)
from pandas.errors import MergeError
from pandas.util._decorators import cache_readonly, set_module
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_object,
    is_bool,
    is_bool_dtype,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_number,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import CategoricalDtype, DatetimeTZDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import isna, na_value_for_dtype
from pandas.core.arrays import (
    ArrowExtensionArray,
    BaseMaskedArray,
    ExtensionArray,
)
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.construction import ensure_wrapped_if_datetimelike, extract_array
from pandas.core.indexes.api import default_index
from pandas.core.sorting import get_group_index, is_int64_overflow_possible

if TYPE_CHECKING:
    from pandas import DataFrame
    from pandas.core import groupby
    from pandas.core.arrays import DatetimeArray
    from pandas.core.indexes.frozen import FrozenList

_factorizers: dict[np.dtype[Any], type] = {
    np.int64: libhashtable.Int64Factorizer,
    np.longlong: libhashtable.Int64Factorizer,
    np.int32: libhashtable.Int32Factorizer,
    np.int16: libhashtable.Int16Factorizer,
    np.int8: libhashtable.Int8Factorizer,
    np.uint64: libhashtable.UInt64Factorizer,
    np.uint32: libhashtable.UInt32Factorizer,
    np.uint16: libhashtable.UInt16Factorizer,
    np.uint8: libhashtable.UInt8Factorizer,
    np.bool_: libhashtable.UInt8Factorizer,
    np.float64: libhashtable.Float64Factorizer,
    np.float32: libhashtable.Float32Factorizer,
    np.complex64: libhashtable.Complex64Factorizer,
    np.complex128: libhashtable.Complex128Factorizer,
    np.object_: libhashtable.ObjectFactorizer,
}
if np.intc is not np.int32:
    if np.dtype(np.intc).itemsize == 4:
        _factorizers[np.intc] = libhashtable.Int32Factorizer
    else:
        _factorizers[np.intc] = libhashtable.Int64Factorizer
if np.uintc is not np.uint32:
    if np.dtype(np.uintc).itemsize == 4:
        _factorizers[np.uintc] = libhashtable.UInt32Factorizer
    else:
        _factorizers[np.uintc] = libhashtable.UInt64Factorizer
_known = (np.ndarray, ExtensionArray, Index, ABCSeries)


@set_module("pandas")
def func_po3mrmad(
    left: Union[DataFrame, Series],
    right: Union[DataFrame, Series],
    how: Literal[
        "left",
        "right",
        "outer",
        "inner",
        "cross",
        "left_anti",
        "right_anti",
    ] = "inner",
    on: Optional[Union[Hashable, List[Hashable]]] = None,
    left_on: Optional[
        Union[Hashable, List[Hashable], List[List[Any]]]
    ] = None,
    right_on: Optional[
        Union[Hashable, List[Hashable], List[List[Any]]]
    ] = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: Tuple[Optional[str], Optional[str]] = ("_x", "_y"),
    copy: Any = lib.no_default,
    indicator: Union[bool, str] = False,
    validate: Optional[str] = None,
) -> DataFrame:
    """
    Merge DataFrame or named Series objects with a database-style join.

    [Docstring omitted for brevity]
    """
    left_df = _validate_operand(left)
    left._check_copy_deprecation(copy)
    right_df = _validate_operand(right)
    if how == "cross":
        return _cross_merge(
            left_df,
            right_df,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            indicator=indicator,
            validate=validate,
        )
    else:
        op = _MergeOperation(
            left_df,
            right_df,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            indicator=indicator,
            validate=validate,
        )
        return op.get_result()


def func_1br7z7i8(
    left: Union[DataFrame, Series],
    right: Union[DataFrame, Series],
    on: Optional[Union[Hashable, List[Hashable]]] = None,
    left_on: Optional[
        Union[Hashable, List[Hashable], List[List[Any]]]
    ] = None,
    right_on: Optional[
        Union[Hashable, List[Hashable], List[List[Any]]]
    ] = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: Tuple[Optional[str], Optional[str]] = ("_x", "_y"),
    indicator: Union[bool, str] = False,
    validate: Optional[str] = None,
) -> DataFrame:
    """
    See merge.__doc__ with how='cross'
    """
    if (
        left_index
        or right_index
        or right_on is not None
        or left_on is not None
        or on is not None
    ):
        raise MergeError(
            "Can not pass on, right_on, left_on or set right_index=True or left_index=True"
        )
    cross_col = f"_cross_{uuid.uuid4()}"
    left = left.assign(**{cross_col: 1})
    right = right.assign(**{cross_col: 1})
    left_on = right_on = [cross_col]
    res = func_po3mrmad(
        left,
        right,
        how="inner",
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        sort=sort,
        suffixes=suffixes,
        indicator=indicator,
        validate=validate,
    )
    del res[cross_col]
    return res


def func_x0xjdcu3(
    by: Union[Hashable, List[Hashable]],
    left: DataFrame,
    right: DataFrame,
    merge_pieces: Callable[[DataFrame, DataFrame], DataFrame],
) -> Tuple[DataFrame, groupby.DataFrameGroupBy]:
    """
    groupby & merge; we are always performing a left-by type operation

    Parameters
    ----------
    by: field to group
    left: DataFrame
    right: DataFrame
    merge_pieces: function for merging
    """
    pieces: List[DataFrame] = []
    if not isinstance(by, (list, tuple)):
        by = [by]
    lby = left.groupby(by, sort=False)
    rby: Optional[groupby.DataFrameGroupBy] = None
    if all(item in right.columns for item in by):
        rby = right.groupby(by, sort=False)
    for key, lhs in lby._grouper.get_iterator(lby._selected_obj):
        if rby is None:
            rhs = right
        else:
            try:
                rhs = right.take(rby.indices[key])
            except KeyError:
                lcols = lhs.columns.tolist()
                cols = lcols + [r for r in right.columns if r not in set(lcols)]
                merged = lhs.reindex(columns=cols)
                merged.index = range(len(merged))
                pieces.append(merged)
                continue
        merged = merge_pieces(lhs, rhs)
        merged[by] = key
        pieces.append(merged)
    from pandas.core.reshape.concat import concat

    result = concat(pieces, ignore_index=True)
    result = result.reindex(columns=pieces[0].columns)
    return result, lby


@set_module("pandas")
def func_fqdj9083(
    left: Union[DataFrame, Series],
    right: Union[DataFrame, Series],
    on: Optional[Union[Hashable, List[Hashable]]] = None,
    left_on: Optional[
        Union[Hashable, List[Hashable], List[List[Any]]]
    ] = None,
    right_on: Optional[
        Union[Hashable, List[Hashable], List[List[Any]]]
    ] = None,
    left_by: Optional[Union[Hashable, List[Hashable]]] = None,
    right_by: Optional[Union[Hashable, List[Hashable]]] = None,
    fill_method: Optional[Literal["ffill"]] = None,
    suffixes: Tuple[Optional[str], Optional[str]] = ("_x", "_y"),
    how: Literal["left", "right", "outer", "inner"] = "outer",
) -> DataFrame:
    """
    Perform a merge for ordered data with optional filling/interpolation.

    [Docstring omitted for brevity]
    """

    def func_x2sl14mw(x: DataFrame, y: DataFrame) -> DataFrame:
        op = _OrderedMerge(
            x,
            y,
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
            fill_method=fill_method,
            how=how,
        )
        return op.get_result()

    if left_by is not None and right_by is not None:
        raise ValueError("Can only group either left or right frames")
    if left_by is not None:
        if isinstance(left_by, str):
            left_by = [left_by]
        check = set(left_by).difference(left.columns)
        if len(check) != 0:
            raise KeyError(f"{check} not found in left columns")
        result, _ = func_x0xjdcu3(left_by, left, right, lambda x, y: func_x2sl14mw(x, y))
    elif right_by is not None:
        if isinstance(right_by, str):
            right_by = [right_by]
        check = set(right_by).difference(right.columns)
        if len(check) != 0:
            raise KeyError(f"{check} not found in right columns")
        result, _ = func_x0xjdcu3(right_by, right, left, lambda x, y: func_x2sl14mw(y, x))
    else:
        result = func_x2sl14mw(left, right)
    return result


@set_module("pandas")
def func_fva53swq(
    left: Union[DataFrame, Series],
    right: Union[DataFrame, Series],
    on: Optional[Union[Hashable, List[Hashable]]] = None,
    left_on: Optional[Union[Hashable, List[Hashable], List[List[Any]]]] = None,
    right_on: Optional[Union[Hashable, List[Hashable], List[List[Any]]]] = None,
    left_index: bool = False,
    right_index: bool = False,
    by: Optional[Union[Hashable, List[Hashable]]] = None,
    left_by: Optional[Union[Hashable, List[Hashable]]] = None,
    right_by: Optional[Union[Hashable, List[Hashable]]] = None,
    suffixes: Tuple[Optional[str], Optional[str]] = ("_x", "_y"),
    tolerance: Optional[Union[int, datetime.timedelta]] = None,
    allow_exact_matches: bool = True,
    direction: Literal["backward", "forward", "nearest"] = "backward",
) -> DataFrame:
    """
    Perform a merge by key distance.

    [Docstring omitted for brevity]
    """
    op = _AsOfMerge(
        left,
        right,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        by=by,
        left_by=left_by,
        right_by=right_by,
        suffixes=suffixes,
        how="asof",
        tolerance=tolerance,
        allow_exact_matches=allow_exact_matches,
        direction=direction,
    )
    return op.get_result()


class _MergeOperation:
    """
    Perform a database (SQL) merge operation between two DataFrame or Series
    objects using either columns as keys or their row indexes
    """

    _merge_type: str = "merge"

    def __init__(
        self,
        left: Union[DataFrame, Series],
        right: Union[DataFrame, Series],
        how: Literal[
            "left",
            "right",
            "outer",
            "inner",
            "left_anti",
            "right_anti",
            "cross",
            "asof",
        ] = "inner",
        on: Optional[Union[Hashable, List[Hashable]]] = None,
        left_on: Optional[
            Union[Hashable, List[Hashable], List[List[Any]]]
        ] = None,
        right_on: Optional[
            Union[Hashable, List[Hashable], List[List[Any]]]
        ] = None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = True,
        suffixes: Tuple[Optional[str], Optional[str]] = ("_x", "_y"),
        indicator: bool = False,
        validate: Optional[str] = None,
    ) -> None:
        _left = _validate_operand(left)
        _right = _validate_operand(right)
        self.left: DataFrame = _left
        self.orig_left: DataFrame = _left
        self.right: DataFrame = _right
        self.orig_right: DataFrame = _right
        self.how, self.anti_join = self._validate_how(how)
        self.on: Optional[List[Hashable]] = com.maybe_make_list(on)
        self.suffixes: Tuple[Optional[str], Optional[str]] = suffixes
        self.sort: bool = sort or how == "outer"
        self.left_index: bool = left_index
        self.right_index: bool = right_index
        self.indicator: bool = indicator
        if not is_bool(left_index):
            raise ValueError(
                f"left_index parameter must be of type bool, not {type(left_index)}"
            )
        if not is_bool(right_index):
            raise ValueError(
                f"right_index parameter must be of type bool, not {type(right_index)}"
            )
        if _left.columns.nlevels != _right.columns.nlevels:
            msg = (
                f"Not allowed to merge between different levels. ({_left.columns.nlevels} levels on the left, {_right.columns.nlevels} on the right)"
            )
            raise MergeError(msg)
        self.left_on, self.right_on = self._validate_left_right_on(left_on, right_on)
        (
            self.left_join_keys,
            self.right_join_keys,
            self.join_names,
            left_drop,
            right_drop,
        ) = self._get_merge_keys()
        if left_drop:
            self.left = self.left._drop_labels_or_levels(left_drop)
        if right_drop:
            self.right = self.right._drop_labels_or_levels(right_drop)
        self._maybe_require_matching_dtypes(self.left_join_keys, self.right_join_keys)
        self._validate_tolerance(self.left_join_keys)
        self._maybe_coerce_merge_keys()
        if validate is not None:
            self._validate_validate_kwd(validate)

    @final
    def _validate_how(
        self, how: str
    ) -> Tuple[Union[JoinHow, Literal['asof']], bool]:
        """
        Validate the 'how' parameter and return the actual join type and whether
        this is an anti join.
        """
        merge_type: set[str] = {
            "left",
            "right",
            "outer",
            "inner",
            "left_anti",
            "right_anti",
            "cross",
            "asof",
        }
        if how not in merge_type:
            raise ValueError(
                f"'{how}' is not a valid Merge type: left, right, inner, outer, left_anti, right_anti, cross, asof"
            )
        anti_join = False
        if how in {"left_anti", "right_anti"}:
            how = how.split("_")[0]
            anti_join = True
        how = cast(Union[JoinHow, Literal["asof"]], how)
        return how, anti_join

    def func_bpeb5s82(
        self, left_join_keys: List[ArrayLike], right_join_keys: List[ArrayLike]
    ) -> None:
        pass

    def func_hufktgb8(self, left_join_keys: List[ArrayLike]) -> None:
        pass

    @final
    def _reindex_and_concat(
        self,
        join_index: Index,
        left_indexer: Optional[np.ndarray],
        right_indexer: Optional[np.ndarray],
    ) -> DataFrame:
        """
        Reindex along index and concat along columns.
        """
        left = self.left[:]
        right = self.right[:]
        llabels, rlabels = _items_overlap_with_suffix(
            self.left._info_axis, self.right._info_axis, self.suffixes
        )
        if left_indexer is not None and not is_range_indexer(left_indexer, len(left)):
            lmgr = left._mgr.reindex_indexer(
                join_index,
                left_indexer,
                axis=1,
                only_slice=True,
                allow_dups=True,
                use_na_proxy=True,
            )
            left = left._constructor_from_mgr(lmgr, axes=lmgr.axes)
        left.index = join_index
        if right_indexer is not None and not is_range_indexer(right_indexer, len(right)):
            rmgr = right._mgr.reindex_indexer(
                join_index,
                right_indexer,
                axis=1,
                only_slice=True,
                allow_dups=True,
                use_na_proxy=True,
            )
            right = right._constructor_from_mgr(rmgr, axes=rmgr.axes)
        right.index = join_index
        from pandas import concat

        left.columns = llabels
        right.columns = rlabels
        result = concat([left, right], axis=1)
        return result

    def get_result(self) -> DataFrame:
        if self.indicator:
            self.left, self.right = self._indicator_pre_merge(self.left, self.right)
        join_index, left_indexer, right_indexer = self._get_join_info()
        result = self._reindex_and_concat(join_index, left_indexer, right_indexer)
        result = result.__finalize__(self, method=self._merge_type)
        if self.indicator:
            result = self._indicator_post_merge(result)
        self._maybe_add_join_keys(result, left_indexer, right_indexer)
        self._maybe_restore_index_levels(result)
        return result.__finalize__(self, method="merge")

    @final
    @cache_readonly
    def _indicator_name(self) -> Optional[str]:
        if isinstance(self.indicator, str):
            return self.indicator
        elif isinstance(self.indicator, bool):
            return "_merge" if self.indicator else None
        else:
            raise ValueError(
                "indicator option can only accept boolean or string arguments"
            )

    @final
    def _indicator_pre_merge(
        self, left: DataFrame, right: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        columns = left.columns.union(right.columns)
        for i in ["_left_indicator", "_right_indicator"]:
            if i in columns:
                raise ValueError(
                    f"Cannot use `indicator=True` option when data contains a column named {i}"
                )
        if self._indicator_name in columns:
            raise ValueError(
                "Cannot use name of an existing column for indicator column"
            )
        left = left.copy()
        right = right.copy()
        left["_left_indicator"] = 1
        left["_left_indicator"] = left["_left_indicator"].astype("int8")
        right["_right_indicator"] = 2
        right["_right_indicator"] = right["_right_indicator"].astype("int8")
        return left, right

    @final
    def _indicator_post_merge(self, result: DataFrame) -> DataFrame:
        result["_left_indicator"] = result["_left_indicator"].fillna(0)
        result["_right_indicator"] = result["_right_indicator"].fillna(0)
        result[self._indicator_name] = Categorical(
            result["_left_indicator"] + result["_right_indicator"],
            categories=[1, 2, 3],
        )
        result[self._indicator_name] = result[self._indicator_name].cat.rename_categories(
            ["left_only", "right_only", "both"]
        )
        result = result.drop(labels=["_left_indicator", "_right_indicator"], axis=1)
        return result

    @final
    def _maybe_restore_index_levels(self, result: DataFrame) -> None:
        """
        Restore index levels specified as `on` parameters

        Here we check for cases where `self.left_on` and `self.right_on` pairs
        each reference an index level in their respective DataFrames. The
        joined columns corresponding to these pairs are then restored to the
        index of `result`.

        **Note:** This method has side effects. It modifies `result` in-place

        Parameters
        ----------
        result: DataFrame
            merge result

        Returns
        -------
        None
        """
        names_to_restore: List[Hashable] = []
        for name, left_key, right_key in zip(
            self.join_names, self.left_on, self.right_on
        ):
            if (
                self.orig_left._is_level_reference(left_key)
                and self.orig_right._is_level_reference(right_key)
                and left_key == right_key
                and name not in result.index.names
            ):
                names_to_restore.append(name)
        if names_to_restore:
            result.set_index(names_to_restore, inplace=True)

    @final
    def _maybe_add_join_keys(
        self,
        result: DataFrame,
        left_indexer: Optional[np.ndarray],
        right_indexer: Optional[np.ndarray],
    ) -> None:
        left_has_missing: Optional[bool] = None
        right_has_missing: Optional[bool] = None
        assert all(
            isinstance(x, (_known)) for x in self.left_join_keys
        ), "All left_join_keys must be known types"
        keys = zip(self.join_names, self.left_on, self.right_on)
        for i, (name, lname, rname) in enumerate(keys):
            if not _should_fill(lname, rname):
                continue
            take_left: Optional[ArrayLike] = None
            take_right: Optional[ArrayLike] = None
            if name in result:
                if left_indexer is not None or right_indexer is not None:
                    if name in self.left:
                        if left_has_missing is None:
                            left_has_missing = False if left_indexer is None else (
                                (left_indexer == -1).any()
                            )
                        if left_has_missing:
                            take_right = self.right_join_keys[i]
                            if result[name].dtype != self.left[name].dtype:
                                take_left = self.left[name]._values
                    elif name in self.right:
                        if right_has_missing is None:
                            right_has_missing = False if right_indexer is None else (
                                (right_indexer == -1).any()
                            )
                        if right_has_missing:
                            take_left = self.left_join_keys[i]
                            if result[name].dtype != self.right[name].dtype:
                                take_right = self.right[name]._values
            else:
                take_left = self.left_join_keys[i]
                take_right = self.right_join_keys[i]
            if take_left is not None or take_right is not None:
                if take_left is None:
                    lvals = result[name]._values
                elif left_indexer is None:
                    lvals = take_left
                else:
                    take_left = extract_array(take_left, extract_numpy=True)
                    lfill = na_value_for_dtype(take_left.dtype)
                    lvals = algos.take_nd(take_left, left_indexer, fill_value=lfill)
                if take_right is None:
                    rvals = result[name]._values
                elif right_indexer is None:
                    rvals = take_right
                else:
                    taker = extract_array(take_right, extract_numpy=True)
                    rfill = na_value_for_dtype(taker.dtype)
                    rvals = algos.take_nd(taker, right_indexer, fill_value=rfill)
                if left_indexer is not None and (left_indexer == -1).all():
                    key_col = Index(rvals)
                    result_dtype = rvals.dtype
                elif right_indexer is not None and (right_indexer == -1).all():
                    key_col = Index(lvals)
                    result_dtype = lvals.dtype
                else:
                    key_col = Index(lvals)
                    if left_indexer is not None:
                        mask_left = left_indexer == -1
                        key_col = key_col.where(~mask_left, rvals)
                    result_dtype = find_common_type([lvals.dtype, rvals.dtype])
                    if (
                        lvals.dtype.kind == "M"
                        and rvals.dtype.kind == "M"
                        and result_dtype.kind == "O"
                    ):
                        result_dtype = key_col.dtype
                if result._is_label_reference(name):
                    result[name] = result._constructor_sliced(
                        key_col, dtype=result_dtype, index=result.index
                    )
                elif result._is_level_reference(name):
                    if isinstance(result.index, MultiIndex):
                        key_col.name = name
                        idx_list = [
                            result.index.get_level_values(level_name)
                            if level_name != name
                            else key_col
                            for level_name in result.index.names
                        ]
                        result.set_index(idx_list, inplace=True)
                    else:
                        result.index = Index(key_col, name=name)
                else:
                    result.insert(i, name or f"key_{i}", key_col)

    @final
    def _get_join_info(
        self,
    ) -> Tuple[Index, Optional[np.ndarray], Optional[np.ndarray]]:
        join_index, left_indexer, right_indexer = self._get_join_indexers()
        if self.anti_join:
            join_index, left_indexer, right_indexer = self._handle_anti_join(
                join_index, left_indexer, right_indexer
            )
        return join_index, left_indexer, right_indexer

    @final
    def _handle_anti_join(
        self,
        join_index: Index,
        left_indexer: Optional[np.ndarray],
        right_indexer: Optional[np.ndarray],
    ) -> Tuple[Index, Optional[np.ndarray], Optional[np.ndarray]]:
        if left_indexer is None:
            left_indexer = np.arange(len(self.left), dtype=np.intp)
        if right_indexer is None:
            right_indexer = np.arange(len(self.right), dtype=np.intp)
        assert self.how in {"left", "right"}, "Invalid how for anti join"
        if self.how == "left":
            filt = right_indexer == -1
        else:
            filt = left_indexer == -1
        join_index = join_index[filt]
        left_indexer = left_indexer[filt]
        right_indexer = right_indexer[filt]
        return join_index, left_indexer, right_indexer

    @final
    def _get_join_indexers(
        self,
    ) -> Tuple[Index, Optional[np.ndarray], Optional[np.ndarray]]:
        left_ax = self.left.index
        right_ax = self.right.index
        if self.left_index and self.right_index and self.how != "asof":
            join_index, left_indexer, right_indexer = left_ax.join(
                right_ax,
                how=self.how,
                return_indexers=True,
                sort=self.sort,
            )
        elif self.right_index and self.how == "left":
            join_index, left_indexer, right_indexer = _left_join_on_index(
                left_ax, right_ax, self.left_join_keys, sort=self.sort
            )
        elif self.left_index and self.how == "right":
            join_index, right_indexer, left_indexer = _left_join_on_index(
                right_ax, left_ax, self.right_join_keys, sort=self.sort
            )
        else:
            left_indexer, right_indexer = self._get_join_indexers_non_asof()
            if self.right_index:
                if len(self.left) > 0:
                    join_index = self._create_join_index(
                        left_ax, right_ax, left_indexer, how="right"
                    )
                elif right_indexer is None:
                    join_index = right_ax.copy()
                else:
                    join_index = right_ax.take(right_indexer)
            elif self.left_index:
                if self.how == "asof":
                    join_index = self._create_join_index(
                        left_ax, right_ax, left_indexer, how="left"
                    )
                elif len(self.right) > 0:
                    join_index = self._create_join_index(
                        right_ax, left_ax, right_indexer, how="left"
                    )
                elif left_indexer is None:
                    join_index = left_ax.copy()
                else:
                    join_index = left_ax.take(left_indexer)
            else:
                n = len(left_ax) if left_indexer is None else len(left_indexer)
                join_index = default_index(n)
        if self.anti_join:
            join_index, left_indexer, right_indexer = self._handle_anti_join(
                join_index, left_indexer, right_indexer
            )
        return join_index, left_indexer, right_indexer

    @final
    def _get_join_indexers_non_asof(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return get_join_indexers(
            self.left_join_keys, self.right_join_keys, sort=self.sort, how=self.how
        )

    def _get_merge_keys(
        self,
    ) -> Tuple[
        List[ArrayLike],
        List[ArrayLike],
        List[Optional[Hashable]],
        List[Hashable],
        List[Hashable],
    ]:
        left_keys: List[ArrayLike] = []
        right_keys: List[ArrayLike] = []
        join_names: List[Optional[Hashable]] = []
        right_drop: List[Hashable] = []
        left_drop: List[Hashable] = []
        left = self.left
        right = self.right
        is_lkey = lambda x: isinstance(x, _known) and len(x) == len(left)
        is_rkey = lambda x: isinstance(x, _known) and len(x) == len(right)
        if _any(self.left_on) and _any(self.right_on):
            for lk, rk in zip(self.left_on, self.right_on):
                lk = extract_array(lk, extract_numpy=True)
                rk = extract_array(rk, extract_numpy=True)
                if is_lkey(lk):
                    lk = cast(ArrayLike, lk)
                    left_keys.append(lk)
                    if is_rkey(rk):
                        rk = cast(ArrayLike, rk)
                        right_keys.append(rk)
                        join_names.append(None)
                    else:
                        rk = cast(Hashable, rk)
                        if rk is not None:
                            right_keys.append(right._get_label_or_level_values(rk))
                            join_names.append(rk)
                        else:
                            right_keys.append(right.index._values)
                            join_names.append(right.index.name)
                else:
                    if not is_rkey(rk):
                        rk = cast(Hashable, rk)
                        if rk is not None:
                            right_keys.append(right._get_label_or_level_values(rk))
                        else:
                            right_keys.append(right.index._values)
                        if lk is not None and lk == rk:
                            right_drop.append(rk)
                    else:
                        rk = cast(ArrayLike, rk)
                        right_keys.append(rk)
                    if lk is not None:
                        lk = cast(Hashable, lk)
                        left_keys.append(left._get_label_or_level_values(lk))
                        join_names.append(lk)
                    else:
                        left_keys.append(left.index._values)
                        join_names.append(left.index.name)
        elif _any(self.left_on):
            for k in self.left_on:
                if is_lkey(k):
                    k = extract_array(k, extract_numpy=True)
                    k = cast(ArrayLike, k)
                    left_keys.append(k)
                    join_names.append(None)
                else:
                    k = cast(Hashable, k)
                    left_keys.append(left._get_label_or_level_values(k))
                    join_names.append(k)
            if isinstance(self.right.index, MultiIndex):
                right_keys = [
                    lev._values.take(lev_codes) for lev, lev_codes in zip(
                        self.right.index.levels, self.right.index.codes
                    )
                ]
            else:
                right_keys = [self.right.index._values]
        elif _any(self.right_on):
            for k in self.right_on:
                k = extract_array(k, extract_numpy=True)
                if is_rkey(k):
                    k = cast(ArrayLike, k)
                    right_keys.append(k)
                    join_names.append(None)
                else:
                    k = cast(Hashable, k)
                    right_keys.append(right._get_label_or_level_values(k))
                    join_names.append(k)
            if isinstance(self.left.index, MultiIndex):
                left_keys = [
                    lev._values.take(lev_codes) for lev, lev_codes in zip(
                        self.left.index.levels, self.left.index.codes
                    )
                ]
            else:
                left_keys = [self.left.index._values]
        return left_keys, right_keys, join_names, left_drop, right_drop

    @final
    def _maybe_require_matching_dtypes(
        self, left_join_keys: List[ArrayLike], right_join_keys: List[ArrayLike]
    ) -> None:
        for i, (lk, rk, name) in enumerate(zip(
            left_join_keys, right_join_keys, self.join_names
        )):
            if len(lk) and not len(rk) or not len(lk) and len(rk):
                continue
            lk = extract_array(lk, extract_numpy=True)
            rk = extract_array(rk, extract_numpy=True)
            lk_is_cat = isinstance(lk.dtype, CategoricalDtype)
            rk_is_cat = isinstance(rk.dtype, CategoricalDtype)
            lk_is_object_or_string = is_object_dtype(lk.dtype) or is_string_dtype(lk.dtype)
            rk_is_object_or_string = is_object_dtype(rk.dtype) or is_string_dtype(rk.dtype)
            if lk_is_cat and rk_is_cat:
                lk = cast(Categorical, lk)
                rk = cast(Categorical, rk)
                if lk._categories_match_up_to_permutation(rk):
                    continue
            elif lk_is_cat or rk_is_cat:
                pass
            elif lk.dtype == rk.dtype:
                continue
            msg = (
                f"You are trying to merge on {lk.dtype} and {rk.dtype} columns for key '{name}'. If you wish to proceed you should use pd.concat"
            )
            if is_numeric_dtype(lk.dtype) and is_numeric_dtype(rk.dtype):
                if lk.dtype.kind == rk.dtype.kind:
                    continue
                if isinstance(lk.dtype, ExtensionDtype) and not isinstance(
                    rk.dtype, ExtensionDtype
                ):
                    ct = find_common_type([lk.dtype, rk.dtype])
                    if isinstance(ct, ExtensionDtype):
                        com_cls = ct.construct_array_type()
                        rk = com_cls._from_sequence(rk, dtype=ct, copy=False)
                    else:
                        rk = rk.astype(ct)
                elif isinstance(rk.dtype, ExtensionDtype):
                    ct = find_common_type([lk.dtype, rk.dtype])
                    if isinstance(ct, ExtensionDtype):
                        com_cls = ct.construct_array_type()
                        lk = com_cls._from_sequence(lk, dtype=ct, copy=False)
                    else:
                        lk = lk.astype(ct)
                if is_integer_dtype(rk.dtype) and is_float_dtype(lk.dtype):
                    with np.errstate(invalid="ignore"):
                        casted = lk.astype(rk.dtype)
                    mask = ~np.isnan(lk)
                    match = lk == casted
                    if not match[mask].all():
                        warnings.warn(
                            "You are merging on int and float columns where the float values are not equal to their int representation.",
                            UserWarning,
                            stacklevel=find_stack_level(),
                        )
                    continue
                if is_float_dtype(rk.dtype) and is_integer_dtype(lk.dtype):
                    with np.errstate(invalid="ignore"):
                        casted = rk.astype(lk.dtype)
                    mask = ~np.isnan(rk)
                    match = rk == casted
                    if not match[mask].all():
                        warnings.warn(
                            "You are merging on int and float columns where the float values are not equal to their int representation.",
                            UserWarning,
                            stacklevel=find_stack_level(),
                        )
                    continue
                if lib.infer_dtype(lk, skipna=False) == lib.infer_dtype(rk, skipna=False):
                    continue
            elif lk_is_object_or_string and is_bool_dtype(rk.dtype) or is_bool_dtype(
                lk.dtype
            ) and rk_is_object_or_string:
                pass
            elif lk_is_object_or_string and is_numeric_dtype(rk.dtype) or is_numeric_dtype(
                lk.dtype
            ) and rk_is_object_or_string:
                inferred_left = lib.infer_dtype(lk, skipna=False)
                inferred_right = lib.infer_dtype(rk, skipna=False)
                bool_types = ["integer", "mixed-integer", "boolean", "empty"]
                string_types = ["string", "unicode", "mixed", "bytes", "empty"]
                if (inferred_left in bool_types and inferred_right in bool_types):
                    pass
                elif inferred_left in string_types and inferred_right not in string_types or inferred_right in string_types and inferred_left not in string_types:
                    raise ValueError(msg)
            elif needs_i8_conversion(lk.dtype) and not needs_i8_conversion(rk.dtype):
                raise ValueError(msg)
            elif not needs_i8_conversion(lk.dtype) and needs_i8_conversion(rk.dtype):
                raise ValueError(msg)
            elif isinstance(lk.dtype, DatetimeTZDtype) and not isinstance(rk.dtype, DatetimeTZDtype):
                raise ValueError(msg)
            elif not isinstance(lk.dtype, DatetimeTZDtype) and isinstance(rk.dtype, DatetimeTZDtype):
                raise ValueError(msg)
            elif isinstance(lk.dtype, DatetimeTZDtype) and isinstance(rk.dtype, DatetimeTZDtype) or lk.dtype.kind == "M" and rk.dtype.kind == "M":
                continue
            elif lk.dtype.kind == "M" and rk.dtype.kind == "m":
                raise ValueError(msg)
            elif lk.dtype.kind == "m" and rk.dtype.kind == "M":
                raise ValueError(msg)
            elif is_object_dtype(lk.dtype) and is_object_dtype(rk.dtype):
                continue
            if name in self.left.columns:
                typ: Union[type, Any] = (
                    cast(Categorical, lk).categories.dtype
                    if lk_is_cat
                    else object
                )
                self.left = self.left.copy()
                self.left[name] = self.left[name].astype(typ)
            if name in self.right.columns:
                typ = (
                    cast(Categorical, rk).categories.dtype
                    if rk_is_cat
                    else object
                )
                self.right = self.right.copy()
                self.right[name] = self.right[name].astype(typ)

    @final
    def _validate_left_right_on(
        self,
        left_on: Optional[
            Union[Hashable, List[Hashable], List[List[Any]]]
        ],
        right_on: Optional[
            Union[Hashable, List[Hashable], List[List[Any]]]
        ],
    ) -> Tuple[Optional[List[Hashable]], Optional[List[Hashable]]]:
        left_on, right_on = func_rwv2fy39(
            com.maybe_make_list(left_on),
            com.maybe_make_list(right_on),
            sort=self.sort,
            how=self.how,
        )
        return left_on, right_on

    def _validate_tolerance(self, left_join_keys: List[ArrayLike]) -> None:
        pass

    def _validate_validate_kwd(self, validate: str) -> None:
        pass

    @final
    def _maybe_coerce_merge_keys(self) -> None:
        pass

    @final
    def _maybe_require_matching_dtypes(self, left_keys: List[ArrayLike], right_keys: List[ArrayLike]) -> None:
        pass

    def _indicator_pre_merge(self, left: DataFrame, right: DataFrame) -> Tuple[DataFrame, DataFrame]:
        pass

    def _indicator_post_merge(self, result: DataFrame) -> DataFrame:
        pass

    def _get_join_info(self) -> Tuple[Index, Optional[np.ndarray], Optional[np.ndarray]]:
        pass

    def _handle_anti_join(
        self,
        join_index: Index,
        left_indexer: Optional[np.ndarray],
        right_indexer: Optional[np.ndarray],
    ) -> Tuple[Index, Optional[np.ndarray], Optional[np.ndarray]]:
        pass

    def _get_join_indexers(self) -> Tuple[Index, Optional[np.ndarray], Optional[np.ndarray]]:
        pass

    def _get_join_indexers_non_asof(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        pass

    def _get_merge_keys(
        self,
    ) -> Tuple[
        List[ArrayLike],
        List[ArrayLike],
        List[Optional[Hashable]],
        List[Hashable],
        List[Hashable],
    ]:
        pass

    def _maybe_add_join_keys(
        self,
        result: DataFrame,
        left_indexer: Optional[np.ndarray],
        right_indexer: Optional[np.ndarray],
    ) -> None:
        pass

    def _maybe_restore_index_levels(self, result: DataFrame) -> None:
        pass


def func_90uuvznk(
    left_keys: List[ArrayLike],
    right_keys: List[ArrayLike],
    sort: bool = False,
    how: Literal["inner", "outer", "left", "right"] = "inner",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parameters
    ----------
    left_keys : list[ndarray, ExtensionArray, Index, Series]
    right_keys : list[ndarray, ExtensionArray, Index, Series]
    sort : bool, default False
    how : {'inner', 'outer', 'left', 'right'}, default 'inner'

    Returns
    -------
    np.ndarray[np.intp] or None
        Indexer into the left_keys.
    np.ndarray[np.intp] or None
        Indexer into the right_keys.
    """
    assert len(left_keys) == len(right_keys), "left_keys and right_keys must be the same length"
    left_n = len(left_keys[0])
    right_n = len(right_keys[0])
    if left_n == 0:
        if how in ["left", "inner"]:
            return _get_empty_indexer()
        elif not sort and how in ["right", "outer"]:
            return _get_no_sort_one_missing_indexer(right_n, True)
    elif right_n == 0:
        if how in ["right", "inner"]:
            return _get_empty_indexer()
        elif not sort and how in ["left", "outer"]:
            return _get_no_sort_one_missing_indexer(left_n, False)
    if len(left_keys) > 1:
        mapped = (
            _factorize_keys(left_keys[n], right_keys[n], sort=sort)
            for n in range(len(left_keys))
        )
        zipped = zip(*mapped)
        llab, rlab, shape = (list(x) for x in zipped)
        lkey, rkey = _get_join_keys(llab, rlab, tuple(shape), sort)
    else:
        lkey = left_keys[0]
        rkey = right_keys[0]
    left = Index(lkey)
    right = Index(rkey)
    if left.is_monotonic_increasing and right.is_monotonic_increasing and (
        left.is_unique or right.is_unique
    ):
        _, lidx, ridx = left.join(right, how=how, return_indexers=True, sort=sort)
    else:
        lidx, ridx = get_join_indexers_non_unique(left._values, right._values, sort, how)
    if lidx is not None and is_range_indexer(lidx, len(left)):
        lidx = None
    if ridx is not None and is_range_indexer(ridx, len(right)):
        ridx = None
    return lidx, ridx


def func_9ln44eyy(
    left_ax: Index,
    right_ax: Index,
    sort: bool = False,
    how: Literal["inner", "outer", "left", "right"] = "inner",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get join indexers for left and right.

    Parameters
    ----------
    left : ArrayLike
    right : ArrayLike
    sort : bool, default False
    how : {'inner', 'outer', 'left', 'right'}, default 'inner'

    Returns
    -------
    np.ndarray[np.intp]
        Indexer into left.
    np.ndarray[np.intp]
        Indexer into right.
    """
    lkey, rkey, count = _factorize_keys(left_ax, right_ax, sort=sort, how=how)
    if count == -1:
        return lkey, rkey
    if how == "left":
        lidx, ridx = libjoin.left_outer_join(lkey, rkey, count, sort=sort)
    elif how == "right":
        ridx, lidx = libjoin.left_outer_join(rkey, lkey, count, sort=sort)
    elif how == "inner":
        lidx, ridx = libjoin.inner_join(lkey, rkey, count, sort=sort)
    elif how == "outer":
        lidx, ridx = libjoin.full_outer_join(lkey, rkey, count)
    return lidx, ridx


def func_bc8mncw8(
    left: MultiIndex,
    right: MultiIndex,
    dropped_level_names: List[str],
    join_index: Index,
    lindexer: Optional[np.ndarray],
    rindexer: Optional[np.ndarray],
) -> Tuple[List[Index], List[np.ndarray], List[Hashable]]:
    """
    *this is an internal non-public method*

    Returns the levels, labels and names of a multi-index to multi-index join.
    Depending on the type of join, this method restores the appropriate
    dropped levels of the joined multi-index.
    The method relies on lindexer, rindexer which hold the index positions of
    left and right, where a join was feasible

    Parameters
    ----------
    left : MultiIndex
        left index
    right : MultiIndex
        right index
    dropped_level_names : str array
        list of non-common level names
    join_index : Index
        the index of the join between the
        common levels of left and right
    lindexer : np.ndarray[np.intp] or None
        left indexer
    rindexer : np.ndarray[np.intp] or None
        right indexer

    Returns
    -------
    levels : list of Index
        levels of combined multiindexes
    labels : List[np.ndarray]
        labels of combined multiindexes
    names : List[Hashable]
        names of combined multiindex levels
    """

    def func_tcr3z0po(index: Index) -> MultiIndex:
        if isinstance(index, MultiIndex):
            return index
        else:
            return MultiIndex.from_arrays([index._values], names=[index.name])

    join_index = func_tcr3z0po(join_index)
    join_levels = join_index.levels
    join_codes = join_index.codes
    join_names = join_index.names
    for dropped_level_name in dropped_level_names:
        if dropped_level_name in left.names:
            idx = left
            indexer = lindexer
        else:
            idx = right
            indexer = rindexer
        name_idx = idx.names.index(dropped_level_name)
        restore_levels = idx.levels[name_idx]
        codes = idx.codes[name_idx]
        if indexer is None:
            restore_codes = codes
        else:
            restore_codes = algos.take_nd(codes, indexer, fill_value=-1)
        join_levels = join_levels + [restore_levels]
        join_codes = join_codes + [restore_codes]
        join_names = join_names + [dropped_level_name]
    return join_levels, join_codes, join_names


class _OrderedMerge(_MergeOperation):
    _merge_type: str = "ordered_merge"

    def __init__(
        self,
        left: DataFrame,
        right: DataFrame,
        on: Optional[Union[Hashable, List[Hashable]]] = None,
        left_on: Optional[
            Union[Hashable, List[Hashable], List[List[Any]]]
        ] = None,
        right_on: Optional[
            Union[Hashable, List[Hashable], List[List[Any]]]
        ] = None,
        left_index: bool = False,
        right_index: bool = False,
        suffixes: Tuple[Optional[str], Optional[str]] = ("_x", "_y"),
        fill_method: Optional[Literal["ffill"]] = None,
        how: Literal["outer", "left", "right", "inner"] = "outer",
    ) -> None:
        self.fill_method: Optional[Literal["ffill"]] = fill_method
        super().__init__(
            left,
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            suffixes=suffixes,
            sort=True,
        )

    def get_result(self) -> DataFrame:
        join_index, left_indexer, right_indexer = self._get_join_info()
        if self.fill_method == "ffill":
            if left_indexer is None:
                left_join_indexer: Optional[np.ndarray] = None
            else:
                left_join_indexer = libjoin.ffill_indexer(left_indexer)
            if right_indexer is None:
                right_join_indexer: Optional[np.ndarray] = None
            else:
                right_join_indexer = libjoin.ffill_indexer(right_indexer)
        elif self.fill_method is None:
            left_join_indexer = left_indexer
            right_join_indexer = right_indexer
        else:
            raise ValueError("fill_method must be 'ffill' or None")
        result = self._reindex_and_concat(
            join_index, left_join_indexer, right_join_indexer
        )
        self._maybe_add_join_keys(result, left_indexer, right_indexer)
        return result


def func_vrwi2ovl(direction: Literal["backward", "forward", "nearest"]) -> Callable[..., Any]:
    name = f"asof_join_{direction}_on_X_by_Y"
    return getattr(libjoin, name, None)


class _AsOfMerge(_OrderedMerge):
    _merge_type: str = "asof_merge"

    def __init__(
        self,
        left: DataFrame,
        right: DataFrame,
        on: Optional[Union[Hashable, List[Hashable]]] = None,
        left_on: Optional[
            Union[Hashable, List[Hashable], List[List[Any]]]
        ] = None,
        right_on: Optional[
            Union[Hashable, List[Hashable], List[List[Any]]]
        ] = None,
        left_index: bool = False,
        right_index: bool = False,
        by: Optional[Union[Hashable, List[Hashable]]] = None,
        left_by: Optional[Union[Hashable, List[Hashable]]] = None,
        right_by: Optional[Union[Hashable, List[Hashable]]] = None,
        suffixes: Tuple[Optional[str], Optional[str]] = ("_x", "_y"),
        how: Literal["asof"] = "asof",
        tolerance: Optional[Union[int, datetime.timedelta]] = None,
        allow_exact_matches: bool = True,
        direction: Literal["backward", "forward", "nearest"] = "backward",
    ) -> None:
        self.by: Optional[Union[Hashable, List[Hashable]]] = by
        self.left_by: Optional[Union[Hashable, List[Hashable]]] = left_by
        self.right_by: Optional[Union[Hashable, List[Hashable]]] = right_by
        self.tolerance: Optional[Union[int, datetime.timedelta]] = tolerance
        self.allow_exact_matches: bool = allow_exact_matches
        self.direction: Literal["backward", "forward", "nearest"] = direction
        if self.direction not in ["backward", "forward", "nearest"]:
            raise MergeError(f"direction invalid: {self.direction}")
        if not is_bool(self.allow_exact_matches):
            msg = (
                f"allow_exact_matches must be boolean, passed {self.allow_exact_matches}"
            )
            raise MergeError(msg)
        super().__init__(
            left,
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            suffixes=suffixes,
            fill_method=None,
        )

    def _validate_left_right_on(
        self,
        left_on: Optional[
            Union[Hashable, List[Hashable], List[List[Any]]]
        ],
        right_on: Optional[
            Union[Hashable, List[Hashable], List[List[Any]]]
        ],
    ) -> Tuple[Optional[List[Hashable]], Optional[List[Hashable]]]:
        left_on, right_on = super()._validate_left_right_on(left_on, right_on)
        if len(left_on) != 1 and not self.left_index:
            raise MergeError("can only asof on a key for left")
        if len(right_on) != 1 and not self.right_index:
            raise MergeError("can only asof on a key for right")
        if self.left_index and isinstance(self.left.index, MultiIndex):
            raise MergeError("left can only have one index")
        if self.right_index and isinstance(self.right.index, MultiIndex):
            raise MergeError("right can only have one index")
        if self.by is not None:
            if self.left_by is not None or self.right_by is not None:
                raise MergeError("Can only pass by OR left_by and right_by")
            self.left_by = self.right_by = self.by
        if self.left_by is None and self.right_by is not None:
            raise MergeError("missing left_by")
        if self.left_by is not None and self.right_by is None:
            raise MergeError("missing right_by")
        if not self.left_index:
            left_on_0 = left_on[0]
            if isinstance(left_on_0, _known):
                lo_dtype = left_on_0.dtype
            else:
                lo_dtype = (
                    self.left._get_label_or_level_values(left_on_0).dtype
                    if left_on_0 in self.left.columns
                    else self.left.index.get_level_values(left_on_0).dtype
                )
        else:
            lo_dtype = self.left.index.dtype
        if not self.right_index:
            right_on_0 = right_on[0]
            if isinstance(right_on_0, _known):
                ro_dtype = right_on_0.dtype
            else:
                ro_dtype = (
                    self.right._get_label_or_level_values(right_on_0).dtype
                    if right_on_0 in self.right.columns
                    else self.right.index.get_level_values(right_on_0).dtype
                )
        else:
            ro_dtype = self.right.index.dtype
        if is_object_dtype(lo_dtype) or is_object_dtype(ro_dtype) or is_string_dtype(
            lo_dtype
        ) or is_string_dtype(ro_dtype):
            raise MergeError(
                f"Incompatible merge dtype, {lo_dtype!r} and {ro_dtype!r}, both sides must have numeric dtype"
            )
        if self.left_by is not None:
            if not is_list_like(self.left_by):
                self.left_by = [self.left_by]
            if not is_list_like(self.right_by):
                self.right_by = [self.right_by]
            if len(self.left_by) != len(self.right_by):
                raise MergeError("left_by and right_by must be the same length")
            left_on = self.left_by + list(left_on)
            right_on = self.right_by + list(right_on)
        return left_on, right_on

    def _validate_tolerance(self, left_join_keys: List[ArrayLike]) -> None:
        if self.tolerance is not None:
            if self.left_index:
                lt = self.left.index._values
            else:
                lt = left_join_keys[-1]
            msg = (
                f"incompatible tolerance {self.tolerance}, must be compat with type {lt.dtype!r}"
            )
            if needs_i8_conversion(lt.dtype) or (
                isinstance(lt, ArrowExtensionArray) and lt.dtype.kind in "mM"
            ):
                if not isinstance(self.tolerance, datetime.timedelta):
                    raise MergeError(msg)
                if self.tolerance < Timedelta(0):
                    raise MergeError("tolerance must be positive")
                if lt.dtype.kind in "mM":
                    if isinstance(lt, ArrowExtensionArray):
                        unit = lt.dtype.pyarrow_dtype.unit
                    else:
                        unit = ensure_wrapped_if_datetimelike(lt)._unit  # type: ignore[attr-defined]
                    self.tolerance = self.tolerance.as_unit(unit)
                self.tolerance = self.tolerance._value
            elif is_integer_dtype(lt.dtype):
                if not is_integer(self.tolerance):
                    raise MergeError(msg)
                if self.tolerance < 0:
                    raise MergeError("tolerance must be positive")
            elif is_float_dtype(lt.dtype):
                if not is_number(self.tolerance):
                    raise MergeError(msg)
                if self.tolerance < 0:
                    raise MergeError("tolerance must be positive")
            else:
                raise MergeError("key must be integer, timestamp or float")

    def _convert_values_for_libjoin(
        self, values: ArrayLike, side: str
    ) -> ArrayLike:
        if not Index(values).is_monotonic_increasing:
            if isna(values).any():
                raise ValueError(f"Merge keys contain null values on {side} side")
            raise ValueError(f"{side} keys must be sorted")
        if isinstance(values, ArrowExtensionArray):
            values = values._maybe_convert_datelike_array()
        if needs_i8_conversion(values.dtype):
            values = values.view("i8")
        elif isinstance(values, BaseMaskedArray):
            values = values._data
        elif isinstance(values, ExtensionArray):
            values = values.to_numpy()
        return values

    def _get_join_info(self) -> Tuple[Index, Optional[np.ndarray], Optional[np.ndarray]]:
        left_values: ArrayLike
        right_values: ArrayLike
        if self.left_index:
            left_values = self.left.index._values
        else:
            left_values = self.left_join_keys[-1]
        if self.right_index:
            right_values = self.right.index._values
        else:
            right_values = self.right_join_keys[-1]
        assert left_values.dtype == right_values.dtype, "Join keys must have the same dtype"

        tolerance = self.tolerance
        if tolerance is not None:
            if needs_i8_conversion(left_values.dtype) or (
                isinstance(left_values, ArrowExtensionArray) and left_values.dtype.kind in "mM"
            ):
                tolerance = Timedelta(tolerance)
                if left_values.dtype.kind in "mM":
                    if isinstance(left_values, ArrowExtensionArray):
                        unit = left_values.dtype.pyarrow_dtype.unit
                    else:
                        unit = ensure_wrapped_if_datetimelike(left_values)._unit  # type: ignore[attr-defined]
                    tolerance = tolerance.as_unit(unit)
                tolerance = tolerance._value
        left_values = self._convert_values_for_libjoin(left_values, "left")
        right_values = self._convert_values_for_libjoin(right_values, "right")
        if self.left_by is not None:
            if self.left_index and self.right_index:
                left_join_keys = self.left_join_keys
                right_join_keys = self.right_join_keys
            else:
                left_join_keys = self.left_join_keys[0:-1]
                right_join_keys = self.right_join_keys[0:-1]
            mapped = [_factorize_keys(left_join_keys[n], right_join_keys[n], sort=False) for n in range(len(left_join_keys))]
            if len(left_join_keys) == 1:
                left_by_values = mapped[0][0]
                right_by_values = mapped[0][1]
            else:
                arrs = [np.concatenate(m[:2]) for m in mapped]
                shape = tuple(m[2] for m in mapped)
                group_index = get_group_index(arrs, shape=shape, sort=False, xnull=False)
                left_len = len(left_join_keys[0])
                left_by_values = group_index[:left_len]
                right_by_values = group_index[left_len:]
            left_by_values = ensure_int64(left_by_values)
            right_by_values = ensure_int64(right_by_values)
            func = func_vrwi2ovl(self.direction)
            return func(
                left_values,
                right_values,
                left_by_values,
                right_by_values,
                self.allow_exact_matches,
                tolerance,
            )
        else:
            func = func_vrwi2ovl(self.direction)
            return func(
                left_values,
                right_values,
                None,
                None,
                self.allow_exact_matches,
                tolerance,
            )


def func_90uuvznk(left_keys: List[ArrayLike], right_keys: List[ArrayLike], sort: bool = False, how: Literal["inner", "outer", "left", "right"] = "inner") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parameters
    ----------
    left_keys : list[ndarray, ExtensionArray, Index, Series]
    right_keys : list[ndarray, ExtensionArray, Index, Series]
    sort : bool, default False
    how : {'inner', 'outer', 'left', 'right'}, default 'inner'

    Returns
    -------
    np.ndarray[np.intp] or None
        Indexer into the left_keys.
    np.ndarray[np.intp] or None
        Indexer into the right_keys.
    """
    assert len(left_keys) == len(right_keys), "left_keys and right_keys must be the same length"
    left_n = len(left_keys[0])
    right_n = len(right_keys[0])
    if left_n == 0:
        if how in ["left", "inner"]:
            return _get_empty_indexer()
        elif not sort and how in ["right", "outer"]:
            return _get_no_sort_one_missing_indexer(right_n, True)
    elif right_n == 0:
        if how in ["right", "inner"]:
            return _get_empty_indexer()
        elif not sort and how in ["left", "outer"]:
            return _get_no_sort_one_missing_indexer(left_n, False)
    if len(left_keys) > 1:
        mapped = (
            _factorize_keys(left_keys[n], right_keys[n], sort=sort)
            for n in range(len(left_keys))
        )
        zipped = zip(*mapped)
        llab, rlab, shape = (list(x) for x in zipped)
        lkey, rkey = _get_join_keys(llab, rlab, tuple(shape), sort)
    else:
        lkey = left_keys[0]
        rkey = right_keys[0]
    left = Index(lkey)
    right = Index(rkey)
    if left.is_monotonic_increasing and right.is_monotonic_increasing and (
        left.is_unique or right.is_unique
    ):
        _, lidx, ridx = left.join(right, how=how, return_indexers=True, sort=sort)
    else:
        lidx, ridx = get_join_indexers_non_unique(left._values, right._values, sort, how)
    if lidx is not None and is_range_indexer(lidx, len(left)):
        lidx = None
    if ridx is not None and is_range_indexer(ridx, len(right)):
        ridx = None
    return lidx, ridx


def func_9ln44eyy(left_ax: Index, right_ax: Index, sort: bool = False, how: Literal["inner", "outer", "left", "right"] = "inner") -> Tuple[np.ndarray, np.ndarray]:
    """
    Get join indexers for left and right.

    Parameters
    ----------
    left : ArrayLike
    right : ArrayLike
    sort : bool, default False
    how : {'inner', 'outer', 'left', 'right'}, default 'inner'

    Returns
    -------
    np.ndarray[np.intp]
        Indexer into left.
    np.ndarray[np.intp]
        Indexer into right.
    """
    lkey, rkey, count = _factorize_keys(left_ax, right_ax, sort=sort, how=how)
    if count == -1:
        return lkey, rkey
    if how == "left":
        lidx, ridx = libjoin.left_outer_join(lkey, rkey, count, sort=sort)
    elif how == "right":
        ridx, lidx = libjoin.left_outer_join(rkey, lkey, count, sort=sort)
    elif how == "inner":
        lidx, ridx = libjoin.inner_join(lkey, rkey, count, sort=sort)
    elif how == "outer":
        lidx, ridx = libjoin.full_outer_join(lkey, rkey, count)
    return lidx, ridx


def func_bc8mncw8(
    left: MultiIndex,
    right: MultiIndex,
    dropped_level_names: List[str],
    join_index: Index,
    lindexer: Optional[np.ndarray],
    rindexer: Optional[np.ndarray],
) -> Tuple[List[Index], List[np.ndarray], List[Hashable]]:
    """
    *this is an internal non-public method*

    [Docstring omitted for brevity]
    """
    def func_tcr3z0po(index: Index) -> MultiIndex:
        if isinstance(index, MultiIndex):
            return index
        else:
            return MultiIndex.from_arrays([index._values], names=[index.name])

    join_index = func_tcr3z0po(join_index)
    join_levels = join_index.levels
    join_codes = join_index.codes
    join_names = join_index.names
    for dropped_level_name in dropped_level_names:
        if dropped_level_name in left.names:
            idx = left
            indexer = lindexer
        else:
            idx = right
            indexer = rindexer
        name_idx = idx.names.index(dropped_level_name)
        restore_levels = idx.levels[name_idx]
        codes = idx.codes[name_idx]
        if indexer is None:
            restore_codes = codes
        else:
            restore_codes = algos.take_nd(codes, indexer, fill_value=-1)
        join_levels = join_levels + [restore_levels]
        join_codes = join_codes + [restore_codes]
        join_names = join_names + [dropped_level_name]
    return join_levels, join_codes, join_names


def func_326dgogl() -> Tuple[np.ndarray, np.ndarray]:
    """Return empty join indexers."""
    return np.array([], dtype=np.intp), np.array([], dtype=np.intp)


def func_tq4tzutc(n: int, left_missing: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return join indexers where all of one side is selected without sorting
    and none of the other side is selected.

    Parameters
    ----------
    n : int
        Length of indexers to create.
    left_missing : bool
        If True, the left indexer will contain only -1's.
        If False, the right indexer will contain only -1's.

    Returns
    -------
    np.ndarray[np.intp]
        Left indexer
    np.ndarray[np.intp]
        Right indexer
    """
    idx = np.arange(n, dtype=np.intp)
    idx_missing = np.full(shape=n, fill_value=-1, dtype=np.intp)
    if left_missing:
        return idx_missing, idx
    return idx, idx_missing


def func_pse8fw9c(
    left_ax: Index,
    right_ax: Index,
    join_keys: List[ArrayLike],
    sort: bool = False,
) -> Tuple[Index, Optional[np.ndarray], Optional[np.ndarray]]:
    if isinstance(right_ax, MultiIndex):
        lkey, rkey = func_rwv2fy39(join_keys, right_ax, sort=sort)
    else:
        lkey = join_keys[0]
        rkey = right_ax._values
    left_key, right_key, count = _factorize_keys(lkey, rkey, sort=sort)
    left_indexer, right_indexer = libjoin.left_outer_join(left_key, right_key, count, sort=sort)
    if sort or len(left_ax) != len(left_indexer):
        join_index = left_ax.take(left_indexer)
        return join_index, left_indexer, right_indexer
    return left_ax, None, right_indexer


def func_p0vpgnoh(
    lk: ArrayLike,
    rk: ArrayLike,
    sort: bool = True,
    how: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Encode left and right keys as enumerated types.

    [Docstring omitted for brevity]
    """
    if isinstance(lk.dtype, DatetimeTZDtype) and isinstance(rk.dtype, DatetimeTZDtype) or lib.is_np_dtype(lk.dtype, "M") and lib.is_np_dtype(rk.dtype, "M"):
        lk, _ = cast(DatetimeArray, lk)._ensure_matching_resos(rk)
        lk = cast(DatetimeArray, lk)._ndarray
        rk = cast(DatetimeArray, rk)._ndarray
    elif isinstance(lk.dtype, CategoricalDtype) and isinstance(rk.dtype, CategoricalDtype) and lk.dtype == rk.dtype:
        assert isinstance(lk, Categorical)
        assert isinstance(rk, Categorical)
        rk = lk._encode_with_my_categories(rk)
        lk = ensure_int64(lk.codes)
        rk = ensure_int64(rk.codes)
    elif isinstance(lk, ExtensionArray) and lk.dtype == rk.dtype:
        if (isinstance(lk.dtype, ArrowDtype) and is_string_dtype(lk.dtype)) or (
            isinstance(lk.dtype, StringDtype) and lk.dtype.storage == "pyarrow"
        ):
            import pyarrow as pa
            import pyarrow.compute as pc

            len_lk = len(lk)
            lk = lk._pa_array
            rk = rk._pa_array
            dc = pa.chunked_array(lk.chunks + rk.chunks).combine_chunks().dictionary_encode()
            llab = pc.fill_null(dc.indices[:len_lk], -1).to_numpy().astype(np.intp, copy=False)
            rlab = pc.fill_null(dc.indices[len_lk:], -1).to_numpy().astype(np.intp, copy=False)
            count = len(dc.dictionary)
            if dc.null_count > 0:
                lmask = llab == -1
                lany = lmask.any()
                rmask = rlab == -1
                rany = rmask.any()
                if lany:
                    llab = np.where(lmask, count, llab)
                if rany:
                    rlab = np.where(rmask, count, rlab)
                count += 1
            return llab, rlab, count
        if not isinstance(lk, BaseMaskedArray) and not (
            isinstance(lk.dtype, ArrowDtype)
            and (is_numeric_dtype(lk.dtype.numpy_dtype) or is_string_dtype(lk.dtype) and not sort)
        ):
            lk, _ = lk._values_for_factorize()
            rk, _ = rk._values_for_factorize()
    if needs_i8_conversion(lk.dtype) and lk.dtype == rk.dtype:
        lk = np.asarray(lk, dtype=np.int64)
        rk = np.asarray(rk, dtype=np.int64)
    klass, lk, rk = _convert_arrays_and_get_rizer_klass(lk, rk)
    rizer = klass(max(len(lk), len(rk)), uses_mask=isinstance(rk, (BaseMaskedArray, ArrowExtensionArray)))
    if isinstance(lk, BaseMaskedArray):
        assert isinstance(rk, BaseMaskedArray)
        lk_data, lk_mask = lk._data, lk._mask
        rk_data, rk_mask = rk._data, rk._mask
    elif isinstance(lk, ArrowExtensionArray):
        assert isinstance(rk, ArrowExtensionArray)
        lk_data = lk.to_numpy(na_value=1, dtype=lk.dtype.numpy_dtype)
        rk_data = rk.to_numpy(na_value=1, dtype=lk.dtype.numpy_dtype)
        lk_mask, rk_mask = lk.isna(), rk.isna()
    else:
        lk_data, rk_data = lk, rk
        lk_mask, rk_mask = None, None
    hash_join_available = (how == "inner" and not sort and lk.dtype.kind in "iufb")
    if hash_join_available:
        rlab = rizer.factorize(rk_data, mask=rk_mask)
        if rizer.get_count() == len(rlab):
            ridx, lidx = rizer.hash_inner_join(lk_data, lk_mask)
            return lidx, ridx, -1
        else:
            llab = rizer.factorize(lk_data, mask=lk_mask)
    else:
        llab = rizer.factorize(lk_data, mask=lk_mask)
        rlab = rizer.factorize(rk_data, mask=rk_mask)
    assert llab.dtype == np.dtype(np.intp), llab.dtype
    assert rlab.dtype == np.dtype(np.intp), rlab.dtype
    count = rizer.get_count()
    if sort:
        uniques = rizer.uniques.to_array()
        llab, rlab = _sort_labels(uniques, llab, rlab)
    lmask = llab == -1
    lany = lmask.any()
    rmask = rlab == -1
    rany = rmask.any()
    if lany or rany:
        if lany:
            llab = np.where(lmask, count, llab)
        if rany:
            rlab = np.where(rmask, count, rlab)
        count += 1
    return llab, rlab, count


def func_wobflk5l(lk: ArrayLike, rk: ArrayLike) -> Tuple[type, ArrayLike, ArrayLike]:
    if is_numeric_dtype(lk.dtype):
        if lk.dtype != rk.dtype:
            dtype = find_common_type([lk.dtype, rk.dtype])
            if isinstance(dtype, ExtensionDtype):
                cls = dtype.construct_array_type()
                if not isinstance(lk, ExtensionArray):
                    lk = cast(ExtensionArray, cls._from_sequence(lk, dtype=dtype, copy=False))
                else:
                    lk = lk.astype(dtype, copy=False)
                if not isinstance(rk, ExtensionArray):
                    rk = cast(ExtensionArray, cls._from_sequence(rk, dtype=dtype, copy=False))
                else:
                    rk = rk.astype(dtype, copy=False)
            else:
                lk = lk.astype(dtype, copy=False)
                rk = rk.astype(dtype, copy=False)
        if isinstance(lk, BaseMaskedArray):
            klass = _factorizers[lk.dtype.type]
        elif isinstance(lk.dtype, ArrowDtype):
            klass = _factorizers[lk.dtype.numpy_dtype.type]
        else:
            klass = _factorizers[lk.dtype.type]
    else:
        klass = libhashtable.ObjectFactorizer
        lk = ensure_object(lk)
        rk = ensure_object(rk)
    return klass, lk, rk


def func_7p2g9ix1(uniques: ArrayLike, left: ArrayLike, right: ArrayLike, sort: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    llab, rlab = _sort_labels(uniques, left, right)
    return llab, rlab


def func_8dtwzjkv(
    llab: List[np.ndarray],
    rlab: List[np.ndarray],
    shape: Tuple[int, ...],
    sort: bool,
) -> Tuple[ArrayLike, ArrayLike]:
    nlev = next(lev for lev in range(len(shape), 0, -1) if not is_int64_overflow_possible(shape[:lev]))
    stride = np.prod(shape[1:nlev], dtype="i8")
    lkey = stride * llab[0].astype("i8", subok=False, copy=False)
    rkey = stride * rlab[0].astype("i8", subok=False, copy=False)
    for i in range(1, nlev):
        with np.errstate(divide="ignore"):
            stride //= shape[i]
        lkey += llab[i] * stride
        rkey += rlab[i] * stride
    if nlev == len(shape):
        return lkey, rkey
    lkey, rkey, count = func_p0vpgnoh(lkey, rkey, sort=sort)
    llab = [lkey] + llab[nlev:]
    rlab = [rkey] + rlab[nlev:]
    shape = (count,) + shape[nlev:]
    return func_8dtwzjkv(llab, rlab, shape, sort)


def func_ex4k80vh(lname: Any, rname: Any) -> bool:
    if not isinstance(lname, str) or not isinstance(rname, str):
        return True
    return lname == rname


def func_q1cjr8t7(x: Any) -> bool:
    return x is not None and com.any_not_none(*x)


def func_v4y2lnvs(obj: Any) -> DataFrame:
    if isinstance(obj, ABCDataFrame):
        return obj
    elif isinstance(obj, ABCSeries):
        if obj.name is None:
            raise ValueError("Cannot merge a Series without a name")
        return obj.to_frame()
    else:
        raise TypeError(
            f"Can only merge Series or DataFrame objects, a {type(obj)} was passed"
        )


def func_79qwl9ih(left: Index, right: Index, suffixes: Tuple[Optional[str], Optional[str]]) -> Tuple[Index, Index]:
    """
    Suffixes type validation.

    If two indices overlap, add suffixes to overlapping entries.

    If corresponding suffix is empty, the entry is simply converted to string.

    """
    if not is_list_like(suffixes, allow_sets=False) or isinstance(suffixes, dict):
        raise TypeError(
            f"Passing 'suffixes' as a {type(suffixes)}, is not supported. Provide 'suffixes' as a tuple instead."
        )
    to_rename = left.intersection(right)
    if len(to_rename) == 0:
        return left, right
    lsuffix, rsuffix = suffixes
    if not lsuffix and not rsuffix:
        raise ValueError(f"columns overlap but no suffix specified: {to_rename}")

    def renamer(x: Hashable, suffix: Optional[str]) -> Hashable:
        """
        Rename the left and right indices.

        If there is overlap, and suffix is not None, add
        suffix, otherwise, leave it as-is.

        Parameters
        ----------
        x : original column name
        suffix : str or None

        Returns
        -------
        x : renamed column name
        """
        if x in to_rename and suffix is not None:
            return f"{x}{suffix}"
        return x

    lrenamer = partial(renamer, suffix=lsuffix)
    rrenamer = partial(renamer, suffix=rsuffix)
    llabels = left._transform_index(lrenamer)
    rlabels = right._transform_index(rrenamer)
    dups: List[Hashable] = []
    if not llabels.is_unique:
        dups.extend(llabels[llabels.duplicated() & ~left.duplicated()].tolist())
    if not rlabels.is_unique:
        dups.extend(rrabels[rrabels.duplicated() & ~right.duplicated()].tolist())
    if dups:
        raise MergeError(
            f"Passing 'suffixes' which cause duplicate columns {set(dups)} is not allowed."
        )
    return llabels, rlabels


# Placeholder implementations for missing functions and classes
def _validate_operand(obj: Union[DataFrame, Series]) -> DataFrame:
    if isinstance(obj, Series):
        if obj.name is None:
            raise ValueError("Cannot merge a Series without a name")
        return obj.to_frame()
    elif isinstance(obj, DataFrame):
        return obj
    else:
        raise TypeError(f"Can only merge Series or DataFrame objects, a {type(obj)} was passed")


def _cross_merge(
    left: DataFrame,
    right: DataFrame,
    on: Optional[Union[Hashable, List[Hashable]]],
    left_on: Optional[Union[Hashable, List[Hashable], List[List[Any]]]],
    right_on: Optional[Union[Hashable, List[Hashable], List[List[Any]]]],
    left_index: bool,
    right_index: bool,
    sort: bool,
    suffixes: Tuple[Optional[str], Optional[str]],
    indicator: Union[bool, str],
    validate: Optional[str],
) -> DataFrame:
    return func_1br7z7i8(left, right, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)


def _get_empty_indexer() -> Tuple[np.ndarray, np.ndarray]:
    return np.array([], dtype=np.intp), np.array([], dtype=np.intp)


def _get_no_sort_one_missing_indexer(n: int, left_missing: bool) -> Tuple[np.ndarray, np.ndarray]:
    return func_tq4tzutc(n, left_missing)


def _factorize_keys(lk: ArrayLike, rk: ArrayLike, sort: bool = True, how: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    return func_p0vpgnoh(lk, rk, sort=sort, how=how)


def get_join_indexers_non_unique(left: ArrayLike, right: ArrayLike, sort: bool, how: Literal["inner", "outer", "left", "right"]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    return func_90uuvznk(left, right, sort=sort, how=how)


def _get_join_keys(llab: List[np.ndarray], rlab: List[np.ndarray], shape: Tuple[int, ...], sort: bool) -> Tuple[np.ndarray, np.ndarray]:
    return func_8dtwzjkv(llab, rlab, shape, sort)


def get_group_index(arrs: List[np.ndarray], shape: Tuple[int, ...], sort: bool, xnull: bool) -> np.ndarray:
    # Placeholder for group_index implementation
    return np.array([])


def _items_overlap_with_suffix(
    left_axis: Index,
    right_axis: Index,
    suffixes: Tuple[Optional[str], Optional[str]],
) -> Tuple[Index, Index]:
    return func_79qwl9ih(left_axis, right_axis, suffixes)


def _should_fill(lname: Any, rname: Any) -> bool:
    return func_ex4k80vh(lname, rname)


def _convert_arrays_and_get_rizer_klass(lk: ArrayLike, rk: ArrayLike) -> Tuple[type, ArrayLike, ArrayLike]:
    return func_wobflk5l(lk, rk)


def _sort_labels(uniques: ArrayLike, llab: np.ndarray, rlab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return func_7p2g9ix1(uniques, llab, rlab)


def _get_join_keys(llab: List[np.ndarray], rlab: List[np.ndarray], shape: Tuple[int, ...], sort: bool) -> Tuple[np.ndarray, np.ndarray]:
    return func_8dtwzjkv(llab, rlab, shape, sort)


def _create_join_index(left_ax: Index, right_ax: Index, indexer: Optional[np.ndarray], how: str) -> Index:
    return func_pse8fw9c(left_ax, right_ax, [], sort=False)[0]


def _should_only_fill(lname: Any, rname: Any) -> bool:
    return func_ex4k80vh(lname, rname)
