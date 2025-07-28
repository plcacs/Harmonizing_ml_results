#!/usr/bin/env python
from __future__ import annotations
import inspect
import re
import warnings
import weakref
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import numpy as np
from pandas._libs import NaT, internals as libinternals, lib
from pandas._libs.internals import BlockPlacement, BlockValuesRefs
from pandas._libs.missing import NA
from pandas._typing import (
    ArrayLike,
    AxisInt,
    DtypeBackend,
    DtypeObj,
    FillnaOptions,
    IgnoreRaise,
    InterpolateOptions,
    QuantileInterpolation,
    Self,
    Shape,
    npt,
)
from pandas.errors import AbstractMethodError, OutOfBoundsDatetime
from pandas.util._decorators import cache_readonly, final
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.astype import astype_array_safe, astype_is_view
from pandas.core.dtypes.cast import LossySetitemError, can_hold_element, convert_dtypes, find_result_type, np_can_hold_element
from pandas.core.dtypes.common import is_1d_only_ea_dtype, is_float_dtype, is_integer_dtype, is_list_like, is_scalar, is_string_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype, ExtensionDtype, IntervalDtype, NumpyEADtype, PeriodDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCNumpyExtensionArray, ABCSeries
from pandas.core.dtypes.inference import is_re
from pandas.core.dtypes.missing import is_valid_na_for_dtype, isna, na_value_for_dtype
from pandas.core import missing
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
    extract_bool_array,
    putmask_inplace,
    putmask_without_repeat,
    setitem_datetimelike_compat,
    validate_putmask,
)
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.replace import compare_or_regex_search, replace_regex, should_use_regex
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays import (
    DatetimeArray,
    ExtensionArray,
    IntervalArray,
    NumpyExtensionArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation import expressions
from pandas.core.construction import ensure_wrapped_if_datetimelike, extract_array
from pandas.core.indexers import check_setitem_lengths
from pandas.core.indexes.base import get_values_for_csv

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Sequence
    from pandas.core.api import Index
    from pandas.core.arrays._mixins import NDArrayBackedExtensionArray

_dtype_obj = np.dtype("object")


class Block(PandasObject, libinternals.Block):
    """
    Canonical n-dimensional unit of homogeneous dtype contained in a pandas
    data structure

    Index-ignorant; let the container take care of that
    """
    __slots__ = ()
    is_numeric: bool = False

    @final
    @cache_readonly
    def _validate_ndim(self) -> bool:
        """
        We validate dimension for blocks that can hold 2D values, which for now
        means numpy dtypes or DatetimeTZDtype.
        """
        dtype = self.dtype
        return not isinstance(dtype, ExtensionDtype) or isinstance(dtype, DatetimeTZDtype)

    @final
    @cache_readonly
    def is_object(self) -> bool:
        return self.values.dtype == _dtype_obj

    @final
    @cache_readonly
    def is_extension(self) -> bool:
        return not lib.is_np_dtype(self.values.dtype)

    @final
    @cache_readonly
    def _can_consolidate(self) -> bool:
        return not self.is_extension

    @final
    @cache_readonly
    def _consolidate_key(self) -> Tuple[bool, str]:
        return (self._can_consolidate, self.dtype.name)

    @final
    @cache_readonly
    def _can_hold_na(self) -> bool:
        """
        Can we store NA values in this Block?
        """
        dtype = self.dtype
        if isinstance(dtype, np.dtype):
            return dtype.kind not in "iub"
        return dtype._can_hold_na

    @final
    @property
    def is_bool(self) -> bool:
        """
        We can be bool if a) we are bool dtype or b) object dtype with bool objects.
        """
        return self.values.dtype == np.dtype(bool)

    @final
    def external_values(self) -> np.ndarray:
        return external_values(self.values)

    @final
    @cache_readonly
    def fill_value(self) -> Any:
        return na_value_for_dtype(self.dtype, compat=False)

    @final
    def _standardize_fill_value(self, value: Any) -> Any:
        if self.dtype != _dtype_obj and is_valid_na_for_dtype(value, self.dtype):
            value = self.fill_value
        return value

    @property
    def mgr_locs(self) -> Any:
        return self._mgr_locs

    @mgr_locs.setter
    def mgr_locs(self, new_mgr_locs: Any) -> None:
        self._mgr_locs = new_mgr_locs

    @final
    def make_block(
        self, values: Union[np.ndarray, ExtensionArray], placement: Optional[Any] = None, refs: Any = None
    ) -> Block:
        """
        Create a new block, with type inference propagate any values that are
        not specified
        """
        if placement is None:
            placement = self._mgr_locs
        if self.is_extension:
            values = ensure_block_shape(values, ndim=self.ndim)
        return new_block(values, placement=placement, ndim=self.ndim, refs=refs)

    @final
    def make_block_same_class(
        self, values: Union[np.ndarray, ExtensionArray], placement: Optional[Any] = None, refs: Any = None
    ) -> Block:
        """Wrap given values in a block of same type as self."""
        if placement is None:
            placement = self._mgr_locs
        return type(self)(values, placement=placement, ndim=self.ndim, refs=refs)

    @final
    def __repr__(self) -> str:
        name = type(self).__name__
        if self.ndim == 1:
            result = f"{name}: {len(self)} dtype: {self.dtype}"
        else:
            shape = " x ".join([str(s) for s in self.shape])
            result = f"{name}: {self.mgr_locs.indexer}, {shape}, dtype: {self.dtype}"
        return result

    @final
    def __len__(self) -> int:
        return len(self.values)

    @final
    def slice_block_columns(self, slc: Any) -> Block:
        """
        Perform __getitem__-like, return result as block.
        """
        new_mgr_locs = self._mgr_locs[slc]
        new_values = self._slice(slc)
        refs = self.refs
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=refs)

    @final
    def take_block_columns(self, indices: Any) -> Block:
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
        new_mgr_locs = self._mgr_locs[indices]
        new_values = self._slice(indices)
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=None)

    @final
    def getitem_block_columns(self, slicer: Any, new_mgr_locs: Any, ref_inplace_op: bool = False) -> Block:
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
        new_values = self._slice(slicer)
        refs = self.refs if (not ref_inplace_op or self.refs.has_reference()) else None
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=refs)

    @final
    def _can_hold_element(self, element: Any) -> bool:
        """require the same dtype as ourselves"""
        element = extract_array(element, extract_numpy=True)
        return can_hold_element(self.values, element)

    @final
    def should_store(self, value: Any) -> bool:
        """
        Should we set self.values[indexer] = value inplace or do we need to cast?
        """
        return value.dtype == self.dtype

    @final
    def apply(self, func: Callable[[Any], Any], **kwargs: Any) -> List[Block]:
        """
        apply the function to my values; return a block if we are not
        one
        """
        result = func(self.values, **kwargs)
        result = maybe_coerce_values(result)
        return self._split_op_result(result)

    @final
    def reduce(self, func: Callable[[Any], Any]) -> List[Block]:
        assert self.ndim == 2
        result = func(self.values)
        if self.values.ndim == 1:
            res_values = result
        else:
            res_values = result.reshape(-1, 1)
        nb = self.make_block(res_values)
        return [nb]

    @final
    def _split_op_result(self, result: Any) -> List[Block]:
        if result.ndim > 1 and isinstance(result.dtype, ExtensionDtype):
            nbs: List[Block] = []
            for i, loc in enumerate(self._mgr_locs):
                if not is_1d_only_ea_dtype(result.dtype):
                    vals = result[i : i + 1]
                else:
                    vals = result[i]
                bp = BlockPlacement(loc)
                block = self.make_block(values=vals, placement=bp)
                nbs.append(block)
            return nbs
        nb = self.make_block(result)
        return [nb]

    @final
    def _split(self) -> Iterator[Block]:
        """
        Split a block into a list of single-column blocks.
        """
        assert self.ndim == 2
        for i, ref_loc in enumerate(self._mgr_locs):
            vals = self.values[slice(i, i + 1)]
            bp = BlockPlacement(ref_loc)
            nb = type(self)(vals, placement=bp, ndim=2, refs=self.refs)
            yield nb

    @final
    def split_and_operate(self, func: Callable[[Block, Any], List[Block]], *args: Any, **kwargs: Any) -> List[Block]:
        """
        Split the block and apply func column-by-column.
        """
        assert self.ndim == 2 and self.shape[0] != 1
        res_blocks: List[Block] = []
        for nb in self._split():
            rbs = func(nb, *args, **kwargs)
            res_blocks.extend(rbs)
        return res_blocks

    @final
    def coerce_to_target_dtype(self, other: Any, raise_on_upcast: bool) -> Block:
        """
        coerce the current block to a dtype compat for other
        we will return a block, possibly object, and not raise
        """
        new_dtype = find_result_type(self.values.dtype, other)
        if new_dtype == self.dtype:
            raise AssertionError(
                "Something has gone wrong, please report a bug at https://github.com/pandas-dev/pandas/issues"
            )
        if is_scalar(other) and is_integer_dtype(self.values.dtype) and isna(other) and (other is not NaT) and (
            not (isinstance(other, (np.datetime64, np.timedelta64)) and np.isnat(other))
        ):
            raise_on_upcast = False
        elif isinstance(other, np.ndarray) and other.ndim == 1 and is_integer_dtype(self.values.dtype) and is_float_dtype(other.dtype) and lib.has_only_ints_or_nan(other):
            raise_on_upcast = False
        if raise_on_upcast:
            raise TypeError(f"Invalid value '{other}' for dtype '{self.values.dtype}'")
        if self.values.dtype == new_dtype:
            raise AssertionError(
                f"Did not expect new dtype {new_dtype} to equal self.dtype {self.values.dtype}. Please report a bug at https://github.com/pandas-dev/pandas/issues."
            )
        try:
            return self.astype(new_dtype)
        except OutOfBoundsDatetime as err:
            raise OutOfBoundsDatetime(
                f"Incompatible (high-resolution) value for dtype='{self.dtype}'. Explicitly cast before operating."
            ) from err

    @final
    def convert(self) -> List[Block]:
        """
        Attempt to coerce any object types to better types. Return a copy
        of the block (if copy = True).
        """
        if not self.is_object:
            return [self.copy(deep=False)]
        if self.ndim != 1 and self.shape[0] != 1:
            blocks = self.split_and_operate(Block.convert)
            if all((blk.dtype.kind == "O" for blk in blocks)):
                return [self.copy(deep=False)]
            return blocks
        values = self.values
        if values.ndim == 2:
            values = values[0]
        res_values = lib.maybe_convert_objects(values, convert_non_numeric=True)
        refs: Any = None
        if res_values is values or (isinstance(res_values, NumpyExtensionArray) and res_values._ndarray is values):
            refs = self.refs
        res_values = ensure_block_shape(res_values, self.ndim)
        res_values = maybe_coerce_values(res_values)
        return [self.make_block(res_values, refs=refs)]

    def convert_dtypes(
        self,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
        convert_floating: bool = True,
        dtype_backend: Union[str, DtypeBackend] = "numpy_nullable",
    ) -> List[Block]:
        if infer_objects and self.is_object:
            blks = self.convert()
        else:
            blks = [self]
        if not any([convert_floating, convert_integer, convert_boolean, convert_string]):
            return [b.copy(deep=False) for b in blks]
        rbs: List[Block] = []
        for blk in blks:
            sub_blks = [blk] if (blk.ndim == 1 or self.shape[0] == 1) else list(blk._split())
            dtypes = [
                convert_dtypes(b.values, convert_string, convert_integer, convert_boolean, convert_floating, infer_objects, dtype_backend)
                for b in sub_blks
            ]
            if all((dtype == self.dtype for dtype in dtypes)):
                rbs.append(blk.copy(deep=False))
                continue
            for dtype, b in zip(dtypes, sub_blks):
                rbs.append(b.astype(dtype=dtype, squeeze=b.ndim != 1))
        return rbs

    @final
    @cache_readonly
    def dtype(self) -> Any:
        return self.values.dtype

    @final
    def astype(self, dtype: Union[np.dtype, ExtensionDtype], errors: str = "raise", squeeze: bool = False) -> Block:
        """
        Coerce to the new dtype.
        """
        values = self.values
        if squeeze and values.ndim == 2 and is_1d_only_ea_dtype(dtype):
            if values.shape[0] != 1:
                raise ValueError("Can not squeeze with more than one column.")
            values = values[0, :]
        new_values = astype_array_safe(values, dtype, errors=errors)
        new_values = maybe_coerce_values(new_values)
        refs: Any = None
        if astype_is_view(values.dtype, new_values.dtype):
            refs = self.refs
        newb = self.make_block(new_values, refs=refs)
        if newb.shape != self.shape:
            raise TypeError(
                f"cannot set astype for dtype ({self.dtype.name} [{self.shape}]) to different shape ({newb.dtype.name} [{newb.shape}])"
            )
        return newb

    @final
    def get_values_for_csv(
        self,
        *,
        float_format: Any,
        date_format: Any,
        decimal: Any,
        na_rep: str = "nan",
        quoting: Any = None,
    ) -> Block:
        """convert to our native types format"""
        result = get_values_for_csv(
            self.values, na_rep=na_rep, quoting=quoting, float_format=float_format, date_format=date_format, decimal=decimal
        )
        return self.make_block(result)

    @final
    def copy(self, deep: bool = True) -> Block:
        """copy constructor"""
        values = self.values
        if deep:
            values = values.copy()
            refs: Any = None
        else:
            refs = self.refs
        return type(self)(values, placement=self._mgr_locs, ndim=self.ndim, refs=refs)

    def _maybe_copy(self, inplace: bool) -> Block:
        if inplace:
            deep = self.refs.has_reference()
            return self.copy(deep=deep)
        return self.copy()

    @final
    def _get_refs_and_copy(self, inplace: bool) -> Tuple[bool, Any]:
        refs: Any = None
        copy_flag = not inplace
        if inplace:
            if self.refs.has_reference():
                copy_flag = True
            else:
                refs = self.refs
        return (copy_flag, refs)

    @final
    def replace(self, to_replace: Any, value: Any, inplace: bool = False, mask: Optional[np.ndarray] = None) -> List[Block]:
        """
        replace the to_replace value with value, possible to create new
        blocks here this is just a call to putmask.
        """
        values = self.values
        if not self._can_hold_element(to_replace):
            return [self.copy(deep=False)]
        if mask is None:
            mask = missing.mask_missing(values, to_replace)
        if not mask.any():
            return [self.copy(deep=False)]
        elif self._can_hold_element(value) or (self.dtype == "string" and is_re(value)):
            blk = self._maybe_copy(inplace)
            putmask_inplace(blk.values, mask, value)
            return [blk]
        elif self.ndim == 1 or self.shape[0] == 1:
            if value is None or value is NA:
                blk = self.astype(np.dtype(object))
            else:
                blk = self.coerce_to_target_dtype(value, raise_on_upcast=False)
            return blk.replace(to_replace=to_replace, value=value, inplace=True, mask=mask)
        else:
            blocks: List[Block] = []
            for i, nb in enumerate(self._split()):
                blocks.extend(
                    type(self).replace(nb, to_replace=to_replace, value=value, inplace=True, mask=mask[i : i + 1])
                )
            return blocks

    @final
    def _replace_regex(self, to_replace: Any, value: Any, inplace: bool = False, mask: Optional[np.ndarray] = None) -> List[Block]:
        """
        Replace elements by the given value.
        """
        if not is_re(to_replace) and (not self._can_hold_element(to_replace)):
            return [self.copy(deep=False)]
        if is_re(to_replace) and self.dtype not in [object, "string"]:
            return [self.copy(deep=False)]
        if not (self._can_hold_element(value) or (self.dtype == "string" and is_re(value))):
            block = self.astype(np.dtype(object))
        else:
            block = self._maybe_copy(inplace)
        rx = re.compile(to_replace)
        replace_regex(block.values, rx, value, mask)  # type: ignore[arg-type]
        return [block]

    @final
    def replace_list(
        self, src_list: Sequence[Any], dest_list: Sequence[Any], inplace: bool = False, regex: bool = False
    ) -> List[Block]:
        """
        See BlockManager.replace_list docstring.
        """
        values = self.values
        pairs = [(x, y) for x, y in zip(src_list, dest_list) if self._can_hold_element(x) or (self.dtype == "string" and is_re(x))]
        if not len(pairs):
            return [self.copy(deep=False)]
        src_len = len(pairs) - 1
        if is_string_dtype(values.dtype):
            na_mask = ~isna(values)
            masks = (
                extract_bool_array(cast(ArrayLike, compare_or_regex_search(values, s[0], regex=regex, mask=na_mask)))
                for s in pairs
            )
        else:
            masks = (missing.mask_missing(values, s[0]) for s in pairs)
        if inplace:
            masks = list(masks)
        rb: List[Block] = [self]
        for i, ((src, dest), mask) in enumerate(zip(pairs, masks)):
            new_rb: List[Block] = []
            for blk_num, blk in enumerate(rb):
                if len(rb) == 1:
                    m = mask
                else:
                    mib = mask
                    assert not isinstance(mib, bool)
                    m = mib[blk_num : blk_num + 1]
                result = blk._replace_coerce(to_replace=src, value=dest, mask=m, inplace=inplace, regex=regex)
                if i != src_len:
                    for b in result:
                        ref = weakref.ref(b)
                        b.refs.referenced_blocks.pop(b.refs.referenced_blocks.index(ref))
                new_rb.extend(result)
            rb = new_rb
        return rb

    @final
    def _replace_coerce(self, to_replace: Any, value: Any, mask: np.ndarray, inplace: bool = True, regex: bool = False) -> List[Block]:
        """
        Replace value corresponding to the given boolean array with another
        value.
        """
        if should_use_regex(regex, to_replace):
            return self._replace_regex(to_replace, value, inplace=inplace, mask=mask)
        else:
            if value is None:
                if mask.any():
                    has_ref = self.refs.has_reference()
                    nb = self.astype(np.dtype(object))
                    if not inplace:
                        nb = nb.copy()
                    elif inplace and has_ref and nb.refs.has_reference():
                        nb = nb.copy()
                    putmask_inplace(nb.values, mask, value)
                    return [nb]
                return [self.copy(deep=False)]
            return self.replace(to_replace=to_replace, value=value, inplace=inplace, mask=mask)

    def _maybe_squeeze_arg(self, arg: Any) -> Any:
        """
        For compatibility with 1D-only ExtensionArrays.
        """
        return arg

    def _unwrap_setitem_indexer(self, indexer: Any) -> Any:
        """
        For compatibility with 1D-only ExtensionArrays.
        """
        return indexer

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.values.shape

    def iget(self, i: Any) -> Any:
        return self.values[i]

    def _slice(self, slicer: Any) -> Any:
        """return a slice of my values"""
        return self.values[slicer]

    def set_inplace(self, locs: Any, values: Any, copy: bool = False) -> None:
        """
        Modify block values in-place with new item value.
        """
        if copy:
            self.values = self.values.copy()
        self.values[locs] = values

    @final
    def take_nd(self, indexer: Any, axis: int, new_mgr_locs: Optional[Any] = None, fill_value: Any = lib.no_default) -> Block:
        """
        Take values according to indexer and return them as a block.
        """
        values = self.values
        if fill_value is lib.no_default:
            fill_value = self.fill_value
            allow_fill = False
        else:
            allow_fill = True
        new_values = algos.take_nd(values, indexer, axis=axis, allow_fill=allow_fill, fill_value=fill_value)
        if isinstance(self, ExtensionBlock):
            assert not (self.ndim == 1 and new_mgr_locs is None)
        assert not (axis == 0 and new_mgr_locs is None)
        if new_mgr_locs is None:
            new_mgr_locs = self._mgr_locs
        if new_values.dtype != self.dtype:
            return self.make_block(new_values, new_mgr_locs)
        else:
            return self.make_block_same_class(new_values, new_mgr_locs)

    def _unstack(self, unstacker: Any, fill_value: Any, new_placement: np.ndarray, needs_masking: Any) -> Tuple[List[Block], Any]:
        """
        Return a list of unstacked blocks of self
        """
        new_values, mask = unstacker.get_new_values(self.values.T, fill_value=fill_value)
        mask = mask.any(0)
        new_values = new_values.T[mask]
        new_placement = new_placement[mask]
        bp = BlockPlacement(new_placement)
        blocks = [new_block_2d(new_values, placement=bp)]
        return (blocks, mask)

    def setitem(self, indexer: Any, value: Any) -> Block:
        """
        Attempt self.values[indexer] = value, possibly creating a new array.
        """
        value = self._standardize_fill_value(value)
        values = self.values  # type: np.ndarray
        if self.ndim == 2:
            values = values.T
        check_setitem_lengths(indexer, value, values)
        if self.dtype != _dtype_obj:
            value = extract_array(value, extract_numpy=True)
        try:
            casted = np_can_hold_element(values.dtype, value)
        except LossySetitemError:
            nb = self.coerce_to_target_dtype(value, raise_on_upcast=True)
            return nb.setitem(indexer, value)
        else:
            if self.dtype == _dtype_obj:
                vi = values[indexer]
                if lib.is_list_like(vi):
                    casted = setitem_datetimelike_compat(values, len(vi), casted)
            self = self._maybe_copy(inplace=True)
            values = self.values.T  # type: np.ndarray
            if isinstance(casted, np.ndarray) and casted.ndim == 1 and (len(casted) == 1):
                casted = casted[0, ...]
            try:
                values[indexer] = casted
            except (TypeError, ValueError) as err:
                if is_list_like(casted):
                    raise ValueError("setting an array element with a sequence.") from err
                raise
        return self

    def putmask(self, mask: Any, new: Any) -> List[Block]:
        """
        putmask the data to the block; it is possible that we may create a
        new dtype of block
        """
        orig_mask = mask
        values = self.values  # type: np.ndarray
        mask, noop = validate_putmask(values.T, mask)
        assert not isinstance(new, (ABCIndex, ABCSeries, ABCDataFrame))
        if new is lib.no_default:
            new = self.fill_value
        new = self._standardize_fill_value(new)
        new = extract_array(new, extract_numpy=True)
        if noop:
            return [self.copy(deep=False)]
        try:
            casted = np_can_hold_element(values.dtype, new)
            self = self._maybe_copy(inplace=True)
            values = self.values  # type: np.ndarray
            putmask_without_repeat(values.T, mask, casted)
            return [self]
        except LossySetitemError:
            if self.ndim == 1 or self.shape[0] == 1:
                if not is_list_like(new):
                    return self.coerce_to_target_dtype(new, raise_on_upcast=True).putmask(mask, new)
                else:
                    indexer = mask.nonzero()[0]
                    nb = self.setitem(indexer, new[indexer])
                    return [nb]
            else:
                is_array = isinstance(new, np.ndarray)
                res_blocks: List[Block] = []
                for i, nb in enumerate(self._split()):
                    n = new
                    if is_array:
                        n = new[:, i : i + 1]
                    submask = orig_mask[:, i : i + 1]
                    rbs = nb.putmask(submask, n)
                    res_blocks.extend(rbs)
                return res_blocks

    def where(self, other: Any, cond: Any) -> List[Block]:
        """
        evaluate the block; return result block(s) from the result
        """
        assert cond.ndim == self.ndim
        assert not isinstance(other, (ABCIndex, ABCSeries, ABCDataFrame))
        transpose = self.ndim == 2
        cond = extract_bool_array(cond)
        values = self.values  # type: np.ndarray
        orig_other = other
        if transpose:
            values = values.T
        icond, noop = validate_putmask(values, ~cond)
        if noop:
            return [self.copy(deep=False)]
        if other is lib.no_default:
            other = self.fill_value
        other = self._standardize_fill_value(other)
        try:
            casted = np_can_hold_element(values.dtype, other)
        except (ValueError, TypeError, LossySetitemError):
            if self.ndim == 1 or self.shape[0] == 1:
                block = self.coerce_to_target_dtype(other, raise_on_upcast=False)
                return block.where(orig_other, cond)
            else:
                is_array = isinstance(other, (np.ndarray, ExtensionArray))
                res_blocks: List[Block] = []
                for i, nb in enumerate(self._split()):
                    oth = other
                    if is_array:
                        oth = other[:, i : i + 1]
                    submask = cond[:, i : i + 1]
                    rbs = nb.where(oth, submask)
                    res_blocks.extend(rbs)
                return res_blocks
        else:
            other = casted
            alt = setitem_datetimelike_compat(values, icond.sum(), other)
            if alt is not other:
                if is_list_like(other) and len(other) < len(values):
                    np.where(~icond, values, other)
                    raise NotImplementedError(
                        "This should not be reached; call to np.where above is expected to raise ValueError. Please report a bug at github.com/pandas-dev/pandas"
                    )
                result = values.copy()
                np.putmask(result, icond, alt)
            else:
                if is_list_like(other) and (not isinstance(other, np.ndarray)) and (len(other) == self.shape[-1]):
                    other = np.array(other).reshape(values.shape)
                result = expressions.where(~icond, values, other)
        if transpose:
            result = result.T
        return [self.make_block(result)]

    def fillna(self, value: Any, limit: Optional[int] = None, inplace: bool = False) -> List[Block]:
        """
        fillna on the block with the value. If we fail, then convert to
        block to hold objects instead and try again
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if not self._can_hold_na:
            noop = True
        else:
            mask = isna(self.values)
            mask, noop = validate_putmask(self.values, mask)
        if noop:
            return [self.copy(deep=False)]
        if limit is not None:
            mask[mask.cumsum(self.values.ndim - 1) > limit] = False
        if inplace:
            nbs = self.putmask(mask.T, value)
        else:
            nbs = self.where(value, ~mask.T)
        return extend_blocks(nbs)

    def pad_or_backfill(
        self, *, method: str, inplace: bool = False, limit: Optional[int] = None, limit_area: Any = None
    ) -> List[Block]:
        if not self._can_hold_na:
            return [self.copy(deep=False)]
        copy, refs = self._get_refs_and_copy(inplace)
        vals = cast(NumpyExtensionArray, self.array_values)
        new_values = vals.T._pad_or_backfill(method=method, limit=limit, limit_area=limit_area, copy=copy).T
        data = extract_array(new_values, extract_numpy=True)
        return [self.make_block_same_class(data, refs=refs)]

    @final
    def interpolate(
        self,
        *,
        method: str,
        index: Any,
        inplace: bool = False,
        limit: Optional[int] = None,
        limit_direction: str = "forward",
        limit_area: Any = None,
        **kwargs: Any,
    ) -> List[Block]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        if method == "asfreq":
            missing.clean_fill_method(method)
        if not self._can_hold_na:
            return [self.copy(deep=False)]
        copy, refs = self._get_refs_and_copy(inplace)
        new_values = self.array_values.interpolate(
            method=method,
            axis=self.ndim - 1,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            copy=copy,
            **kwargs,
        )
        data = extract_array(new_values, extract_numpy=True)
        return [self.make_block_same_class(data, refs=refs)]

    @final
    def diff(self, n: int) -> List[Block]:
        """return block for the diff of the values"""
        new_values = algos.diff(self.values.T, n, axis=0).T
        return [self.make_block(values=new_values)]

    def shift(self, periods: int, fill_value: Optional[Any] = None) -> List[Block]:
        """shift the block by periods, possibly upcast"""
        axis = self.ndim - 1
        if not lib.is_scalar(fill_value) and self.dtype != _dtype_obj:
            raise ValueError("fill_value must be a scalar")
        fill_value = self._standardize_fill_value(fill_value)
        try:
            casted = np_can_hold_element(self.dtype, fill_value)
        except LossySetitemError:
            nb = self.coerce_to_target_dtype(fill_value, raise_on_upcast=False)
            return nb.shift(periods, fill_value=fill_value)
        else:
            values = self.values  # type: np.ndarray
            new_values = shift(values, periods, axis, casted)
            return [self.make_block_same_class(new_values)]

    @final
    def quantile(self, qs: Any, interpolation: str = "linear") -> Block:
        """
        compute the quantiles of the
        """
        assert self.ndim == 2
        assert is_list_like(qs)
        result = quantile_compat(self.values, np.asarray(qs._values), interpolation)
        result = ensure_block_shape(result, ndim=2)
        return new_block_2d(result, placement=self._mgr_locs)

    @final
    def round(self, decimals: int) -> Block:
        """
        Rounds the values.
        """
        if not self.is_numeric or self.is_bool:
            return self.copy(deep=False)
        values = self.values.round(decimals)
        refs: Any = None
        if values is self.values:
            refs = self.refs
        return self.make_block_same_class(values, refs=refs)

    def delete(self, loc: Union[int, Sequence[int]]) -> List[Block]:
        """Deletes the locs from the block."""
        if not is_list_like(loc):
            loc = [loc]
        if self.ndim == 1:
            values = self.values  # type: np.ndarray
            values = np.delete(values, loc)
            mgr_locs = self._mgr_locs.delete(loc)
            return [type(self)(values, placement=mgr_locs, ndim=self.ndim)]
        if np.max(loc) >= self.values.shape[0]:
            raise IndexError
        loc = np.concatenate([loc, [self.values.shape[0]]])
        mgr_locs_arr = self._mgr_locs.as_array
        new_blocks: List[Block] = []
        previous_loc = -1
        refs: Any = self.refs if self.refs.has_reference() else None
        for idx in loc:
            if idx == previous_loc + 1:
                pass
            else:
                values = self.values[previous_loc + 1 : idx, :]
                locs = mgr_locs_arr[previous_loc + 1 : idx]
                nb = type(self)(values, placement=BlockPlacement(locs), ndim=self.ndim, refs=refs)
                new_blocks.append(nb)
            previous_loc = idx
        return new_blocks

    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""
        raise AbstractMethodError(self)

    @property
    def array_values(self) -> ExtensionArray:
        """
        The array that Series.array returns. Always an ExtensionArray.
        """
        raise AbstractMethodError(self)

    def get_values(self, dtype: Optional[Any] = None) -> Any:
        """
        return an internal format, currently just the ndarray
        """
        raise AbstractMethodError(self)


class EABackedBlock(Block):
    """
    Mixin for Block subclasses backed by ExtensionArray.
    """

    @final
    def shift(self, periods: int, fill_value: Optional[Any] = None) -> List[Block]:
        """
        Shift the block by `periods`.
        """
        new_values = self.values.T.shift(periods=periods, fill_value=fill_value).T
        return [self.make_block_same_class(new_values)]

    @final
    def setitem(self, indexer: Any, value: Any) -> Block:
        """
        Attempt self.values[indexer] = value, possibly creating a new array.
        """
        orig_indexer = indexer
        orig_value = value
        indexer = self._unwrap_setitem_indexer(indexer)
        value = self._maybe_squeeze_arg(value)
        values = self.values
        if values.ndim == 2:
            values = values.T
        check_setitem_lengths(indexer, value, values)
        try:
            values[indexer] = value
        except (ValueError, TypeError):
            if isinstance(self.dtype, IntervalDtype):
                nb = self.coerce_to_target_dtype(orig_value, raise_on_upcast=True)
                return nb.setitem(orig_indexer, orig_value)
            elif isinstance(self, NDArrayBackedExtensionBlock):
                nb = self.coerce_to_target_dtype(orig_value, raise_on_upcast=True)
                return nb.setitem(orig_indexer, orig_value)
            else:
                raise
        else:
            return self

    @final
    def where(self, other: Any, cond: Any) -> List[Block]:
        arr = self.values.T
        cond = extract_bool_array(cond)
        orig_other = other
        orig_cond = cond
        other = self._maybe_squeeze_arg(other)
        cond = self._maybe_squeeze_arg(cond)
        if other is lib.no_default:
            other = self.fill_value
        icond, noop = validate_putmask(arr, ~cond)
        if noop:
            return [self.copy(deep=False)]
        try:
            res_values = arr._where(cond, other).T
        except (ValueError, TypeError):
            if self.ndim == 1 or self.shape[0] == 1:
                if isinstance(self.dtype, (IntervalDtype, StringDtype)):
                    blk = self.coerce_to_target_dtype(orig_other, raise_on_upcast=False)
                    if self.ndim == 2 and isinstance(orig_cond, np.ndarray) and (orig_cond.ndim == 1) and (not is_1d_only_ea_dtype(blk.dtype)):
                        orig_cond = orig_cond[:, None]
                    return blk.where(orig_other, orig_cond)
                elif isinstance(self, NDArrayBackedExtensionBlock):
                    blk = self.coerce_to_target_dtype(orig_other, raise_on_upcast=False)
                    return blk.where(orig_other, orig_cond)
                else:
                    raise
            else:
                is_array = isinstance(orig_other, (np.ndarray, ExtensionArray))
                res_blocks: List[Block] = []
                for i, nb in enumerate(self._split()):
                    n = orig_other
                    if is_array:
                        n = orig_other[:, i : i + 1]
                    submask = orig_cond[:, i : i + 1]
                    rbs = nb.where(n, submask)
                    res_blocks.extend(rbs)
                return res_blocks
        nb = self.make_block_same_class(res_values)
        return [nb]

    @final
    def putmask(self, mask: Any, new: Any) -> List[Block]:
        mask = extract_bool_array(mask)
        if new is lib.no_default:
            new = self.fill_value
        orig_new = new
        orig_mask = mask
        new = self._maybe_squeeze_arg(new)
        mask = self._maybe_squeeze_arg(mask)
        if not mask.any():
            return [self.copy(deep=False)]
        self = self._maybe_copy(inplace=True)
        values = self.values
        if values.ndim == 2:
            values = values.T
        try:
            values._putmask(mask, new)
        except (TypeError, ValueError):
            if self.ndim == 1 or self.shape[0] == 1:
                if isinstance(self.dtype, IntervalDtype):
                    blk = self.coerce_to_target_dtype(orig_new, raise_on_upcast=True)
                    return blk.putmask(orig_mask, orig_new)
                elif isinstance(self, NDArrayBackedExtensionBlock):
                    blk = self.coerce_to_target_dtype(orig_new, raise_on_upcast=True)
                    return blk.putmask(orig_mask, orig_new)
                else:
                    raise
            else:
                is_array = isinstance(orig_new, (np.ndarray, ExtensionArray))
                res_blocks: List[Block] = []
                for i, nb in enumerate(self._split()):
                    n = orig_new
                    if is_array:
                        n = orig_new[:, i : i + 1]
                    submask = orig_mask[:, i : i + 1]
                    rbs = nb.putmask(submask, n)
                    res_blocks.extend(rbs)
                return res_blocks
        return [self]

    @final
    def delete(self, loc: Union[int, Sequence[int]]) -> List[Block]:
        if self.ndim == 1:
            values = self.values.delete(loc)
            mgr_locs = self._mgr_locs.delete(loc)
            return [type(self)(values, placement=mgr_locs, ndim=self.ndim)]
        elif self.values.ndim == 1:
            return []
        return super().delete(loc)

    @final
    @cache_readonly
    def array_values(self) -> ExtensionArray:
        return self.values

    @final
    def get_values(self, dtype: Optional[Any] = None) -> np.ndarray:
        values = self.values
        if dtype == _dtype_obj:
            values = values.astype(object)
        return np.asarray(values).reshape(self.shape)

    @final
    def pad_or_backfill(
        self, *, method: str, inplace: bool = False, limit: Optional[int] = None, limit_area: Any = None
    ) -> List[Block]:
        values = self.values
        kwargs = {"method": method, "limit": limit}
        if "limit_area" in inspect.signature(values._pad_or_backfill).parameters:
            kwargs["limit_area"] = limit_area
        elif limit_area is not None:
            raise NotImplementedError(
                f"{type(values).__name__} does not implement limit_area (added in pandas 2.2). 3rd-party ExtensionArray authors need to add this argument to _pad_or_backfill."
            )
        if values.ndim == 2:
            new_values = values.T._pad_or_backfill(**kwargs).T
        else:
            new_values = values._pad_or_backfill(**kwargs)
        return [self.make_block_same_class(new_values)]


class ExtensionBlock(EABackedBlock):
    """
    Block for holding extension types.
    """

    def fillna(self, value: Any, limit: Optional[int] = None, inplace: bool = False) -> List[Block]:
        if isinstance(self.dtype, (IntervalDtype, StringDtype)):
            if isinstance(self.dtype, IntervalDtype) and limit is not None:
                raise ValueError("limit must be None")
            return super().fillna(value=value, limit=limit, inplace=inplace)
        if self._can_hold_na and (not self.values._hasna):
            refs: Any = self.refs
            new_values = self.values
        else:
            copy, refs = self._get_refs_and_copy(inplace)
            try:
                new_values = self.values.fillna(value=value, limit=limit, copy=copy)
            except TypeError:
                refs = None
                new_values = self.values.fillna(value=value, limit=limit)
                warnings.warn(
                    f"ExtensionArray.fillna added a 'copy' keyword in pandas 2.1.0. In a future version, ExtensionArray subclasses will need to implement this keyword or an exception will be raised. In the interim, the keyword is ignored by {type(self.values).__name__}.",
                    DeprecationWarning,
                    stacklevel=find_stack_level(),
                )
        return [self.make_block_same_class(new_values, refs=refs)]

    @cache_readonly
    def shape(self) -> Tuple[int, ...]:
        if self.ndim == 1:
            return (len(self.values),)
        return (len(self._mgr_locs), len(self.values))

    def iget(self, i: Any) -> Any:
        if isinstance(i, tuple):
            col, loc = i
            if not com.is_null_slice(col) and col != 0:
                raise IndexError(f"{self} only contains one item")
            if isinstance(col, slice):
                if loc < 0:
                    loc += len(self.values)
                return self.values[loc : loc + 1]
            return self.values[loc]
        else:
            if i != 0:
                raise IndexError(f"{self} only contains one item")
            return self.values

    def set_inplace(self, locs: Any, values: Any, copy: bool = False) -> None:
        if copy:
            self.values = self.values.copy()
        self.values[:] = values

    def _maybe_squeeze_arg(self, arg: Any) -> Any:
        if isinstance(arg, (np.ndarray, ExtensionArray)) and arg.ndim == self.values.ndim + 1:
            assert arg.shape[1] == 1
            arg = arg[:, 0]
        elif isinstance(arg, ABCDataFrame):
            assert arg.shape[1] == 1
            arg = arg._ixs(0, axis=1)._values
        return arg

    def _unwrap_setitem_indexer(self, indexer: Any) -> Any:
        if isinstance(indexer, tuple) and len(indexer) == 2:
            if all((isinstance(x, np.ndarray) and x.ndim == 2 for x in indexer)):
                first, second = indexer
                if not (second.size == 1 and (second == 0).all() and (first.shape[1] == 1)):
                    raise NotImplementedError("This should not be reached. Please report a bug at github.com/pandas-dev/pandas/")
                indexer = first[:, 0]
            elif lib.is_integer(indexer[1]) and indexer[1] == 0:
                indexer = indexer[0]
            elif com.is_null_slice(indexer[1]):
                indexer = indexer[0]
            elif is_list_like(indexer[1]) and indexer[1][0] == 0:
                indexer = indexer[0]
            else:
                raise NotImplementedError("This should not be reached. Please report a bug at github.com/pandas-dev/pandas/")
        return indexer

    @property
    def is_view(self) -> bool:
        """Extension arrays are never treated as views."""
        return False

    @cache_readonly
    def is_numeric(self) -> bool:
        return self.values.dtype._is_numeric

    def _slice(self, slicer: Any) -> ExtensionArray:
        if self.ndim == 2:
            if not isinstance(slicer, slice):
                raise AssertionError("invalid slicing for a 1-ndim ExtensionArray", slicer)
            new_locs = range(1)[slicer]
            if not len(new_locs):
                raise AssertionError("invalid slicing for a 1-ndim ExtensionArray", slicer)
            slicer = slice(None)
        return self.values[slicer]

    @final
    def slice_block_rows(self, slicer: Any) -> Block:
        new_values = self.values[slicer]
        return type(self)(new_values, self._mgr_locs, ndim=self.ndim, refs=self.refs)

    def _unstack(self, unstacker: Any, fill_value: Any, new_placement: np.ndarray, needs_masking: Any) -> Tuple[List[Block], Any]:
        new_values, mask = unstacker.arange_result
        new_values = new_values.T[mask]
        new_placement = new_placement[mask]
        blocks = [
            type(self)(
                self.values.take(indices, allow_fill=needs_masking[i], fill_value=fill_value),
                BlockPlacement(place),
                ndim=2,
            )
            for i, (indices, place) in enumerate(zip(new_values, new_placement))
        ]
        return (blocks, mask)


class NumpyBlock(Block):
    __slots__ = ()

    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""
        return self.values.base is not None

    @property
    def array_values(self) -> NumpyExtensionArray:
        return NumpyExtensionArray(self.values)

    def get_values(self, dtype: Optional[Any] = None) -> np.ndarray:
        if dtype == _dtype_obj:
            return self.values.astype(_dtype_obj)
        return self.values

    @cache_readonly
    def is_numeric(self) -> bool:
        dtype = self.values.dtype
        kind = dtype.kind
        return kind in "fciub"


class NDArrayBackedExtensionBlock(EABackedBlock):
    """
    Block backed by an NDArrayBackedExtensionArray
    """

    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""
        return self.values._ndarray.base is not None


class DatetimeLikeBlock(NDArrayBackedExtensionBlock):
    """Block for datetime64[ns], timedelta64[ns]."""
    __slots__ = ()
    is_numeric: bool = False


def maybe_coerce_values(values: Union[np.ndarray, ExtensionArray]) -> Union[np.ndarray, ExtensionArray]:
    """
    Ensure that datetime64/timedelta64 dtypes are in nanoseconds
    """
    if isinstance(values, np.ndarray):
        values = ensure_wrapped_if_datetimelike(values)
        if issubclass(values.dtype.type, str):
            values = np.array(values, dtype=object)
    if isinstance(values, (DatetimeArray, TimedeltaArray)) and values.freq is not None:
        values = values._with_freq(None)
    return values


def get_block_type(dtype: Any) -> Any:
    """
    Find the appropriate Block subclass to use for the given values and dtype.
    """
    if isinstance(dtype, DatetimeTZDtype):
        return DatetimeLikeBlock
    elif isinstance(dtype, PeriodDtype):
        return NDArrayBackedExtensionBlock
    elif isinstance(dtype, ExtensionDtype):
        return ExtensionBlock
    kind = dtype.kind
    if kind in "Mm":
        return DatetimeLikeBlock
    return NumpyBlock


def new_block_2d(values: Union[np.ndarray, ExtensionArray], placement: Any, refs: Optional[Any] = None) -> Block:
    klass = get_block_type(values.dtype)
    values = maybe_coerce_values(values)
    return klass(values, ndim=2, placement=placement, refs=refs)


def new_block(values: Union[np.ndarray, ExtensionArray], placement: Any, *, ndim: int, refs: Optional[Any] = None) -> Block:
    klass = get_block_type(values.dtype)
    return klass(values, ndim=ndim, placement=placement, refs=refs)


def check_ndim(values: np.ndarray, placement: Any, ndim: int) -> None:
    if values.ndim > ndim:
        raise ValueError(f"Wrong number of dimensions. values.ndim > ndim [{values.ndim} > {ndim}]")
    if not is_1d_only_ea_dtype(values.dtype):
        if values.ndim != ndim:
            raise ValueError(f"Wrong number of dimensions. values.ndim != ndim [{values.ndim} != {ndim}]")
        if len(placement) != len(values):
            raise ValueError(f"Wrong number of items passed {len(values)}, placement implies {len(placement)}")
    elif ndim == 2 and len(placement) != 1:
        raise ValueError("need to split")


def extract_pandas_array(values: Any, dtype: Any, ndim: int) -> Tuple[Any, Any]:
    if isinstance(values, ABCNumpyExtensionArray):
        values = values.to_numpy()
        if ndim and ndim > 1:
            values = np.atleast_2d(values)
    if isinstance(dtype, NumpyEADtype):
        dtype = dtype.numpy_dtype
    return (values, dtype)


def extend_blocks(result: Union[List[Any], Block], blocks: Optional[List[Block]] = None) -> List[Block]:
    if blocks is None:
        blocks = []
    if isinstance(result, list):
        for r in result:
            if isinstance(r, list):
                blocks.extend(r)
            else:
                blocks.append(r)
    else:
        assert isinstance(result, Block), type(result)
        blocks.append(result)
    return blocks


def ensure_block_shape(values: Union[np.ndarray, ExtensionArray], ndim: int = 1) -> Union[np.ndarray, ExtensionArray]:
    if values.ndim < ndim:
        if not is_1d_only_ea_dtype(values.dtype):
            values = values.reshape(1, -1)
    return values


def external_values(values: Union[np.ndarray, PeriodArray, IntervalArray, DatetimeArray, TimedeltaArray]) -> np.ndarray:
    if isinstance(values, (PeriodArray, IntervalArray)):
        return values.astype(object)
    elif isinstance(values, (DatetimeArray, TimedeltaArray)):
        values = values._ndarray
    if isinstance(values, np.ndarray):
        values = values.view()
        values.flags.writeable = False
    return values
