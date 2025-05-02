from __future__ import annotations
import inspect
import re
from typing import TYPE_CHECKING, Any, Literal, cast, final, Callable, Generator, Iterable, Sequence, Union, Optional, List, Tuple, Dict, TypeVar, overload
import warnings
import weakref
import numpy as np
from numpy.typing import NDArray
from pandas._libs import NaT, internals as libinternals, lib
from pandas._libs.internals import BlockPlacement, BlockValuesRefs
from pandas._libs.missing import NA
from pandas._typing import ArrayLike, AxisInt, DtypeBackend, DtypeObj, FillnaOptions, IgnoreRaise, InterpolateOptions, QuantileInterpolation, Self, Shape, npt
from pandas.errors import AbstractMethodError, OutOfBoundsDatetime
from pandas.util._decorators import cache_readonly
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
from pandas.core.array_algos.putmask import extract_bool_array, putmask_inplace, putmask_without_repeat, setitem_datetimelike_compat, validate_putmask
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.replace import compare_or_regex_search, replace_regex, should_use_regex
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays import DatetimeArray, ExtensionArray, IntervalArray, NumpyExtensionArray, PeriodArray, TimedeltaArray
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
_dtype_obj = np.dtype('object')

class Block(PandasObject, libinternals.Block):
    """
    Canonical n-dimensional unit of homogeneous dtype contained in a pandas
    data structure

    Index-ignorant; let the container take care of that
    """
    values: Union[np.ndarray, ExtensionArray]
    ndim: int
    refs: BlockValuesRefs
    __init__: Callable
    __slots__ = ()
    is_numeric = False

    @final
    @cache_readonly
    def _validate_ndim(self):
        """
        We validate dimension for blocks that can hold 2D values, which for now
        means numpy dtypes or DatetimeTZDtype.
        """
        dtype = self.dtype
        return not isinstance(dtype, ExtensionDtype) or isinstance(dtype, DatetimeTZDtype)

    @final
    @cache_readonly
    def is_object(self):
        return self.values.dtype == _dtype_obj

    @final
    @cache_readonly
    def is_extension(self):
        return not lib.is_np_dtype(self.values.dtype)

    @final
    @cache_readonly
    def _can_consolidate(self):
        return not self.is_extension

    @final
    @cache_readonly
    def _consolidate_key(self):
        return (self._can_consolidate, self.dtype.name)

    @final
    @cache_readonly
    def _can_hold_na(self):
        """
        Can we store NA values in this Block?
        """
        dtype = self.dtype
        if isinstance(dtype, np.dtype):
            return dtype.kind not in 'iub'
        return dtype._can_hold_na

    @final
    @property
    def is_bool(self):
        """
        We can be bool if a) we are bool dtype or b) object dtype with bool objects.
        """
        return self.values.dtype == np.dtype(bool)

    @final
    def external_values(self):
        return external_values(self.values)

    @final
    @cache_readonly
    def fill_value(self):
        return na_value_for_dtype(self.dtype, compat=False)

    @final
    def _standardize_fill_value(self, value):
        if self.dtype != _dtype_obj and is_valid_na_for_dtype(value, self.dtype):
            value = self.fill_value
        return value

    @property
    def mgr_locs(self):
        return self._mgr_locs

    @mgr_locs.setter
    def mgr_locs(self, new_mgr_locs):
        self._mgr_locs = new_mgr_locs

    @final
    def make_block(self, values, placement=None, refs=None):
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
    def make_block_same_class(self, values, placement=None, refs=None):
        """Wrap given values in a block of same type as self."""
        if placement is None:
            placement = self._mgr_locs
        return type(self)(values, placement=placement, ndim=self.ndim, refs=refs)

    @final
    def __repr__(self):
        name = type(self).__name__
        if self.ndim == 1:
            result = f'{name}: {len(self)} dtype: {self.dtype}'
        else:
            shape = ' x '.join([str(s) for s in self.shape])
            result = f'{name}: {self.mgr_locs.indexer}, {shape}, dtype: {self.dtype}'
        return result

    @final
    def __len__(self):
        return len(self.values)

    @final
    def slice_block_columns(self, slc):
        """
        Perform __getitem__-like, return result as block.
        """
        new_mgr_locs = self._mgr_locs[slc]
        new_values = self._slice(slc)
        refs = self.refs
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=refs)

    @final
    def take_block_columns(self, indices):
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
        new_mgr_locs = self._mgr_locs[indices]
        new_values = self._slice(indices)
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=None)

    @final
    def getitem_block_columns(self, slicer, new_mgr_locs, ref_inplace_op=False):
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
        new_values = self._slice(slicer)
        refs = self.refs if not ref_inplace_op or self.refs.has_reference() else None
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=refs)

    @final
    def _can_hold_element(self, element):
        """require the same dtype as ourselves"""
        element = extract_array(element, extract_numpy=True)
        return can_hold_element(self.values, element)

    @final
    def should_store(self, value):
        """
        Should we set self.values[indexer] = value inplace or do we need to cast?

        Parameters
        ----------
        value : np.ndarray or ExtensionArray

        Returns
        -------
        bool
        """
        return value.dtype == self.dtype

    @final
    def apply(self, func, **kwargs):
        """
        apply the function to my values; return a block if we are not
        one
        """
        result = func(self.values, **kwargs)
        result = maybe_coerce_values(result)
        return self._split_op_result(result)

    @final
    def reduce(self, func):
        assert self.ndim == 2
        result = func(self.values)
        if self.values.ndim == 1:
            res_values = result
        else:
            res_values = result.reshape(-1, 1)
        nb = self.make_block(res_values)
        return [nb]

    @final
    def _split_op_result(self, result):
        if result.ndim > 1 and isinstance(result.dtype, ExtensionDtype):
            nbs = []
            for i, loc in enumerate(self._mgr_locs):
                if not is_1d_only_ea_dtype(result.dtype):
                    vals = result[i:i + 1]
                else:
                    vals = result[i]
                bp = BlockPlacement(loc)
                block = self.make_block(values=vals, placement=bp)
                nbs.append(block)
            return nbs
        nb = self.make_block(result)
        return [nb]

    @final
    def _split(self):
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
    def split_and_operate(self, func, *args, **kwargs):
        """
        Split the block and apply func column-by-column.

        Parameters
        ----------
        func : Block method
        *args
        **kwargs

        Returns
        -------
        List[Block]
        """
        assert self.ndim == 2 and self.shape[0] != 1
        res_blocks = []
        for nb in self._split():
            rbs = func(nb, *args, **kwargs)
            res_blocks.extend(rbs)
        return res_blocks

    @final
    def coerce_to_target_dtype(self, other, raise_on_upcast):
        """
        coerce the current block to a dtype compat for other
        we will return a block, possibly object, and not raise

        we can also safely try to coerce to the same dtype
        and will receive the same block
        """
        new_dtype = find_result_type(self.values.dtype, other)
        if new_dtype == self.dtype:
            raise AssertionError('Something has gone wrong, please report a bug at https://github.com/pandas-dev/pandas/issues')
        if is_scalar(other) and is_integer_dtype(self.values.dtype) and isna(other) and (other is not NaT) and (not (isinstance(other, (np.datetime64, np.timedelta64)) and np.isnat(other))):
            raise_on_upcast = False
        elif isinstance(other, np.ndarray) and other.ndim == 1 and is_integer_dtype(self.values.dtype) and is_float_dtype(other.dtype) and lib.has_only_ints_or_nan(other):
            raise_on_upcast = False
        if raise_on_upcast:
            raise TypeError(f"Invalid value '{other}' for dtype '{self.values.dtype}'")
        if self.values.dtype == new_dtype:
            raise AssertionError(f'Did not expect new dtype {new_dtype} to equal self.dtype {self.values.dtype}. Please report a bug at https://github.com/pandas-dev/pandas/issues.')
        try:
            return self.astype(new_dtype)
        except OutOfBoundsDatetime as err:
            raise OutOfBoundsDatetime(f"Incompatible (high-resolution) value for dtype='{self.dtype}'. Explicitly cast before operating.") from err

    @final
    def convert(self):
        """
        Attempt to coerce any object types to better types. Return a copy
        of the block (if copy = True).
        """
        if not self.is_object:
            return [self.copy(deep=False)]
        if self.ndim != 1 and self.shape[0] != 1:
            blocks = self.split_and_operate(Block.convert)
            if all((blk.dtype.kind == 'O' for blk in blocks)):
                return [self.copy(deep=False)]
            return blocks
        values = self.values
        if values.ndim == 2:
            values = values[0]
        res_values = lib.maybe_convert_objects(values, convert_non_numeric=True)
        refs = None
        if res_values is values or (isinstance(res_values, NumpyExtensionArray) and res_values._ndarray is values):
            refs = self.refs
        res_values = ensure_block_shape(res_values, self.ndim)
        res_values = maybe_coerce_values(res_values)
        return [self.make_block(res_values, refs=refs)]

    def convert_dtypes(self, infer_objects=True, convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True, dtype_backend='numpy_nullable'):
        if infer_objects and self.is_object:
            blks = self.convert()
        else:
            blks = [self]
        if not any([convert_floating, convert_integer, convert_boolean, convert_string]):
            return [b.copy(deep=False) for b in blks]
        rbs = []
        for blk in blks:
            sub_blks = [blk] if blk.ndim == 1 or self.shape[0] == 1 else list(blk._split())
            dtypes = [convert_dtypes(b.values, convert_string, convert_integer, convert_boolean, convert_floating, infer_objects, dtype_backend) for b in sub_blks]
            if all((dtype == self.dtype for dtype in dtypes)):
                rbs.append(blk.copy(deep=False))
                continue
            for dtype, b in zip(dtypes, sub_blks):
                rbs.append(b.astype(dtype=dtype, squeeze=b.ndim != 1))
        return rbs

    @final
    @cache_readonly
    def dtype(self):
        return self.values.dtype

    @final
    def astype(self, dtype, errors='raise', squeeze=False):
        """
        Coerce to the new dtype.

        Parameters
        ----------
        dtype : np.dtype or ExtensionDtype
        errors : str, {'raise', 'ignore'}, default 'raise'
            - ``raise`` : allow exceptions to be raised
            - ``ignore`` : suppress exceptions. On error return original object
        squeeze : bool, default False
            squeeze values to ndim=1 if only one column is given

        Returns
        -------
        Block
        """
        values = self.values
        if squeeze and values.ndim == 2 and is_1d_only_ea_dtype(dtype):
            if values.shape[0] != 1:
                raise ValueError('Can not squeeze with more than one column.')
            values = values[0, :]
        new_values = astype_array_safe(values, dtype, errors=errors)
        new_values = maybe_coerce_values(new_values)
        refs = None
        if astype_is_view(values.dtype, new_values.dtype):
            refs = self.refs
        newb = self.make_block(new_values, refs=refs)
        if newb.shape != self.shape:
            raise TypeError(f'cannot set astype for dtype ({self.dtype.name} [{self.shape}]) to different shape ({newb.dtype.name} [{newb.shape}])')
        return newb

    @final
    def get_values_for_csv(self, *, float_format: Optional[str], date_format: Optional[str], decimal: str, na_rep: str='nan', quoting=None):
        """convert to our native types format"""
        result = get_values_for_csv(self.values, na_rep=na_rep, quoting=quoting, float_format=float_format, date_format=date_format, decimal=decimal)
        return self.make_block(result)

    @final
    def copy(self, deep=True):
        """copy constructor"""
        values = self.values
        refs: Optional[BlockValuesRefs]
        if deep:
            values = values.copy()
            refs = None
        else:
            refs = self.refs
        return type(self)(values, placement=self._mgr_locs, ndim=self.ndim, refs=refs)

    def _maybe_copy(self, inplace):
        if inplace:
            deep = self.refs.has_reference()
            return self.copy(deep=deep)
        return self.copy()

    @final
    def _get_refs_and_copy(self, inplace):
        refs = None
        copy = not inplace
        if inplace:
            if self.refs.has_reference():
                copy = True
            else:
                refs = self.refs
        return (copy, refs)

    @final
    def replace(self, to_replace, value, inplace=False, mask=None):
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
        elif self._can_hold_element(value) or (self.dtype == 'string' and is_re(value)):
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
            blocks = []
            for i, nb in enumerate(self._split()):
                blocks.extend(type(self).replace(nb, to_replace=to_replace, value=value, inplace=True, mask=mask[i:i + 1]))
            return blocks

    @final
    def _replace_regex(self, to_replace, value, inplace=False, mask=None):
        """
        Replace elements by the given value.

        Parameters
        ----------
        to_replace : object or pattern
            Scalar to replace or regular expression to match.
        value : object
            Replacement object.
        inplace : bool, default False
            Perform inplace modification.
        mask : array-like of bool, optional
            True indicate corresponding element is ignored.

        Returns
        -------
        List[Block]
        """
        if not is_re(to_replace) and (not self._can_hold_element(to_replace)):
            return [self.copy(deep=False)]
        if is_re(to_replace) and self.dtype not in [object, 'string']:
            return [self.copy(deep=False)]
        if not (self._can_hold_element(value) or (self.dtype == 'string' and is_re(value))):
            block = self.astype(np.dtype(object))
        else:
            block = self._maybe_copy(inplace)
        rx = re.compile(to_replace)
        replace_regex(block.values, rx, value, mask)
        return [block]

    @final
    def replace_list(self, src_list, dest_list, inplace=False, regex=False):
        """
        See BlockManager.replace_list docstring.
        """
        values = self.values
        pairs = [(x, y) for x, y in zip(src_list, dest_list) if self._can_hold_element(x) or (self.dtype == 'string' and is_re(x))]
        if not len(pairs):
            return [self.copy(deep=False)]
        src_len = len(pairs) - 1
        if is_string_dtype(values.dtype):
            na_mask = ~isna(values)
            masks: Iterable[NDArray[np.bool_]] = (extract_bool_array(cast(ArrayLike, compare_or_regex_search(values, s[0], regex=regex, mask=na_mask))) for s in pairs)
        else:
            masks = (missing.mask_missing(values, s[0]) for s in pairs)
        if inplace:
            masks = list(masks)
        rb = [self]
        for i, ((src, dest), mask) in enumerate(zip(pairs, masks)):
            new_rb: List[Block] = []
            for blk_num, blk in enumerate(rb):
                if len(rb) == 1:
                    m = mask
                else:
                    mib = mask
                    assert not isinstance(mib, bool)
                    m = mib[blk_num:blk_num + 1]
                result = blk._replace_coerce(to_replace=src, value=dest, mask=m, inplace=inplace, regex=regex)
                if i != src_len:
                    for b in result:
                        ref = weakref.ref(b)
                        b.refs.referenced_blocks.pop(b.refs.referenced_blocks.index(ref))
                new_rb.extend(result)
            rb = new_rb
        return rb

    @final
    def _replace_coerce(self, to_replace, value, mask, inplace=True, regex=False):
        """
        Replace value corresponding to the given boolean array with another
        value.

        Parameters
        ----------
        to_replace : object or pattern
            Scalar to replace or regular expression to match.
        value : object
            Replacement object.
        mask : np.ndarray[bool]
            True indicate corresponding element is ignored.
        inplace : bool, default True
            Perform inplace modification.
        regex : bool, default False
            If true, perform regular expression substitution.

        Returns
        -------
        List[Block]
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

    def _maybe_squeeze_arg(self, arg):
        """
        For compatibility with 1D-only ExtensionArrays.
        """
        return arg

    def _unwrap_setitem_indexer(self, indexer):
        """
        For compatibility with 1D-only ExtensionArrays.
        """
        return indexer

    @property
    def shape(self):
        return self.values.shape

    def iget(self, i):
        return self.values[i]

    def _slice(self, slicer):
        """return a slice of my values"""
        return self.values[slicer]

    def set_inplace(self, locs, values, copy=False):
        """
        Modify block values in-place with new item value.

        If copy=True, first copy the underlying values in place before modifying
        (for Copy-on-Write).

        Notes
        -----
        `set_inplace` never creates a new array or new Block, whereas `setitem`
        _may_ create a new array and always creates a new Block.

        Caller is responsible for checking values.dtype == self.dtype.
        """
        if copy:
            self.values = self.values.copy()
        self.values[locs] = values

    @final
    def take_nd(self, indexer, axis, new_mgr_locs=None, fill_value=lib.no_default):
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

    def _unstack(self, unstacker, fill_value, new_placement, needs_masking):
        """
        Return a list of unstacked blocks of self

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : int
            Only used in ExtensionBlock._unstack
        new_placement : np.ndarray[np.intp]
        allow_fill : bool
        needs_masking : np.ndarray[bool]

        Returns
        -------
        blocks : list of Block
            New blocks of unstacked values.
        mask : array-like of bool
            The mask of columns of `blocks` we should keep.
        """
        new_values, mask = unstacker.get_new_values(self.values.T, fill_value=fill_value)
        mask = mask.any(0)