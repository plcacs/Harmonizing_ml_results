from __future__ import annotations
import inspect
import re
from typing import TYPE_CHECKING, Any, Literal, cast, final, Optional, Union, List, Tuple, Dict, Sequence, Generator, Iterable, Callable
import warnings
import weakref
import numpy as np
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
    __slots__ = ()
    is_numeric: bool = False

    @final
    @cache_readonly
    def _validate_ndim(self) -> bool:
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
        dtype = self.dtype
        if isinstance(dtype, np.dtype):
            return dtype.kind not in 'iub'
        return dtype._can_hold_na

    @final
    @property
    def is_bool(self) -> bool:
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
    def mgr_locs(self) -> BlockPlacement:
        return self._mgr_locs

    @mgr_locs.setter
    def mgr_locs(self, new_mgr_locs: BlockPlacement) -> None:
        self._mgr_locs = new_mgr_locs

    @final
    def make_block(self, values: ArrayLike, placement: Optional[BlockPlacement] = None, refs: Optional[BlockValuesRefs] = None) -> Block:
        if placement is None:
            placement = self._mgr_locs
        if self.is_extension:
            values = ensure_block_shape(values, ndim=self.ndim)
        return new_block(values, placement=placement, ndim=self.ndim, refs=refs)

    @final
    def make_block_same_class(self, values: ArrayLike, placement: Optional[BlockPlacement] = None, refs: Optional[BlockValuesRefs] = None) -> Block:
        if placement is None:
            placement = self._mgr_locs
        return type(self)(values, placement=placement, ndim=self.ndim, refs=refs)

    @final
    def __repr__(self) -> str:
        name = type(self).__name__
        if self.ndim == 1:
            result = f'{name}: {len(self)} dtype: {self.dtype}'
        else:
            shape = ' x '.join([str(s) for s in self.shape])
            result = f'{name}: {self.mgr_locs.indexer}, {shape}, dtype: {self.dtype}'
        return result

    @final
    def __len__(self) -> int:
        return len(self.values)

    @final
    def slice_block_columns(self, slc: slice) -> Block:
        new_mgr_locs = self._mgr_locs[slc]
        new_values = self._slice(slc)
        refs = self.refs
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=refs)

    @final
    def take_block_columns(self, indices: Sequence[int]) -> Block:
        new_mgr_locs = self._mgr_locs[indices]
        new_values = self._slice(indices)
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=None)

    @final
    def getitem_block_columns(self, slicer: Any, new_mgr_locs: BlockPlacement, ref_inplace_op: bool = False) -> Block:
        new_values = self._slice(slicer)
        refs = self.refs if not ref_inplace_op or self.refs.has_reference() else None
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=refs)

    @final
    def _can_hold_element(self, element: Any) -> bool:
        element = extract_array(element, extract_numpy=True)
        return can_hold_element(self.values, element)

    @final
    def should_store(self, value: ArrayLike) -> bool:
        return value.dtype == self.dtype

    @final
    def apply(self, func: Callable, **kwargs: Any) -> List[Block]:
        result = func(self.values, **kwargs)
        result = maybe_coerce_values(result)
        return self._split_op_result(result)

    @final
    def reduce(self, func: Callable) -> List[Block]:
        assert self.ndim == 2
        result = func(self.values)
        if self.values.ndim == 1:
            res_values = result
        else:
            res_values = result.reshape(-1, 1)
        nb = self.make_block(res_values)
        return [nb]

    @final
    def _split_op_result(self, result: ArrayLike) -> List[Block]:
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
    def _split(self) -> Generator[Block, None, None]:
        assert self.ndim == 2
        for i, ref_loc in enumerate(self._mgr_locs):
            vals = self.values[slice(i, i + 1)]
            bp = BlockPlacement(ref_loc)
            nb = type(self)(vals, placement=bp, ndim=2, refs=self.refs)
            yield nb

    @final
    def split_and_operate(self, func: Callable, *args: Any, **kwargs: Any) -> List[Block]:
        assert self.ndim == 2 and self.shape[0] != 1
        res_blocks = []
        for nb in self._split():
            rbs = func(nb, *args, **kwargs)
            res_blocks.extend(rbs)
        return res_blocks

    @final
    def coerce_to_target_dtype(self, other: Any, raise_on_upcast: bool) -> Block:
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
    def convert(self) -> List[Block]:
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

    def convert_dtypes(self, infer_objects: bool = True, convert_string: bool = True, convert_integer: bool = True, convert_boolean: bool = True, convert_floating: bool = True, dtype_backend: str = 'numpy_nullable') -> List[Block]:
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
    def dtype(self) -> DtypeObj:
        return self.values.dtype

    @final
    def astype(self, dtype: DtypeObj, errors: str = 'raise', squeeze: bool = False) -> Block:
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
    def get_values_for_csv(self, *, float_format: Optional[str], date_format: Optional[str], decimal: str, na_rep: str = 'nan', quoting: Optional[int] = None) -> Block:
        result = get_values_for_csv(self.values, na_rep=na_rep, quoting=quoting, float_format=float_format, date_format=date_format, decimal=decimal)
        return self.make_block(result)

    @final
    def copy(self, deep: bool = True) -> Block:
        values = self.values
        if deep:
            values = values.copy()
            refs = None
        else:
            refs = self.refs
        return type(self)(values, placement=self._mgr_locs, ndim=self.ndim, refs=refs)

    def _maybe_copy(self, inplace: bool) -> Block:
        if inplace:
            deep = self.refs.has_reference()
            return self.copy(deep=deep)
        return self.copy()

    @final
    def _get_refs_and_copy(self, inplace: bool) -> Tuple[bool, Optional[BlockValuesRefs]]:
        refs = None
        copy = not inplace
        if inplace:
            if self.refs.has_reference():
                copy = True
            else:
                refs = self.refs
        return (copy, refs)

    @final
    def replace(self, to_replace: Any, value: Any, inplace: bool = False, mask: Optional[np.ndarray] = None) -> List[Block]:
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
    def _replace_regex(self, to_replace: Any, value: Any, inplace: bool = False, mask: Optional[np.ndarray] = None) -> List[Block]:
        if not is_re(to_replace) and (not self._can_hold_element(to_replace)):
            return [self.copy(deep=False)]
        if is_re(to_replace) and self.dtype not in [object, 'string']:
            return [self.copy(deep=False)]
        if not (self._can_hold_element(value) or (self.dtype == 'string' and is_re(value))):
            block = self.astype(np.dtype(object))
        else:
            block = self._maybe_copy(inplace)
        rx = re.compile(to_replace)
       