from __future__ import annotations
from collections.abc import Callable, Hashable, Sequence, Generator, Iterable
import itertools
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast, final, Optional, Union, Tuple, List, Dict, Set, TypeVar, Generic, overload
import warnings
import numpy as np
from pandas._config.config import get_option
from pandas._libs import algos as libalgos, internals as libinternals, lib
from pandas._libs.internals import BlockPlacement, BlockValuesRefs
from pandas._libs.tslibs import Timestamp
from pandas.errors import AbstractMethodError, PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import find_common_type, infer_dtype_from_scalar, np_can_hold_element
from pandas.core.dtypes.common import ensure_platform_int, is_1d_only_ea_dtype, is_list_like
from pandas.core.dtypes.dtypes import DatetimeTZDtype, ExtensionDtype, SparseDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import array_equals, isna
import pandas.core.algorithms as algos
from pandas.core.arrays import DatetimeArray
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import ensure_wrapped_if_datetimelike, extract_array
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import Index, default_index, ensure_index
from pandas.core.internals.blocks import Block, NumpyBlock, ensure_block_shape, extend_blocks, get_block_type, maybe_coerce_values, new_block, new_block_2d
from pandas.core.internals.ops import blockwise_all, operate_blockwise

if TYPE_CHECKING:
    from pandas._typing import ArrayLike, AxisInt, DtypeObj, QuantileInterpolation, Self, Shape, npt
    from pandas.api.extensions import ExtensionArray

T = TypeVar('T')

def interleaved_dtype(dtypes: Sequence[DtypeObj]) -> Optional[DtypeObj]:
    """
    Find the common dtype for `blocks`.

    Parameters
    ----------
    blocks : List[DtypeObj]

    Returns
    -------
    dtype : np.dtype, ExtensionDtype, or None
        None is returned when `blocks` is empty.
    """
    if not len(dtypes):
        return None
    return find_common_type(dtypes)

def ensure_np_dtype(dtype: DtypeObj) -> np.dtype:
    if isinstance(dtype, SparseDtype):
        dtype = dtype.subtype
        dtype = cast(np.dtype, dtype)
    elif isinstance(dtype, ExtensionDtype):
        dtype = np.dtype('object')
    elif dtype == np.dtype(str):
        dtype = np.dtype('object')
    return dtype

class BaseBlockManager(PandasObject):
    """
    Core internal data structure to implement DataFrame, Series, etc.
    """
    __slots__ = ()

    @property
    def ndim(self) -> int:
        raise NotImplementedError

    def __init__(self, blocks: Sequence[Block], axes: Sequence[Index], verify_integrity: bool = True) -> None:
        raise NotImplementedError

    @final
    def __len__(self) -> int:
        return len(self.items)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple((len(ax) for ax in self.axes))

    @classmethod
    def from_blocks(cls, blocks: Sequence[Block], axes: Sequence[Index]) -> Self:
        raise NotImplementedError

    @property
    def blknos(self) -> np.ndarray:
        """
        Suppose we want to find the array corresponding to our i'th column.

        blknos[i] identifies the block from self.blocks that contains this column.

        blklocs[i] identifies the column of interest within
        self.blocks[self.blknos[i]]
        """
        if self._blknos is None:
            self._rebuild_blknos_and_blklocs()
        return self._blknos

    @property
    def blklocs(self) -> np.ndarray:
        """
        See blknos.__doc__
        """
        if self._blklocs is None:
            self._rebuild_blknos_and_blklocs()
        return self._blklocs

    def make_empty(self, axes: Optional[Sequence[Index]] = None) -> BaseBlockManager:
        """return an empty BlockManager with the items axis of len 0"""
        if axes is None:
            axes = [default_index(0)] + self.axes[1:]
        if self.ndim == 1:
            assert isinstance(self, SingleBlockManager)
            blk = self.blocks[0]
            arr = blk.values[:0]
            bp = BlockPlacement(slice(0, 0))
            nb = blk.make_block_same_class(arr, placement=bp)
            blocks = [nb]
        else:
            blocks = []
        return type(self).from_blocks(blocks, axes)

    def __bool__(self) -> bool:
        return True

    def set_axis(self, axis: int, new_labels: Index) -> None:
        self._validate_set_axis(axis, new_labels)
        self.axes[axis] = new_labels

    @final
    def _validate_set_axis(self, axis: int, new_labels: Index) -> None:
        old_len = len(self.axes[axis])
        new_len = len(new_labels)
        if axis == 1 and len(self.items) == 0:
            pass
        elif new_len != old_len:
            raise ValueError(f'Length mismatch: Expected axis has {old_len} elements, new values have {new_len} elements')

    @property
    def is_single_block(self) -> bool:
        return len(self.blocks) == 1

    @property
    def items(self) -> Index:
        return self.axes[0]

    def _has_no_reference(self, i: int) -> bool:
        """
        Check for column `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the column has no references.
        """
        blkno = self.blknos[i]
        return self._has_no_reference_block(blkno)

    def _has_no_reference_block(self, blkno: int) -> bool:
        """
        Check for block `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the block has no references.
        """
        return not self.blocks[blkno].refs.has_reference()

    def add_references(self, mgr: BaseBlockManager) -> None:
        """
        Adds the references from one manager to another. We assume that both
        managers have the same block structure.
        """
        if len(self.blocks) != len(mgr.blocks):
            return
        for i, blk in enumerate(self.blocks):
            blk.refs = mgr.blocks[i].refs
            blk.refs.add_reference(blk)

    def references_same_values(self, mgr: BaseBlockManager, blkno: int) -> bool:
        """
        Checks if two blocks from two different block managers reference the
        same underlying values.
        """
        blk = self.blocks[blkno]
        return any((blk is ref() for ref in mgr.blocks[blkno].refs.referenced_blocks))

    def get_dtypes(self) -> np.ndarray:
        dtypes = np.array([blk.dtype for blk in self.blocks], dtype=object)
        return dtypes.take(self.blknos)

    @property
    def arrays(self) -> List[ArrayLike]:
        """
        Quick access to the backing arrays of the Blocks.
        """
        return [blk.values for blk in self.blocks]

    def __repr__(self) -> str:
        output = type(self).__name__
        for i, ax in enumerate(self.axes):
            if i == 0:
                output += f'\nItems: {ax}'
            else:
                output += f'\nAxis {i}: {ax}'
        for block in self.blocks:
            output += f'\n{block}'
        return output

    def _equal_values(self, other: BaseBlockManager) -> bool:
        """
        To be implemented by the subclasses. Only check the column values
        assuming shape and indexes have already been checked.
        """
        raise AbstractMethodError(self)

    @final
    def equals(self, other: Any) -> bool:
        """
        Implementation for DataFrame.equals
        """
        if not isinstance(other, type(self)):
            return False
        self_axes, other_axes = (self.axes, other.axes)
        if len(self_axes) != len(other_axes):
            return False
        if not all((ax1.equals(ax2) for ax1, ax2 in zip(self_axes, other_axes))):
            return False
        return self._equal_values(other)

    def apply(self, f: Union[str, Callable], align_keys: Optional[List[str]] = None, **kwargs: Any) -> BaseBlockManager:
        """
        Iterate over the blocks, collect and create a new BlockManager.
        """
        assert 'filter' not in kwargs
        align_keys = align_keys or []
        result_blocks = []
        aligned_args = {k: kwargs[k] for k in align_keys}
        for b in self.blocks:
            if aligned_args:
                for k, obj in aligned_args.items():
                    if isinstance(obj, (ABCSeries, ABCDataFrame)):
                        if obj.ndim == 1:
                            kwargs[k] = obj.iloc[b.mgr_locs.indexer]._values
                        else:
                            kwargs[k] = obj.iloc[:, b.mgr_locs.indexer]._values
                    else:
                        kwargs[k] = obj[b.mgr_locs.indexer]
            if callable(f):
                applied = b.apply(f, **kwargs)
            else:
                applied = getattr(b, f)(**kwargs)
            result_blocks = extend_blocks(applied, result_blocks)
        out = type(self).from_blocks(result_blocks, self.axes)
        return out

    @final
    def isna(self, func: Callable) -> BaseBlockManager:
        return self.apply('apply', func=func)

    @final
    def fillna(self, value: Any, limit: Optional[int], inplace: bool) -> BaseBlockManager:
        if limit is not None:
            limit = libalgos.validate_limit(None, limit=limit)
        return self.apply('fillna', value=value, limit=limit, inplace=inplace)

    @final
    def where(self, other: Any, cond: Any, align: bool) -> BaseBlockManager:
        if align:
            align_keys = ['other', 'cond']
        else:
            align_keys = ['cond']
            other = extract_array(other, extract_numpy=True)
        return self.apply('where', align_keys=align_keys, other=other, cond=cond)

    @final
    def putmask(self, mask: Any, new: Any, align: bool = True) -> BaseBlockManager:
        if align:
            align_keys = ['new', 'mask']
        else:
            align_keys = ['mask']
            new = extract_array(new, extract_numpy=True)
        return self.apply('putmask', align_keys=align_keys, mask=mask, new=new)

    @final
    def round(self, decimals: int) -> BaseBlockManager:
        return self.apply('round', decimals=decimals)

    @final
    def replace(self, to_replace: Any, value: Any, inplace: bool) -> BaseBlockManager:
        inplace = validate_bool_kwarg(inplace, 'inplace')
        assert not lib.is_list_like(to_replace)
        assert not lib.is_list_like(value)
        return self.apply('replace', to_replace=to_replace, value=value, inplace=inplace)

    @final
    def replace_regex(self, **kwargs: Any) -> BaseBlockManager:
        return self.apply('_replace_regex', **kwargs)

    @final
    def replace_list(self, src_list: Sequence[Any], dest_list: Sequence[Any], inplace: bool = False, regex: bool = False) -> BaseBlockManager:
        """do a list replace"""
        inplace = validate_bool_kwarg(inplace, 'inplace')
        bm = self.apply('replace_list', src_list=src_list, dest_list=dest_list, inplace=inplace, regex=regex)
        bm._consolidate_inplace()
        return bm

    def interpolate(self, inplace: bool, **kwargs: Any) -> BaseBlockManager:
        return self.apply('interpolate', inplace=inplace, **kwargs)

    def pad_or_backfill(self, inplace: bool, **kwargs: Any) -> BaseBlockManager:
        return self.apply('pad_or_backfill', inplace=inplace, **kwargs)

    def shift(self, periods: int, fill_value: Any) -> BaseBlockManager:
        if fill_value is lib.no_default:
            fill_value = None
        return self.apply('shift', periods=periods, fill_value=fill_value)

    def setitem(self, indexer: Any, value: Any) -> BaseBlockManager:
        """
        Set values with indexer.
        """
        if isinstance(indexer, np.ndarray) and indexer.ndim > self.ndim:
            raise ValueError(f'Cannot set values with ndim > {self.ndim}')
        if not self._has_no_reference(0):
            if self.ndim == 2 and isinstance(indexer, tuple):
                blk_loc = self.blklocs[indexer[1]]
                if is_list_like(blk_loc) and blk_loc.ndim == 2:
                    blk_loc = np.squeeze(blk_loc, axis=0)
                elif not is_list_like(blk_loc):
                    blk_loc = [blk_loc]
                if len(blk_loc) == 0:
                    return self.copy(deep=False)
                values = self.blocks[0].values
                if values.ndim == 2:
                    values = values[blk_loc]
                    self._iset_split_block(0, blk_loc, values)
                    self.blocks[0].setitem((indexer[0], np.arange(len(blk_loc))), value)
                    return self
            self = self.copy()
        return self.apply('setitem', indexer=indexer, value=value)

    def diff(self, n: int) -> BaseBlockManager:
        return self.apply('diff', n=n)

    def astype(self, dtype: DtypeObj, errors: str = 'raise') -> BaseBlockManager:
        return self.apply('astype', dtype=dtype, errors=errors)

    def convert(self) -> BaseBlockManager:
        return self.apply('convert')

    def convert_dtypes(self, **kwargs: Any) -> BaseBlockManager:
        return self.apply('convert_dtypes', **kwargs)

    def get_values_for_csv(self, *, float_format: Optional[str], date_format: Optional[str], decimal: str, na_rep: str = 'nan', quoting: Optional[int] = None) -> BaseBlockManager:
        """
        Convert values to native types (strings / python objects) that are used
        in formatting (repr / csv).
        """
        return self.apply('get_values_for_csv', na_rep=na_rep, quoting=quoting, float_format=float_format, date_format=date_format, decimal=decimal)

    @property
    def any_extension_types(self) -> bool:
        """Whether any of the blocks in this manager are extension blocks"""
        return any((block.is_extension for block in self.blocks))

    @property
    def is_view(self) -> bool:
        """return a boolean if we are a single block and are a view"""
        if len(self.blocks) == 1:
            return self.blocks[0].is_view
        return False

    def _get_data_subset(self, predicate: Callable) -> BaseBlockManager:
        blocks = [blk for blk in self.blocks if predicate(blk.values)]
        return self._combine(blocks)

    def get_bool_data(self) -> BaseBlockManager:
        """
        Select blocks that are bool-dtype and columns from object-dtype blocks
        that are all-bool.
        """
        new_blocks = []
        for blk in self.blocks:
            if blk.dtype == bool:
                new_blocks.append(blk)
            elif blk.is_object:
                new_blocks.extend((nb for nb in blk._split() if nb.is_bool))
        return self._combine(new_blocks)

    def get_numeric_data(self) -> BaseBlockManager:
        numeric_blocks = [blk for blk in self.blocks if blk.is_numeric]
        if len(numeric_blocks) == len(self.blocks):
            return self
        return self._combine(numeric_blocks)

    def _combine(self, blocks: Sequence[Block], index: Optional[Index] = None) -> BaseBlockManager:
        """return a new manager with the blocks"""
        if len(blocks) == 0:
            if self.ndim == 2:
                if index is not None:
                    axes = [self.items[:0], index]
                else:
                    axes = [self.items[:0]] + self.axes[1:]
                return self.make_empty(axes)
            return self.make_empty()
        indexer = np.sort(np.concatenate([b.mgr_locs.as_array for b in blocks]))
        inv_indexer = lib.get_reverse_indexer(indexer, self.shape[0])
        new_blocks = []
        for b in blocks:
            nb = b.copy(deep=False)
            nb.mgr_locs = BlockPlacement(inv_indexer[nb.mgr_locs.indexer])
            new_blocks.append(nb)
        axes = list(self.axes)
        if index is not None:
            axes[-1] = index
        axes[0] = self.items.take(indexer)
        return type(self).from_blocks(new_blocks, axes)

    @property
    def nblocks(self) -> int:
        return len(self.blocks)

    def copy(self,