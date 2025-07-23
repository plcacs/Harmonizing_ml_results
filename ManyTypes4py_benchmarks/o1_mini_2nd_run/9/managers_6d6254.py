from __future__ import annotations
from collections.abc import Callable, Hashable, Sequence
import itertools
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    NoReturn,
    Optional,
    Tuple,
    Union,
    cast,
    final,
)
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
from pandas.core.dtypes.cast import (
    find_common_type,
    infer_dtype_from_scalar,
    np_can_hold_element,
)
from pandas.core.dtypes.common import ensure_platform_int, is_1d_only_ea_dtype, is_list_like
from pandas.core.dtypes.dtypes import DatetimeTZDtype, ExtensionDtype, SparseDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import array_equals, isna
import pandas.core.algorithms as algos
from pandas.core.arrays import DatetimeArray
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import Index, default_index, ensure_index
from pandas.core.internals.blocks import (
    Block,
    NumpyBlock,
    ensure_block_shape,
    extend_blocks,
    get_block_type,
    maybe_coerce_values,
    new_block,
    new_block_2d,
)
from pandas.core.internals.ops import blockwise_all, operate_blockwise

if TYPE_CHECKING:
    from collections.abc import Generator
    from pandas._typing import ArrayLike, AxisInt, DtypeObj, QuantileInterpolation, Self, Shape, npt
    from pandas.api.extensions import ExtensionArray


def interleaved_dtype(dtypes: List[DtypeObj]) -> Optional[Union[np.dtype, ExtensionDtype]]:
    """
    Find the common dtype for `blocks`.

    Parameters
    ----------
    blocks : List[DtypeObj]

    Returns
    -------
    dtype : np.dtype, ExtensionDtype, or None
        None is returned when `blocks` are empty.
    """
    if not len(dtypes):
        return None
    return find_common_type(dtypes)


def ensure_np_dtype(dtype: DtypeObj) -> np.dtype:
    if isinstance(dtype, SparseDtype):
        dtype = dtype.subtype
        dtype = cast(np.dtype, dtype)
    elif isinstance(dtype, ExtensionDtype):
        dtype = np.dtype("object")
    elif dtype == np.dtype(str):
        dtype = np.dtype("object")
    return dtype


class BaseBlockManager(PandasObject):
    """
    Core internal data structure to implement DataFrame, Series, etc.

    Manage a bunch of labeled 2D mixed-type ndarrays. Essentially it's a
    lightweight blocked set of labeled data to be manipulated by the DataFrame
    public API class

    Attributes
    ----------
    shape
    ndim
    axes
    values
    items

    Methods
    -------
    set_axis(axis, new_labels)
    copy(deep=True)

    get_dtypes

    apply(func, axes, block_filter_fn)

    get_bool_data
    get_numeric_data

    get_slice(slice_like, axis)
    get(label)
    iget(loc)

    take(indexer, axis)
    reindex_axis(new_labels, axis)
    reindex_indexer(new_labels, indexer, axis)

    delete(label)
    insert(loc, label, value)
    set(label, value)

    Parameters
    ----------
    blocks: Sequence of Block
    axes: Sequence of Index
    verify_integrity: bool, default True

    Notes
    -----
    This is *not* a public API class
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
    def from_blocks(cls: type[BaseBlockManager], blocks: Sequence[Block], axes: Sequence[Index]) -> BaseBlockManager:
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
            raise ValueError(
                f"Length mismatch: Expected axis has {old_len} elements, new values have {new_len} elements"
            )

    @property
    def is_single_block(self) -> bool:
        return len(self.blocks) == 1

    @property
    def items(self) -> Index:
        return self.axes[0]

    def _has_no_reference(self, i: int = 0) -> bool:
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
    def arrays(self) -> List[np.ndarray]:
        """
        Quick access to the backing arrays of the Blocks.

        Only for compatibility with ArrayManager for testing convenience.
        Not to be used in actual code, and return value is not the same as the
        ArrayManager method (list of 1D arrays vs iterator of 2D ndarrays / 1D EAs).

        Warning! The returned arrays don't handle Copy-on-Write, so this should
        be used with caution (only in read-mode).
        """
        return [blk.values for blk in self.blocks]

    def __repr__(self) -> str:
        output = type(self).__name__
        for i, ax in enumerate(self.axes):
            if i == 0:
                output += f"\nItems: {ax}"
            else:
                output += f"\nAxis {i}: {ax}"
        for block in self.blocks:
            output += f"\n{block}"
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

    def apply(
        self,
        f: Union[str, Callable[..., Any]],
        align_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseBlockManager:
        """
        Iterate over the blocks, collect and create a new BlockManager.

        Parameters
        ----------
        f : str or callable
            Name of the Block method to apply.
        align_keys: List[str] or None, default None
        **kwargs
            Keywords to pass to `f`

        Returns
        -------
        BlockManager
        """
        assert "filter" not in kwargs
        align_keys = align_keys or []
        result_blocks: List[Block] = []
        aligned_args: Dict[str, Any] = {k: kwargs[k] for k in align_keys}
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
    def isna(self, func: Callable[..., Any]) -> BaseBlockManager:
        return self.apply("apply", func=func)

    @final
    def fillna(
        self, value: Any, limit: Optional[int], inplace: bool
    ) -> BaseBlockManager:
        if limit is not None:
            limit = libalgos.validate_limit(None, limit=limit)
        return self.apply("fillna", value=value, limit=limit, inplace=inplace)

    @final
    def where(
        self, other: Any, cond: Any, align: bool
    ) -> BaseBlockManager:
        if align:
            align_keys = ["other", "cond"]
        else:
            align_keys = ["cond"]
            other = extract_array(other, extract_numpy=True)
        return self.apply("where", align_keys=align_keys, other=other, cond=cond)

    @final
    def putmask(
        self, mask: Any, new: Any, align: bool
    ) -> BaseBlockManager:
        if align:
            align_keys = ["new", "mask"]
        else:
            align_keys = ["mask"]
            new = extract_array(new, extract_numpy=True)
        return self.apply("putmask", align_keys=align_keys, mask=mask, new=new)

    @final
    def round(self, decimals: Union[int, Sequence[int]]) -> BaseBlockManager:
        return self.apply("round", decimals=decimals)

    @final
    def replace(
        self, to_replace: Any, value: Any, inplace: bool
    ) -> BaseBlockManager:
        inplace = validate_bool_kwarg(inplace, "inplace")
        assert not lib.is_list_like(to_replace)
        assert not lib.is_list_like(value)
        return self.apply("replace", to_replace=to_replace, value=value, inplace=inplace)

    @final
    def replace_regex(self, **kwargs: Any) -> BaseBlockManager:
        return self.apply("_replace_regex", **kwargs)

    @final
    def replace_list(
        self,
        src_list: List[Any],
        dest_list: List[Any],
        inplace: bool = False,
        regex: bool = False,
    ) -> BaseBlockManager:
        """do a list replace"""
        inplace = validate_bool_kwarg(inplace, "inplace")
        bm = self.apply(
            "replace_list",
            src_list=src_list,
            dest_list=dest_list,
            inplace=inplace,
            regex=regex,
        )
        bm._consolidate_inplace()
        return bm

    def interpolate(
        self, inplace: bool, **kwargs: Any
    ) -> BaseBlockManager:
        return self.apply("interpolate", inplace=inplace, **kwargs)

    def pad_or_backfill(
        self, inplace: bool, **kwargs: Any
    ) -> BaseBlockManager:
        return self.apply("pad_or_backfill", inplace=inplace, **kwargs)

    def shift(
        self, periods: int, fill_value: Any
    ) -> BaseBlockManager:
        if fill_value is lib.no_default:
            fill_value = None
        return self.apply(
            "shift", periods=periods, fill_value=fill_value
        )

    def setitem(
        self,
        indexer: Union[int, slice, Tuple[Union[int, slice, np.ndarray], ...]],
        value: Any,
    ) -> BaseBlockManager:
        """
        Set values with indexer.

        For SingleBlockManager, this backs s[indexer] = value
        """
        if isinstance(indexer, np.ndarray) and indexer.ndim > self.ndim:
            raise ValueError(f"Cannot set values with ndim > {self.ndim}")
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
                    self.blocks[0].setitem(
                        (indexer[0], np.arange(len(blk_loc))), value
                    )
                    return self
            self = self.copy()
        return self.apply("setitem", indexer=indexer, value=value)

    def diff(self, n: int) -> BaseBlockManager:
        return self.apply("diff", n=n)

    def astype(
        self, dtype: DtypeObj, errors: Literal["raise", "ignore"] = "raise"
    ) -> BaseBlockManager:
        return self.apply("astype", dtype=dtype, errors=errors)

    def convert(self) -> BaseBlockManager:
        return self.apply("convert")

    def convert_dtypes(self, **kwargs: Any) -> BaseBlockManager:
        return self.apply("convert_dtypes", **kwargs)

    def get_values_for_csv(
        self,
        *,
        float_format: Optional[str],
        date_format: Optional[str],
        decimal: str,
        na_rep: str = "nan",
        quoting: Optional[int] = None,
    ) -> BaseBlockManager:
        """
        Convert values to native types (strings / python objects) that are used
        in formatting (repr / csv).
        """
        return self.apply(
            "get_values_for_csv",
            na_rep=na_rep,
            quoting=quoting,
            float_format=float_format,
            date_format=date_format,
            decimal=decimal,
        )

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

    def _get_data_subset(self, predicate: Callable[[np.ndarray], bool]) -> BaseBlockManager:
        blocks = [blk for blk in self.blocks if predicate(blk.values)]
        return self._combine(blocks)

    def get_bool_data(self) -> BaseBlockManager:
        """
        Select blocks that are bool-dtype and columns from object-dtype blocks
        that are all-bool.
        """
        new_blocks: List[Block] = []
        for blk in self.blocks:
            if blk.dtype == bool:
                new_blocks.append(blk)
            elif blk.is_object:
                new_blocks.extend((nb for nb in blk._split() if nb.is_bool))
        return self._combine(new_blocks)

    def get_numeric_data(self) -> BaseBlockManager:
        """
        Select blocks that are numeric dtype.
        """
        numeric_blocks = [blk for blk in self.blocks if blk.is_numeric]
        if len(numeric_blocks) == len(self.blocks):
            return self
        return self._combine(numeric_blocks)

    def _combine(
        self, blocks: List[Block], index: Optional[Index] = None
    ) -> BaseBlockManager:
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
        new_blocks: List[Block] = []
        for b in blocks:
            nb = b.copy(deep=False)
            nb.mgr_locs = BlockPlacement(inv_indexer[b.mgr_locs.indexer])
            new_blocks.append(nb)
        axes = list(self.axes)
        if index is not None:
            axes[-1] = index
        axes[0] = self.items.take(indexer)
        return type(self).from_blocks(new_blocks, axes)

    @property
    def nblocks(self) -> int:
        return len(self.blocks)

    def copy(self, deep: Union[bool, Literal["all"]] = True) -> BaseBlockManager:
        """
        Make deep or shallow copy of BlockManager

        Parameters
        ----------
        deep : bool, string or None, default True
            If False or None, return a shallow copy (do not copy data)
            If 'all', copy data and a deep copy of the index

        Returns
        -------
        BlockManager
        """
        if deep:
            def copy_func(ax: Index) -> Index:
                return ax.copy(deep=True) if deep == "all" else ax.view()

            new_axes = [copy_func(ax) for ax in self.axes]
        else:
            new_axes = [ax.view() for ax in self.axes]
        res = self.apply("copy", deep=deep)
        res.axes = new_axes
        if self.ndim > 1:
            blknos = self._blknos
            if blknos is not None:
                res._blknos = blknos.copy()
                res._blklocs = self._blklocs.copy()
        if deep:
            res._consolidate_inplace()
        return res

    def is_consolidated(self) -> bool:
        return True

    def consolidate(self) -> BaseBlockManager:
        """
        Join together blocks having same dtype

        Returns
        -------
        y : BlockManager
        """
        if self.is_consolidated():
            return self
        bm = type(self)(self.blocks, self.axes, verify_integrity=False)
        bm._is_consolidated = False
        bm._consolidate_inplace()
        return bm

    def _consolidate_inplace(self) -> None:
        return

    @final
    def reindex_axis(
        self,
        new_index: Index,
        axis: int,
        fill_value: Any = None,
        only_slice: bool = False,
    ) -> BaseBlockManager:
        """
        Conform data manager to new index.
        """
        new_index, indexer = self.axes[axis].reindex(new_index)
        return self.reindex_indexer(
            new_axis=new_index,
            indexer=indexer,
            axis=axis,
            fill_value=fill_value,
            only_slice=only_slice,
        )

    def reindex_indexer(
        self,
        new_axis: Index,
        indexer: Optional[np.ndarray],
        axis: int,
        fill_value: Any = None,
        allow_dups: bool = False,
        only_slice: bool = False,
        *,
        use_na_proxy: bool = False,
    ) -> BaseBlockManager:
        """
        Parameters
        ----------
        new_axis : Index
        indexer : ndarray[intp] or None
        axis : int
        fill_value : object, default None
        allow_dups : bool, default False
        only_slice : bool, default False
            Whether to take views, not copies, along columns.
        use_na_proxy : bool, default False
            Whether to use a np.void ndarray for newly introduced columns.

        pandas-indexer with -1's only.
        """
        if indexer is None:
            if new_axis is self.axes[axis]:
                return self
            result = self.copy(deep=False)
            result.axes = list(self.axes)
            result.axes[axis] = new_axis
            return result
        assert isinstance(indexer, np.ndarray)
        if not allow_dups:
            self.axes[axis]._validate_can_reindex(indexer)
        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")
        if axis == 0:
            new_blocks = list(
                self._slice_take_blocks_ax0(
                    indexer, fill_value=fill_value, only_slice=only_slice, use_na_proxy=use_na_proxy
                )
            )
        else:
            new_blocks = [
                blk.take_nd(
                    indexer,
                    axis=1,
                    fill_value=fill_value if fill_value is not None else blk.fill_value,
                )
                for blk in self.blocks
            ]
        new_axes = list(self.axes)
        new_axes[axis] = new_axis
        new_mgr = type(self).from_blocks(new_blocks, new_axes)
        if axis == 1:
            new_mgr._blknos = self.blknos.copy()
            new_mgr._blklocs = self.blklocs.copy()
        return new_mgr

    def _slice_take_blocks_ax0(
        self,
        slice_or_indexer: Union[slice, np.ndarray],
        fill_value: Any = lib.no_default,
        only_slice: bool = False,
        *,
        use_na_proxy: bool = False,
        ref_inplace_op: bool = False,
    ) -> Iterator[Block]:
        """
        Slice/take blocks along axis=0.

        Overloaded for SingleBlock

        Parameters
        ----------
        slice_or_indexer : slice or np.ndarray[int64]
        fill_value : scalar, default lib.no_default
        only_slice : bool, default False
            If True, we always return views on existing arrays, never copies.
            This is used when called from ops.blockwise.operate_blockwise.
        use_na_proxy : bool, default False
            Whether to use a np.void ndarray for newly introduced columns.
        ref_inplace_op: bool, default False
            Don't track refs if True because we operate inplace

        Yields
        ------
        Block : New Block
        """
        allow_fill = fill_value is not lib.no_default
        sl_type, slobj, sllen = _preprocess_slice_or_indexer(slice_or_indexer, self.shape[0], allow_fill=allow_fill)
        if self.is_single_block:
            blk = self.blocks[0]
            if sl_type == "slice":
                if sllen == 0:
                    return
                bp = BlockPlacement(slice(0, sllen))
                yield blk.getitem_block_columns(slobj, new_mgr_locs=bp)
                return
            elif not allow_fill or self.ndim == 1:
                if allow_fill and fill_value is None:
                    fill_value = blk.fill_value
                if not allow_fill and only_slice:
                    for i, ml in enumerate(slobj):
                        yield blk.getitem_block_columns(
                            slice(ml, ml + 1), new_mgr_locs=BlockPlacement(i), ref_inplace_op=ref_inplace_op
                        )
                else:
                    bp = BlockPlacement(slice(0, sllen))
                    yield blk.take_nd(slobj, axis=0, new_mgr_locs=bp, fill_value=fill_value)
                return
        if sl_type == "slice":
            blknos = self.blknos[slobj]
            blklocs = self.blklocs[slobj]
        else:
            blknos = algos.take_nd(
                self.blknos, slice_or_indexer, fill_value=-1, allow_fill=allow_fill
            )
            blklocs = algos.take_nd(
                self.blklocs, slice_or_indexer, fill_value=-1, allow_fill=allow_fill
            )
        group = not only_slice
        for blkno, mgr_locs in libinternals.get_blkno_placements(blknos, group=group):
            if blkno == -1:
                yield self._make_na_block(
                    placement=mgr_locs, fill_value=fill_value, use_na_proxy=use_na_proxy
                )
            else:
                blk = self.blocks[blkno]
                if not blk._can_consolidate and (not blk._validate_ndim):
                    deep = False
                    for mgr_loc in mgr_locs:
                        newblk = blk.copy(deep=deep)
                        newblk.mgr_locs = BlockPlacement(slice(mgr_loc, mgr_loc + 1))
                        yield newblk
                else:
                    taker = blklocs[mgr_locs.indexer]
                    max_len = max(len(mgr_locs), taker.max() + 1)
                    taker = lib.maybe_indices_to_slice(taker, max_len)
                    if isinstance(taker, slice):
                        nb = blk.getitem_block_columns(taker, new_mgr_locs=mgr_locs)
                        yield nb
                    elif only_slice:
                        for i, ml in zip(taker, mgr_locs):
                            slc = slice(i, i + 1)
                            bp = BlockPlacement(ml)
                            nb = blk.getitem_block_columns(slc, new_mgr_locs=bp)
                            yield nb
                    else:
                        nb = blk.take_nd(taker, axis=0, new_mgr_locs=mgr_locs)
                        yield nb

    def _make_na_block(
        self,
        placement: BlockPlacement,
        fill_value: Any = None,
        use_na_proxy: bool = False,
    ) -> Block:
        if use_na_proxy:
            assert fill_value is None
            shape = (len(placement), self.shape[1])
            vals = np.empty(shape, dtype=np.void)
            nb = NumpyBlock(vals, placement, ndim=2)
            return nb
        if fill_value is None or fill_value is np.nan:
            fill_value = np.nan
            dtype = interleaved_dtype([blk.dtype for blk in self.blocks])
            if dtype is not None and np.issubdtype(dtype.type, np.floating):
                fill_value = dtype.type(fill_value)
        shape = (len(placement), self.shape[1])
        dtype, fill_value = infer_dtype_from_scalar(fill_value)
        block_values = make_na_array(dtype, shape, fill_value)
        return new_block_2d(block_values, placement=placement)

    def take(
        self,
        indexer: np.ndarray,
        axis: int = 1,
        verify: bool = True,
    ) -> BaseBlockManager:
        """
        Take items along any axis.

        indexer : np.ndarray[np.intp]
        axis : int, default 1
        verify : bool, default True
            Check that all entries are between 0 and len(self) - 1, inclusive.
            Pass verify=False if this check has been done by the caller.

        Returns
        -------
        BlockManager
        """
        n = self.shape[axis]
        indexer = maybe_convert_indices(indexer, n, verify=verify)
        new_labels = self.axes[axis].take(indexer)
        return self.reindex_indexer(
            new_axis=new_labels, indexer=indexer, axis=axis, allow_dups=True
        )


class BlockManager(libinternals.BlockManager, BaseBlockManager):
    """
    BaseBlockManager that holds 2D blocks.
    """
    ndim: int = 2

    def __init__(
        self,
        blocks: Sequence[Block],
        axes: Sequence[Index],
        verify_integrity: bool = True,
    ) -> None:
        if verify_integrity:
            for block in blocks:
                if self.ndim != block.ndim:
                    raise AssertionError(
                        f"Number of Block dimensions ({block.ndim}) must equal number of axes ({self.ndim})"
                    )
            self._verify_integrity()

    def _verify_integrity(self) -> None:
        mgr_shape = self.shape
        tot_items = sum((len(x.mgr_locs) for x in self.blocks))
        for block in self.blocks:
            if block.shape[1:] != mgr_shape[1:]:
                raise_construction_error(
                    tot_items, block.shape[1:], self.axes
                )
        if len(self.items) != tot_items:
            raise AssertionError(
                f"Number of manager items must equal union of block items\n# manager items: {len(self.items)}, # tot_items: {tot_items}"
            )

    @classmethod
    def from_blocks(
        cls: type[BlockManager],
        blocks: Sequence[Block],
        axes: Sequence[Index],
    ) -> BlockManager:
        """
        Constructor for BlockManager and SingleBlockManager with same signature.
        """
        return cls(blocks, axes, verify_integrity=False)

    def fast_xs(self, loc: int) -> Union[np.ndarray, ExtensionArray]:
        """
        Return the array corresponding to `frame.iloc[loc]`.

        Parameters
        ----------
        loc : int

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        if len(self.blocks) == 1:
            result = self.blocks[0].iget((slice(None), loc))
            bp = BlockPlacement(slice(0, len(result)))
            block = new_block(result, placement=bp, ndim=1, refs=self.blocks[0].refs)
            return SingleBlockManager(block, self.axes[0])
        dtype = interleaved_dtype([blk.dtype for blk in self.blocks])
        n = len(self)
        if isinstance(dtype, ExtensionDtype):
            result = np.empty(n, dtype=object)
        else:
            result = np.empty(n, dtype=dtype)
            result = ensure_wrapped_if_datetimelike(result)
        for blk in self.blocks:
            for i, rl in enumerate(blk.mgr_locs):
                result[rl] = blk.iget((i, loc))
        if isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            result = cls._from_sequence(result, dtype=dtype)
        bp = BlockPlacement(slice(0, len(result)))
        block = new_block(result, placement=bp, ndim=1)
        return SingleBlockManager(block, self.axes[0])

    def iget(self, i: int, track_ref: bool = True) -> SingleBlockManager:
        """
        Return the data as a SingleBlockManager.
        """
        block = self.blocks[self.blknos[i]]
        values = block.iget(self.blklocs[i])
        bp = BlockPlacement(slice(0, len(values)))
        nb = type(block)(
            values,
            placement=bp,
            ndim=1,
            refs=block.refs if track_ref else None,
        )
        return SingleBlockManager(nb, self.axes[1])

    def iget_values(self, i: int) -> Union[np.ndarray, ExtensionArray]:
        """
        Return the data for column i as the values (ndarray or ExtensionArray).

        Warning! The returned array is a view but doesn't handle Copy-on-Write,
        so this should be used with caution.
        """
        block = self.blocks[self.blknos[i]]
        values = block.iget(self.blklocs[i])
        return values

    @property
    def column_arrays(self) -> List[Union[np.ndarray, ExtensionArray]]:
        """
        Used in the JSON C code to access column arrays.
        This optimizes compared to using `iget_values` by converting each

        Warning! This doesn't handle Copy-on-Write, so should be used with
        caution (current use case of consuming this in the JSON code is fine).
        """
        result: List[Union[np.ndarray, ExtensionArray]] = [None] * len(self.items)
        for blk in self.blocks:
            mgr_locs = blk._mgr_locs
            values = blk.array_values._values_for_json()
            if values.ndim == 1:
                result[mgr_locs[0]] = values
            else:
                for i, loc in enumerate(mgr_locs):
                    result[loc] = values[i]
        return result

    def iset(
        self,
        loc: int,
        value: Union[np.ndarray, ExtensionArray],
        inplace: bool = False,
        refs: Optional[BlockValuesRefs] = None,
    ) -> None:
        """
        Set new item in-place. Does not consolidate. Adds new Block if not
        contained in the current set of items
        """
        value_is_extension_type = is_1d_only_ea_dtype(value.dtype)
        if not value_is_extension_type:
            if value.ndim == 2:
                value = value.T
            else:
                value = ensure_block_shape(value, ndim=2)
            if value.shape[1:] != self.shape[1:]:
                raise AssertionError("Shape of new values must be compatible with manager shape")
        if lib.is_integer(loc):
            loc = cast(int, loc)
            blkno = self.blknos[loc]
            blk = self.blocks[blkno]
            if len(blk._mgr_locs) == 1:
                return self._iset_single(
                    loc,
                    value,
                    inplace=inplace,
                    blkno=blkno,
                    blk=blk,
                    refs=refs,
                )
            loc = [loc]
        if value_is_extension_type:

            def value_getitem(placement: BlockPlacement) -> Union[np.ndarray, ExtensionArray]:
                return value
        else:

            def value_getitem(placement: BlockPlacement) -> np.ndarray:
                return value[placement.indexer]
        blknos = self.blknos[loc]
        blklocs = self.blklocs[loc].copy()
        unfit_mgr_locs: List[np.ndarray] = []
        unfit_val_locs: List[BlockPlacement] = []
        removed_blknos: List[int] = []
        for blkno_l, val_locs in libinternals.get_blkno_placements(blknos, group=True):
            blk = self.blocks[blkno_l]
            blk_locs = blklocs[val_locs.indexer]
            if inplace and blk.should_store(value):
                if not self._has_no_reference_block(blkno_l):
                    self._iset_split_block(blkno_l, blk_locs, value_getitem(val_locs), refs=refs)
                else:
                    blk.set_inplace(blk_locs, value_getitem(val_locs))
                    continue
            else:
                unfit_mgr_locs.append(blk.mgr_locs.as_array[blk_locs])
                unfit_val_locs.append(val_locs)
                if len(val_locs) == len(blk.mgr_locs):
                    removed_blknos.append(blkno_l)
                    continue
                else:
                    self._iset_split_block(blkno_l, blk_locs, refs=refs)
        if len(removed_blknos):
            is_deleted = np.zeros(self.nblocks, dtype=np.bool_)
            is_deleted[removed_blknos] = True
            new_blknos = np.empty(self.nblocks, dtype=np.intp)
            new_blknos.fill(-1)
            new_blknos[~is_deleted] = np.arange(self.nblocks - len(removed_blknos))
            self._blknos = new_blknos[self._blknos]
            self.blocks = tuple((blk for i, blk in enumerate(self.blocks) if i not in set(removed_blknos)))
        if unfit_val_locs:
            unfit_idxr = np.concatenate(unfit_mgr_locs)
            unfit_count = len(unfit_idxr)
            new_blocks: List[Block] = []
            if value_is_extension_type:
                new_blocks.extend(
                    (
                        new_block_2d(
                            values=value,
                            placement=BlockPlacement(slice(mgr_loc, mgr_loc + 1)),
                            refs=refs,
                        )
                        for mgr_loc in unfit_idxr
                    )
                )
                self._blknos[unfit_idxr] = np.arange(unfit_count) + len(self.blocks)
                self._blklocs[unfit_idxr] = 0
            else:
                unfit_val_items = unfit_val_locs[0].append(*unfit_val_locs[1:])
                new_blocks.append(
                    new_block_2d(
                        values=value_getitem(unfit_val_items),
                        placement=BlockPlacement(unfit_idxr),
                        refs=refs,
                    )
                )
                self._blknos[unfit_idxr] = len(self.blocks)
                self._blklocs[unfit_idxr] = np.arange(unfit_count)
            self.blocks += tuple(new_blocks)
            self._known_consolidated = False

    def _iset_split_block(
        self,
        blkno_l: int,
        blk_locs: List[int],
        value: Optional[Union[np.ndarray, ExtensionArray]] = None,
        refs: Optional[BlockValuesRefs] = None,
    ) -> None:
        """Removes columns from a block by splitting the block.

        Avoids copying the whole block through slicing and updates the manager
        after determining the new block structure. Optionally adds a new block,
        otherwise has to be done by the caller.

        Parameters
        ----------
        blkno_l: The block number to operate on, relevant for updating the manager
        blk_locs: The locations of our block that should be deleted.
        value: The value to set as a replacement.
        refs: The reference tracking object of the value to set.
        """
        blk = self.blocks[blkno_l]
        if self._blklocs is None:
            self._rebuild_blknos_and_blklocs()
        nbs_tup = tuple(blk.delete(blk_locs))
        if value is not None:
            locs = blk.mgr_locs.as_array[blk_locs]
            first_nb = new_block_2d(value, BlockPlacement(locs), refs=refs)
        else:
            first_nb = nbs_tup[0]
            nbs_tup = tuple(nbs_tup[1:])
        nr_blocks = len(self.blocks)
        blocks_tup = self.blocks[:blkno_l] + (first_nb,) + self.blocks[blkno_l + 1 :] + nbs_tup
        self.blocks = blocks_tup
        if not nbs_tup and value is not None:
            return
        self._blklocs[first_nb.mgr_locs.indexer] = np.arange(len(first_nb))
        for i, nb in enumerate(nbs_tup):
            self._blklocs[nb.mgr_locs.indexer] = np.arange(len(nb))
            self._blknos[nb.mgr_locs.indexer] = i + nr_blocks

    def _iset_single(
        self,
        loc: int,
        value: Union[np.ndarray, ExtensionArray],
        inplace: bool,
        blkno: int,
        blk: Block,
        refs: Optional[BlockValuesRefs] = None,
    ) -> None:
        """
        Fastpath for iset when we are only setting a single position and
        the Block currently in that position is itself single-column.

        In this case we can swap out the entire Block and blklocs and blknos
        are unaffected.
        """
        if inplace and blk.should_store(value):
            copy = not self._has_no_reference_block(blkno)
            iloc = self.blklocs[loc]
            blk.set_inplace(slice(iloc, iloc + 1), value, copy=copy)
            return
        nb = new_block_2d(value, placement=blk._mgr_locs, refs=refs)
        old_blocks = self.blocks
        new_blocks = old_blocks[:blkno] + (nb,) + old_blocks[blkno + 1 :]
        self.blocks = new_blocks
        return

    def column_setitem(
        self,
        loc: int,
        idx: Any,
        value: Any,
        inplace_only: bool = False,
    ) -> None:
        """
        Set values ("setitem") into a single column (not setting the full column).

        This is a method on the BlockManager level, to avoid creating an
        intermediate Series at the DataFrame level (`s = df[loc]; s[idx] = value`)
        """
        if not self._has_no_reference(loc):
            blkno = self.blknos[loc]
            blk_loc = self.blklocs[loc]
            values = self.blocks[blkno].values
            if values.ndim == 1:
                values = values.copy()
            else:
                values = values[[blk_loc]]
            self._iset_split_block(blkno, [blk_loc], values)
        col_mgr = self.iget(loc, track_ref=False)
        if inplace_only:
            col_mgr.setitem_inplace(idx, value)
        else:
            new_mgr = col_mgr.setitem((idx,), value)
            self.iset(loc, new_mgr._block.values, inplace=True)

    def insert(
        self,
        loc: int,
        item: Hashable,
        value: Union[np.ndarray, ExtensionArray],
        refs: Optional[BlockValuesRefs] = None,
    ) -> None:
        """
        Insert item at selected position.

        Parameters
        ----------
        loc : int
        item : hashable
        value : np.ndarray or ExtensionArray
        refs : The reference tracking object of the value to set.
        """
        new_axis = self.items.insert(loc, item)
        if value.ndim == 2:
            value = value.T
            if len(value) > 1:
                raise ValueError(
                    f"Expected a 1D array, got an array with shape {value.T.shape}"
                )
        else:
            value = ensure_block_shape(value, ndim=self.ndim)
        bp = BlockPlacement(slice(loc, loc + 1))
        block = new_block_2d(values=value, placement=bp, refs=refs)
        if not len(self.blocks):
            self._blklocs = np.array([0], dtype=np.intp)
            self._blknos = np.array([0], dtype=np.intp)
        else:
            self._insert_update_mgr_locs(loc)
            self._insert_update_blklocs_and_blknos(loc)
        self.axes[0] = new_axis
        self.blocks += (block,)
        self._known_consolidated = False
        if get_option("performance_warnings") and sum(
            (not block.is_extension for block in self.blocks)
        ) > 100:
            warnings.warn(
                "DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`",
                PerformanceWarning,
                stacklevel=find_stack_level(),
            )

    def _insert_update_mgr_locs(self, loc: int) -> None:
        """
        When inserting a new Block at location 'loc', we increment
        all of the mgr_locs of blocks above that by one.
        """
        blknos = np.bincount(self.blknos[loc:]).nonzero()[0]
        for blkno in blknos:
            blk = self.blocks[blkno]
            blk._mgr_locs = blk._mgr_locs.increment_above(loc)

    def _insert_update_blklocs_and_blknos(self, loc: int) -> None:
        """
        When inserting a new Block at location 'loc', we update our
        _blklocs and _blknos.
        """
        if loc == self.blklocs.shape[0]:
            self._blklocs = np.append(self._blklocs, 0)
            self._blknos = np.append(self._blknos, len(self.blocks))
        elif loc == 0:
            self._blklocs = np.concatenate([[0], self._blklocs])
            self._blknos = np.concatenate([[len(self.blocks)], self._blknos])
        else:
            new_blklocs, new_blknos = libinternals.update_blklocs_and_blknos(
                self.blklocs, self.blknos, loc, len(self.blocks)
            )
            self._blklocs = new_blklocs
            self._blknos = new_blknos

    def idelete(self, indexer: np.ndarray) -> BlockManager:
        """
        Delete selected locations, returning a new BlockManager.
        """
        is_deleted = np.zeros(self.shape[0], dtype=np.bool_)
        is_deleted[indexer] = True
        taker = (~is_deleted).nonzero()[0]
        nbs = list(
            self._slice_take_blocks_ax0(taker, only_slice=True, ref_inplace_op=True)
        )
        new_columns = self.items[~is_deleted]
        axes = [new_columns, self.axes[1]]
        return type(self)(tuple(nbs), axes, verify_integrity=False)

    def grouped_reduce(self, func: Callable[[np.ndarray], Any]) -> BlockManager:
        """
        Apply grouped reduction function blockwise, returning a new BlockManager.

        Parameters
        ----------
        func : grouped reduction function

        Returns
        -------
        BlockManager
        """
        result_blocks: List[Block] = []
        for blk in self.blocks:
            if blk.is_object:
                for sb in blk._split():
                    applied = sb.apply(func)
                    result_blocks = extend_blocks(applied, result_blocks)
            else:
                applied = blk.apply(func)
                result_blocks = extend_blocks(applied, result_blocks)
        if len(result_blocks) == 0:
            nrows = 0
        else:
            nrows = result_blocks[0].values.shape[-1]
        index = default_index(nrows)
        return type(self).from_blocks(result_blocks, [self.axes[0], index])

    def reduce(self, func: Callable[[np.ndarray], Any]) -> BlockManager:
        """
        Apply reduction function blockwise, returning a single-row BlockManager.

        Parameters
        ----------
        func : reduction function

        Returns
        -------
        BlockManager
        """
        assert self.ndim == 2
        res_blocks: List[Block] = []
        for blk in self.blocks:
            nbs = blk.reduce(func)
            res_blocks.extend(nbs)
        index = Index([None])
        new_mgr = type(self).from_blocks(res_blocks, [self.items, index])
        return new_mgr

    def operate_blockwise(
        self,
        other: BaseBlockManager,
        array_op: Callable[[Any, Any], Any],
    ) -> BaseBlockManager:
        """
        Apply array_op blockwise with another (aligned) BlockManager.
        """
        return operate_blockwise(self, other, array_op)

    def _equal_values(self, other: BaseBlockManager) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
        return blockwise_all(self, other, array_equals)

    def quantile(
        self,
        *,
        qs: Sequence[float],
        interpolation: QuantileInterpolation = "linear",
    ) -> BlockManager:
        """
        Iterate over blocks applying quantile reduction.
        This routine is intended for reduction type operations and
        will do inference on the generated blocks.

        Parameters
        ----------
        interpolation : type of interpolation, default 'linear'
        qs : list of the quantiles to be computed

        Returns
        -------
        BlockManager
        """
        assert self.ndim >= 2
        assert is_list_like(qs)
        new_axes = list(self.axes)
        new_axes[1] = Index(qs, dtype=np.float64)
        blocks = [blk.quantile(qs=qs, interpolation=interpolation) for blk in self.blocks]
        return type(self)(blocks, new_axes)

    def unstack(
        self,
        unstacker: Any,  # Assuming reshape._Unstacker type
        fill_value: Any,
    ) -> BlockManager:
        """
        Return a BlockManager with all blocks unstacked.

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : Any
            fill_value for newly introduced missing values.

        Returns
        -------
        unstacked : BlockManager
        """
        new_columns = unstacker.get_new_columns(self.items)
        new_index = unstacker.new_index
        allow_fill = not unstacker.mask_all
        if allow_fill:
            new_mask2D = (~unstacker.mask).reshape(*unstacker.full_shape)
            needs_masking = new_mask2D.any(axis=0)
        else:
            needs_masking = np.zeros(unstacker.full_shape[1], dtype=bool)
        new_blocks: List[Block] = []
        columns_mask: List[bool] = []
        if len(self.items) == 0:
            factor = 1
        else:
            fac = len(new_columns) / len(self.items)
            assert fac == int(fac)
            factor = int(fac)
        for blk in self.blocks:
            mgr_locs = blk.mgr_locs
            new_placement = mgr_locs.tile_for_unstack(factor)
            blocks, mask = blk._unstack(
                unstacker,
                fill_value,
                new_placement=new_placement,
                needs_masking=needs_masking,
            )
            new_blocks.extend(blocks)
            columns_mask.extend(mask)
            assert mask.sum() == sum((len(nb._mgr_locs) for nb in blocks))
        new_columns = new_columns[columns_mask]
        bm = BlockManager(new_blocks, [new_columns, new_index], verify_integrity=False)
        return bm

    def to_iter_dict(self) -> Iterator[Tuple[str, BlockManager]]:
        """
        Yield a tuple of (str(dtype), BlockManager)

        Returns
        -------
        values : a tuple of (str(dtype), BlockManager)
        """
        key = lambda block: str(block.dtype)
        for dtype, blocks in itertools.groupby(sorted(self.blocks, key=key), key=key):
            yield (dtype, self._combine(list(blocks)))

    def as_array(
        self,
        dtype: Optional[DtypeObj] = None,
        copy: bool = False,
        na_value: Any = lib.no_default,
    ) -> np.ndarray:
        """
        Convert the blockmanager data into an numpy array.

        Parameters
        ----------
        dtype : np.dtype or None, default None
            Data type of the return array.
        copy : bool, default False
            If True then guarantee that a copy is returned. A value of
            False does not guarantee that the underlying data is not
            copied.
        na_value : object, default lib.no_default
            Value to be used as the missing value sentinel.

        Returns
        -------
        arr : ndarray
        """
        passed_nan = lib.is_float(na_value) and isna(na_value)
        if len(self.blocks) == 0:
            arr = np.empty(self.shape, dtype=float)
            return arr.transpose()
        if self.is_single_block:
            blk = self.blocks[0]
            if na_value is not lib.no_default:
                if lib.is_np_dtype(blk.dtype, "f") and passed_nan:
                    pass
                else:
                    copy = True
            if blk.is_extension:
                arr = blk.values.to_numpy(dtype=dtype, na_value=na_value, copy=copy).reshape(
                    blk.shape
                )
            elif not copy:
                arr = np.asarray(blk.values, dtype=dtype)
            else:
                arr = np.array(blk.values, dtype=dtype, copy=copy)
            if not copy:
                arr = arr.view()
                arr.flags.writeable = False
        else:
            arr = self._interleave(dtype=dtype, na_value=na_value)
        if na_value is lib.no_default:
            pass
        elif arr.dtype.kind == "f" and passed_nan:
            pass
        else:
            arr[isna(arr)] = na_value
        return arr.transpose()

    def _interleave(
        self,
        dtype: Optional[DtypeObj] = None,
        na_value: Any = lib.no_default,
    ) -> np.ndarray:
        """
        Return ndarray from blocks with specified item order
        Items must be contained in the blocks
        """
        if not dtype:
            dtype = interleaved_dtype([blk.dtype for blk in self.blocks])
        dtype = ensure_np_dtype(dtype)
        result = np.empty(self.shape, dtype=dtype)
        itemmask = np.zeros(self.shape[0])
        if dtype == np.dtype("object") and na_value is lib.no_default:
            for blk in self.blocks:
                rl = blk.mgr_locs
                arr = blk.get_values(dtype)
                result[rl.indexer] = arr
                itemmask[rl.indexer] = 1
            return result
        for blk in self.blocks:
            rl = blk.mgr_locs
            if blk.is_extension:
                arr = blk.values.to_numpy(dtype=dtype, na_value=na_value)
            else:
                arr = blk.get_values(dtype)
            result[rl.indexer] = arr
            itemmask[rl.indexer] = 1
        if not itemmask.all():
            raise AssertionError("Some items were not contained in blocks")
        return result

    def is_consolidated(self) -> bool:
        """
        Return True if more than one block with the same dtype
        """
        if not self._known_consolidated:
            self._consolidate_check()
        return self._is_consolidated

    def _consolidate_check(self) -> None:
        if len(self.blocks) == 1:
            self._is_consolidated = True
            self._known_consolidated = True
            return
        dtypes = [blk.dtype for blk in self.blocks if blk._can_consolidate]
        self._is_consolidated = len(dtypes) == len(set(dtypes))
        self._known_consolidated = True

    def _consolidate_inplace(self) -> None:
        if not self.is_consolidated():
            self.blocks = _consolidate(self.blocks)
            self._is_consolidated = True
            self._known_consolidated = True
            self._rebuild_blknos_and_blklocs()

    @classmethod
    def concat_horizontal(
        cls: type[BlockManager],
        mgrs: Iterable[BaseBlockManager],
        axes: Sequence[Index],
    ) -> BlockManager:
        """
        Concatenate uniformly-indexed BlockManagers horizontally.
        """
        offset = 0
        blocks: List[Block] = []
        for mgr in mgrs:
            for blk in mgr.blocks:
                nb = blk.slice_block_columns(slice(None))
                nb._mgr_locs = nb._mgr_locs.add(offset)
                blocks.append(nb)
            offset += len(mgr.items)
        new_mgr = cls(tuple(blocks), axes)
        return new_mgr

    @classmethod
    def concat_vertical(
        cls: type[BlockManager],
        mgrs: Iterable[BaseBlockManager],
        axes: Sequence[Index],
    ) -> BlockManager:
        """
        Concatenate uniformly-indexed BlockManagers vertically.
        """
        raise NotImplementedError("This logic lives (for now) in internals.concat")


class SingleBlockManager(BaseBlockManager):
    """manage a single block with"""

    @property
    def ndim(self) -> int:
        return 1

    _is_consolidated: bool = True
    _known_consolidated: bool = True
    __slots__ = ()

    is_single_block: bool = True

    def __init__(
        self,
        block: Block,
        axis: Index,
        verify_integrity: bool = False,
    ) -> None:
        self.axes = [axis]
        self.blocks = (block,)

    @classmethod
    def from_blocks(
        cls: type[SingleBlockManager],
        blocks: Sequence[Block],
        axes: Sequence[Index],
    ) -> SingleBlockManager:
        """
        Constructor for BlockManager and SingleBlockManager with same signature.
        """
        assert len(blocks) == 1
        assert len(axes) == 1
        return cls(blocks[0], axes[0], verify_integrity=False)

    @classmethod
    def from_array(
        cls: type[SingleBlockManager],
        array: Union[np.ndarray, ExtensionArray],
        index: Index,
        refs: Optional[BlockValuesRefs] = None,
    ) -> SingleBlockManager:
        """
        Constructor for if we have an array that is not yet a Block.
        """
        array = maybe_coerce_values(array)
        bp = BlockPlacement(slice(0, len(index)))
        block = new_block(array, placement=bp, ndim=1, refs=refs)
        return cls(block, index)

    def to_2d_mgr(self, columns: Index) -> BlockManager:
        """
        Manager analogue of Series.to_frame
        """
        blk = self.blocks[0]
        arr = ensure_block_shape(blk.values, ndim=2)
        bp = BlockPlacement(0)
        new_blk = type(blk)(arr, placement=bp, ndim=2, refs=blk.refs)
        axes = [columns, self.axes[0]]
        return BlockManager([new_blk], axes=axes, verify_integrity=False)

    def _has_no_reference(self, i: int = 0) -> bool:
        """
        Check for column `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the column has no references.
        """
        return not self.blocks[0].refs.has_reference()

    def __getstate__(self) -> Tuple[List[Index], List[np.ndarray], List[Index], Dict[str, Any]]:
        block_values = [b.values for b in self.blocks]
        block_items = [self.items[b.mgr_locs.indexer] for b in self.blocks]
        axes_array = list(self.axes)
        extra_state: Dict[str, Any] = {
            "0.14.1": {
                "axes": axes_array,
                "blocks": [{"values": b.values, "mgr_locs": b.mgr_locs.indexer} for b in self.blocks],
            }
        }
        return (axes_array, block_values, block_items, extra_state)

    def __setstate__(self, state: Any) -> None:
        def unpickle_block(
            values: np.ndarray,
            mgr_locs: Union[List[int], BlockPlacement],
            ndim: int,
        ) -> Block:
            values = extract_array(values, extract_numpy=True)
            if not isinstance(mgr_locs, BlockPlacement):
                mgr_locs = BlockPlacement(mgr_locs)
            values = maybe_coerce_values(values)
            return new_block(values, placement=mgr_locs, ndim=ndim)

        if (
            isinstance(state, tuple)
            and len(state) >= 4
            and "0.14.1" in state[3]
        ):
            state = cast(Dict[str, Any], state[3]["0.14.1"])
            self.axes = [ensure_index(ax) for ax in state["axes"]]
            ndim = len(self.axes)
            self.blocks = tuple(
                (
                    unpickle_block(b["values"], b["mgr_locs"], ndim=ndim)
                    for b in state["blocks"]
                )
            )
        else:
            raise NotImplementedError("pre-0.14.1 pickles are no longer supported")
        self._post_setstate()

    def _post_setstate(self) -> None:
        pass

    @cache_readonly
    def _block(self) -> Block:
        return self.blocks[0]

    @final
    @property
    def array(self) -> Union[np.ndarray, ExtensionArray]:
        """
        Quick access to the backing array of the Block.
        """
        return self.blocks[0].values

    @property
    def _blknos(self) -> Optional[np.ndarray]:
        """compat with BlockManager"""
        return None

    @property
    def _blklocs(self) -> Optional[np.ndarray]:
        """compat with BlockManager"""
        return None

    def get_rows_with_mask(
        self,
        indexer: Union[np.ndarray, List[bool]],
    ) -> SingleBlockManager:
        blk = self._block
        if len(indexer) > 0 and indexer.all():
            return type(self)(blk.copy(deep=False), self.index)
        array = blk.values[indexer]
        if isinstance(indexer, np.ndarray) and indexer.dtype.kind == "b":
            refs: Optional[BlockValuesRefs] = None
        else:
            refs = blk.refs
        bp = BlockPlacement(slice(0, len(array)))
        block = type(blk)(
            array,
            placement=bp,
            ndim=1,
            refs=refs,
        )
        new_idx = self.index[indexer]
        return type(self)(block, new_idx)

    def get_slice(
        self,
        slobj: slice,
        axis: int = 0,
    ) -> SingleBlockManager:
        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")
        blk = self._block
        array = blk.values[slobj]
        bp = BlockPlacement(slice(0, len(array)))
        block = type(blk)(
            array,
            placement=bp,
            ndim=1,
            refs=blk.refs,
        )
        new_index = self.index._getitem_slice(slobj)
        return type(self)(block, new_index)

    @property
    def index(self) -> Index:
        return self.axes[0]

    @property
    def dtype(self) -> DtypeObj:
        return self._block.dtype

    def get_dtypes(self) -> np.ndarray:
        return np.array([self._block.dtype], dtype=object)

    def external_values(self) -> ArrayLike:
        """The array that Series.values returns"""
        return self._block.external_values()

    def internal_values(self) -> ArrayLike:
        """The array that Series._values returns"""
        return self._block.values

    def array_values(self) -> NDArrayBackedExtensionArray:
        """The array that Series.array returns"""
        return self._block.array_values()

    def get_numeric_data(self) -> SingleBlockManager:
        if self._block.is_numeric:
            return self.copy(deep=False)
        return self.make_empty()

    @property
    def _can_hold_na(self) -> bool:
        return self._block._can_hold_na

    def setitem_inplace(
        self,
        indexer: Any,
        value: Any,
    ) -> None:
        """
        Set values with indexer.

        For SingleBlockManager, this backs s[indexer] = value

        This is an inplace version of `setitem()`, mutating the manager/values
        in place, not returning a new Manager (and Block), and thus never changing
        the dtype.
        """
        if not self._has_no_reference(0):
            self.blocks = (self._block.copy(),)
            self._reset_cache()
        arr = self.array
        if isinstance(arr, np.ndarray):
            value = np_can_hold_element(arr.dtype, value)
        if (
            isinstance(value, np.ndarray)
            and value.ndim == 1
            and (len(value) == 1)
        ):
            value = value[0, ...]
        arr[indexer] = value

    def idelete(self, indexer: int) -> SingleBlockManager:
        """
        Delete single location from SingleBlockManager.

        Ensures that self.blocks doesn't become empty.
        """
        nb = self._block.delete(indexer)[0]
        self.blocks = (nb,)
        self.axes[0] = self.axes[0].delete(indexer)
        self._reset_cache()
        return self

    def fast_xs(self, loc: Any) -> Any:
        """
        fast path for getting a cross-section
        return a view of the data
        """
        raise NotImplementedError("Use series._values[loc] instead")

    def set_values(
        self,
        values: Union[np.ndarray, ExtensionArray],
    ) -> None:
        """
        Set the values of the single block in place.

        Use at your own risk! This does not check if the passed values are
        valid for the current Block/SingleBlockManager (length, dtype, etc),
        and this does not properly keep track of references.
        """
        self.blocks[0].values = values
        self.blocks[0]._mgr_locs = BlockPlacement(slice(len(values)))

    def _equal_values(self, other: SingleBlockManager) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
        if other.ndim != 1:
            return False
        left = self.blocks[0].values
        right = other.blocks[0].values
        return array_equals(left, right)

    def grouped_reduce(
        self,
        func: Callable[[np.ndarray], Any],
    ) -> SingleBlockManager:
        arr = self.array
        res = func(arr)
        index = default_index(len(res))
        mgr = type(self).from_array(res, index)
        return mgr


def create_block_manager_from_blocks(
    blocks: Sequence[Block],
    axes: Sequence[Index],
    consolidate: bool = True,
    verify_integrity: bool = True,
) -> BlockManager:
    try:
        mgr = BlockManager(blocks, axes, verify_integrity=verify_integrity)
    except ValueError as err:
        arrays = [blk.values for blk in blocks]
        tot_items = sum((arr.shape[0] for arr in arrays))
        raise_construction_error(
            tot_items, arrays[0].shape[1:], axes, err
        )
    if consolidate:
        mgr._consolidate_inplace()
    return mgr


def create_block_manager_from_column_arrays(
    arrays: Sequence[ArrayLike],
    axes: Sequence[Index],
    consolidate: bool,
    refs: List[Optional[BlockValuesRefs]],
) -> BlockManager:
    try:
        blocks = _form_blocks(arrays, consolidate, refs)
        mgr = BlockManager(blocks, axes, verify_integrity=False)
    except ValueError as e:
        raise_construction_error(len(arrays), arrays[0].shape, axes, e)
    if consolidate:
        mgr._consolidate_inplace()
    return mgr


def raise_construction_error(
    tot_items: int,
    block_shape: Tuple[int, ...],
    axes: Sequence[Index],
    e: Optional[Exception] = None,
) -> NoReturn:
    """raise a helpful message about our construction"""
    passed = tuple(map(int, [tot_items] + list(block_shape)))
    if len(passed) <= 2:
        passed = passed[::-1]
    implied = tuple((len(ax) for ax in axes))
    if len(implied) <= 2:
        implied = implied[::-1]
    if passed == implied and e is not None:
        raise e
    if block_shape[0] == 0:
        raise ValueError("Empty data passed with indices specified.")
    raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")


def _grouping_func(tup: Tuple[int, Block]) -> Tuple[int, Any]:
    dtype = tup[1].dtype
    if is_1d_only_ea_dtype(dtype):
        sep = id(dtype)
    else:
        sep = 0
    return (sep, dtype)


def _form_blocks(
    arrays: Sequence[ArrayLike],
    consolidate: bool,
    refs: List[Optional[BlockValuesRefs]],
) -> List[Block]:
    tuples = enumerate(arrays)
    if not consolidate:
        return _tuples_to_blocks_no_consolidate(tuples, refs)
    grouper = itertools.groupby(tuples, _grouping_func)
    nbs: List[Block] = []
    for (_, dtype), tup_block in grouper:
        block_type = get_block_type(dtype)
        if isinstance(dtype, np.dtype):
            is_dtlike = dtype.kind in "mM"
            if issubclass(dtype.type, (str, bytes)):
                dtype = np.dtype("object")
            values, placement = _stack_arrays(tup_block, dtype)
            if is_dtlike:
                values = ensure_wrapped_if_datetimelike(values)
            blk = block_type(
                values,
                placement=BlockPlacement(placement),
                ndim=2,
            )
            nbs.append(blk)
        elif is_1d_only_ea_dtype(dtype):
            dtype_blocks = [
                block_type(
                    x[1],
                    placement=BlockPlacement(x[0]),
                    ndim=2,
                )
                for x in tup_block
            ]
            nbs.extend(dtype_blocks)
        else:
            dtype_blocks = [
                block_type(
                    ensure_block_shape(x[1], 2),
                    placement=BlockPlacement(x[0]),
                    ndim=2,
                )
                for x in tup_block
            ]
            nbs.extend(dtype_blocks)
    return nbs


def _tuples_to_blocks_no_consolidate(
    tuples: Iterable[Tuple[int, ArrayLike]],
    refs: List[Optional[BlockValuesRefs]],
) -> List[Block]:
    return [
        new_block_2d(
            ensure_block_shape(arr, ndim=2),
            placement=BlockPlacement(i),
            refs=ref,
        )
        for (i, arr), ref in zip(tuples, refs)
    ]


def _stack_arrays(
    tuples: Iterable[Tuple[int, ArrayLike]],
    dtype: np.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    placement, arrays = zip(*tuples)
    first = arrays[0]
    shape = (len(arrays),) + first.shape
    stacked = np.empty(shape, dtype=dtype)
    for i, arr in enumerate(arrays):
        stacked[i] = arr
    return (stacked, placement)


def _consolidate(blocks: List[Block]) -> Tuple[Block, ...]:
    """
    Merge blocks having same dtype, exclude non-consolidating blocks
    """
    gkey = lambda x: x._consolidate_key
    grouper = itertools.groupby(sorted(blocks, key=gkey), key=gkey)
    new_blocks: List[Block] = []
    for (_can_consolidate, dtype), group_blocks in grouper:
        merged_blocks, _ = _merge_blocks(
            list(group_blocks), dtype=dtype, can_consolidate=_can_consolidate
        )
        new_blocks = extend_blocks(merged_blocks, new_blocks)
    return tuple(new_blocks)


def _merge_blocks(
    blocks: List[Block],
    dtype: Any,
    can_consolidate: bool,
) -> Tuple[List[Block], bool]:
    if len(blocks) == 1:
        return (blocks, False)
    if can_consolidate:
        new_mgr_locs = np.concatenate([b.mgr_locs.as_array for b in blocks])
        if isinstance(blocks[0].dtype, np.dtype):
            new_values = np.vstack([b.values for b in blocks])
        else:
            bvals = [blk.values for blk in blocks]
            bvals2 = cast(Sequence[NDArrayBackedExtensionArray], bvals)
            new_values = bvals2[0]._concat_same_type(bvals2, axis=0)
        argsort = np.argsort(new_mgr_locs)
        new_values = new_values[argsort]
        new_mgr_locs = new_mgr_locs[argsort]
        bp = BlockPlacement(new_mgr_locs)
        return ([new_block_2d(new_values, placement=bp)], True)
    return (blocks, False)


def _preprocess_slice_or_indexer(
    slice_or_indexer: Union[slice, np.ndarray],
    length: int,
    allow_fill: bool,
) -> Tuple[str, Union[slice, np.ndarray], int]:
    if isinstance(slice_or_indexer, slice):
        return ("slice", slice_or_indexer, libinternals.slice_len(slice_or_indexer, length))
    else:
        if not isinstance(slice_or_indexer, np.ndarray) or slice_or_indexer.dtype.kind != "i":
            dtype = getattr(slice_or_indexer, "dtype", None)
            raise TypeError(type(slice_or_indexer), dtype)
        indexer = ensure_platform_int(slice_or_indexer)
        if not allow_fill:
            indexer = maybe_convert_indices(indexer, length)
        return ("fancy", indexer, len(indexer))


def make_na_array(
    dtype: DtypeObj,
    shape: Tuple[int, ...],
    fill_value: Any,
) -> Union[np.ndarray, ExtensionArray]:
    if isinstance(dtype, DatetimeTZDtype):
        ts = Timestamp(fill_value).as_unit(dtype.unit)
        i8values = np.full(shape, ts._value)
        dt64values = i8values.view(f"M8[{dtype.unit}]")
        return DatetimeArray._simple_new(dt64values, dtype=dtype)
    elif is_1d_only_ea_dtype(dtype):
        dtype = cast(ExtensionDtype, dtype)
        cls = dtype.construct_array_type()
        missing_arr = cls._from_sequence([], dtype=dtype)
        ncols, nrows = shape
        assert ncols == 1, ncols
        empty_arr = -1 * np.ones((nrows,), dtype=np.intp)
        return missing_arr.take(empty_arr, allow_fill=True, fill_value=fill_value)
    elif isinstance(dtype, ExtensionDtype):
        cls = dtype.construct_array_type()
        missing_arr = cls._empty(shape=shape, dtype=dtype)
        missing_arr[:] = fill_value
        return missing_arr
    else:
        missing_arr_np = np.empty(shape, dtype=dtype)
        missing_arr_np.fill(fill_value)
        if dtype.kind in "mM":
            missing_arr_np = ensure_wrapped_if_datetimelike(missing_arr_np)
        return missing_arr_np


def _consolidate(blocks: List[Block]) -> Tuple[Block, ...]:
    """
    Merge blocks having same dtype, exclude non-consolidating blocks
    """
    gkey = lambda x: x._consolidate_key
    grouper = itertools.groupby(sorted(blocks, key=gkey), key=gkey)
    new_blocks: List[Block] = []
    for (_can_consolidate, dtype), group_blocks in grouper:
        merged_blocks, _ = _merge_blocks(
            list(group_blocks), dtype=dtype, can_consolidate=_can_consolidate
        )
        new_blocks = extend_blocks(merged_blocks, new_blocks)
    return tuple(new_blocks)


def _merge_blocks(
    blocks: List[Block],
    dtype: Any,
    can_consolidate: bool,
) -> Tuple[List[Block], bool]:
    if len(blocks) == 1:
        return (blocks, False)
    if can_consolidate:
        new_mgr_locs = np.concatenate([b.mgr_locs.as_array for b in blocks])
        if isinstance(blocks[0].dtype, np.dtype):
            new_values = np.vstack([b.values for b in blocks])
        else:
            bvals = [blk.values for blk in blocks]
            bvals2 = cast(Sequence[NDArrayBackedExtensionArray], bvals)
            new_values = bvals2[0]._concat_same_type(bvals2, axis=0)
        argsort = np.argsort(new_mgr_locs)
        new_values = new_values[argsort]
        new_mgr_locs = new_mgr_locs[argsort]
        bp = BlockPlacement(new_mgr_locs)
        return ([new_block_2d(new_values, placement=bp)], True)
    return (blocks, False)
