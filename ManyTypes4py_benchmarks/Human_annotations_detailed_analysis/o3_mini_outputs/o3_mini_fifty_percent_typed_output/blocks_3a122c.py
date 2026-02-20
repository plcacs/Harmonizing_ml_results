from __future__ import annotations

import inspect
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Literal,
    Sequence,
    cast,
)
import warnings
import weakref

import numpy as np

from pandas._libs import (
    NaT,
    internals as libinternals,
    lib,
)
from pandas._libs.internals import (
    BlockPlacement,
    BlockValuesRefs,
)
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
from pandas.errors import (
    AbstractMethodError,
    OutOfBoundsDatetime,
)
from pandas.util._decorators import cache_readonly, final
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg

from pandas.core.dtypes.astype import (
    astype_array_safe,
    astype_is_view,
)
from pandas.core.dtypes.cast import (
    LossySetitemError,
    can_hold_element,
    convert_dtypes,
    find_result_type,
    np_can_hold_element,
)
from pandas.core.dtypes.common import (
    is_1d_only_ea_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_list_like,
    is_scalar,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    NumpyEADtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCNumpyExtensionArray,
    ABCSeries,
)
from pandas.core.dtypes.inference import is_re
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
)

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
from pandas.core.array_algos.replace import (
    compare_or_regex_search,
    replace_regex,
    should_use_regex,
)
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
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import check_setitem_lengths
from pandas.core.indexes.base import get_values_for_csv

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
    )

    from pandas.core.api import Index
    from pandas.core.arrays._mixins import NDArrayBackedExtensionArray

# comparison is faster than is_object_dtype
_dtype_obj = np.dtype("object")


class Block(PandasObject, libinternals.Block):
    """
    Canonical n-dimensional unit of homogeneous dtype contained in a pandas
    data structure

    Index-ignorant; let the container take care of that
    """

    values: np.ndarray | ExtensionArray
    ndim: int
    refs: BlockValuesRefs
    __init__: Callable[..., Any]

    __slots__ = ()
    is_numeric = False

    @final
    @cache_readonly
    def _validate_ndim(self) -> bool:
        """
        We validate dimension for blocks that can hold 2D values, which for now
        means numpy dtypes or DatetimeTZDtype.
        """
        dtype = self.dtype
        return not isinstance(dtype, ExtensionDtype) or isinstance(
            dtype, DatetimeTZDtype
        )

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
        # We _could_ consolidate for DatetimeTZDtype but don't for now.
        return not self.is_extension

    @final
    @cache_readonly
    def _consolidate_key(self) -> tuple[bool, str]:
        return self._can_consolidate, self.dtype.name

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
    def external_values(self) -> ArrayLike:
        return external_values(self.values)

    @final
    @cache_readonly
    def fill_value(self) -> Any:
        # Used in reindex_indexer
        return na_value_for_dtype(self.dtype, compat=False)

    @final
    def _standardize_fill_value(self, value: Any) -> Any:
        # if we are passed a scalar None, convert it here
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
    def make_block(
        self,
        values: np.ndarray | ExtensionArray,
        placement: BlockPlacement | None = None,
        refs: BlockValuesRefs | None = None,
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
        self,
        values: np.ndarray | ExtensionArray,
        placement: BlockPlacement | None = None,
        refs: BlockValuesRefs | None = None,
    ) -> Self:
        """Wrap given values in a block of same type as self."""
        # Pre-2.0 we called ensure_wrapped_if_datetimelike because fastparquet
        #  relied on it, as of 2.0 the caller is responsible for this.
        if placement is None:
            placement = self._mgr_locs

        # We assume maybe_coerce_values has already been called
        return type(self)(values, placement=placement, ndim=self.ndim, refs=refs)

    @final
    def __repr__(self) -> str:
        # don't want to print out all of the items here
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
    def slice_block_columns(self, slc: slice) -> Self:
        """
        Perform __getitem__-like, return result as block.
        """
        new_mgr_locs: BlockPlacement = self._mgr_locs[slc]

        new_values: ArrayLike = self._slice(slc)
        refs: BlockValuesRefs = self.refs
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=refs)

    @final
    def take_block_columns(self, indices: npt.NDArray[np.intp]) -> Self:
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
        # Note: only called from is from internals.concat, and we can verify
        #  that never happens with 1-column blocks, i.e. never for ExtensionBlock.

        new_mgr_locs: BlockPlacement = self._mgr_locs[indices]

        new_values: ArrayLike = self._slice(indices)
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=None)

    @final
    def getitem_block_columns(
        self, slicer: Any, new_mgr_locs: BlockPlacement, ref_inplace_op: bool = False
    ) -> Self:
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
        new_values: ArrayLike = self._slice(slicer)
        refs: BlockValuesRefs | None = self.refs if not ref_inplace_op or self.refs.has_reference() else None
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

        Parameters
        ----------
        value : np.ndarray or ExtensionArray

        Returns
        -------
        bool
        """
        return value.dtype == self.dtype

    # ---------------------------------------------------------------------
    # Apply/Reduce and Helpers

    @final
    def apply(self, func: Callable[..., Any], **kwargs: Any) -> list[Block]:
        """
        apply the function to my values; return a block if we are not
        one
        """
        result = func(self.values, **kwargs)

        result = maybe_coerce_values(result)
        return self._split_op_result(result)

    @final
    def reduce(self, func: Callable[[Any], Any]) -> list[Block]:
        # We will apply the function and reshape the result into a single-row
        #  Block with the same mgr_locs; squeezing will be done at a higher level
        assert self.ndim == 2

        result = func(self.values)

        if self.values.ndim == 1:
            res_values = result
        else:
            res_values = result.reshape(-1, 1)

        nb = self.make_block(res_values)
        return [nb]

    @final
    def _split_op_result(self, result: np.ndarray | ExtensionArray) -> list[Block]:
        # See also: split_and_operate
        if result.ndim > 1 and isinstance(result.dtype, ExtensionDtype):
            # TODO(EA2D): unnecessary with 2D EAs
            # if we get a 2D ExtensionArray, we need to split it into 1D pieces
            nbs: list[Block] = []
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
    def _split(self) -> Generator[Block, None, None]:
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
    def split_and_operate(self, func: Callable[..., list[Block]], *args: Any, **kwargs: Any) -> list[Block]:
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

        res_blocks: list[Block] = []
        for nb in self._split():
            rbs = func(nb, *args, **kwargs)
            res_blocks.extend(rbs)
        return res_blocks

    # ---------------------------------------------------------------------
    # Up/Down-casting

    @final
    def coerce_to_target_dtype(self, other: Any, raise_on_upcast: bool) -> Block:
        """
        coerce the current block to a dtype compat for other
        we will return a block, possibly object, and not raise

        we can also safely try to coerce to the same dtype
        and will receive the same block
        """
        new_dtype = find_result_type(self.values.dtype, other)
        if new_dtype == self.dtype:
            # GH#52927 avoid RecursionError
            raise AssertionError(
                "Something has gone wrong, please report a bug at "
                "https://github.com/pandas-dev/pandas/issues"
            )

        # In a future version of pandas, the default will be that
        # setting `nan` into an integer series won't raise.
        if (
            is_scalar(other)
            and is_integer_dtype(self.values.dtype)
            and isna(other)
            and other is not NaT
            and not (
                isinstance(other, (np.datetime64, np.timedelta64)) and np.isnat(other)
            )
        ):
            raise_on_upcast = False
        elif (
            isinstance(other, np.ndarray)
            and other.ndim == 1
            and is_integer_dtype(self.values.dtype)
            and is_float_dtype(other.dtype)
            and lib.has_only_ints_or_nan(other)
        ):
            raise_on_upcast = False

        if raise_on_upcast:
            raise TypeError(f"Invalid value '{other}' for dtype '{self.values.dtype}'")
        if self.values.dtype == new_dtype:
            raise AssertionError(
                f"Did not expect new dtype {new_dtype} to equal self.dtype "
                f"{self.values.dtype}. Please report a bug at "
                "https://github.com/pandas-dev/pandas/issues."
            )
        try:
            return self.astype(new_dtype)
        except OutOfBoundsDatetime as err:
            raise OutOfBoundsDatetime(
                f"Incompatible (high-resolution) value for dtype='{self.dtype}'. "
                "Explicitly cast before operating."
            ) from err

    @final
    def convert(self) -> list[Block]:
        """
        Attempt to coerce any object types to better types. Return a copy
        of the block (if copy = True).
        """
        if not self.is_object:
            return [self.copy(deep=False)]

        if self.ndim != 1 and self.shape[0] != 1:
            blocks = self.split_and_operate(Block.convert)
            if all(blk.dtype.kind == "O" for blk in blocks):
                return [self.copy(deep=False)]
            return blocks

        values = self.values
        if values.ndim == 2:
            values = values[0]

        res_values = lib.maybe_convert_objects(
            values,  # type: ignore[arg-type]
            convert_non_numeric=True,
        )
        refs: BlockValuesRefs | None = None
        if res_values is values or (
            isinstance(res_values, NumpyExtensionArray)
            and res_values._ndarray is values
        ):
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
        dtype_backend: DtypeBackend = "numpy_nullable",
    ) -> list[Block]:
        if infer_objects and self.is_object:
            blks = self.convert()
        else:
            blks = [self]

        if not any(
            [convert_floating, convert_integer, convert_boolean, convert_string]
        ):
            return [b.copy(deep=False) for b in blks]

        rbs: list[Block] = []
        for blk in blks:
            sub_blks: list[Block]
            if blk.ndim == 1 or self.shape[0] == 1:
                sub_blks = [blk]
            else:
                sub_blks = list(blk._split())
            dtypes = [
                convert_dtypes(
                    b.values,
                    convert_string,
                    convert_integer,
                    convert_boolean,
                    convert_floating,
                    infer_objects,
                    dtype_backend,
                )
                for b in sub_blks
            ]
            if all(dtype == self.dtype for dtype in dtypes):
                rbs.append(blk.copy(deep=False))
                continue

            for dtype, b in zip(dtypes, sub_blks):
                rbs.append(b.astype(dtype=dtype, squeeze=b.ndim != 1))
        return rbs

    # ---------------------------------------------------------------------
    # Array-Like Methods

    @final
    @cache_readonly
    def dtype(self) -> DtypeObj:
        return self.values.dtype

    @final
    def astype(
        self,
        dtype: DtypeObj,
        errors: IgnoreRaise = "raise",
        squeeze: bool = False,
    ) -> Block:
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
                raise ValueError("Can not squeeze with more than one column.")
            values = values[0, :]

        new_values = astype_array_safe(values, dtype, errors=errors)

        new_values = maybe_coerce_values(new_values)

        refs: BlockValuesRefs | None = None
        if astype_is_view(values.dtype, new_values.dtype):
            refs = self.refs

        newb: Block = self.make_block(new_values, refs=refs)
        if newb.shape != self.shape:
            raise TypeError(
                f"cannot set astype for dtype "
                f"({self.dtype.name} [{self.shape}]) to different shape "
                f"({newb.dtype.name} [{newb.shape}])"
            )
        return newb

    @final
    def get_values_for_csv(
        self, *, float_format: str, date_format: str, decimal: str, na_rep: str = "nan", quoting: Any = None
    ) -> Block:
        """convert to our native types format"""
        result = get_values_for_csv(
            self.values,
            na_rep=na_rep,
            quoting=quoting,
            float_format=float_format,
            date_format=date_format,
            decimal=decimal,
        )
        return self.make_block(result)

    @final
    def copy(self, deep: bool = True) -> Self:
        """copy constructor"""
        values = self.values
        refs: BlockValuesRefs | None
        if deep:
            values = values.copy()
            refs = None
        else:
            refs = self.refs
        return type(self)(values, placement=self._mgr_locs, ndim=self.ndim, refs=refs)

    # ---------------------------------------------------------------------
    # Copy-on-Write Helpers

    def _maybe_copy(self, inplace: bool) -> Self:
        if inplace:
            deep: bool = self.refs.has_reference()
            return self.copy(deep=deep)
        return self.copy()

    @final
    def _get_refs_and_copy(self, inplace: bool) -> tuple[bool, BlockValuesRefs | None]:
        refs: BlockValuesRefs | None = None
        copy: bool = not inplace
        if inplace:
            if self.refs.has_reference():
                copy = True
            else:
                refs = self.refs
        return copy, refs

    # ---------------------------------------------------------------------
    # Replace

    @final
    def replace(
        self,
        to_replace: Any,
        value: Any,
        inplace: bool = False,
        mask: Any = None,
    ) -> list[Block]:
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
            blk: Block = self._maybe_copy(inplace)
            putmask_inplace(blk.values, mask, value)
            return [blk]

        elif self.ndim == 1 or self.shape[0] == 1:
            if value is None or value is NA:
                blk = self.astype(np.dtype(object))
            else:
                blk = self.coerce_to_target_dtype(value, raise_on_upcast=False)
            return blk.replace(
                to_replace=to_replace,
                value=value,
                inplace=True,
                mask=mask,
            )

        else:
            blocks: list[Block] = []
            for i, nb in enumerate(self._split()):
                blocks.extend(
                    type(self).replace(
                        nb,
                        to_replace=to_replace,
                        value=value,
                        inplace=True,
                        mask=mask[i : i + 1],
                    )
                )
            return blocks

    @final
    def _replace_regex(
        self,
        to_replace: Any,
        value: Any,
        inplace: bool = False,
        mask: Any = None,
    ) -> list[Block]:
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
        if not is_re(to_replace) and not self._can_hold_element(to_replace):
            return [self.copy(deep=False)]

        if is_re(to_replace) and self.dtype not in [object, "string"]:
            return [self.copy(deep=False)]

        if not (
            self._can_hold_element(value) or (self.dtype == "string" and is_re(value))
        ):
            block = self.astype(np.dtype(object))
        else:
            block = self._maybe_copy(inplace)

        rx = re.compile(to_replace)

        replace_regex(block.values, rx, value, mask)
        return [block]

    @final
    def replace_list(
        self,
        src_list: Iterable[Any],
        dest_list: Sequence[Any],
        inplace: bool = False,
        regex: bool = False,
    ) -> list[Block]:
        """
        See BlockManager.replace_list docstring.
        """
        values = self.values

        pairs = [
            (x, y)
            for x, y in zip(src_list, dest_list)
            if (self._can_hold_element(x) or (self.dtype == "string" and is_re(x)))
        ]
        if not len(pairs):
            return [self.copy(deep=False)]

        src_len: int = len(pairs) - 1

        if is_string_dtype(values.dtype):
            na_mask = ~isna(values)
            masks: Iterable[npt.NDArray[np.bool_]] = (
                extract_bool_array(
                    cast(
                        ArrayLike,
                        compare_or_regex_search(
                            values, s[0], regex=regex, mask=na_mask
                        ),
                    )
                )
                for s in pairs
            )
        else:
            masks = (missing.mask_missing(values, s[0]) for s in pairs)
        if inplace:
            masks = list(masks)

        rb: list[Block] = [self]

        for i, ((src, dest), mask) in enumerate(zip(pairs, masks)):
            new_rb: list[Block] = []

            for blk_num, blk in enumerate(rb):
                if len(rb) == 1:
                    m = mask
                else:
                    mib = mask
                    assert not isinstance(mib, bool)
                    m = mib[blk_num : blk_num + 1]

                result = blk._replace_coerce(
                    to_replace=src,
                    value=dest,
                    mask=m,
                    inplace=inplace,
                    regex=regex,
                )

                if i != src_len:
                    for b in result:
                        ref = weakref.ref(b)
                        b.refs.referenced_blocks.pop(
                            b.refs.referenced_blocks.index(ref)
                        )
                new_rb.extend(result)
            rb = new_rb
        return rb

    @final
    def _replace_coerce(
        self,
        to_replace: Any,
        value: Any,
        mask: npt.NDArray[np.bool_],
        inplace: bool = True,
        regex: bool = False,
    ) -> list[Block]:
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
            return self._replace_regex(
                to_replace,
                value,
                inplace=inplace,
                mask=mask,
            )
        else:
            if value is None:
                if mask.any():
                    has_ref: bool = self.refs.has_reference()
                    nb = self.astype(np.dtype(object))
                    if not inplace:
                        nb = nb.copy()
                    elif inplace and has_ref and nb.refs.has_reference():
                        nb = nb.copy()
                    putmask_inplace(nb.values, mask, value)
                    return [nb]
                return [self.copy(deep=False)]
            return self.replace(
                to_replace=to_replace,
                value=value,
                inplace=inplace,
                mask=mask,
            )

    # ---------------------------------------------------------------------
    # 2D Methods - Shared by NumpyBlock and NDArrayBackedExtensionBlock
    #  but not ExtensionBlock

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
    def shape(self) -> Shape:
        return self.values.shape

    def iget(self, i: int | tuple[int, int] | tuple[slice, int]) -> np.ndarray:
        return self.values[i]  # type: ignore[index]

    def _slice(
        self, slicer: Any
    ) -> ArrayLike:
        """return a slice of my values"""

        return self.values[slicer]

    def set_inplace(self, locs: Any, values: ArrayLike, copy: bool = False) -> None:
        """
        Modify block values in-place with new item value.

        If copy=True, first copy the underlying values in place before modifying
        (for Copy-on-Write).

        Notes
        -----
        `set_inplace` never creates a new array or new Block, whereas `setitem`
        _may_ create a new array and always creates a new Block.
        """
        if copy:
            self.values = self.values.copy()
        self.values[locs] = values

    @final
    def take_nd(
        self,
        indexer: Any,
        axis: AxisInt,
        new_mgr_locs: BlockPlacement | None = None,
        fill_value: Any = lib.no_default,
    ) -> Block:
        """
        Take values according to indexer and return them as a block.
        """
        values = self.values

        if fill_value is lib.no_default:
            fill_value = self.fill_value
            allow_fill: bool = False
        else:
            allow_fill = True

        new_values = algos.take_nd(
            values, indexer, axis=axis, allow_fill=allow_fill, fill_value=fill_value
        )

        if isinstance(self, ExtensionBlock):
            assert not (self.ndim == 1 and new_mgr_locs is None)
        assert not (axis == 0 and new_mgr_locs is None)

        if new_mgr_locs is None:
            new_mgr_locs = self._mgr_locs

        if new_values.dtype != self.dtype:
            return self.make_block(new_values, new_mgr_locs)
        else:
            return self.make_block_same_class(new_values, new_mgr_locs)

    def _unstack(
        self,
        unstacker: Any,
        fill_value: Any,
        new_placement: npt.NDArray[np.intp],
        needs_masking: Any,
    ) -> tuple[list[Block], npt.NDArray[np.bool_]]:
        """
        Return a list of unstacked blocks of self

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : int
            Only used in ExtensionBlock._unstack
        new_placement : np.ndarray[np.intp]
        needs_masking : np.ndarray[bool]

        Returns
        -------
        blocks : list of Block
            New blocks of unstacked values.
        mask : array-like of bool
            The mask of columns of `blocks` we should keep.
        """
        new_values, mask = unstacker.get_new_values(
            self.values.T, fill_value=fill_value
        )

        mask = mask.any(0)
        new_values = new_values.T[mask]
        new_placement = new_placement[mask]

        bp = BlockPlacement(new_placement)
        blocks = [new_block_2d(new_values, placement=bp)]
        return blocks, mask

    def setitem(self, indexer: Any, value: Any) -> Block:
        """
        Attempt self.values[indexer] = value, possibly creating a new array.
        """
        value = self._standardize_fill_value(value)

        values = cast(np.ndarray, self.values)
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
            values = cast(np.ndarray, self.values.T)
            if isinstance(casted, np.ndarray) and casted.ndim == 1 and len(casted) == 1:
                casted = casted[0, ...]
            try:
                values[indexer] = casted
            except (TypeError, ValueError) as err:
                if is_list_like(casted):
                    raise ValueError(
                        "setting an array element with a sequence."
                    ) from err
                raise
        return self

    def putmask(self, mask: Any, new: Any) -> list[Block]:
        """
        putmask the data to the block; it is possible that we may create a
        new dtype of block

        Return the resulting block(s).

        Parameters
        ----------
        mask : np.ndarray[bool], SparseArray[bool], or BooleanArray
        new : a ndarray/object

        Returns
        -------
        List[Block]
        """
        orig_mask = mask
        values = cast(np.ndarray, self.values)
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
            values = cast(np.ndarray, self.values)

            putmask_without_repeat(values.T, mask, casted)
            return [self]
        except LossySetitemError:
            if self.ndim == 1 or self.shape[0] == 1:
                if not is_list_like(new):
                    return self.coerce_to_target_dtype(
                        new, raise_on_upcast=True
                    ).putmask(mask, new)
                else:
                    indexer = mask.nonzero()[0]
                    nb = self.setitem(indexer, new[indexer])
                    return [nb]

            else:
                is_array = isinstance(new, np.ndarray)

                res_blocks: list[Block] = []
                for i, nb in enumerate(self._split()):
                    n: Any = new
                    if is_array:
                        n = new[:, i : i + 1]

                    submask = orig_mask[:, i : i + 1]
                    rbs = nb.putmask(submask, n)
                    res_blocks.extend(rbs)
                return res_blocks

    def where(self, other: Any, cond: Any) -> list[Block]:
        """
        evaluate the block; return result block(s) from the result

        Parameters
        ----------
        other : a ndarray/object
        cond : np.ndarray[bool], SparseArray[bool], or BooleanArray

        Returns
        -------
        List[Block]
        """
        assert cond.ndim == self.ndim
        assert not isinstance(other, (ABCIndex, ABCSeries, ABCDataFrame))

        transpose: bool = self.ndim == 2

        cond = extract_bool_array(cond)

        values = cast(np.ndarray, self.values)
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

                res_blocks: list[Block] = []
                for i, nb in enumerate(self._split()):
                    oth: Any = other
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
                        "This should not be reached; call to np.where above is "
                        "expected to raise ValueError. Please report a bug at "
                        "github.com/pandas-dev/pandas"
                    )
                result = values.copy()
                np.putmask(result, icond, alt)
            else:
                if (
                    is_list_like(other)
                    and not isinstance(other, np.ndarray)
                    and len(other) == self.shape[-1]
                ):
                    other = np.array(other).reshape(values.shape)
                result = expressions.where(~icond, values, other)
            if transpose:
                result = result.T

        return [self.make_block(result)]

    def fillna(
        self,
        value: Any,
        limit: int | None = None,
        inplace: bool = False,
    ) -> list[Block]:
        """
        fillna on the block with the value. If we fail, then convert to
        block to hold objects instead and try again
        """
        inplace = validate_bool_kwarg(inplace, "inplace")

        if not self._can_hold_na:
            noop: bool = True
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
        self,
        *,
        method: FillnaOptions,
        inplace: bool = False,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ) -> list[Block]:
        if not self._can_hold_na:
            return [self.copy(deep=False)]

        copy, refs = self._get_refs_and_copy(inplace)

        vals = cast(NumpyExtensionArray, self.array_values)
        new_values = vals.T._pad_or_backfill(
            method=method,
            limit=limit,
            limit_area=limit_area,
            copy=copy,
        ).T

        data = extract_array(new_values, extract_numpy=True)
        return [self.make_block_same_class(data, refs=refs)]

    @final
    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        index: Any,
        inplace: bool = False,
        limit: Any = None,
        limit_direction: Literal["forward", "backward", "both"] = "forward",
        limit_area: Literal["inside", "outside"] | None = None,
        **kwargs: Any,
    ) -> list[Block]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        if method == "asfreq":  # type: ignore[comparison-overlap]
            missing.clean_fill_method(method)

        if not self._can_hold_na:
            return [self.copy(deep=False)]

        if self.dtype == _dtype_obj:
            name = {1: "Series", 2: "DataFrame"}[self.ndim]
            raise TypeError(f"{name} cannot interpolate with object dtype.")

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
    def diff(self, n: int) -> list[Block]:
        """return block for the diff of the values"""
        new_values = algos.diff(self.values.T, n, axis=0).T
        return [self.make_block(values=new_values)]

    def shift(self, periods: int, fill_value: Any = None) -> list[Block]:
        """shift the block by periods, possibly upcast"""
        axis: int = self.ndim - 1

        if not lib.is_scalar(fill_value) and self.dtype != _dtype_obj:
            raise ValueError("fill_value must be a scalar")

        fill_value = self._standardize_fill_value(fill_value)

        try:
            casted = np_can_hold_element(
                self.dtype,  
                fill_value,
            )
        except LossySetitemError:
            nb = self.coerce_to_target_dtype(fill_value, raise_on_upcast=False)
            return nb.shift(periods, fill_value=fill_value)
        else:
            values = cast(np.ndarray, self.values)
            new_values = shift(values, periods, axis, casted)
            return [self.make_block_same_class(new_values)]

    @final
    def quantile(
        self,
        qs: Any,
        interpolation: str = "linear",
    ) -> Block:
        """
        compute the quantiles of the

        Parameters
        ----------
        qs : Index
            The quantiles to be computed in float64.
        interpolation : str, default 'linear'
            Type of interpolation.

        Returns
        -------
        Block
        """
        assert self.ndim == 2
        assert is_list_like(qs)
        result = quantile_compat(self.values, np.asarray(qs._values), interpolation)
        result = ensure_block_shape(result, ndim=2)
        return new_block_2d(result, placement=self._mgr_locs)

    @final
    def round(self, decimals: int) -> Self:
        """
        Rounds the values.
        If the block is not of an integer or float dtype, nothing happens.
        This is consistent with DataFrame.round behavior.
        (Note: Series.round would raise)

        Parameters
        ----------
        decimals: int,
            Number of decimal places to round to.
        """
        if not self.is_numeric or self.is_bool:
            return self.copy(deep=False)
        values = self.values.round(decimals)  # type: ignore[union-attr]

        refs: BlockValuesRefs | None = None
        if values is self.values:
            refs = self.refs

        return self.make_block_same_class(values, refs=refs)

    # ---------------------------------------------------------------------
    # Abstract Methods Overridden By EABackedBlock and NumpyBlock

    def delete(self, loc: Any) -> list[Block]:
        """Deletes the locs from the block.
        """
        if not is_list_like(loc):
            loc = [loc]

        if self.ndim == 1:
            values = cast(np.ndarray, self.values)
            values = np.delete(values, loc)
            mgr_locs = self._mgr_locs.delete(loc)
            return [type(self)(values, placement=mgr_locs, ndim=self.ndim)]

        if np.max(loc) >= self.values.shape[0]:
            raise IndexError

        loc = np.concatenate([loc, [self.values.shape[0]]])
        mgr_locs_arr = self._mgr_locs.as_array
        new_blocks: list[Block] = []

        previous_loc: int = -1
        refs: BlockValuesRefs | None = self.refs if self.refs.has_reference() else None
        for idx in loc:
            if idx == previous_loc + 1:
                pass
            else:
                values = self.values[previous_loc + 1 : idx, :]  # type: ignore[call-overload]
                locs = mgr_locs_arr[previous_loc + 1 : idx]
                nb = type(self)(
                    values, placement=BlockPlacement(locs), ndim=self.ndim, refs=refs
                )
                new_blocks.append(nb)

            previous_loc = idx

        return new_blocks

    @property
    def is_view(self) -> bool:
        raise AbstractMethodError(self)

    @property
    def array_values(self) -> ExtensionArray:
        raise AbstractMethodError(self)

    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray:
        raise AbstractMethodError(self)


class EABackedBlock(Block):
    """
    Mixin for Block subclasses backed by ExtensionArray.
    """

    values: ExtensionArray

    @final
    def shift(self, periods: int, fill_value: Any = None) -> list[Block]:
        new_values = self.values.T.shift(periods=periods, fill_value=fill_value).T
        return [self.make_block_same_class(new_values)]

    @final
    def setitem(self, indexer: Any, value: Any) -> Block:
        orig_indexer: Any = indexer
        orig_value: Any = value

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
    def where(self, other: Any, cond: Any) -> list[Block]:
        arr = self.values.T

        cond = extract_bool_array(cond)

        orig_other: Any = other
        orig_cond: Any = cond
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
                    if (
                        self.ndim == 2
                        and isinstance(orig_cond, np.ndarray)
                        and orig_cond.ndim == 1
                        and not is_1d_only_ea_dtype(blk.dtype)
                    ):
                        orig_cond = orig_cond[:, None]
                    return blk.where(orig_other, orig_cond)
                elif isinstance(self, NDArrayBackedExtensionBlock):
                    blk = self.coerce_to_target_dtype(orig_other, raise_on_upcast=False)
                    return blk.where(orig_other, orig_cond)
                else:
                    raise
            else:
                is_array = isinstance(orig_other, (np.ndarray, ExtensionArray))

                res_blocks: list[Block] = []
                for i, nb in enumerate(self._split()):
                    n: Any = orig_other
                    if is_array:
                        n = orig_other[:, i : i + 1]
                    submask = orig_cond[:, i : i + 1]
                    rbs = nb.where(n, submask)
                    res_blocks.extend(rbs)
                return res_blocks

        nb = self.make_block_same_class(res_values)
        return [nb]

    @final
    def putmask(self, mask: Any, new: Any) -> list[Block]:
        mask = extract_bool_array(mask)
        if new is lib.no_default:
            new = self.fill_value

        orig_new: Any = new
        orig_mask: Any = mask
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

                res_blocks: list[Block] = []
                for i, nb in enumerate(self._split()):
                    n: Any = orig_new
                    if is_array:
                        n = orig_new[:, i : i + 1]
                    submask = orig_mask[:, i : i + 1]
                    rbs = nb.putmask(submask, n)
                    res_blocks.extend(rbs)
                return res_blocks

        return [self]

    @final
    def delete(self, loc: Any) -> list[Block]:
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
    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray:
        values: ArrayLike = self.values
        if dtype == _dtype_obj:
            values = values.astype(object)
        return np.asarray(values).reshape(self.shape)

    @final
    def pad_or_backfill(
        self,
        *,
        method: Any,
        inplace: bool = False,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ) -> list[Block]:
        values = self.values

        kwargs: dict[str, Any] = {"method": method, "limit": limit}
        if "limit_area" in inspect.signature(values._pad_or_backfill).parameters:
            kwargs["limit_area"] = limit_area
        elif limit_area is not None:
            raise NotImplementedError(
                f"{type(values).__name__} does not implement limit_area "
                "(added in pandas 2.2). 3rd-party ExtensionArray authors "
                "need to add this argument to _pad_or_backfill."
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

    values: ExtensionArray

    def fillna(
        self,
        value: Any,
        limit: int | None = None,
        inplace: bool = False,
    ) -> list[Block]:
        if isinstance(self.dtype, (IntervalDtype, StringDtype)):
            if isinstance(self.dtype, IntervalDtype) and limit is not None:
                raise ValueError("limit must be None")
            return super().fillna(
                value=value,
                limit=limit,
                inplace=inplace,
            )
        if self._can_hold_na and not self.values._hasna:
            refs: BlockValuesRefs | None = self.refs
            new_values: ExtensionArray = self.values
        else:
            copy, refs = self._get_refs_and_copy(inplace)
            try:
                new_values = self.values.fillna(value=value, limit=limit, copy=copy)
            except TypeError:
                refs = None
                new_values = self.values.fillna(value=value, limit=limit)
                warnings.warn(
                    "ExtensionArray.fillna added a 'copy' keyword in pandas "
                    "2.1.0. In a future version, ExtensionArray subclasses will "
                    "need to implement this keyword or an exception will be "
                    "raised. In the interim, the keyword is ignored by "
                    f"{type(self.values).__name__}.",
                    DeprecationWarning,
                    stacklevel=find_stack_level(),
                )

        return [self.make_block_same_class(new_values, refs=refs)]

    @cache_readonly
    def shape(self) -> Shape:
        if self.ndim == 1:
            return (len(self.values),)
        return len(self._mgr_locs), len(self.values)

    def iget(self, i: int | tuple[int, int] | tuple[slice, int]) -> Any:
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
        if (
            isinstance(arg, (np.ndarray, ExtensionArray))
            and arg.ndim == self.values.ndim + 1
        ):
            assert arg.shape[1] == 1
            arg = arg[:, 0]  # type: ignore[call-overload]
        elif isinstance(arg, ABCDataFrame):
            assert arg.shape[1] == 1
            arg = arg._ixs(0, axis=1)._values
        return arg

    def _unwrap_setitem_indexer(self, indexer: Any) -> Any:
        if isinstance(indexer, tuple) and len(indexer) == 2:
            if all(isinstance(x, np.ndarray) and x.ndim == 2 for x in indexer):
                first, second = indexer
                if not (
                    second.size == 1 and (second == 0).all() and first.shape[1] == 1
                ):
                    raise NotImplementedError(
                        "This should not be reached. Please report a bug at "
                        "github.com/pandas-dev/pandas/"
                    )
                indexer = first[:, 0]
            elif lib.is_integer(indexer[1]) and indexer[1] == 0:
                indexer = indexer[0]
            elif com.is_null_slice(indexer[1]):
                indexer = indexer[0]
            elif is_list_like(indexer[1]) and indexer[1][0] == 0:
                indexer = indexer[0]
            else:
                raise NotImplementedError(
                    "This should not be reached. Please report a bug at "
                    "github.com/pandas-dev/pandas/"
                )
        return indexer

    @property
    def is_view(self) -> bool:
        return False

    @cache_readonly
    def is_numeric(self) -> bool:  # type: ignore[override]
        return self.values.dtype._is_numeric

    def _slice(
        self, slicer: slice | npt.NDArray[np.bool_] | npt.NDArray[np.intp]
    ) -> ExtensionArray:
        if self.ndim == 2:
            if not isinstance(slicer, slice):
                raise AssertionError(
                    "invalid slicing for a 1-ndim ExtensionArray", slicer
                )
            new_locs = range(1)[slicer]
            if not len(new_locs):
                raise AssertionError(
                    "invalid slicing for a 1-ndim ExtensionArray", slicer
                )
            slicer = slice(None)
        return self.values[slicer]

    @final
    def slice_block_rows(self, slicer: slice) -> Self:
        new_values: ExtensionArray = self.values[slicer]
        return type(self)(new_values, self._mgr_locs, ndim=self.ndim, refs=self.refs)

    def _unstack(
        self,
        unstacker: Any,
        fill_value: Any,
        new_placement: npt.NDArray[np.intp],
        needs_masking: Any,
    ) -> tuple[list[Block], npt.NDArray[np.bool_]]:
        new_values, mask = unstacker.arange_result
        new_values = new_values.T[mask]
        new_placement = new_placement[mask]

        blocks = [
            type(self)(
                self.values.take(
                    indices, allow_fill=needs_masking[i], fill_value=fill_value
                ),
                BlockPlacement(place),
                ndim=2,
            )
            for i, (indices, place) in enumerate(zip(new_values, new_placement))
        ]
        return blocks, mask


class NumpyBlock(Block):
    values: np.ndarray
    __slots__ = ()

    @property
    def is_view(self) -> bool:
        return self.values.base is not None

    @property
    def array_values(self) -> ExtensionArray:
        return NumpyExtensionArray(self.values)

    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray:
        if dtype == _dtype_obj:
            return self.values.astype(_dtype_obj)
        return self.values

    @cache_readonly
    def is_numeric(self) -> bool:  # type: ignore[override]
        dtype = self.values.dtype
        kind = dtype.kind

        return kind in "fciub"


class NDArrayBackedExtensionBlock(EABackedBlock):
    """
    Block backed by an NDArrayBackedExtensionArray
    """

    values: NDArrayBackedExtensionArray

    @property
    def is_view(self) -> bool:
        return self.values._ndarray.base is not None


class DatetimeLikeBlock(NDArrayBackedExtensionBlock):
    """Block for datetime64[ns], timedelta64[ns]."""

    __slots__ = ()
    is_numeric = False
    values: DatetimeArray | TimedeltaArray


def maybe_coerce_values(values: ArrayLike) -> ArrayLike:
    """
    Input validation for values passed to __init__. Ensure that
    any datetime64/timedelta64 dtypes are in nanoseconds.  Ensure
    that we do not have string dtypes.
    """
    if isinstance(values, np.ndarray):
        values = ensure_wrapped_if_datetimelike(values)

        if issubclass(values.dtype.type, str):
            values = np.array(values, dtype=object)

    if isinstance(values, (DatetimeArray, TimedeltaArray)) and values.freq is not None:
        values = values._with_freq(None)

    return values


def get_block_type(dtype: DtypeObj) -> type[Block]:
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


def new_block_2d(
    values: ArrayLike, placement: BlockPlacement, refs: BlockValuesRefs | None = None
) -> Block:
    klass = get_block_type(values.dtype)

    values = maybe_coerce_values(values)
    return klass(values, ndim=2, placement=placement, refs=refs)


def new_block(
    values: ArrayLike,
    placement: BlockPlacement,
    *,
    ndim: int,
    refs: BlockValuesRefs | None = None,
) -> Block:
    klass = get_block_type(values.dtype)
    return klass(values, ndim=ndim, placement=placement, refs=refs)


def check_ndim(values: ArrayLike, placement: BlockPlacement, ndim: int) -> None:
    if values.ndim > ndim:
        raise ValueError(
            f"Wrong number of dimensions. values.ndim > ndim [{values.ndim} > {ndim}]"
        )

    if not is_1d_only_ea_dtype(values.dtype):
        if values.ndim != ndim:
            raise ValueError(
                "Wrong number of dimensions. "
                f"values.ndim != ndim [{values.ndim} != {ndim}]"
            )
        if len(placement) != len(values):
            raise ValueError(
                f"Wrong number of items passed {len(values)}, "
                f"placement implies {len(placement)}"
            )
    elif ndim == 2 and len(placement) != 1:
        raise ValueError("need to split")


def extract_pandas_array(
    values: ArrayLike, dtype: DtypeObj | None, ndim: int
) -> tuple[ArrayLike, DtypeObj | None]:
    if isinstance(values, ABCNumpyExtensionArray):
        values = values.to_numpy()
        if ndim and ndim > 1:
            values = np.atleast_2d(values)

    if isinstance(dtype, NumpyEADtype):
        dtype = dtype.numpy_dtype

    return values, dtype


def extend_blocks(result: Block | list[Block], blocks: list[Block] | None = None) -> list[Block]:
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


def ensure_block_shape(values: ArrayLike, ndim: int = 1) -> ArrayLike:
    if values.ndim < ndim:
        if not is_1d_only_ea_dtype(values.dtype):
            values = cast("np.ndarray | DatetimeArray | TimedeltaArray", values)
            values = values.reshape(1, -1)

    return values


def external_values(values: ArrayLike) -> ArrayLike:
    if isinstance(values, (PeriodArray, IntervalArray)):
        return values.astype(object)
    elif isinstance(values, (DatetimeArray, TimedeltaArray)):
        values = values._ndarray

    if isinstance(values, np.ndarray):
        values = values.view()
        values.flags.writeable = False

    return values