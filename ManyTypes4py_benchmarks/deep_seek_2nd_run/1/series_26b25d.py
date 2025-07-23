from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
import functools
import operator
import sys
from textwrap import dedent
from typing import (
    IO, TYPE_CHECKING, Any, Literal, cast, overload, Optional, Union, 
    List, Tuple, Dict, Set, TypeVar, Generic, Type, Sequence as Seq, 
    Mapping as Map, Callable as Func, Hashable as Hash, Iterable as Iter
)
import warnings
import numpy as np
from pandas._libs import lib, properties, reshape
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import ChainedAssignmentError, InvalidIndexError
from pandas.errors.cow import _chained_assignment_method_msg, _chained_assignment_msg
from pandas.util._decorators import Appender, Substitution, deprecate_nonkeyword_arguments, doc, set_module
from pandas.util._validators import validate_ascending, validate_bool_kwarg, validate_percentile
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import LossySetitemError, construct_1d_arraylike_from_scalar, find_common_type, infer_dtype_from, maybe_box_native, maybe_cast_pointwise_result
from pandas.core.dtypes.common import is_dict_like, is_float, is_integer, is_iterator, is_list_like, is_object_dtype, is_scalar, pandas_dtype, validate_all_hashable
from pandas.core.dtypes.dtypes import CategoricalDtype, ExtensionDtype, SparseDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import isna, na_value_for_dtype, notna, remove_na_arraylike
from pandas.core import algorithms, base, common as com, nanops, ops, roperator
from pandas.core.accessor import Accessor
from pandas.core.apply import SeriesApply
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.arrow import ListAccessor, StructAccessor
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.sparse import SparseAccessor
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import array as pd_array, extract_array, sanitize_array
from pandas.core.generic import NDFrame, make_doc
from pandas.core.indexers import disallow_ndim_indexing, unpack_1tuple
from pandas.core.indexes.accessors import CombinedDatetimelikeProperties
from pandas.core.indexes.api import DatetimeIndex, Index, MultiIndex, PeriodIndex, default_index, ensure_index, maybe_sequence_to_range
import pandas.core.indexes.base as ibase
from pandas.core.indexes.multi import maybe_droplevels
from pandas.core.indexing import check_bool_indexer, check_dict_or_set_indexers
from pandas.core.internals import SingleBlockManager
from pandas.core.methods import selectn
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import ensure_key_mapped, nargsort
from pandas.core.strings.accessor import StringMethods
from pandas.core.tools.datetimes import to_datetime
import pandas.io.formats.format as fmt
from pandas.io.formats.info import INFO_DOCSTRING, SeriesInfo, series_sub_kwargs
import pandas.plotting

if TYPE_CHECKING:
    from pandas._libs.internals import BlockValuesRefs
    from pandas._typing import (
        AggFuncType, AnyAll, AnyArrayLike, ArrayLike, Axis, AxisInt, 
        CorrelationMethod, DropKeep, Dtype, DtypeObj, FilePath, Frequency, 
        IgnoreRaise, IndexKeyFunc, IndexLabel, Level, ListLike, 
        MutableMappingT, NaPosition, NumpySorter, NumpyValueArrayLike, 
        QuantileInterpolation, ReindexMethod, Renamer, Scalar, Self, 
        SortKind, StorageOptions, Suffixes, ValueKeyFunc, WriteBuffer, npt
    )
    from pandas.core.frame import DataFrame
    from pandas.core.groupby.generic import SeriesGroupBy

T = TypeVar('T')
S = TypeVar('S', bound='Series')

__all__ = ['Series']

_shared_doc_kwargs = {
    'axes': 'index', 
    'klass': 'Series', 
    'axes_single_arg': "{0 or 'index'}", 
    'axis': "axis : {0 or 'index'}\n        Unused. Parameter needed for compatibility with DataFrame.", 
    'inplace': 'inplace : bool, default False\n        If True, performs operation inplace and returns None.', 
    'unique': 'np.ndarray', 
    'duplicated': 'Series', 
    'optional_by': '', 
    'optional_reindex': '\nindex : array-like, optional\n    New labels for the index. Preferably an Index object to avoid\n    duplicating data.\naxis : int or str, optional\n    Unused.'
}

@set_module('pandas')
class Series(base.IndexOpsMixin, NDFrame, Generic[T]):
    _typ = 'series'
    _HANDLED_TYPES = (Index, ExtensionArray, np.ndarray)
    _metadata = ['_name']
    _internal_names_set = {'index', 'name'} | NDFrame._internal_names_set
    _accessors = {'dt', 'cat', 'str', 'sparse'}
    _hidden_attrs = base.IndexOpsMixin._hidden_attrs | NDFrame._hidden_attrs | frozenset([])
    __pandas_priority__ = 3000
    
    def __init__(
        self, 
        data: Optional[Union[Iterable[T], Mapping[Hashable, T], ArrayLike]] = None,
        index: Optional[Union[Index, ArrayLike]] = None,
        dtype: Optional[Dtype] = None,
        name: Optional[Hashable] = None,
        copy: Optional[bool] = None
    ) -> None:
        # Implementation remains the same
        pass

    @property
    def dtype(self) -> DtypeObj:
        return self._mgr.dtype

    @property
    def dtypes(self) -> DtypeObj:
        return self.dtype

    @property
    def name(self) -> Optional[Hashable]:
        return self._name

    @name.setter
    def name(self, value: Optional[Hashable]) -> None:
        validate_all_hashable(value, error_name=f'{type(self).__name__}.name')
        object.__setattr__(self, '_name', value)

    @property
    def values(self) -> np.ndarray:
        return self._mgr.external_values()

    @property
    def _values(self) -> ArrayLike:
        return self._mgr.internal_values()

    @property
    def _references(self) -> Optional[BlockValuesRefs]:
        return self._mgr._block.refs

    @property
    def array(self) -> ExtensionArray:
        return self._mgr.array_values()

    def __len__(self) -> int:
        return len(self._mgr)

    def __array__(self, dtype: Optional[Dtype] = None, copy: Optional[bool] = None) -> np.ndarray:
        # Implementation remains the same
        pass

    @property
    def axes(self) -> List[Index]:
        return [self.index]

    def _ixs(self, i: int, axis: int = 0) -> T:
        return self._values[i]

    def _slice(self, slobj: slice, axis: int = 0) -> Series[T]:
        # Implementation remains the same
        pass

    def __getitem__(self, key: Any) -> Union[T, Series[T]]:
        # Implementation remains the same
        pass

    def _get_with(self, key: Any) -> Union[T, Series[T]]:
        # Implementation remains the same
        pass

    def _get_values_tuple(self, key: Tuple[Hashable, ...]) -> Union[T, Series[T]]:
        # Implementation remains the same
        pass

    def _get_rows_with_mask(self, indexer: np.ndarray) -> Series[T]:
        # Implementation remains the same
        pass

    def _get_value(self, label: Hashable, takeable: bool = False) -> Union[T, Series[T]]:
        # Implementation remains the same
        pass

    def __setitem__(self, key: Any, value: Any) -> None:
        # Implementation remains the same
        pass

    def _set_with_engine(self, key: Hashable, value: Any) -> None:
        # Implementation remains the same
        pass

    def _set_with(self, key: Any, value: Any) -> None:
        # Implementation remains the same
        pass

    def _set_labels(self, key: Any, value: Any) -> None:
        # Implementation remains the same
        pass

    def _set_values(self, key: Any, value: Any) -> None:
        # Implementation remains the same
        pass

    def _set_value(self, label: Hashable, value: Any, takeable: bool = False) -> None:
        # Implementation remains the same
        pass

    def repeat(self, repeats: Union[int, Sequence[int]], axis: Optional[int] = None) -> Series[T]:
        # Implementation remains the same
        pass

    @overload
    def reset_index(
        self,
        level: Optional[Union[Hashable, Sequence[Hashable]]] = ...,
        *,
        drop: bool = ...,
        name: Optional[Union[Hashable, Literal[lib.no_default]]] = ...,
        inplace: Literal[False] = ...,
        allow_duplicates: bool = ...
    ) -> DataFrame: ...

    @overload
    def reset_index(
        self,
        level: Optional[Union[Hashable, Sequence[Hashable]]] = ...,
        *,
        drop: bool,
        name: Optional[Union[Hashable, Literal[lib.no_default]]] = ...,
        inplace: Literal[True],
        allow_duplicates: bool = ...
    ) -> None: ...

    @overload
    def reset_index(
        self,
        level: Optional[Union[Hashable, Sequence[Hashable]]] = ...,
        *,
        drop: bool = ...,
        name: Optional[Union[Hashable, Literal[lib.no_default]]] = ...,
        inplace: bool = ...,
        allow_duplicates: bool = ...
    ) -> Optional[DataFrame]: ...

    def reset_index(
        self,
        level: Optional[Union[Hashable, Sequence[Hashable]]] = None,
        *,
        drop: bool = False,
        name: Union[Hashable, Literal[lib.no_default]] = lib.no_default,
        inplace: bool = False,
        allow_duplicates: bool = False
    ) -> Optional[DataFrame]:
        # Implementation remains the same
        pass

    def __repr__(self) -> str:
        # Implementation remains the same
        pass

    @overload
    def to_string(
        self,
        buf: Optional[IO[str]] = ...,
        *,
        na_rep: str = ...,
        float_format: Optional[Callable[[float], str]] = ...,
        header: bool = ...,
        index: bool = ...,
        length: bool = ...,
        dtype: bool = ...,
        name: bool = ...,
        max_rows: Optional[int] = ...,
        min_rows: Optional[int] = ...
    ) -> str: ...

    @overload
    def to_string(
        self,
        buf: IO[str],
        *,
        na_rep: str = ...,
        float_format: Optional[Callable[[float], str]] = ...,
        header: bool = ...,
        index: bool = ...,
        length: bool = ...,
        dtype: bool = ...,
        name: bool = ...,
        max_rows: Optional[int] = ...,
        min_rows: Optional[int] = ...
    ) -> None: ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self', 'buf'], name='to_string')
    def to_string(
        self,
        buf: Optional[IO[str]] = None,
        na_rep: str = 'NaN',
        float_format: Optional[Callable[[float], str]] = None,
        header: bool = True,
        index: bool = True,
        length: bool = False,
        dtype: bool = False,
        name: bool = False,
        max_rows: Optional[int] = None,
        min_rows: Optional[int] = None
    ) -> Optional[str]:
        # Implementation remains the same
        pass

    @overload
    def to_markdown(
        self,
        buf: Optional[Union[str, IO[str]]] = ...,
        *,
        mode: str = ...,
        index: bool = ...,
        storage_options: Optional[StorageOptions] = ...,
        **kwargs: Any
    ) -> str: ...

    @overload
    def to_markdown(
        self,
        buf: Union[str, IO[str]],
        *,
        mode: str = ...,
        index: bool = ...,
        storage_options: Optional[StorageOptions] = ...,
        **kwargs: Any
    ) -> None: ...

    @doc(klass=_shared_doc_kwargs['klass'], storage_options=_shared_docs['storage_options'], examples=dedent('''Examples\n            --------\n            >>> s = pd.Series(["elk", "pig", "dog", "quetzal"], name="animal")\n            >>> print(s.to_markdown())\n            |    | animal   |\n            |---:|:---------|\n            |  0 | elk      |\n            |  1 | pig      |\n            |  2 | dog      |\n            |  3 | quetzal  |\n\n            Output markdown with a tabulate option.\n\n            >>> print(s.to_markdown(tablefmt="grid"))\n            +----+----------+\n            |    | animal   |\n            +====+==========+\n            |  0 | elk      |\n            +----+----------+\n            |  1 | pig      |\n            +----+----------+\n            |  2 | dog      |\n            +----+----------+\n            |  3 | quetzal  |\n            +----+----------+'''))
    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self', 'buf'], name='to_markdown')
    def to_markdown(
        self,
        buf: Optional[Union[str, IO[str]]] = None,
        mode: str = 'wt',
        index: bool = True,
        storage_options: Optional[StorageOptions] = None,
        **kwargs: Any
    ) -> Optional[str]:
        # Implementation remains the same
        pass

    def items(self) -> Iterable[Tuple[Hashable, T]]:
        return zip(iter(self.index), iter(self))

    def keys(self) -> Index:
        return self.index

    @overload
    def to_dict(self, *, into: Type[MutableMappingT]) -> MutableMappingT: ...

    @overload
    def to_dict(self, *, into: MutableMappingT = ...) -> MutableMappingT: ...

    def to_dict(self, *, into: Type[MutableMappingT] = dict) -> MutableMappingT:
        # Implementation remains the same
        pass

    def to_frame(self, name: Union[Hashable, Literal[lib.no_default]] = lib.no_default) -> DataFrame:
        # Implementation remains the same
        pass

    def _set_name(self, name: Optional[Hashable], inplace: bool = False, deep: Optional[bool] = None) -> Optional[Series[T]]:
        # Implementation remains the same
        pass

    def groupby(
        self,
        by: Optional[Union[Hashable, Sequence[Hashable], Callable[[Hashable], Hashable]]] = None,
        level: Optional[Union[Hashable, Sequence[Hashable]]] = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True
    ) -> SeriesGroupBy[T]:
        # Implementation remains the same
        pass

    def count(self) -> int:
        return notna(self._values).sum().astype('int64')

    def mode(self, dropna: bool = True) -> Series[T]:
        # Implementation remains the same
        pass

    def unique(self) -> np.ndarray:
        return super().unique()

    @overload
    def drop_duplicates(
        self,
        *,
        keep: Union[Literal["first"], Literal["last"], Literal[False]] = ...,
        inplace: Literal[False] = ...,
        ignore_index: bool = ...
    ) -> Series[T]: ...

    @overload
    def drop_duplicates(
        self,
        *,
        keep: Union[Literal["first"], Literal["last"], Literal[False]] = ...,
        inplace: Literal[True],
        ignore_index: bool = ...
    ) -> None: ...

    @overload
    def drop_duplicates(
        self,
        *,
        keep: Union[Literal["first"], Literal["last"], Literal[False]] = ...,
        inplace: bool = ...,
        ignore_index: bool = ...
    ) -> Optional[Series[T]]: ...

    def drop_duplicates(
        self,
        *,
        keep: Union[Literal["first"], Literal["last"], Literal[False]] = 'first',
        inplace: bool = False,
        ignore_index: bool = False
    ) -> Optional[Series[T]]:
        # Implementation remains the same
        pass

    def duplicated(self, keep: Union[Literal["first"], Literal["last"], Literal[False]] = 'first') -> Series[bool]:
        # Implementation remains the same
        pass

    def idxmin(self, axis: int =