from __future__ import annotations
import abc
from collections import defaultdict
from collections.abc import Callable, Generator, Hashable, Iterable, MutableMapping, Sequence
import functools
from functools import partial
import inspect
from typing import TYPE_CHECKING, Any, Literal, cast, Optional, Union, Dict, List, Tuple, DefaultDict
import numpy as np
from pandas._libs.internals import BlockValuesRefs
from pandas._typing import AggFuncType, AggFuncTypeBase, AggFuncTypeDict, AggObjType, Axis, AxisInt, NDFrameT, npt
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SpecificationError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import is_nested_object
from pandas.core.dtypes.common import is_dict_like, is_extension_array_dtype, is_list_like, is_numeric_dtype, is_sequence
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCNDFrame, ABCSeries
from pandas.core._numba.executor import generate_apply_looper
import pandas.core.common as com
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.util.numba_ import get_jit_arguments, prepare_function_arguments

if TYPE_CHECKING:
    from pandas import DataFrame, Index, Series
    from pandas.core.groupby import GroupBy
    from pandas.core.resample import Resampler
    from pandas.core.window.rolling import BaseWindow

ResType = Dict[int, Any]

def frame_apply(
    obj: Union[DataFrame, Series],
    func: Union[str, Callable, List[Union[str, Callable]], Dict[Hashable, Union[str, Callable]]],
    axis: Axis = 0,
    raw: bool = False,
    result_type: Optional[Literal['reduce', 'broadcast', 'expand']] = None,
    by_row: Union[bool, Literal['compat', '_compat'] = 'compat',
    engine: str = 'python',
    engine_kwargs: Optional[Dict[str, Any]] = None,
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None
) -> Union[FrameRowApply, FrameColumnApply]:
    """construct and return a row or column based frame apply object"""
    _, func, columns, _ = reconstruct_func(func, **(kwargs or {}))
    axis_num = obj._get_axis_number(axis)
    if axis_num == 0:
        klass = FrameRowApply
    elif axis_num == 1:
        if columns:
            raise NotImplementedError(f'Named aggregation is not supported when axis={axis!r}.')
        klass = FrameColumnApply
    return klass(
        obj, func, raw=raw, result_type=result_type, by_row=by_row,
        engine=engine, engine_kwargs=engine_kwargs, args=args, kwargs=kwargs
    )

class Apply(metaclass=abc.ABCMeta):
    def __init__(
        self,
        obj: Union[DataFrame, Series, GroupBy, Resampler, BaseWindow],
        func: Union[str, Callable, List[Union[str, Callable]], Dict[Hashable, Union[str, Callable]]],
        raw: bool,
        result_type: Optional[Literal['reduce', 'broadcast', 'expand']],
        *,
        by_row: Union[bool, Literal['compat', '_compat']] = 'compat',
        engine: str = 'python',
        engine_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[Tuple[Any, ...]],
        kwargs: Optional[Dict[str, Any]]
    ) -> None:
        self.obj = obj
        self.raw = raw
        assert by_row is False or by_row in ['compat', '_compat']
        self.by_row = by_row
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.engine = engine
        self.engine_kwargs = {} if engine_kwargs is None else engine_kwargs
        if result_type not in [None, 'reduce', 'broadcast', 'expand']:
            raise ValueError("invalid value for result_type, must be one of {None, 'reduce', 'broadcast', 'expand'}")
        self.result_type = result_type
        self.func = func

    @abc.abstractmethod
    def apply(self) -> Any:
        pass

    @abc.abstractmethod
    def agg_or_apply_list_like(self, op_name: str) -> Any:
        pass

    @abc.abstractmethod
    def agg_or_apply_dict_like(self, op_name: str) -> Any:
        pass

    def agg(self) -> Optional[Any]:
        """
        Provide an implementation for the aggregators.

        Returns
        -------
        Result of aggregation, or None if agg cannot be performed by
        this method.
        """
        func = self.func
        if isinstance(func, str):
            return self.apply_str()
        if is_dict_like(func):
            return self.agg_dict_like()
        elif is_list_like(func):
            return self.agg_list_like()
        return None

    def transform(self) -> Any:
        """
        Transform a DataFrame or Series.

        Returns
        -------
        DataFrame or Series
            Result of applying ``func`` along the given axis of the
            Series or DataFrame.

        Raises
        ------
        ValueError
            If the transform function fails or does not transform.
        """
        obj = self.obj
        func = self.func
        axis = self.axis
        args = self.args
        kwargs = self.kwargs
        is_series = obj.ndim == 1
        if obj._get_axis_number(axis) == 1:
            assert not is_series
            return obj.T.transform(func, 0, *args, **kwargs).T
        if is_list_like(func) and (not is_dict_like(func)):
            func = cast(List[AggFuncTypeBase], func)
            if is_series:
                func = {com.get_callable_name(v) or v: v for v in func}
            else:
                func = {col: func for col in obj}
        if is_dict_like(func):
            func = cast(AggFuncTypeDict, func)
            return self.transform_dict_like(func)
        func = cast(AggFuncTypeBase, func)
        try:
            result = self.transform_str_or_callable(func)
        except TypeError:
            raise
        except Exception as err:
            raise ValueError('Transform function failed') from err
        if isinstance(result, (ABCSeries, ABCDataFrame)) and result.empty and (not obj.empty):
            raise ValueError('Transform function failed')
        if not isinstance(result, (ABCSeries, ABCDataFrame)) or not result.index.equals(obj.index):
            raise ValueError('Function did not transform')
        return result

    def transform_dict_like(self, func: AggFuncTypeDict) -> Any:
        """
        Compute transform in the case of a dict-like func
        """
        from pandas.core.reshape.concat import concat
        obj = self.obj
        args = self.args
        kwargs = self.kwargs
        assert isinstance(obj, ABCNDFrame)
        if len(func) == 0:
            raise ValueError('No transform functions were provided')
        func = self.normalize_dictlike_arg('transform', obj, func)
        results = {}
        for name, how in func.items():
            colg = obj._gotitem(name, ndim=1)
            results[name] = colg.transform(how, 0, *args, **kwargs)
        return concat(results, axis=1)

    def transform_str_or_callable(self, func: Union[str, Callable]) -> Any:
        """
        Compute transform in the case of a string or callable func
        """
        obj = self.obj
        args = self.args
        kwargs = self.kwargs
        if isinstance(func, str):
            return self._apply_str(obj, func, *args, **kwargs)
        try:
            return obj.apply(func, args=args, **kwargs)
        except Exception:
            return func(obj, *args, **kwargs)

    def agg_list_like(self) -> Any:
        """
        Compute aggregation in the case of a list-like argument.

        Returns
        -------
        Result of aggregation.
        """
        return self.agg_or_apply_list_like(op_name='agg')

    def compute_list_like(
        self,
        op_name: str,
        selected_obj: Union[DataFrame, Series],
        kwargs: Dict[str, Any]
    ) -> Tuple[Union[List[Hashable], Index], List[Any]]:
        """
        Compute agg/apply results for like-like input.

        Parameters
        ----------
        op_name : {"agg", "apply"}
            Operation being performed.
        selected_obj : Series or DataFrame
            Data to perform operation on.
        kwargs : dict
            Keyword arguments to pass to the functions.

        Returns
        -------
        keys : list[Hashable] or Index
            Index labels for result.
        results : list
            Data for result. When aggregating with a Series, this can contain any
            Python objects.
        """
        func = cast(List[AggFuncTypeBase], self.func)
        obj = self.obj
        results = []
        keys = []
        if selected_obj.ndim == 1:
            for a in func:
                colg = obj._gotitem(selected_obj.name, ndim=1, subset=selected_obj)
                args = [self.axis, *self.args] if include_axis(op_name, colg) else self.args
                new_res = getattr(colg, op_name)(a, *args, **kwargs)
                results.append(new_res)
                name = com.get_callable_name(a) or a
                keys.append(name)
        else:
            indices = []
            for index, col in enumerate(selected_obj):
                colg = obj._gotitem(col, ndim=1, subset=selected_obj.iloc[:, index])
                args = [self.axis, *self.args] if include_axis(op_name, colg) else self.args
                new_res = getattr(colg, op_name)(func, *args, **kwargs)
                results.append(new_res)
                indices.append(index)
            keys = selected_obj.columns.take(indices)
        return (keys, results)

    def wrap_results_list_like(self, keys: Union[List[Hashable], Index], results: List[Any]) -> Any:
        from pandas.core.reshape.concat import concat
        obj = self.obj
        try:
            return concat(results, keys=keys, axis=1, sort=False)
        except TypeError as err:
            from pandas import Series
            result = Series(results, index=keys, name=obj.name)
            if is_nested_object(result):
                raise ValueError('cannot combine transform and aggregation operations') from err
            return result

    def agg_dict_like(self) -> Any:
        """
        Compute aggregation in the case of a dict-like argument.

        Returns
        -------
        Result of aggregation.
        """
        return self.agg_or_apply_dict_like(op_name='agg')

    def compute_dict_like(
        self,
        op_name: str,
        selected_obj: Union[DataFrame, Series],
        selection: Optional[Union[Hashable, Sequence[Hashable]]],
        kwargs: Dict[str, Any]
    ) -> Tuple[List[Hashable], List[Any]]:
        """
        Compute agg/apply results for dict-like input.

        Parameters
        ----------
        op_name : {"agg", "apply"}
            Operation being performed.
        selected_obj : Series or DataFrame
            Data to perform operation on.
        selection : hashable or sequence of hashables
            Used by GroupBy, Window, and Resample if selection is applied to the object.
        kwargs : dict
            Keyword arguments to pass to the functions.

        Returns
        -------
        keys : list[hashable]
            Index labels for result.
        results : list
            Data for result. When aggregating with a Series, this can contain any
            Python object.
        """
        from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy
        obj = self.obj
        is_groupby = isinstance(obj, (DataFrameGroupBy, SeriesGroupBy))
        func = cast(AggFuncTypeDict, self.func)
        func = self.normalize_dictlike_arg(op_name, selected_obj, func)
        is_non_unique_col = selected_obj.ndim == 2 and selected_obj.columns.nunique() < len(selected_obj.columns)
        if selected_obj.ndim == 1:
            colg = obj._gotitem(selection, ndim=1)
            results = [getattr(colg, op_name)(how, **kwargs) for _, how in func.items()]
            keys = list(func.keys())
        elif not is_groupby and is_non_unique_col:
            results = []
            keys = []
            for key, how in func.items():
                indices = selected_obj.columns.get_indexer_for([key])
                labels = selected_obj.columns.take(indices)
                label_to_indices = defaultdict(list)
                for index, label in zip(indices, labels):
                    label_to_indices[label].append(index)
                key_data = [getattr(selected_obj._ixs(indice, axis=1), op_name)(how, **kwargs) for label, indices in label_to_indices.items() for indice in indices]
                keys += [key] * len(key_data)
                results += key_data
        elif is_groupby:
            df = selected_obj
            results, keys = ([], [])
            for key, how in func.items():
                cols = df[key]
                if cols.ndim == 1:
                    series = obj._gotitem(key, ndim=1, subset=cols)
                    results.append(getattr(series, op_name)(how, **kwargs))
                    keys.append(key)
                else:
                    for _, col in cols.items():
                        series = obj._gotitem(key, ndim=1, subset=col)
                        results.append(getattr(series, op_name)(how, **kwargs))
                        keys.append(key)
        else:
            results = [getattr(obj._gotitem(key, ndim=1), op_name)(how, **kwargs) for key, how in func.items()]
            keys = list(func.keys())
        return (keys, results)

    def wrap_results_dict_like(
        self,
        selected_obj: Union[DataFrame, Series],
        result_index: Index,
        result_data: List[Any]
    ) -> Any:
        from pandas import Index
        from pandas.core.reshape.concat import concat
        obj = self.obj
        is_ndframe = [isinstance(r, ABCNDFrame) for r in result_data]
        if all(is_ndframe):
            results = [result for result in result_data if not result.empty]
            keys_to_use = [k for k, v in zip(result_index, result_data) if not v.empty]
            if keys_to_use == []:
                keys_to_use = result_index
                results = result_data
            if selected_obj.ndim == 2:
                ktu = Index(keys_to_use)
                ktu._set_names(selected_obj.columns.names)
                keys_to_use = ktu
            axis = 0 if isinstance(obj, ABCSeries) else 1
            result = concat(results, axis=axis, keys=keys_to_use)
        elif any(is_ndframe):
            raise ValueError('cannot perform both aggregation and transformation operations simultaneously')
        else:
            from pandas import Series
            if obj.ndim == 1:
                obj = cast('Series', obj)
                name = obj.name
            else:
                name = None
            result = Series(result_data, index=result_index, name=name)
        return result

    def apply_str(self) -> Any:
        """
        Compute apply in case of a string.

        Returns
        -------
        result: Series or DataFrame
        """
        func = cast(str, self.func)
        obj = self.obj
        from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy
        method = getattr(obj, func, None)
        if callable(method):
            sig = inspect.getfullargspec(method)
            arg_names = (*sig.args, *sig.kwonlyargs)
            if self.axis != 0 and ('axis' not in arg_names or func in ('corrwith', 'skew')):
                raise ValueError(f'Operation {func} does not support axis=1')
            if 'axis' in arg_names and (not isinstance(obj, (SeriesGroupBy, DataFrameGroupBy))):
                self.kwargs['axis'] = self.axis
        return self._apply_str(obj, func, *self.args, **self.kwargs)

    def apply_list_or_dict_like(self) -> Optional[Any]:
        """
        Compute apply in case of a list-like or dict-like.

        Returns
        -------
        result: Series, DataFrame, or None
            Result when self.func is a list-like or dict-like, None otherwise.
        """
        if self.engine == 'numba':
            raise NotImplementedError("The 'numba' engine doesn't support list-like/dict likes of callables yet.")
        if self.axis == 1 and isinstance(self.obj, ABCDataFrame):
            return self.obj.T.apply(self.func, 0, args=self.args, **self.kwargs).T
        func = self.func
        kwargs = self.kwargs
        if is_dict_like(func):
            result = self.agg_or_apply_dict_like(op_name='apply')
        else:
            result = self.agg_or_apply_list_like(op_name='apply')
        result = reconstruct_and_relabel_result(result, func, **kwargs)
        return result

    def normalize_dictlike_arg(
        self,
        how: Literal['apply', 'agg', 'transform'],
        obj: Union[DataFrame, Series],
        func: AggFuncTypeDict
    ) -> AggFuncTypeDict:
        """
        Handler for dict-like argument.

        Ensures that necessary columns exist if obj is a DataFrame, and
        that a nested renamer is not passed. Also normalizes to all lists
        when values consists of a mix of list and non-lists.
        """
        assert how in ('apply', 'agg', 'transform')
        if how == 'agg' and isinstance(obj, ABCSeries) and any((is_list_like(v) for _, v in func.items())) or any((is_dict_like(v) for _, v in func.items