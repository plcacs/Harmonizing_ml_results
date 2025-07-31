from __future__ import annotations
import abc
from collections import defaultdict
from collections.abc import Callable
import functools
from functools import partial
import inspect
from typing import TYPE_CHECKING, Any, Literal, cast, Optional, Tuple, List, Dict, Union
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
    from collections.abc import Generator, Hashable, Iterable, MutableMapping, Sequence
    from pandas import DataFrame, Index, Series
    from pandas.core.groupby import GroupBy
    from pandas.core.resample import Resampler
    from pandas.core.window.rolling import BaseWindow

ResType = Dict[int, Any]

def frame_apply(obj: NDFrameT,
                func: Optional[Union[str, Callable[..., Any], List[AggFuncTypeBase], Dict[Any, Any]]],
                axis: Axis = 0,
                raw: bool = False,
                result_type: Optional[Literal['reduce', 'broadcast', 'expand']] = None,
                by_row: Union[Literal['compat'], bool] = 'compat',
                engine: Literal['python', 'numba'] = 'python',
                engine_kwargs: Optional[Dict[str, Any]] = None,
                args: Optional[Tuple[Any, ...]] = None,
                kwargs: Optional[Dict[str, Any]] = None) -> Union[FrameRowApply, FrameColumnApply]:
    """construct and return a row or column based frame apply object"""
    _, func, columns, _ = reconstruct_func(func, **(kwargs or {}))
    axis = obj._get_axis_number(axis)
    if axis == 0:
        klass: type[FrameApply] = FrameRowApply
    elif axis == 1:
        if columns:
            raise NotImplementedError(f'Named aggregation is not supported when axis={axis!r}.')
        klass = FrameColumnApply
    return klass(obj, func, raw=raw, result_type=result_type, by_row=by_row, engine=engine, engine_kwargs=engine_kwargs, args=args, kwargs=kwargs)

class Apply(metaclass=abc.ABCMeta):
    def __init__(self,
                 obj: NDFrameT,
                 func: Any,
                 raw: bool,
                 result_type: Optional[Literal['reduce', 'broadcast', 'expand']],
                 *,
                 by_row: Union[Literal['compat'], bool] = 'compat',
                 engine: Literal['python', 'numba'] = 'python',
                 engine_kwargs: Optional[Dict[str, Any]] = None,
                 args: Optional[Tuple[Any, ...]] = None,
                 kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.obj: NDFrameT = obj
        self.raw: bool = raw
        assert by_row is False or by_row in ['compat', '_compat']
        self.by_row: Union[Literal['compat'], bool] = by_row
        self.args: Tuple[Any, ...] = args or ()
        self.kwargs: Dict[str, Any] = kwargs or {}
        self.engine: Literal['python', 'numba'] = engine
        self.engine_kwargs: Dict[str, Any] = {} if engine_kwargs is None else engine_kwargs
        if result_type not in [None, 'reduce', 'broadcast', 'expand']:
            raise ValueError("invalid value for result_type, must be one of {None, 'reduce', 'broadcast', 'expand'}")
        self.result_type: Optional[Literal['reduce', 'broadcast', 'expand']] = result_type
        self.func: Any = func

    @abc.abstractmethod
    def apply(self) -> Any:
        pass

    @abc.abstractmethod
    def agg_or_apply_list_like(self, op_name: str) -> Any:
        pass

    @abc.abstractmethod
    def agg_or_apply_dict_like(self, op_name: str) -> Any:
        pass

    def agg(self) -> Any:
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
        axis = self.axis  # type: ignore
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
        results: Dict[Any, Any] = {}
        for name, how in func.items():
            colg = obj._gotitem(name, ndim=1)
            results[name] = colg.transform(how, 0, *args, **kwargs)
        return concat(results, axis=1)

    def transform_str_or_callable(self, func: Union[str, Callable[..., Any]]) -> Any:
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

    def compute_list_like(self, op_name: str, selected_obj: Any, kwargs: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
        """
        Compute agg/apply results for list-like input.

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
        results: List[Any] = []
        keys: List[Any] = []
        if selected_obj.ndim == 1:
            for a in func:
                colg = obj._gotitem(selected_obj.name, ndim=1, subset=selected_obj)
                args = [self.axis, *self.args] if include_axis(op_name, colg) else self.args
                new_res = getattr(colg, op_name)(a, *args, **kwargs)
                results.append(new_res)
                name = com.get_callable_name(a) or a
                keys.append(name)
        else:
            indices: List[int] = []
            for index, col in enumerate(selected_obj):
                colg = obj._gotitem(col, ndim=1, subset=selected_obj.iloc[:, index])
                args = [self.axis, *self.args] if include_axis(op_name, colg) else self.args
                new_res = getattr(colg, op_name)(func, *args, **kwargs)
                results.append(new_res)
                indices.append(index)
            keys = list(selected_obj.columns.take(indices))
        return (keys, results)

    def wrap_results_list_like(self, keys: List[Any], results: List[Any]) -> Any:
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

    def compute_dict_like(self, op_name: str, selected_obj: Any, selection: Any, kwargs: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
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
            Python objects.
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
                label_to_indices: Dict[Any, List[int]] = defaultdict(list)
                for index, label in zip(indices, labels):
                    label_to_indices[label].append(index)
                key_data = [getattr(selected_obj._ixs(indice, axis=1), op_name)(how, **kwargs)
                            for label, indices in label_to_indices.items() for indice in indices]
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

    def wrap_results_dict_like(self, selected_obj: Any, result_index: List[Any], result_data: List[Any]) -> Any:
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
            axis_val = 0 if isinstance(obj, ABCSeries) else 1
            result = concat(results, axis=axis_val, keys=keys_to_use)
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
                self.kwargs['axis'] = self.axis  # type: ignore
        return self._apply_str(obj, func, *self.args, **self.kwargs)

    def apply_list_or_dict_like(self) -> Any:
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

    def normalize_dictlike_arg(self, how: str, obj: NDFrameT, func: Any) -> Any:
        """
        Handler for dict-like argument.

        Ensures that necessary columns exist if obj is a DataFrame, and
        that a nested renamer is not passed. Also normalizes to all lists
        when values consists of a mix of list and non-lists.
        """
        assert how in ('apply', 'agg', 'transform')
        if how == 'agg' and isinstance(obj, ABCSeries) and any((is_list_like(v) for _, v in func.items())) or any((is_dict_like(v) for _, v in func.items())):
            raise SpecificationError('nested renamer is not supported')
        if obj.ndim != 1:
            from pandas import Index
            cols = Index(list(func.keys())).difference(obj.columns, sort=True)
            if len(cols) > 0:
                raise KeyError(f'Label(s) {list(cols)} do not exist')
        aggregator_types = (list, tuple, dict)
        if any((isinstance(x, aggregator_types) for _, x in func.items())):
            new_func: Dict[Any, Any] = {}
            for k, v in func.items():
                if not isinstance(v, aggregator_types):
                    new_func[k] = [v]
                else:
                    new_func[k] = v
            func = new_func
        return func

    def _apply_str(self, obj: Any, func: str, *args: Any, **kwargs: Any) -> Any:
        """
        if arg is a string, then try to operate on it:
        - try to find a function (or attribute) on obj
        - try to find a numpy function
        - raise
        """
        assert isinstance(func, str)
        if hasattr(obj, func):
            f = getattr(obj, func)
            if callable(f):
                return f(*args, **kwargs)
            assert len(args) == 0
            assert not any((kwarg == 'axis' for kwarg in kwargs))
            return f
        elif hasattr(np, func) and hasattr(obj, '__array__'):
            f = getattr(np, func)
            return f(obj, *args, **kwargs)
        else:
            msg = f"'{func}' is not a valid function for '{type(obj).__name__}' object"
            raise AttributeError(msg)

class NDFrameApply(Apply):
    """
    Methods shared by FrameApply and SeriesApply but
    not GroupByApply or ResamplerWindowApply
    """

    @property
    def index(self) -> Any:
        return self.obj.index

    @property
    def agg_axis(self) -> Any:
        return self.obj._get_agg_axis(self.axis)  # type: ignore

    def agg_or_apply_list_like(self, op_name: str) -> Any:
        obj = self.obj
        kwargs = self.kwargs
        if op_name == 'apply':
            if isinstance(self, FrameApply):
                by_row = self.by_row
            elif isinstance(self, SeriesApply):
                by_row = '_compat' if self.by_row else False
            else:
                by_row = False
            kwargs = {**kwargs, 'by_row': by_row}
        if getattr(obj, 'axis', 0) == 1:
            raise NotImplementedError('axis other than 0 is not supported')
        keys, results = self.compute_list_like(op_name, obj, kwargs)
        result = self.wrap_results_list_like(keys, results)
        return result

    def agg_or_apply_dict_like(self, op_name: str) -> Any:
        assert op_name in ['agg', 'apply']
        obj = self.obj
        kwargs: Dict[str, Any] = {}
        if op_name == 'apply':
            by_row = '_compat' if self.by_row else False
            kwargs.update({'by_row': by_row})
        if getattr(obj, 'axis', 0) == 1:
            raise NotImplementedError('axis other than 0 is not supported')
        selection = None
        result_index, result_data = self.compute_dict_like(op_name, obj, selection, kwargs)
        result = self.wrap_results_dict_like(obj, result_index, result_data)
        return result

class FrameApply(NDFrameApply):
    def __init__(self,
                 obj: NDFrameT,
                 func: Any,
                 raw: bool,
                 result_type: Optional[Literal['reduce', 'broadcast', 'expand']],
                 *,
                 by_row: Union[Literal['compat'], bool] = False,
                 engine: Literal['python', 'numba'] = 'python',
                 engine_kwargs: Optional[Dict[str, Any]] = None,
                 args: Optional[Tuple[Any, ...]] = None,
                 kwargs: Optional[Dict[str, Any]] = None) -> None:
        if by_row is not False and by_row != 'compat':
            raise ValueError(f'by_row={by_row} not allowed')
        super().__init__(obj, func, raw, result_type, by_row=by_row, engine=engine, engine_kwargs=engine_kwargs, args=args, kwargs=kwargs)

    @property
    @abc.abstractmethod
    def result_index(self) -> Any:
        pass

    @property
    @abc.abstractmethod
    def result_columns(self) -> Any:
        pass

    @property
    @abc.abstractmethod
    def series_generator(self) -> Any:
        pass

    @staticmethod
    @functools.cache
    @abc.abstractmethod
    def generate_numba_apply_func(func: Callable[..., Any],
                                  nogil: bool = True,
                                  nopython: bool = True,
                                  parallel: bool = False) -> Callable[..., Any]:
        pass

    @abc.abstractmethod
    def apply_with_numba(self) -> Any:
        pass

    def validate_values_for_numba(self) -> None:
        for colname, dtype in self.obj.dtypes.items():
            if not is_numeric_dtype(dtype):
                raise ValueError(f"Column {colname} must have a numeric dtype. Found '{dtype}' instead")
            if is_extension_array_dtype(dtype):
                raise ValueError(f'Column {colname} is backed by an extension array, which is not supported by the numba engine.')

    @abc.abstractmethod
    def wrap_results_for_axis(self, results: Any, res_index: Any) -> Any:
        pass

    @property
    def res_columns(self) -> Any:
        return self.result_columns

    @property
    def columns(self) -> Any:
        return self.obj.columns

    @cache_readonly
    def values(self) -> Any:
        return self.obj.values

    def apply(self) -> Any:
        """compute the results"""
        if is_list_like(self.func):
            if self.engine == 'numba':
                raise NotImplementedError("the 'numba' engine doesn't support lists of callables yet")
            return self.apply_list_or_dict_like()
        if len(self.columns) == 0 and len(self.index) == 0:
            return self.apply_empty_result()
        if isinstance(self.func, str):
            if self.engine == 'numba':
                raise NotImplementedError("the 'numba' engine doesn't support using a string as the callable function")
            return self.apply_str()
        elif isinstance(self.func, np.ufunc):
            if self.engine == 'numba':
                raise NotImplementedError("the 'numba' engine doesn't support using a numpy ufunc as the callable function")
            with np.errstate(all='ignore'):
                results = self.obj._mgr.apply('apply', func=self.func)
            return self.obj._constructor_from_mgr(results, axes=results.axes)
        if self.result_type == 'broadcast':
            if self.engine == 'numba':
                raise NotImplementedError("the 'numba' engine doesn't support result_type='broadcast'")
            return self.apply_broadcast(self.obj)
        elif not all(self.obj.shape):
            return self.apply_empty_result()
        elif self.raw:
            return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
        return self.apply_standard()

    def agg(self) -> Any:
        obj = self.obj
        axis = self.axis  # type: ignore
        self.obj = self.obj if self.axis == 0 else self.obj.T  # type: ignore
        self.axis = 0  # type: ignore
        result = None
        try:
            result = super().agg()
        finally:
            self.obj = obj
            self.axis = axis  # type: ignore
        if axis == 1:
            result = result.T if result is not None else result
        if result is None:
            result = self.obj.apply(self.func, axis, args=self.args, **self.kwargs)
        return result

    def apply_empty_result(self) -> Any:
        """
        we have an empty result; at least 1 axis is 0

        we will try to apply the function to an empty
        series in order to see if this is a reduction function
        """
        assert callable(self.func)
        if self.result_type not in ['reduce', None]:
            return self.obj.copy()
        should_reduce = self.result_type == 'reduce'
        from pandas import Series
        if not should_reduce:
            try:
                if self.axis == 0:
                    r = self.func(Series([], dtype=np.float64), *self.args, **self.kwargs)
                else:
                    r = self.func(Series(index=self.columns, dtype=np.float64), *self.args, **self.kwargs)
            except Exception:
                pass
            else:
                should_reduce = not isinstance(r, Series)
        if should_reduce:
            if len(self.agg_axis):
                r = self.func(Series([], dtype=np.float64), *self.args, **self.kwargs)
            else:
                r = np.nan
            return self.obj._constructor_sliced(r, index=self.agg_axis)
        else:
            return self.obj.copy()

    def apply_raw(self, engine: Literal['python', 'numba'] = 'python', engine_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """apply to the values as a numpy array"""
        def wrap_function(func: Callable[..., Any]) -> Callable[..., Any]:
            """
            Wrap user supplied function to work around numpy issue.

            see https://github.com/numpy/numpy/issues/8352
            """
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = func(*args, **kwargs)
                if isinstance(result, str):
                    result = np.array(result, dtype=object)
                return result
            return wrapper
        if engine == 'numba':
            args, kwargs_inner = prepare_function_arguments(self.func, self.args, self.kwargs, num_required_args=1)
            nb_looper = generate_apply_looper(self.func, **get_jit_arguments(engine_kwargs))
            result = nb_looper(self.values, self.axis, *args)
            result = np.squeeze(result)
        else:
            result = np.apply_along_axis(wrap_function(self.func), self.axis, self.values, *self.args, **self.kwargs)
        if result.ndim == 2:
            return self.obj._constructor(result, index=self.index, columns=self.columns)
        else:
            return self.obj._constructor_sliced(result, index=self.agg_axis)

    def apply_broadcast(self, target: NDFrameT) -> Any:
        assert callable(self.func)
        result_values = np.empty_like(target.values)
        result_compare = target.shape[0]
        for i, col in enumerate(target.columns):
            res = self.func(target[col], *self.args, **self.kwargs)
            ares = np.asarray(res).ndim
            if ares > 1:
                raise ValueError('too many dims to broadcast')
            if ares == 1:
                if result_compare != len(res):
                    raise ValueError('cannot broadcast result')
            result_values[:, i] = res
        result = self.obj._constructor(result_values, index=target.index, columns=target.columns)
        return result

    def apply_standard(self) -> Any:
        if self.engine == 'python':
            results, res_index = self.apply_series_generator()
        else:
            results, res_index = self.apply_series_numba()
        return self.wrap_results(results, res_index)

    def apply_series_generator(self) -> Tuple[Dict[int, Any], Any]:
        assert callable(self.func)
        series_gen = self.series_generator
        res_index = self.result_index
        results: Dict[int, Any] = {}
        for i, v in enumerate(series_gen):
            results[i] = self.func(v, *self.args, **self.kwargs)
            if isinstance(results[i], ABCSeries):
                results[i] = results[i].copy(deep=False)
        return (results, res_index)

    def apply_series_numba(self) -> Tuple[Any, Any]:
        if self.engine_kwargs.get('parallel', False):
            raise NotImplementedError("Parallel apply is not supported when raw=False and engine='numba'")
        if not self.obj.index.is_unique or not self.columns.is_unique:
            raise NotImplementedError("The index/columns must be unique when raw=False and engine='numba'")
        self.validate_values_for_numba()
        results = self.apply_with_numba()
        return (results, self.result_index)

    def wrap_results(self, results: Any, res_index: Any) -> Any:
        from pandas import Series
        if len(results) > 0 and 0 in results and is_sequence(results[0]):
            return self.wrap_results_for_axis(results, res_index)
        constructor_sliced = self.obj._constructor_sliced
        if len(results) == 0 and constructor_sliced is Series:
            result = constructor_sliced(results, dtype=np.float64)
        else:
            result = constructor_sliced(results)
        result.index = res_index
        return result

    def apply_str(self) -> Any:
        if self.func == 'size':
            obj = self.obj
            value = obj.shape[self.axis]  # type: ignore
            return obj._constructor_sliced(value, index=self.agg_axis)
        return super().apply_str()

class FrameRowApply(FrameApply):
    axis: AxisInt = 0

    @property
    def series_generator(self) -> Any:
        return (self.obj._ixs(i, axis=1) for i in range(len(self.columns)))

    @staticmethod
    @functools.cache
    def generate_numba_apply_func(func: Callable[..., Any],
                                  nogil: bool = True,
                                  nopython: bool = True,
                                  parallel: bool = False) -> Callable[..., Any]:
        numba = import_optional_dependency('numba')
        from pandas import Series
        from pandas.core._numba.extensions import maybe_cast_str
        jitted_udf = numba.extending.register_jitable(func)

        @numba.jit(nogil=nogil, nopython=nopython, parallel=parallel)
        def numba_func(values: np.ndarray, col_names: Any, df_index: Any, *args: Any) -> Any:
            results: Dict[int, Any] = {}
            for j in range(values.shape[1]):
                ser = Series(values[:, j], index=df_index, name=maybe_cast_str(col_names[j]))
                results[j] = jitted_udf(ser, *args)
            return results
        return numba_func

    def apply_with_numba(self) -> Any:
        func = cast(Callable[..., Any], self.func)
        args, kwargs_inner = prepare_function_arguments(func, self.args, self.kwargs, num_required_args=1)
        nb_func = self.generate_numba_apply_func(func, **get_jit_arguments(self.engine_kwargs))
        from pandas.core._numba.extensions import set_numba_data
        index = self.obj.index
        columns = self.obj.columns
        with set_numba_data(index) as index, set_numba_data(columns) as columns:
            res = dict(nb_func(self.values, columns, index, *args))
        return res

    @property
    def result_index(self) -> Any:
        return self.columns

    @property
    def result_columns(self) -> Any:
        return self.index

    def wrap_results_for_axis(self, results: Any, res_index: Any) -> Any:
        """return the results for the rows"""
        if self.result_type == 'reduce':
            res = self.obj._constructor_sliced(results)
            res.index = res_index
            return res
        elif self.result_type is None and all((isinstance(x, dict) for x in results.values())):
            res = self.obj._constructor_sliced(results)
            res.index = res_index
            return res
        try:
            result = self.obj._constructor(data=results)
        except ValueError as err:
            if 'All arrays must be of the same length' in str(err):
                res = self.obj._constructor_sliced(results)
                res.index = res_index
                return res
            else:
                raise
        if not isinstance(results[0], ABCSeries):
            if len(result.index) == len(self.res_columns):
                result.index = self.res_columns
        if len(result.columns) == len(res_index):
            result.columns = res_index
        return result

class FrameColumnApply(FrameApply):
    axis: AxisInt = 1

    def apply_broadcast(self, target: NDFrameT) -> Any:
        result = super().apply_broadcast(target.T)
        return result.T

    @property
    def series_generator(self) -> Any:
        values = self.values
        values = ensure_wrapped_if_datetimelike(values)
        assert len(values) > 0
        ser = self.obj._ixs(0, axis=0)
        mgr = ser._mgr
        is_view = mgr.blocks[0].refs.has_reference()
        if isinstance(ser.dtype, ExtensionDtype):
            obj = self.obj
            for i in range(len(obj)):
                yield obj._ixs(i, axis=0)
        else:
            for arr, name in zip(values, self.index):
                ser._mgr = mgr
                mgr.set_values(arr)
                object.__setattr__(ser, '_name', name)
                if not is_view:
                    mgr.blocks[0].refs = BlockValuesRefs(mgr.blocks[0])
                yield ser

    @staticmethod
    @functools.cache
    def generate_numba_apply_func(func: Callable[..., Any],
                                  nogil: bool = True,
                                  nopython: bool = True,
                                  parallel: bool = False) -> Callable[..., Any]:
        numba = import_optional_dependency('numba')
        from pandas import Series
        from pandas.core._numba.extensions import maybe_cast_str
        jitted_udf = numba.extending.register_jitable(func)

        @numba.jit(nogil=nogil, nopython=nopython, parallel=parallel)
        def numba_func(values: np.ndarray, col_names_index: Any, index: Any, *args: Any) -> Any:
            results: Dict[int, Any] = {}
            for i in range(values.shape[0]):
                ser = Series(values[i].copy(), index=col_names_index, name=maybe_cast_str(index[i]))
                results[i] = jitted_udf(ser, *args)
            return results
        return numba_func

    def apply_with_numba(self) -> Any:
        func = cast(Callable[..., Any], self.func)
        args, kwargs_inner = prepare_function_arguments(func, self.args, self.kwargs, num_required_args=1)
        nb_func = self.generate_numba_apply_func(func, **get_jit_arguments(self.engine_kwargs))
        from pandas.core._numba.extensions import set_numba_data
        with set_numba_data(self.obj.index) as index, set_numba_data(self.columns) as columns:
            res = dict(nb_func(self.values, columns, index, *args))
        return res

    @property
    def result_index(self) -> Any:
        return self.index

    @property
    def result_columns(self) -> Any:
        return self.columns

    def wrap_results_for_axis(self, results: Any, res_index: Any) -> Any:
        """return the results for the columns"""
        if self.result_type == 'expand':
            result = self.infer_to_same_shape(results, res_index)
        elif not isinstance(results[0], ABCSeries):
            result = self.obj._constructor_sliced(results)
            result.index = res_index
        else:
            result = self.infer_to_same_shape(results, res_index)
        return result

    def infer_to_same_shape(self, results: Any, res_index: Any) -> Any:
        """infer the results to the same shape as the input object"""
        result = self.obj._constructor(data=results)
        result = result.T
        result.index = res_index
        result = result.infer_objects()
        return result

class SeriesApply(NDFrameApply):
    axis: AxisInt = 0

    def __init__(self,
                 obj: NDFrameT,
                 func: Any,
                 *,
                 by_row: Union[Literal['compat'], bool] = 'compat',
                 args: Optional[Tuple[Any, ...]] = None,
                 kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(obj, func, raw=False, result_type=None, by_row=by_row, args=args, kwargs=kwargs)

    def apply(self) -> Any:
        obj = self.obj
        if len(obj) == 0:
            return self.apply_empty_result()
        if is_list_like(self.func):
            return self.apply_list_or_dict_like()
        if isinstance(self.func, str):
            return self.apply_str()
        if self.by_row == '_compat':
            return self.apply_compat()
        return self.apply_standard()

    def agg(self) -> Any:
        result = super().agg()
        if result is None:
            obj = self.obj
            func = self.func
            assert callable(func)
            result = func(obj, *self.args, **self.kwargs)
        return result

    def apply_empty_result(self) -> Any:
        obj = self.obj
        return obj._constructor(dtype=obj.dtype, index=obj.index).__finalize__(obj, method='apply')

    def apply_compat(self) -> Any:
        """compat apply method for funcs in listlikes and dictlikes.
         Used for each callable when giving listlikes and dictlikes of callables to
         apply. Needed for compatibility with Pandas < v2.1.
        .. versionadded:: 2.1.0
        """
        obj = self.obj
        func = self.func
        if callable(func):
            f = com.get_cython_func(func)
            if f and (not self.args) and (not self.kwargs):
                return obj.apply(func, by_row=False)
        try:
            result = obj.apply(func, by_row='compat')
        except (ValueError, AttributeError, TypeError):
            result = obj.apply(func, by_row=False)
        return result

    def apply_standard(self) -> Any:
        func = cast(Callable[..., Any], self.func)
        obj = self.obj
        if isinstance(func, np.ufunc):
            with np.errstate(all='ignore'):
                return func(obj, *self.args, **self.kwargs)
        elif not self.by_row:
            return func(obj, *self.args, **self.kwargs)
        if self.args or self.kwargs:
            def curried(x: Any) -> Any:
                return func(x, *self.args, **self.kwargs)
        else:
            curried = func
        mapped = obj._map_values(mapper=curried)
        if len(mapped) and isinstance(mapped[0], ABCSeries):
            return obj._constructor_expanddim(list(mapped), index=obj.index)
        else:
            return obj._constructor(mapped, index=obj.index).__finalize__(obj, method='apply')

class GroupByApply(Apply):
    def __init__(self,
                 obj: Any,
                 func: Any,
                 *,
                 args: Optional[Tuple[Any, ...]],
                 kwargs: Optional[Dict[str, Any]]) -> None:
        kwargs = kwargs.copy() if kwargs is not None else {}
        self.axis: Axis = obj.obj._get_axis_number(kwargs.get('axis', 0))
        super().__init__(obj, func, raw=False, result_type=None, args=args, kwargs=kwargs)

    def apply(self) -> Any:
        raise NotImplementedError

    def transform(self) -> Any:
        raise NotImplementedError

    def agg_or_apply_list_like(self, op_name: str) -> Any:
        obj = self.obj
        kwargs = self.kwargs
        if op_name == 'apply':
            kwargs = {**kwargs, 'by_row': False}
        if getattr(obj, 'axis', 0) == 1:
            raise NotImplementedError('axis other than 0 is not supported')
        if obj._selected_obj.ndim == 1:
            selected_obj = obj._selected_obj
        else:
            selected_obj = obj._obj_with_exclusions
        with com.temp_setattr(obj, 'as_index', True, condition=hasattr(obj, 'as_index')):
            keys, results = self.compute_list_like(op_name, selected_obj, kwargs)
        result = self.wrap_results_list_like(keys, results)
        return result

    def agg_or_apply_dict_like(self, op_name: str) -> Any:
        from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy
        assert op_name in ['agg', 'apply']
        obj = self.obj
        kwargs: Dict[str, Any] = {}
        if op_name == 'apply':
            by_row = '_compat' if self.by_row else False
            kwargs.update({'by_row': by_row})
        if getattr(obj, 'axis', 0) == 1:
            raise NotImplementedError('axis other than 0 is not supported')
        selected_obj = obj._selected_obj
        selection = obj._selection
        is_groupby = isinstance(obj, (DataFrameGroupBy, SeriesGroupBy))
        if is_groupby:
            engine = self.kwargs.get('engine', None)
            engine_kwargs = self.kwargs.get('engine_kwargs', None)
            kwargs.update({'engine': engine, 'engine_kwargs': engine_kwargs})
        with com.temp_setattr(obj, 'as_index', True, condition=hasattr(obj, 'as_index')):
            result_index, result_data = self.compute_dict_like(op_name, selected_obj, selection, kwargs)
        result = self.wrap_results_dict_like(selected_obj, result_index, result_data)
        return result

class ResamplerWindowApply(GroupByApply):
    axis: AxisInt = 0

    def __init__(self,
                 obj: Any,
                 func: Any,
                 *,
                 args: Optional[Tuple[Any, ...]],
                 kwargs: Optional[Dict[str, Any]]) -> None:
        super(GroupByApply, self).__init__(obj, func, raw=False, result_type=None, args=args, kwargs=kwargs)

    def apply(self) -> Any:
        raise NotImplementedError

    def transform(self) -> Any:
        raise NotImplementedError

def reconstruct_func(func: Optional[Any], **kwargs: Any) -> Tuple[bool, Any, Optional[Tuple[Any, ...]], Optional[Any]]:
    """
    This is the internal function to reconstruct func given if there is relabeling
    or not and also normalize the keyword to get new order of columns.

    If named aggregation is applied, `func` will be None, and kwargs contains the
    column and aggregation function information to be parsed;
    If named aggregation is not applied, `func` is either string (e.g. 'min') or
    Callable, or list of them (e.g. ['min', np.max]), or the dictionary of column name
    and str/Callable/list of them (e.g. {'A': 'min'}, or {'A': [np.min, lambda x: x]})

    If relabeling is True, will return relabeling, reconstructed func, column
    names, and the reconstructed order of columns.
    If relabeling is False, the columns and order will be None.

    Parameters
    ----------
    func: agg function (e.g. 'min' or Callable) or list of agg functions
        (e.g. ['min', np.max]) or dictionary (e.g. {'A': ['min', np.max]}).
    **kwargs: dict, kwargs used in is_multi_agg_with_relabel and
        normalize_keyword_aggregation function for relabelling

    Returns
    -------
    relabelling: bool, if there is relabelling or not
    func: normalized and mangled func
    columns: tuple of column names
    order: array of columns indices
    """
    relabeling: bool = func is None and is_multi_agg_with_relabel(**kwargs)
    columns: Optional[Tuple[Any, ...]] = None
    order: Optional[Any] = None
    if not relabeling:
        if isinstance(func, list) and len(func) > len(set(func)):
            raise SpecificationError('Function names must be unique if there is no new column names assigned')
        if func is None:
            raise TypeError("Must provide 'func' or tuples of '(column, aggfunc).")
    if relabeling:
        func, columns, order = normalize_keyword_aggregation(kwargs)
    assert func is not None
    return (relabeling, func, columns, order)

def is_multi_agg_with_relabel(**kwargs: Any) -> bool:
    """
    Check whether kwargs passed to .agg look like multi-agg with relabeling.

    Parameters
    ----------
    **kwargs : dict

    Returns
    -------
    bool
    """
    return all((isinstance(v, tuple) and len(v) == 2 for v in kwargs.values())) and len(kwargs) > 0

def normalize_keyword_aggregation(kwargs: Dict[str, Any]) -> Tuple[Any, Tuple[str, ...], Any]:
    """
    Normalize user-provided "named aggregation" kwargs.
    Transforms from the new ``Mapping[str, NamedAgg]`` style kwargs
    to the old Dict[str, List[scalar]]].

    Parameters
    ----------
    kwargs : dict

    Returns
    -------
    aggspec : dict
        The transformed kwargs.
    columns : tuple[str, ...]
        The user-provided keys.
    col_idx_order : List[int]
        List of columns indices.
    """
    from pandas.core.indexes.base import Index
    aggspec: Dict[Any, List[Any]] = defaultdict(list)
    order: List[Tuple[Any, Any]] = []
    columns: Tuple[str, ...] = tuple(kwargs.keys())
    for column, aggfunc in kwargs.values():
        aggspec[column].append(aggfunc)
        order.append((column, com.get_callable_name(aggfunc) or aggfunc))
    uniquified_order = _make_unique_kwarg_list(order)
    aggspec_order = [(column, com.get_callable_name(aggfunc) or aggfunc) for column, aggfuncs in aggspec.items() for aggfunc in aggfuncs]
    uniquified_aggspec = _make_unique_kwarg_list(aggspec_order)
    col_idx_order = Index(uniquified_aggspec).get_indexer(uniquified_order)
    return (aggspec, columns, col_idx_order)

def _make_unique_kwarg_list(seq: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
    """
    Uniquify aggfunc name of the pairs in the order list
    """
    return [(pair[0], f'{pair[1]}_{seq[:i].count(pair)}') if seq.count(pair) > 1 else pair for i, pair in enumerate(seq)]

def relabel_result(result: Any, func: Dict[Any, Any], columns: Tuple[str, ...], order: Any) -> Any:
    """
    Internal function to reorder result if relabelling is True for
    dataframe.agg, and return the reordered result in dict.
    """
    from pandas.core.indexes.base import Index
    reordered_indexes = [pair[0] for pair in sorted(zip(columns, order), key=lambda t: t[1])]
    reordered_result_in_dict: Dict[Any, Any] = {}
    idx = 0
    reorder_mask: bool = not isinstance(result, ABCSeries) and len(result.columns) > 1
    for col, fun in func.items():
        s = result[col].dropna()
        if reorder_mask:
            fun = [com.get_callable_name(f) if not isinstance(f, str) else f for f in fun]
            col_idx_order = Index(s.index).get_indexer(fun)
            valid_idx = col_idx_order != -1
            if valid_idx.any():
                s = s.iloc[col_idx_order[valid_idx]]
        if not s.empty:
            s.index = reordered_indexes[idx:idx + len(fun)]
        reordered_result_in_dict[col] = s.reindex(columns)
        idx = idx + len(fun)
    return reordered_result_in_dict

def reconstruct_and_relabel_result(result: Any, func: Any, **kwargs: Any) -> Any:
    from pandas import DataFrame
    relabeling, func, columns, order = reconstruct_func(func, **kwargs)
    if relabeling:
        assert columns is not None
        assert order is not None
        result_in_dict = relabel_result(result, func, columns, order)
        result = DataFrame(result_in_dict, index=columns)
    return result

def _managle_lambda_list(aggfuncs: Sequence[Any]) -> Sequence[Any]:
    """
    Possibly mangle a list of aggfuncs.
    """
    if len(aggfuncs) <= 1:
        return aggfuncs
    i = 0
    mangled_aggfuncs: List[Any] = []
    for aggfunc in aggfuncs:
        if com.get_callable_name(aggfunc) == '<lambda>':
            aggfunc = partial(aggfunc)
            aggfunc.__name__ = f'<lambda_{i}>'
            i += 1
        mangled_aggfuncs.append(aggfunc)
    return mangled_aggfuncs

def maybe_mangle_lambdas(agg_spec: Any) -> Any:
    """
    Make new lambdas with unique names.
    """
    is_dict = is_dict_like(agg_spec)
    if not (is_dict or is_list_like(agg_spec)):
        return agg_spec
    if is_dict:
        mangled_aggspec: Dict[Any, Any] = type(agg_spec)()
        for key, aggfuncs in agg_spec.items():
            if is_list_like(aggfuncs) and (not is_dict_like(aggfuncs)):
                mangled_aggfuncs = _managle_lambda_list(aggfuncs)
            else:
                mangled_aggfuncs = aggfuncs
            mangled_aggspec[key] = mangled_aggfuncs
    else:
        mangled_aggspec = _managle_lambda_list(agg_spec)
    return mangled_aggspec

def validate_func_kwargs(kwargs: Dict[str, Any]) -> Tuple[List[str], List[Union[str, Callable[..., Any]]]]:
    """
    Validates types of user-provided "named aggregation" kwargs.
    """
    tuple_given_message = 'func is expected but received {} in **kwargs.'
    columns: List[str] = list(kwargs)
    func: List[Union[str, Callable[..., Any]]] = []
    for col_func in kwargs.values():
        if not (isinstance(col_func, str) or callable(col_func)):
            raise TypeError(tuple_given_message.format(type(col_func).__name__))
        func.append(col_func)
    if not columns:
        no_arg_message = "Must provide 'func' or named aggregation **kwargs."
        raise TypeError(no_arg_message)
    return (columns, func)

def include_axis(op_name: str, colg: Any) -> bool:
    return isinstance(colg, ABCDataFrame) or (isinstance(colg, ABCSeries) and op_name == 'agg')