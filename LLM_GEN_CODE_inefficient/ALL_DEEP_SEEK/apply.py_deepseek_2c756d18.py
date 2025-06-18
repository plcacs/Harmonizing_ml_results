from __future__ import annotations

import abc
from collections import defaultdict
from collections.abc import Callable, Generator, Hashable, Iterable, MutableMapping, Sequence
import functools
from functools import partial
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    Optional,
    Union,
    Dict,
    List,
    Tuple,
    TypeVar,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from pandas._libs.internals import BlockValuesRefs
from pandas._typing import (
    AggFuncType,
    AggFuncTypeBase,
    AggFuncTypeDict,
    AggObjType,
    Axis,
    AxisInt,
    NDFrameT,
    npt,
)
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SpecificationError
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.cast import is_nested_object
from pandas.core.dtypes.common import (
    is_dict_like,
    is_extension_array_dtype,
    is_list_like,
    is_numeric_dtype,
    is_sequence,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCNDFrame,
    ABCSeries,
)

from pandas.core._numba.executor import generate_apply_looper
import pandas.core.common as com
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.util.numba_ import (
    get_jit_arguments,
    prepare_function_arguments,
)

if TYPE_CHECKING:
    from pandas import (
        DataFrame,
        Index,
        Series,
    )
    from pandas.core.groupby import GroupBy
    from pandas.core.resample import Resampler
    from pandas.core.window.rolling import BaseWindow

ResType = Dict[int, Any]

NDFrameT = TypeVar("NDFrameT", bound="NDFrame")

def frame_apply(
    obj: DataFrame,
    func: AggFuncType,
    axis: Axis = 0,
    raw: bool = False,
    result_type: Optional[str] = None,
    by_row: Literal[False, "compat"] = "compat",
    engine: str = "python",
    engine_kwargs: Optional[Dict[str, bool]] = None,
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> FrameApply:
    """construct and return a row or column based frame apply object"""
    _, func, columns, _ = reconstruct_func(func, **kwargs)

    axis = obj._get_axis_number(axis)
    klass: type[FrameApply]
    if axis == 0:
        klass = FrameRowApply
    elif axis == 1:
        if columns:
            raise NotImplementedError(
                f"Named aggregation is not supported when {axis=}."
            )
        klass = FrameColumnApply

    return klass(
        obj,
        func,
        raw=raw,
        result_type=result_type,
        by_row=by_row,
        engine=engine,
        engine_kwargs=engine_kwargs,
        args=args,
        kwargs=kwargs,
    )


class Apply(metaclass=abc.ABCMeta):
    axis: AxisInt

    def __init__(
        self,
        obj: AggObjType,
        func: AggFuncType,
        raw: bool,
        result_type: Optional[str],
        *,
        by_row: Literal[False, "compat", "_compat"] = "compat",
        engine: str = "python",
        engine_kwargs: Optional[Dict[str, bool]] = None,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.obj = obj
        self.raw = raw

        assert by_row is False or by_row in ["compat", "_compat"]
        self.by_row = by_row

        self.args = args or ()
        self.kwargs = kwargs or {}

        self.engine = engine
        self.engine_kwargs = {} if engine_kwargs is None else engine_kwargs

        if result_type not in [None, "reduce", "broadcast", "expand"]:
            raise ValueError(
                "invalid value for result_type, must be one "
                "of {None, 'reduce', 'broadcast', 'expand'}"
            )

        self.result_type = result_type

        self.func = func

    @abc.abstractmethod
    def apply(self) -> Union[DataFrame, Series]:
        pass

    @abc.abstractmethod
    def agg_or_apply_list_like(
        self, op_name: Literal["agg", "apply"]
    ) -> Union[DataFrame, Series]:
        pass

    @abc.abstractmethod
    def agg_or_apply_dict_like(
        self, op_name: Literal["agg", "apply"]
    ) -> Union[DataFrame, Series]:
        pass

    def agg(self) -> Optional[Union[DataFrame, Series]]:
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
            # we require a list, but not a 'str'
            return self.agg_list_like()

        # caller can react
        return None

    def transform(self) -> Union[DataFrame, Series]:
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

        if is_list_like(func) and not is_dict_like(func):
            func = cast(List[AggFuncTypeBase], func)
            # Convert func equivalent dict
            if is_series:
                func = {com.get_callable_name(v) or v: v for v in func}
            else:
                func = {col: func for col in obj}

        if is_dict_like(func):
            func = cast(AggFuncTypeDict, func)
            return self.transform_dict_like(func)

        # func is either str or callable
        func = cast(AggFuncTypeBase, func)
        try:
            result = self.transform_str_or_callable(func)
        except TypeError:
            raise
        except Exception as err:
            raise ValueError("Transform function failed") from err

        # Functions that transform may return empty Series/DataFrame
        # when the dtype is not appropriate
        if (
            isinstance(result, (ABCSeries, ABCDataFrame))
            and result.empty
            and not obj.empty
        ):
            raise ValueError("Transform function failed")
        if not isinstance(result, (ABCSeries, ABCDataFrame)) or not result.index.equals(
            obj.index
        ):
            raise ValueError("Function did not transform")

        return result

    def transform_dict_like(self, func: AggFuncTypeDict) -> DataFrame:
        """
        Compute transform in the case of a dict-like func
        """
        from pandas.core.reshape.concat import concat

        obj = self.obj
        args = self.args
        kwargs = self.kwargs

        # transform is currently only for Series/DataFrame
        assert isinstance(obj, ABCNDFrame)

        if len(func) == 0:
            raise ValueError("No transform functions were provided")

        func = self.normalize_dictlike_arg("transform", obj, func)

        results: Dict[Hashable, Union[DataFrame, Series]] = {}
        for name, how in func.items():
            colg = obj._gotitem(name, ndim=1)
            results[name] = colg.transform(how, 0, *args, **kwargs)
        return concat(results, axis=1)

    def transform_str_or_callable(self, func: AggFuncTypeBase) -> Union[DataFrame, Series]:
        """
        Compute transform in the case of a string or callable func
        """
        obj = self.obj
        args = self.args
        kwargs = self.kwargs

        if isinstance(func, str):
            return self._apply_str(obj, func, *args, **kwargs)

        # Two possible ways to use a UDF - apply or call directly
        try:
            return obj.apply(func, args=args, **kwargs)
        except Exception:
            return func(obj, *args, **kwargs)

    def agg_list_like(self) -> Union[DataFrame, Series]:
        """
        Compute aggregation in the case of a list-like argument.

        Returns
        -------
        Result of aggregation.
        """
        return self.agg_or_apply_list_like(op_name="agg")

    def compute_list_like(
        self,
        op_name: Literal["agg", "apply"],
        selected_obj: Union[Series, DataFrame],
        kwargs: Dict[str, Any],
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

        # degenerate case
        if selected_obj.ndim == 1:
            for a in func:
                colg = obj._gotitem(selected_obj.name, ndim=1, subset=selected_obj)
                args = (
                    [self.axis, *self.args]
                    if include_axis(op_name, colg)
                    else self.args
                )
                new_res = getattr(colg, op_name)(a, *args, **kwargs)
                results.append(new_res)

                # make sure we find a good name
                name = com.get_callable_name(a) or a
                keys.append(name)

        else:
            indices = []
            for index, col in enumerate(selected_obj):
                colg = obj._gotitem(col, ndim=1, subset=selected_obj.iloc[:, index])
                args = (
                    [self.axis, *self.args]
                    if include_axis(op_name, colg)
                    else self.args
                )
                new_res = getattr(colg, op_name)(func, *args, **kwargs)
                results.append(new_res)
                indices.append(index)
            # error: Incompatible types in assignment (expression has type "Any |
            # Index", variable has type "list[Any | Callable[..., Any] | str]")
            keys = selected_obj.columns.take(indices)  # type: ignore[assignment]

        return keys, results

    def wrap_results_list_like(
        self, keys: Iterable[Hashable], results: List[Union[Series, DataFrame]]
    ) -> Union[DataFrame, Series]:
        from pandas.core.reshape.concat import concat

        obj = self.obj

        try:
            return concat(results, keys=keys, axis=1, sort=False)
        except TypeError as err:
            # we are concatting non-NDFrame objects,
            # e.g. a list of scalars
            from pandas import Series

            result = Series(results, index=keys, name=obj.name)
            if is_nested_object(result):
                raise ValueError(
                    "cannot combine transform and aggregation operations"
                ) from err
            return result

    def agg_dict_like(self) -> Union[DataFrame, Series]:
        """
        Compute aggregation in the case of a dict-like argument.

        Returns
        -------
        Result of aggregation.
        """
        return self.agg_or_apply_dict_like(op_name="agg")

    def compute_dict_like(
        self,
        op_name: Literal["agg", "apply"],
        selected_obj: Union[Series, DataFrame],
        selection: Union[Hashable, Sequence[Hashable]],
        kwargs: Dict[str, Any],
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
        from pandas.core.groupby.generic import (
            DataFrameGroupBy,
            SeriesGroupBy,
        )

        obj = self.obj
        is_groupby = isinstance(obj, (DataFrameGroupBy, SeriesGroupBy))
        func = cast(AggFuncTypeDict, self.func)
        func = self.normalize_dictlike_arg(op_name, selected_obj, func)

        is_non_unique_col = (
            selected_obj.ndim == 2
            and selected_obj.columns.nunique() < len(selected_obj.columns)
        )

        if selected_obj.ndim == 1:
            # key only used for output
            colg = obj._gotitem(selection, ndim=1)
            results = [getattr(colg, op_name)(how, **kwargs) for _, how in func.items()]
            keys = list(func.keys())
        elif not is_groupby and is_non_unique_col:
            # key used for column selection and output
            # GH#51099
            results = []
            keys = []
            for key, how in func.items():
                indices = selected_obj.columns.get_indexer_for([key])
                labels = selected_obj.columns.take(indices)
                label_to_indices = defaultdict(list)
                for index, label in zip(indices, labels):
                    label_to_indices[label].append(index)

                key_data = [
                    getattr(selected_obj._ixs(indice, axis=1), op_name)(how, **kwargs)
                    for label, indices in label_to_indices.items()
                    for indice in indices
                ]

                keys += [key] * len(key_data)
                results += key_data
        elif is_groupby:
            # key used for column selection and output

            df = selected_obj
            results, keys = [], []
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
            results = [
                getattr(obj._gotitem(key, ndim=1), op_name)(how, **kwargs)
                for key, how in func.items()
            ]
            keys = list(func.keys())

        return keys, results

    def wrap_results_dict_like(
        self,
        selected_obj: Union[Series, DataFrame],
        result_index: List[Hashable],
        result_data: List[Any],
    ) -> Union[DataFrame, Series]:
        from pandas import Index
        from pandas.core.reshape.concat import concat

        obj = self.obj

        # Avoid making two isinstance calls in all and any below
        is_ndframe = [isinstance(r, ABCNDFrame) for r in result_data]

        if all(is_ndframe):
            results = [result for result in result_data if not result.empty]
            keys_to_use: Iterable[Hashable]
            keys_to_use = [k for k, v in zip(result_index, result_data) if not v.empty]
            # Have to check, if at least one DataFrame is not empty.
            if keys_to_use == []:
                keys_to_use = result_index
                results = result_data

            if selected_obj.ndim == 2:
                # keys are columns, so we can preserve names
                ktu = Index(keys_to_use)
                ktu._set_names(selected_obj.columns.names)
                keys_to_use = ktu

            axis: AxisInt = 0 if isinstance(obj, ABCSeries) else 1
            result = concat(
                results,
                axis=axis,
                keys=keys_to_use,
            )
        elif any(is_ndframe):
            # There is a mix of NDFrames and scalars
            raise ValueError(
                "cannot perform both aggregation "
                "and transformation operations "
                "simultaneously"
            )
        else:
            from pandas import Series

            # we have a list of scalars
            # GH 36212 use name only if obj is a series
            if obj.ndim == 1:
                obj = cast("Series", obj)
                name = obj.name
            else:
                name = None

            result = Series(result_data, index=result_index, name=name)

        return result

    def apply_str(self) -> Union[DataFrame, Series]:
        """
        Compute apply in case of a string.

        Returns
        -------
        result: Series or DataFrame
        """
        # Caller is responsible for checking isinstance(self.f, str)
        func = cast(str, self.func)

        obj = self.obj

        from pandas.core.groupby.generic import (
            DataFrameGroupBy,
            SeriesGroupBy,
        )

        # Support for `frame.transform('method')`
        # Some methods (shift, etc.) require the axis argument, others
        # don't, so inspect and insert if necessary.
        method = getattr(obj, func, None)
        if callable(method):
            sig = inspect.getfullargspec(method)
            arg_names = (*sig.args, *sig.kwonlyargs)
            if self.axis != 0 and (
                "axis" not in arg_names or func in ("corrwith", "skew")
            ):
                raise ValueError(f"Operation {func} does not support axis=1")
            if "axis" in arg_names and not isinstance(
                obj, (SeriesGroupBy, DataFrameGroupBy)
            ):
                self.kwargs["axis"] = self.axis
        return self._apply_str(obj, func, *self.args, **self.kwargs)

    def apply_list_or_dict_like(self) -> Union[DataFrame, Series]:
        """
        Compute apply in case of a list-like or dict-like.

        Returns
        -------
        result: Series, DataFrame, or None
            Result when self.func is a list-like or dict-like, None otherwise.
        """

        if self.engine == "numba":
            raise NotImplementedError(
                "The 'numba' engine doesn't support list-like/"
                "dict likes of callables yet."
            )

        if self.axis == 1 and isinstance(self.obj, ABCDataFrame):
            return self.obj.T.apply(self.func, 0, args=self.args, **self.kwargs).T

        func = self.func
        kwargs = self.kwargs

        if is_dict_like(func):
            result = self.agg_or_apply_dict_like(op_name="apply")
        else:
            result = self.agg_or_apply_list_like(op_name="apply")

        result = reconstruct_and_relabel_result(result, func, **kwargs)

        return result

    def normalize_dictlike_arg(
        self, how: str, obj: Union[DataFrame, Series], func: AggFuncTypeDict
    ) -> AggFuncTypeDict:
        """
        Handler for dict-like argument.

        Ensures that necessary columns exist if obj is a DataFrame, and
        that a nested renamer is not passed. Also normalizes to all lists
        when values consists of a mix of list and non-lists.
        """
        assert how in ("apply", "agg", "transform")

        # Can't use func.values(); wouldn't work for a Series
        if (
            how == "agg"
            and isinstance(obj, ABCSeries)
            and any(is_list_like(v) for _, v in func.items())
        ) or (any(is_dict_like(v) for _, v in func.items())):
            # GH 15931 - deprecation of renaming keys
            raise SpecificationError("nested renamer is not supported")

        if obj.ndim != 1:
            # Check for missing columns on a frame
            from pandas import Index

            cols = Index(list(func.keys())).difference(obj.columns, sort=True)
            if len(cols) > 0:
                # GH 58474
                raise KeyError(f"Label(s) {list(cols)} do not exist")

        aggregator_types = (list, tuple, dict)

        # if we have a dict of any non-scalars
        # eg. {'A' : ['mean']}, normalize all to
        # be list-likes
        # Cannot use func.values() because arg may be a Series
        if any(isinstance(x, aggregator_types) for _, x in func.items()):
            new_func: AggFuncTypeDict = {}
            for k, v in func.items():
                if not isinstance(v, aggregator_types):
                    new_func[k] = [v]
                else:
                    new_func[k] = v
            func = new_func
        return func

    def _apply_str(self, obj: Union[DataFrame, Series], func: str, *args: Any, **kwargs: Any) -> Union[DataFrame, Series]:
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

            # people may aggregate on a non-callable attribute
            # but don't let them think they can pass args to it
            assert len(args) == 0
            assert not any(kwarg == "axis" for kwarg in kwargs)
            return f
        elif hasattr(np, func) and hasattr(obj, "__array__"):
            # in particular exclude Window
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

    obj: Union[DataFrame, Series]

    @property
    def index(self) -> Index:
        return self.obj.index

    @property
    def agg_axis(self) -> Index:
        return self.obj._get_agg_axis(self.axis)

    def agg_or_apply_list_like(
        self, op_name: Literal["agg", "apply"]
    ) -> Union[DataFrame, Series]:
        obj = self.obj
        kwargs = self.kwargs

        if op_name == "apply":
            if isinstance(self, FrameApply):
                by_row = self.by_row

            elif isinstance(self, SeriesApply):
                by_row = "_compat" if self.by_row else False
            else:
                by_row = False
            kwargs = {**kwargs, "by_row": by_row}

        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        keys, results = self.compute_list_like(op_name, obj, kwargs)
        result = self.wrap_results_list_like(keys, results)
        return result

    def agg_or_apply_dict_like(
        self, op_name: Literal["agg", "apply"]
    ) -> Union[DataFrame, Series]:
        assert op_name in ["agg", "apply"]
        obj = self.obj

        kwargs = {}
        if op_name == "apply":
            by_row = "_compat" if self.by_row else False
            kwargs.update({"by_row": by_row})

        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        selection = None
        result_index, result_data = self.compute_dict_like(
            op_name, obj, selection, kwargs
        )
        result = self.wrap_results_dict_like(obj, result_index, result_data)
        return result


class FrameApply(NDFrameApply):
    obj: DataFrame

    def __init__(
        self,
        obj: AggObjType,
        func: AggFuncType,
        raw: bool,
        result_type: Optional[str],
        *,
        by_row: Literal[False, "compat"] = False,
        engine: str = "python",
        engine_kwargs: Optional[Dict[str, bool]] = None,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if by_row is not False and by_row != "compat":
            raise ValueError(f"by_row={by_row} not allowed")
        super().__init__(
            obj,
            func,
            raw,
            result_type,
            by_row=by_row,
            engine=engine,
            engine_kwargs=engine_kwargs,
            args=args,
            kwargs=kwargs,
        )

    # ---------------------------------------------------------------
    # Abstract Methods

    @property
    @abc.abstractmethod
    def result_index(self) -> Index:
        pass

    @property
    @abc.abstractmethod
    def result_columns(self) -> Index:
        pass

    @property
    @abc.abstractmethod
    def series_generator(self) -> Generator[Series]:
        pass

    @staticmethod
    @functools.cache
    @abc.abstractmethod
    def generate_numba_apply_func(
        func: Callable[..., Any], nogil: bool = True, nopython: bool = True, parallel: bool = False
    ) -> Callable[[NDArray[Any], Index, Index], Dict[int, Any]]:
        pass

    @abc.abstractmethod
    def apply_with_numba(self) -> Dict[int, Any]:
        pass

    def validate_values_for_numba(self) -> None:
        # Validate column dtyps all OK
        for colname, dtype in self.obj.dtypes.items():
            if not is_numeric_dtype(dtype):
                raise ValueError(
                    f"Column {colname} must have a numeric dtype. "
                    f"Found '{dtype}' instead"
                )
            if is_extension_array_dtype(dtype):
                raise ValueError(
                    f"Column {colname} is backed by an extension array, "
                    f"which is not supported by the numba engine."
                )

    @abc.abstractmethod
    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> Union[DataFrame, Series]:
        pass

    # ---------------------------------------------------------------

    @property
    def res_columns(self) -> Index:
        return self.result_columns

    @property
    def columns(self) -> Index:
        return self.obj.columns

    @cache_readonly
    def values(self) -> NDArray[Any]:
        return self.obj.values

    def apply(self) -> Union[DataFrame, Series]:
        """compute the results"""

        # dispatch to handle list-like or dict-like
        if is_list_like(self.func):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support lists of callables yet"
                )
            return self.apply_list_or_dict_like()

        # all empty
        if len(self.columns) == 0 and len(self.index) == 0:
            return self.apply_empty_result()

        # string dispatch
        if isinstance(self.func, str):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support using "
                    "a string as the callable function"
                )
            return self.apply_str()

        # ufunc
        elif isinstance(self.func, np.ufunc):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support "
                    "using a numpy ufunc as the callable function"
                )
            with np.errstate(all="ignore"):
                results = self.obj._mgr.apply("apply", func=self.func)
            # _constructor will retain self.index and self.columns
            return self.obj._constructor_from_mgr(results, axes=results.axes)

        # broadcasting
        if self.result_type == "broadcast":
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support result_type='broadcast'"
                )
            return self.apply_broadcast(self.obj)

        # one axis empty
        elif not all(self.obj.shape):
            return self.apply_empty_result()

        # raw
        elif self.raw:
            return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)

        return self.apply_standard()

    def agg(self) -> Union[DataFrame, Series]:
        obj = self.obj
        axis = self.axis

        # TODO: Avoid having to change state
        self.obj = self.obj if self.axis == 0 else self.obj.T
        self.axis = 0

        result = None
        try:
            result = super().agg()
        finally:
            self.obj = obj
            self.axis = axis

        if axis == 1:
            result = result.T if result is not None else result

        if result is None:
            result = self.obj.apply(self.func, axis, args=self.args, **self.kwargs)

        return result

    def apply_empty_result(self) -> Union[DataFrame, Series]:
        """
        we have an empty result; at least 1 axis is 0

        we will try to apply the function to an empty
        series in order to see if this is a reduction function
        """
        assert callable(self.func)

        # we are not asked to reduce or infer reduction
        # so just return a copy of the existing object
        if self.result_type not in ["reduce", None]:
            return self.obj.copy()

        # we may need to infer
        should_reduce = self.result_type == "reduce"

        from pandas import Series

        if not should_reduce:
            try:
                if self.axis == 0:
                    r = self.func(
                        Series([], dtype=np.float64), *self.args, **self.kwargs
                    )
                else:
                    r = self.func(
                        Series(index=self.columns, dtype=np.float64),
                        *self.args,
                        **self.kwargs,
                    )
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

    def apply_raw(self, engine: str = "python", engine_kwargs: Optional[Dict[str, bool]] = None) -> Union[DataFrame, Series]:
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

        if engine == "numba":
            args, kwargs = prepare_function_arguments(
                self.func,  # type: ignore[arg-type]
                self.args,
                self.kwargs,
                num_required_args=1,
            )
            # error: Argument 1 to "__call__" of "_lru_cache_wrapper" has
            # incompatible type "Callable[..., Any] | str | list[Callable
            # [..., Any] | str] | dict[Hashable,Callable[..., Any] | str |
            # list[Callable[..., Any] | str]]"; expected "Hashable"
            nb_looper = generate_apply_looper(
                self.func,  # type: ignore[arg-type]
                **get_jit_arguments(engine_kwargs),
            )
            result = nb_looper(self.values, self.axis, *args)
            # If we made the result 2-D, squeeze it back to 1-D
            result = np.squeeze(result)
        else:
            result = np.apply_along_axis(
                wrap_function(self.func),
                self.axis,
                self.values,
                *self.args,
                **self.kwargs,
            )

        # TODO: mixed type case
        if result.ndim == 2:
            return self.obj._constructor(result, index=self.index, columns=self.columns)
        else:
            return self.obj._constructor_sliced(result, index=self.agg_axis)

    def apply_broadcast(self, target: DataFrame) -> DataFrame:
        assert callable(self.func)

        result_values = np.empty_like(target.values)

        # axis which we want to compare compliance
        result_compare = target.shape[0]

        for i, col in enumerate(target.columns):
            res = self.func(target[col], *self.args, **self.kwargs)
            ares = np.asarray(res).ndim

            # must be a scalar or 1d
            if ares > 1:
                raise ValueError("too many dims to broadcast")
            if ares == 1:
                # must match return dim
                if result_compare != len(res):
                    raise ValueError("cannot broadcast result")

            result_values[:, i] = res

        # we *always* preserve the original index / columns
        result = self.obj._constructor(
            result_values, index=target.index, columns=target.columns
        )
        return result

    def apply_standard(self) -> Union[DataFrame, Series]:
        if self.engine == "python":
            results, res_index = self.apply_series_generator()
        else:
            results, res_index = self.apply_series_numba()

        # wrap results
        return self.wrap_results(results, res_index)

    def apply_series_generator(self) -> Tuple[ResType, Index]:
        assert callable(self.func)

        series_gen = self.series_generator
        res_index = self.result_index

        results = {}

        for i, v in enumerate(series_gen):
            results[i] = self.func(v, *self.args, **self.kwargs)
            if isinstance(results[i], ABCSeries):
                # If we have a view on v, we need to make a copy because
                #  series_generator will swap out the underlying data
                results[i] = results[i].copy(deep=False)

        return results, res_index

