import functools
import importlib
import inspect
import threading
import time
from types import ModuleType
from typing import Any, Callable, List, Optional, Set, Tuple, Union
import pandas as pd
from databricks.koalas import config, namespace, sql
from databricks.koalas.accessors import KoalasFrameMethods
from databricks.koalas.frame import DataFrame
from databricks.koalas.datetimes import DatetimeMethods
from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy
from databricks.koalas.indexes.base import Index
from databricks.koalas.indexes.category import CategoricalIndex
from databricks.koalas.indexes.datetimes import DatetimeIndex
from databricks.koalas.indexes.multi import MultiIndex
from databricks.koalas.indexes.numeric import Float64Index, Int64Index
from databricks.koalas.missing.frame import _MissingPandasLikeDataFrame
from databricks.koalas.missing.groupby import MissingPandasLikeDataFrameGroupBy, MissingPandasLikeSeriesGroupBy
from databricks.koalas.missing.indexes import (
    MissingPandasLikeCategoricalIndex,
    MissingPandasLikeDatetimeIndex,
    MissingPandasLikeIndex,
    MissingPandasLikeMultiIndex,
)
from databricks.koalas.missing.series import MissingPandasLikeSeries
from databricks.koalas.missing.window import (
    MissingPandasLikeExpanding,
    MissingPandasLikeRolling,
    MissingPandasLikeExpandingGroupby,
    MissingPandasLikeRollingGroupby,
)
from databricks.koalas.series import Series
from databricks.koalas.spark.accessors import (
    CachedSparkFrameMethods,
    SparkFrameMethods,
    SparkIndexOpsMethods,
)
from databricks.koalas.strings import StringMethods
from databricks.koalas.window import Expanding, ExpandingGroupby, Rolling, RollingGroupby

def attach(logger_module: Union[str, ModuleType]) -> None:
    """
    Attach the usage logger.

    Parameters
    ----------
    logger_module : the module or module name contains the usage logger.
        The module needs to provide `get_logger` function as an entry point of the plug-in
        returning the usage logger.

    See Also
    --------
    usage_logger : the reference implementation of the usage logger.
    """
    if isinstance(logger_module, str):
        logger_module = importlib.import_module(logger_module)
    logger = getattr(logger_module, 'get_logger')()
    modules: List[ModuleType] = [config, namespace]
    classes: List[Any] = [
        DataFrame,
        Series,
        Index,
        MultiIndex,
        Int64Index,
        Float64Index,
        CategoricalIndex,
        DatetimeIndex,
        DataFrameGroupBy,
        SeriesGroupBy,
        DatetimeMethods,
        StringMethods,
        Expanding,
        ExpandingGroupby,
        Rolling,
        RollingGroupby,
        CachedSparkFrameMethods,
        SparkFrameMethods,
        SparkIndexOpsMethods,
        KoalasFrameMethods,
    ]
    try:
        from databricks.koalas import mlflow
        modules.append(mlflow)
        classes.append(mlflow.PythonModelWrapper)
    except ImportError:
        pass
    sql._CAPTURE_SCOPES = 3
    modules.append(sql)
    for target_module in modules:
        target_name: str = target_module.__name__.split('.')[-1]
        module_all: List[str] = getattr(target_module, '__all__', [])
        for name in module_all:
            func: Any = getattr(target_module, name)
            if not inspect.isfunction(func):
                continue
            wrapped_func: Callable[..., Any] = _wrap_function(target_name, name, func, logger)
            setattr(target_module, name, wrapped_func)
    special_functions: Set[str] = {
        '__init__',
        '__repr__',
        '__str__',
        '_repr_html_',
        '__len__',
        '__getitem__',
        '__setitem__',
        '__getattr__',
    }
    for target_class in classes:
        for name, func in inspect.getmembers(target_class, inspect.isfunction):
            if name.startswith('_') and name not in special_functions:
                continue
            wrapped_func: Callable[..., Any] = _wrap_function(target_class.__name__, name, func, logger)
            setattr(target_class, name, wrapped_func)
        for name, prop in inspect.getmembers(target_class, lambda o: isinstance(o, property)):
            if name.startswith('_'):
                continue
            wrapped_prop = _wrap_property(target_class.__name__, name, prop, logger)
            setattr(target_class, name, wrapped_prop)
    for original, missing in [
        (pd.DataFrame, _MissingPandasLikeDataFrame),
        (pd.Series, MissingPandasLikeSeries),
        (pd.Index, MissingPandasLikeIndex),
        (pd.MultiIndex, MissingPandasLikeMultiIndex),
        (pd.CategoricalIndex, MissingPandasLikeCategoricalIndex),
        (pd.DatetimeIndex, MissingPandasLikeDatetimeIndex),
        (pd.core.groupby.DataFrameGroupBy, MissingPandasLikeDataFrameGroupBy),
        (pd.core.groupby.SeriesGroupBy, MissingPandasLikeSeriesGroupBy),
        (pd.core.window.Expanding, MissingPandasLikeExpanding),
        (pd.core.window.Rolling, MissingPandasLikeRolling),
        (pd.core.window.ExpandingGroupby, MissingPandasLikeExpandingGroupby),
        (pd.core.window.RollingGroupby, MissingPandasLikeRollingGroupby),
    ]:
        for name, func in inspect.getmembers(missing, inspect.isfunction):
            wrapped_func = _wrap_missing_function(original.__name__, name, func, original, logger)
            setattr(missing, name, wrapped_func)
        for name, prop in inspect.getmembers(missing, lambda o: isinstance(o, property)):
            wrapped_prop = _wrap_missing_property(original.__name__, name, prop, logger)
            setattr(missing, name, wrapped_prop)

_local = threading.local()

def _wrap_function(
    class_name: str,
    function_name: str,
    func: Callable[..., Any],
    logger: Any
) -> Callable[..., Any]:
    signature: inspect.Signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if hasattr(_local, 'logging') and _local.logging:
            return func(*args, **kwargs)
        _local.logging = True
        try:
            start: float = time.perf_counter()
            try:
                res: Any = func(*args, **kwargs)
                logger.log_success(class_name, function_name, time.perf_counter() - start, signature)
                return res
            except Exception as ex:
                logger.log_failure(class_name, function_name, ex, time.perf_counter() - start, signature)
                raise
        finally:
            _local.logging = False
    return wrapper

def _wrap_property(
    class_name: str,
    property_name: str,
    prop: property,
    logger: Any
) -> property:
    
    @property
    def wrapper(self: Any) -> Any:
        if hasattr(_local, 'logging') and _local.logging:
            return prop.fget(self)
        _local.logging = True
        try:
            start: float = time.perf_counter()
            try:
                res: Any = prop.fget(self)
                logger.log_success(class_name, property_name, time.perf_counter() - start)
                return res
            except Exception as ex:
                logger.log_failure(class_name, property_name, ex, time.perf_counter() - start)
                raise
        finally:
            _local.logging = False

    wrapper.__doc__ = prop.__doc__
    if prop.fset is not None:
        wrapped_setter = wrapper.setter(
            _wrap_function(class_name, prop.fset.__name__, prop.fset, logger)
        )
        return wrapped_setter
    return wrapper

def _wrap_missing_function(
    class_name: str,
    function_name: str,
    func: Callable[..., Any],
    original: Any,
    logger: Any
) -> Callable[..., Any]:
    if not hasattr(original, function_name):
        return func
    original_func: Callable[..., Any] = getattr(original, function_name)
    signature: inspect.Signature = inspect.signature(original_func)
    is_deprecated: bool = func.__name__ == 'deprecated_function'

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        finally:
            logger.log_missing(class_name, function_name, is_deprecated, signature)
    return wrapper

def _wrap_missing_property(
    class_name: str,
    property_name: str,
    prop: property,
    logger: Any
) -> property:
    is_deprecated: bool = prop.fget.__name__ == 'deprecated_property'

    @property
    def wrapper(self: Any) -> Any:
        try:
            return prop.fget(self)
        finally:
            logger.log_missing(class_name, property_name, is_deprecated)
    return wrapper
