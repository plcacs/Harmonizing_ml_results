import functools
import importlib
import inspect
import threading
import time
from types import ModuleType
from typing import Any, Callable, Union, Type
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
from databricks.koalas.missing.indexes import MissingPandasLikeCategoricalIndex, MissingPandasLikeDatetimeIndex, MissingPandasLikeIndex, MissingPandasLikeMultiIndex
from databricks.koalas.missing.series import MissingPandasLikeSeries
from databricks.koalas.missing.window import MissingPandasLikeExpanding, MissingPandasLikeRolling, MissingPandasLikeExpandingGroupby, MissingPandasLikeRollingGroupby
from databricks.koalas.series import Series
from databricks.koalas.spark.accessors import CachedSparkFrameMethods, SparkFrameMethods, SparkIndexOpsMethods
from databricks.koalas.strings import StringMethods
from databricks.koalas.window import Expanding, ExpandingGroupby, Rolling, RollingGroupby

from typing import get_type_hints  # just in case

_local: threading.local = threading.local()

def attach(logger_module: Union[str, ModuleType]) -> None:
    """
    Attach the usage logger.

    Parameters
    ----------
    logger_module : Union[str, ModuleType]
        The module or module name that contains the usage logger.
        The module needs to provide a `get_logger` function as an entry point of the plug-in
        returning the usage logger.

    See Also
    --------
    usage_logger : the reference implementation of the usage logger.
    """
    if isinstance(logger_module, str):
        logger_module = importlib.import_module(logger_module)
    logger: Any = getattr(logger_module, 'get_logger')()
    modules: list[Any] = [config, namespace]
    classes: list[Any] = [DataFrame, Series, Index, MultiIndex, Int64Index, Float64Index, CategoricalIndex, DatetimeIndex, DataFrameGroupBy, SeriesGroupBy, DatetimeMethods, StringMethods, Expanding, ExpandingGroupby, Rolling, RollingGroupby, CachedSparkFrameMethods, SparkFrameMethods, SparkIndexOpsMethods, KoalasFrameMethods]
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
        for name in getattr(target_module, '__all__'):
            func: Any = getattr(target_module, name)
            if not inspect.isfunction(func):
                continue
            setattr(target_module, name, _wrap_function(target_name, name, func, logger))
    special_functions: set[str] = set(['__init__', '__repr__', '__str__', '_repr_html_', '__len__', '__getitem__', '__setitem__', '__getattr__'])
    for target_class in classes:
        for name, func in inspect.getmembers(target_class, inspect.isfunction):
            if name.startswith('_') and name not in special_functions:
                continue
            setattr(target_class, name, _wrap_function(target_class.__name__, name, func, logger))
        for name, prop in inspect.getmembers(target_class, lambda o: isinstance(o, property)):
            if name.startswith('_'):
                continue
            setattr(target_class, name, _wrap_property(target_class.__name__, name, prop, logger))
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
        (pd.core.window.RollingGroupby, MissingPandasLikeRollingGroupby)
    ]:
        for name, func in inspect.getmembers(missing, inspect.isfunction):
            setattr(missing, name, _wrap_missing_function(original.__name__, name, func, original, logger))
        for name, prop in inspect.getmembers(missing, lambda o: isinstance(o, property)):
            setattr(missing, name, _wrap_missing_property(original.__name__, name, prop, logger))

def _wrap_function(class_name: str, function_name: str, func: Callable[..., Any], logger: Any) -> Callable[..., Any]:
    signature = inspect.signature(func)

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

def _wrap_property(class_name: str, property_name: str, prop: property, logger: Any) -> property:

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
        wrapper = wrapper.setter(_wrap_function(class_name, prop.fset.__name__, prop.fset, logger))
    return wrapper

def _wrap_missing_function(class_name: str, function_name: str, func: Callable[..., Any], original: Type[Any], logger: Any) -> Callable[..., Any]:
    if not hasattr(original, function_name):
        return func
    signature = inspect.signature(getattr(original, function_name))
    is_deprecated: bool = func.__name__ == 'deprecated_function'

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        finally:
            logger.log_missing(class_name, function_name, is_deprecated, signature)
    return wrapper

def _wrap_missing_property(class_name: str, property_name: str, prop: property, logger: Any) -> property:
    is_deprecated: bool = prop.fget.__name__ == 'deprecated_property'

    @property
    def wrapper(self: Any) -> Any:
        try:
            return prop.fget(self)
        finally:
            logger.log_missing(class_name, property_name, is_deprecated)
    return wrapper