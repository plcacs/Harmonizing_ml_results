"""
Commonly used utils in Koalas.
"""
import functools
from collections import OrderedDict
from contextlib import contextmanager
from distutils.version import LooseVersion
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Iterator, Set, cast
import warnings
import pyarrow
import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, DataType
from pyspark.sql import Column
import pandas as pd
from pandas.api.types import is_list_like
from databricks import koalas as ks
from databricks.koalas.typedef.typehints import as_spark_type, extension_dtypes, spark_type_to_pandas_dtype

if TYPE_CHECKING:
    from databricks.koalas.base import IndexOpsMixin
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.internal import InternalFrame
    from databricks.koalas.series import Series

ERROR_MESSAGE_CANNOT_COMBINE = "Cannot combine the series or dataframe because it comes from a different dataframe. In order to allow this operation, enable 'compute.ops_on_diff_frames' option."

if LooseVersion(pyspark.__version__) < LooseVersion('3.0'):
    SPARK_CONF_ARROW_ENABLED = 'spark.sql.execution.arrow.enabled'
else:
    SPARK_CONF_ARROW_ENABLED = 'spark.sql.execution.arrow.pyspark.enabled'

def same_anchor(this: Union['DataFrame', 'IndexOpsMixin', 'InternalFrame'], 
               that: Union['DataFrame', 'IndexOpsMixin', 'InternalFrame']) -> bool:
    """
    Check if the anchors of the given DataFrame or Series are the same or not.
    """
    from databricks.koalas.base import IndexOpsMixin
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.internal import InternalFrame
    
    if isinstance(this, InternalFrame):
        this_internal = this
    else:
        assert isinstance(this, (DataFrame, IndexOpsMixin)), type(this)
        this_internal = this._internal
    
    if isinstance(that, InternalFrame):
        that_internal = that
    else:
        assert isinstance(that, (DataFrame, IndexOpsMixin)), type(that)
        that_internal = that._internal
    
    return (this_internal.spark_frame is that_internal.spark_frame and 
            this_internal.index_level == that_internal.index_level and 
            all((this_scol._jc.equals(that_scol._jc) for this_scol, that_scol in 
                zip(this_internal.index_spark_columns, that_internal.index_spark_columns)))

def combine_frames(this: 'DataFrame', *args: Union['Series', 'DataFrame'], 
                  how: str = 'full', preserve_order_column: bool = False) -> 'DataFrame':
    """
    This method combines `this` DataFrame with a different `that` DataFrame or
    Series from a different DataFrame.
    """
    from databricks.koalas.config import get_option
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.internal import InternalFrame, HIDDEN_COLUMNS, NATURAL_ORDER_COLUMN_NAME, SPARK_INDEX_NAME_FORMAT
    from databricks.koalas.series import Series

    if all((isinstance(arg, Series) for arg in args):
        assert all((same_anchor(arg, args[0]) for arg in args), 'Currently only one different DataFrame (from given Series) is supported'
        assert not same_anchor(this, args[0]), "We don't need to combine. All series is in this."
        that = args[0]._kdf[list(args)]
    elif len(args) == 1 and isinstance(args[0], DataFrame):
        assert isinstance(args[0], DataFrame)
        assert not same_anchor(this, args[0]), "We don't need to combine. `this` and `that` are same."
        that = args[0]
    else:
        raise AssertionError('args should be single DataFrame or single/multiple Series')

    if get_option('compute.ops_on_diff_frames'):
        def resolve(internal: InternalFrame, side: str) -> InternalFrame:
            rename = lambda col: '__{}_{}'.format(side, col)
            internal = internal.resolved_copy
            sdf = internal.spark_frame
            sdf = internal.spark_frame.select(
                [scol_for(sdf, col).alias(rename(col)) for col in sdf.columns if col not in HIDDEN_COLUMNS] + 
                list(HIDDEN_COLUMNS))
            return internal.copy(
                spark_frame=sdf, 
                index_spark_columns=[scol_for(sdf, rename(col)) for col in internal.index_spark_column_names],
                data_spark_columns=[scol_for(sdf, rename(col)) for col in internal.data_spark_column_names])

        this_internal = resolve(this._internal, 'this')
        that_internal = resolve(that._internal, 'that')
        
        this_index_map = list(zip(this_internal.index_spark_column_names, this_internal.index_names, this_internal.index_dtypes))
        that_index_map = list(zip(that_internal.index_spark_column_names, that_internal.index_names, that_internal.index_dtypes))
        
        assert len(this_index_map) == len(that_index_map)
        
        join_scols: List[Column] = []
        merged_index_scols: List[Column] = []
        this_and_that_index_map = list(zip(this_index_map, that_index_map))
        
        this_sdf = this_internal.spark_frame.alias('this')
        that_sdf = that_internal.spark_frame.alias('that')
        
        index_column_names: List[str] = []
        index_use_extension_dtypes: List[bool] = []
        
        for i, ((this_column, this_name, this_dtype), (that_column, that_name, that_dtype)) in enumerate(this_and_that_index_map):
            if this_name == that_name:
                this_scol = scol_for(this_sdf, this_column)
                that_scol = scol_for(that_sdf, that_column)
                join_scol = this_scol == that_scol
                join_scols.append(join_scol)
                column_name = SPARK_INDEX_NAME_FORMAT(i)
                index_column_names.append(column_name)
                index_use_extension_dtypes.append(
                    any(isinstance(dtype, extension_dtypes) for dtype in [this_dtype, that_dtype]))
                merged_index_scols.append(
                    F.when(this_scol.isNotNull(), this_scol).otherwise(that_scol).alias(column_name))
            else:
                raise ValueError('Index names must be exactly matched currently.')

        assert len(join_scols) > 0, 'cannot join with no overlapping index names'
        
        joined_df = this_sdf.join(that_sdf, on=join_scols, how=how)
        
        if preserve_order_column:
            order_column = [scol_for(this_sdf, NATURAL_ORDER_COLUMN_NAME)]
        else:
            order_column = []
            
        joined_df = joined_df.select(
            merged_index_scols + 
            [scol_for(this_sdf, this_internal.spark_column_name_for(label)) for label in this_internal.column_labels] + 
            [scol_for(that_sdf, that_internal.spark_column_name_for(label)) for label in that_internal.column_labels] + 
            order_column)
        
        index_spark_columns = [scol_for(joined_df, col) for col in index_column_names]
        index_dtypes = [
            spark_type_to_pandas_dtype(field.dataType, use_extension_dtypes=use_extension_dtypes) 
            for field, use_extension_dtypes in zip(joined_df.select(index_spark_columns).schema, index_use_extension_dtypes)]
        
        index_columns = set(index_column_names)
        new_data_columns = [col for col in joined_df.columns if col not in index_columns and col != NATURAL_ORDER_COLUMN_NAME]
        data_dtypes = this_internal.data_dtypes + that_internal.data_dtypes
        level = max(this_internal.column_labels_level, that_internal.column_labels_level)

        def fill_label(label: Optional[Tuple[str, ...]]) -> List[Optional[str]]:
            if label is None:
                return [''] * (level - 1) + [None]
            else:
                return [''] * (level - len(label)) + list(label)

        column_labels = (
            [tuple(['this'] + fill_label(label)) for label in this_internal.column_labels] + 
            [tuple(['that'] + fill_label(label)) for label in that_internal.column_labels])
        
        column_label_names = [None] * (1 + level - this_internal.column_labels_level) + this_internal.column_label_names
        
        return DataFrame(
            InternalFrame(
                spark_frame=joined_df,
                index_spark_columns=index_spark_columns,
                index_names=this_internal.index_names,
                index_dtypes=index_dtypes,
                column_labels=column_labels,
                data_spark_columns=[scol_for(joined_df, col) for col in new_data_columns],
                data_dtypes=data_dtypes,
                column_label_names=column_label_names))
    else:
        raise ValueError(ERROR_MESSAGE_CANNOT_COMBINE)

def align_diff_frames(
    resolve_func: Callable[['DataFrame', List[Tuple[str, ...]], List[Tuple[str, ...]]], 
    this: 'DataFrame', 
    that: 'DataFrame', 
    fillna: bool = True, 
    how: str = 'full', 
    preserve_order_column: bool = False
) -> 'DataFrame':
    """
    This method aligns two different DataFrames with a given `func`.
    """
    from databricks.koalas.frame import DataFrame
    
    assert how in ('full', 'left', 'inner')
    
    this_column_labels = this._internal.column_labels
    that_column_labels = that._internal.column_labels
    common_column_labels = set(this_column_labels).intersection(that_column_labels)
    
    combined = combine_frames(this, that, how=how, preserve_order_column=preserve_order_column)
    combined_column_labels = combined._internal.column_labels
    
    that_columns_to_apply: List[Tuple[str, ...]] = []
    this_columns_to_apply: List[Tuple[str, ...]] = []
    additional_that_columns: List[Tuple[str, ...]] = []
    columns_to_keep: List[Column] = []
    column_labels_to_keep: List[Tuple[str, ...]] = []
    
    for combined_label in combined_column_labels:
        for common_label in common_column_labels:
            if combined_label == tuple(['this', *common_label]):
                this_columns_to_apply.append(combined_label)
                break
            elif combined_label == tuple(['that', *common_label]):
                that_columns_to_apply.append(combined_label)
                break
        else:
            if how == 'left' and combined_label in [tuple(['that', *label]) for label in that_column_labels]:
                additional_that_columns.append(combined_label)
            elif fillna:
                columns_to_keep.append(F.lit(None).cast(DoubleType()).alias(str(combined_label)))
                column_labels_to_keep.append(combined_label)
            else:
                columns_to_keep.append(combined._kser_for(combined_label))
                column_labels_to_keep.append(combined_label)
    
    that_columns_to_apply += additional_that_columns
    
    if len(this_columns_to_apply) > 0 or len(that_columns_to_apply) > 0:
        kser_set, column_labels_applied = zip(*resolve_func(combined, this_columns_to_apply, that_columns_to_apply))
        columns_applied = list(kser_set)
        column_labels_applied = list(column_labels_applied)
    else:
        columns_applied = []
        column_labels_applied = []
    
    applied = DataFrame(
        combined._internal.with_new_columns(
            columns_applied + columns_to_keep,
            column_labels=column_labels_applied + column_labels_to_keep))
    
    this_labels = OrderedDict()
    for this_label in this_column_labels:
        for new_label in applied._internal.column_labels:
            if new_label[1:] not in this_labels and this_label == new_label[1:]:
                this_labels[new_label[1:]] = new_label
    
    other_labels = OrderedDict()
    for new_label in applied._internal.column_labels:
        if new_label[1:] not in this_labels:
            other_labels[new_label[1:]] = new_label
    
    kdf = applied[list(this_labels.values()) + list(other_labels.values())]
    kdf.columns = kdf.columns.droplevel()
    return kdf

def is_testing() -> bool:
    """ Indicates whether Koalas is currently running tests. """
    return 'KOALAS_TESTING' in os.environ

def default_session(conf: Optional[Dict[str, Any]] = None) -> spark.SparkSession:
    if conf is None:
        conf = dict()
    
    should_use_legacy_ipc = False
    if LooseVersion(pyarrow.__version__) >= LooseVersion('0.15') and LooseVersion(pyspark.__version__) < LooseVersion('3.0'):
        conf['spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT'] = '1'
        conf['spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT'] = '1'
        conf['spark.mesos.driverEnv.ARROW_PRE_0_15_IPC_FORMAT'] = '1'
        conf['spark.kubernetes.driverEnv.ARROW_PRE_0_15_IPC_FORMAT'] = '1'
        should_use_legacy_ipc = True
    
    builder = spark.SparkSession.builder.appName('Koalas')
    for key, value in conf.items():
        builder = builder.config(key, value)
    
    builder.config('spark.sql.analyzer.failAmbiguousSelfJoin', False)
    
    if LooseVersion(pyspark.__version__) >= LooseVersion('3.0.1') and is_testing():
        builder.config('spark.executor.allowSparkContext', False)
    
    session = builder.getOrCreate()
    
    if not should_use_legacy_ipc:
        is_legacy_ipc_set = any(
            v == '1' for v in [
                session.conf.get('spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT', None),
                session.conf.get('spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT', None),
                session.conf.get('spark.mesos.driverEnv.ARROW_PRE_0_15_IPC_FORMAT', None),
                session.conf.get('spark.kubernetes.driverEnv.ARROW_PRE_0_15_IPC_FORMAT', None)])
        
        if is_legacy_ipc_set:
            raise RuntimeError(
                "Please explicitly unset 'ARROW_PRE_0_15_IPC_FORMAT' environment variable in both "
                "driver and executor sides. Check your spark.executorEnv.*, "
                "spark.yarn.appMasterEnv.*, spark.mesos.driverEnv.* and "
                "spark.kubernetes.driverEnv.* configurations. It is required to set this "
                "environment variable only when you use pyarrow>=0.15 and pyspark<3.0.")
    
    return session

@contextmanager
def sql_conf(pairs: Dict[str, Any], *, spark: Optional[spark.SparkSession] = None) -> Iterator[None]:
    """
    A convenient context manager to set `value` to the Spark SQL configuration `key` and
    then restores it back when it exits.
    """
    assert isinstance(pairs, dict), 'pairs should be a dictionary.'
    
    if spark is None:
        spark = default_session()
    
    keys = pairs.keys()
    new_values = pairs.values()
    old_values = [spark.conf.get(key, None) for key in keys]
    
    for key, new_value in zip(keys, new_values):
        spark.conf.set(key, new_value)
    
    try:
        yield
    finally:
        for key, old_value in zip(keys, old_values):
            if old_value is None:
                spark.conf.unset(key)
            else:
                spark.conf.set(key, old_value)

def validate_arguments_and_invoke_function(
    pobj: Union[pd.DataFrame, pd.Series],
    koalas_func: Callable[..., Any],
    pandas_func: Callable[..., Any],
    input_args: Dict[str, Any]
) -> Any:
    """
    Invokes a pandas function.
    """
    import inspect
    
    args = input_args.copy()
    del args['self']
    
    if 'kwargs' in args:
        kwargs = args['kwargs']
        del args['kwargs']
        args = {**args, **kwargs}
    
    koalas_params = inspect.signature(koalas_func).parameters
    pandas_params = inspect.signature(pandas_func).parameters
    
    for param in koalas_params.values():
        if param.name not in pandas_params:
            if args[param.name] == param.default:
                del args[param.name]
            else:
                raise TypeError(
                    ("The pandas version [%s] available does not support parameter '%s' " + 
                     "for function '%s'.") % 
                    (pd.__version__, param.name, pandas_func.__name__))
    
    args['self'] = pobj
    return pandas_func(**args)

def lazy_property(fn: Callable[..., Any]) -> Any:
    """
    Decorator that makes a property lazy-evaluated.
    """
    attr_name = '_lazy_' + fn.__name__

    @property
    @functools.wraps(fn)
    def wrapped_lazy_property(self: Any) -> Any:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    def deleter(self: Any) -> None:
        if hasattr(self, attr_name):
            delattr(self, attr_name)
    
    return wrapped_lazy_pro