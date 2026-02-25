from abc import ABCMeta, abstractmethod
import sys
import inspect
from collections import OrderedDict, namedtuple
from collections.abc import Callable
from distutils.version import LooseVersion
from functools import partial
from itertools import product
from typing import Any, List, Set, Tuple, Union, cast, Dict, Optional, Sequence
import pandas as pd
from pandas.api.types import is_hashable, is_list_like
from pyspark.sql import Window, functions as F
from pyspark.sql.types import FloatType, DoubleType, NumericType, StructField, StructType, StringType
from pyspark.sql.functions import PandasUDFType, pandas_udf, Column
from databricks import koalas as ks
from databricks.koalas.typedef import infer_return_type, DataFrameType, ScalarType, SeriesType
from databricks.koalas.frame import DataFrame
from databricks.koalas.internal import InternalFrame, HIDDEN_COLUMNS, NATURAL_ORDER_COLUMN_NAME, SPARK_INDEX_NAME_FORMAT, SPARK_DEFAULT_SERIES_NAME
from databricks.koalas.missing.groupby import MissingPandasLikeDataFrameGroupBy, MissingPandasLikeSeriesGroupBy
from databricks.koalas.series import Series, first_series
from databricks.koalas.config import get_option
from databricks.koalas.utils import align_diff_frames, is_name_like_tuple, is_name_like_value, name_like_string, same_anchor, scol_for, verify_temp_column_name
from databricks.koalas.spark.utils import as_nullable_spark_type, force_decimal_precision_scale
from databricks.koalas.window import RollingGroupby, ExpandingGroupby
from databricks.koalas.exceptions import DataError
from databricks.koalas.spark import functions as SF

NamedAgg = namedtuple('NamedAgg', ['column', 'aggfunc'])

class GroupBy(object, metaclass=ABCMeta):
    def __init__(self, kdf: DataFrame, groupkeys: List[Series], as_index: bool, dropna: bool, column_labels_to_exlcude: Set[Tuple], agg_columns_selected: bool, agg_columns: List[Series]) -> None:
        self._kdf: DataFrame = kdf
        self._groupkeys: List[Series] = groupkeys
        self._as_index: bool = as_index
        self._dropna: bool = dropna
        self._column_labels_to_exlcude: Set[Tuple] = column_labels_to_exlcude
        self._agg_columns_selected: bool = agg_columns_selected
        self._agg_columns: List[Series] = agg_columns

    @property
    def _groupkeys_scols(self) -> List[Column]:
        return [s.spark.column for s in self._groupkeys]

    @property
    def _agg_columns_scols(self) -> List[Column]:
        return [s.spark.column for s in self._agg_columns]

    @abstractmethod
    def _apply_series_op(self, op: Callable, should_resolve: bool = False, numeric_only: bool = False) -> Union[DataFrame, Series]:
        pass

    def aggregate(self, func_or_funcs: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]] = None, *args: Any, **kwargs: Any) -> Union[DataFrame, Series]:
        if func_or_funcs is None and kwargs is None:
            raise ValueError('No aggregation argument or function specified.')
        relabeling: bool = func_or_funcs is None and is_multi_agg_with_relabel(**kwargs)
        if relabeling:
            func_or_funcs, columns, order = normalize_keyword_aggregation(kwargs)
        if not isinstance(func_or_funcs, (str, list)):
            if not isinstance(func_or_funcs, dict) or not all((is_name_like_value(key) and (isinstance(value, str) or (isinstance(value, list) and all((isinstance(v, str) for v in value)))) for key, value in func_or_funcs.items())):
                raise ValueError('aggs must be a dict mapping from column name to aggregate functions (string or list of strings).')
        else:
            agg_cols = [col.name for col in self._agg_columns]
            func_or_funcs = OrderedDict([(col, func_or_funcs) for col in agg_cols])
        kdf = DataFrame(GroupBy._spark_groupby(self._kdf, func_or_funcs, self._groupkeys))
        if self._dropna:
            kdf = DataFrame(kdf._internal.with_new_sdf(kdf._internal.spark_frame.dropna(subset=kdf._internal.index_spark_column_names)))
        if not self._as_index:
            should_drop_index = set((i for i, gkey in enumerate(self._groupkeys) if gkey._kdf is not self._kdf))
            if len(should_drop_index) > 0:
                kdf = kdf.reset_index(level=should_drop_index, drop=True)
            if len(should_drop_index) < len(self._groupkeys):
                kdf = kdf.reset_index()
        if relabeling:
            kdf = kdf[order]
            kdf.columns = columns
        return kdf

    agg = aggregate

    @staticmethod
    def _spark_groupby(kdf: DataFrame, func: Dict[str, Union[str, List[str]]], groupkeys: Sequence[Series] = ()) -> InternalFrame:
        groupkey_names = [SPARK_INDEX_NAME_FORMAT(i) for i in range(len(groupkeys))]
        groupkey_scols = [s.spark.column.alias(name) for s, name in zip(groupkeys, groupkey_names)]
        multi_aggs = any((isinstance(v, list) for v in func.values()))
        reordered = []
        data_columns = []
        column_labels = []
        for key, value in func.items():
            label = key if is_name_like_tuple(key) else (key,)
            if len(label) != kdf._internal.column_labels_level:
                raise TypeError('The length of the key must be the same as the column label level.')
            for aggfunc in [value] if isinstance(value, str) else value:
                column_label = tuple(list(label) + [aggfunc]) if multi_aggs else label
                column_labels.append(column_label)
                data_col = name_like_string(column_label)
                data_columns.append(data_col)
                col_name = kdf._internal.spark_column_name_for(label)
                if aggfunc == 'nunique':
                    reordered.append(F.expr('count(DISTINCT `{0}`) as `{1}`'.format(col_name, data_col)))
                elif aggfunc == 'quartiles':
                    reordered.append(F.expr('percentile_approx(`{0}`, array(0.25, 0.5, 0.75)) as `{1}`'.format(col_name, data_col)))
                else:
                    reordered.append(F.expr('{1}(`{0}`) as `{2}`'.format(col_name, aggfunc, data_col)))
        sdf = kdf._internal.spark_frame.select(groupkey_scols + kdf._internal.data_spark_columns)
        sdf = sdf.groupby(*groupkey_names).agg(*reordered)
        if len(groupkeys) > 0:
            index_spark_column_names = groupkey_names
            index_names = [kser._column_label for kser in groupkeys]
            index_dtypes = [kser.dtype for kser in groupkeys]
        else:
            index_spark_column_names = []
            index_names = []
            index_dtypes = []
        return InternalFrame(spark_frame=sdf, index_spark_columns=[scol_for(sdf, col) for col in index_spark_column_names], index_names=index_names, index_dtypes=index_dtypes, column_labels=column_labels, data_spark_columns=[scol_for(sdf, col) for col in data_columns])

    def count(self) -> Union[DataFrame, Series]:
        return self._reduce_for_stat_function(F.count, only_numeric=False)

    def first(self) -> Union[DataFrame, Series]:
        return self._reduce_for_stat_function(F.first, only_numeric=False)

    def last(self) -> Union[DataFrame, Series]:
        return self._reduce_for_stat_function(lambda col: F.last(col, ignorenulls=True), only_numeric=False)

    def max(self) -> Union[DataFrame, Series]:
        return self._reduce_for_stat_function(F.max, only_numeric=False)

    def mean(self) -> Union[DataFrame, Series]:
        return self._reduce_for_stat_function(F.mean, only_numeric=True)

    def min(self) -> Union[DataFrame, Series]:
        return self._reduce_for_stat_function(F.min, only_numeric=False)

    def std(self, ddof: int = 1) -> Union[DataFrame, Series]:
        assert ddof in (0, 1)
        return self._reduce_for_stat_function(F.stddev_pop if ddof == 0 else F.stddev_samp, only_numeric=True)

    def sum(self) -> Union[DataFrame, Series]:
        return self._reduce_for_stat_function(F.sum, only_numeric=True)

    def var(self, ddof: int = 1) -> Union[DataFrame, Series]:
        assert ddof in (0, 1)
        return self._reduce_for_stat_function(F.var_pop if ddof == 0 else F.var_samp, only_numeric=True)

    def all(self) -> Union[DataFrame, Series]:
        return self._reduce_for_stat_function(lambda col: F.min(F.coalesce(col.cast('boolean'), F.lit(True))), only_numeric=False)

    def any(self) -> Union[DataFrame, Series]:
        return self._reduce_for_stat_function(lambda col: F.max(F.coalesce(col.cast('boolean'), F.lit(False))), only_numeric=False)

    def size(self) -> Series:
        groupkeys = self._groupkeys
        groupkey_names = [SPARK_INDEX_NAME_FORMAT(i) for i in range(len(groupkeys))]
        groupkey_scols = [s.spark.column.alias(name) for s, name in zip(groupkeys, groupkey_names)]
        sdf = self._kdf._internal.spark_frame.select(groupkey_scols + self._kdf._internal.data_spark_columns)
        sdf = sdf.groupby(*groupkey_names).count()
        internal = InternalFrame(spark_frame=sdf, index_spark_columns=[scol_for(sdf, col) for col in groupkey_names], index_names=[kser._column_label for kser in groupkeys], index_dtypes=[kser.dtype for kser in groupkeys], column_labels=[None], data_spark_columns=[scol_for(sdf, 'count')])
        return first_series(DataFrame(internal))

    def diff(self, periods: int = 1) -> Union[DataFrame, Series]:
        return self._apply_series_op(lambda sg: sg._kser._diff(periods, part_cols=sg._groupkeys_scols), should_resolve=True)

    def cumcount(self, ascending: bool = True) -> Series:
        ret = self._groupkeys[0].rename().spark.transform(lambda _: F.lit(0))._cum(F.count, True, part_cols=self._groupkeys_scols, ascending=ascending) - 1
        internal = ret._internal.resolved_copy
        return first_series(DataFrame(internal))

    def cummax(self) -> Union[DataFrame, Series]:
        return self._apply_series_op(lambda sg: sg._kser._cum(F.max, True, part_cols=sg._groupkeys_scols), should_resolve=True, numeric_only=True)

    def cummin(self) -> Union[DataFrame, Series]:
        return self._apply_series_op(lambda sg: sg._kser._cum(F.min, True, part_cols=sg._groupkeys_scols), should_resolve=True, numeric_only=True)

    def cumprod(self) -> Union[DataFrame, Series]:
        return self._apply_series_op(lambda sg: sg._kser._cumprod(True, part_cols=sg._groupkeys_scols), should_resolve=True, numeric_only=True)

    def cumsum(self) -> Union[DataFrame, Series]:
        return self._apply_series_op(lambda sg: sg._kser._cumsum(True, part_cols=sg._groupkeys_scols), should_resolve=True, numeric_only=True)

    def apply(self, func: Callable, *args: Any, **kwargs: Any) -> Union[DataFrame, Series]:
        if LooseVersion(pd.__version__) >= LooseVersion('1.3.0'):
            from pandas.core.common import _builtin_table
        else:
            from pandas.core.base import SelectionMixin
            _builtin_table = SelectionMixin._builtin_table
        if not isinstance(func, Callable):
            raise TypeError('%s object is not callable' % type(func).__name__)
        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get('return', None)
        should_infer_schema = return_sig is None
        is_series_groupby = isinstance(self, SeriesGroupBy)
        kdf = self._kdf
        if self._agg_columns_selected:
            agg_columns = self._agg_columns
        else:
            agg_columns = [kdf._kser_for(label) for label in kdf._internal.column_labels if label not in self._column_labels_to_exlcude]
        kdf, groupkey_labels, groupkey_names = GroupBy._prepare_group_map_apply(kdf, self._groupkeys, agg_columns)
        if is_series_groupby:
            name = kdf.columns[-1]
            pandas_apply = _builtin_table.get(func, func)
        else:
            f = _builtin_table.get(func, func)

            def pandas_apply(pdf: pd.DataFrame, *a: Any, **k: Any) -> pd.DataFrame:
                return f(pdf.drop(groupkey_names, axis=1), *a, **k)
        should_return_series = False
        if should_infer_schema:
            limit = get_option('compute.shortcut_limit')
            pdf = kdf.head(limit + 1)._to_internal_pandas()
            groupkeys = [pdf[groupkey_name].rename(kser.name) for groupkey_name, kser in zip(groupkey_names, self._groupkeys)]
            if is_series_groupby:
                pser_or_pdf = pdf.groupby(groupkeys)[name].apply(pandas_apply, *args, **kwargs)
            else:
                pser_or_pdf = pdf.groupby(groupkeys).apply(pandas_apply, *args, **kwargs)
            kser_or_kdf = ks.from_pandas(pser_or_pdf)
            if len(pdf) <= limit:
                if isinstance(kser_or_kdf, ks.Series) and is_series_groupby:
                    kser_or_kdf = kser_or_kdf.rename(cast(SeriesGroupBy, self)._kser.name)
                return cast(Union[Series, DataFrame], kser_or_kdf)
            if isinstance(kser_or_kdf, Series):
                should_return_series = True
                kdf_from_pandas = kser_or_kdf._kdf
            else:
                kdf_from_pandas = cast(DataFrame, kser_or_kdf)
            return_schema = force_decimal_precision_scale(as_nullable_spark_type(kdf_from_pandas._internal.spark_frame.drop(*HIDDEN_COLUMNS).schema))
        else:
            return_type = infer_return_type(func)
            if not is_series_groupby and isinstance(return_type, SeriesType):
                raise TypeError('Series as a return type hint at frame groupby is not supported currently; however got [%s]. Use DataFrame type hint instead.' % return_sig)
            if isinstance(return_type, DataFrameType):
                return_schema = cast(DataFrameType, return_type).spark_type
                data_dtypes = cast(DataFrameType, return_type).dtypes
            else:
                should_return_series = True
                return_schema = cast(Union[SeriesType, ScalarType], return_type).spark_type
                if is_series_groupby:
                    return_schema = StructType([StructField(name, return_schema)])
                else:
                    return_schema = StructType([StructField(SPARK_DEFAULT_SERIES_NAME, return_schema)])
                data_dtypes = [cast(Union[SeriesType, ScalarType], return_type).dtype]

        def pandas_groupby_apply(pdf: pd.DataFrame) -> pd.DataFrame:
            if not is_series_groupby and LooseVersion(pd.__version__) < LooseVersion('0.25'):
                should_skip_first_call = True

                def wrapped_func(df: pd.DataFrame, *a: Any, **k: Any) -> pd.DataFrame:
                    nonlocal should_skip_first_call
                    if should_skip_first_call:
                        should_skip_first_call = False
                        if should_return_series:
                            return pd.Series()
                        else:
                            return pd.DataFrame()
                    else:
                        return pandas_apply(df, *a, **k)
            else:
                wrapped_func = pandas_apply
            if is_series_groupby:
                pdf_or_ser = pdf.groupby(groupkey_names)[name].apply(wrapped_func, *args, **kwargs)
            else:
                pdf_or_ser = pdf.groupby(groupkey_names).apply(wrapped_func, *args, **kwargs)
            if not isinstance(pdf_or_ser, pd.DataFrame):
                return pd.DataFrame(pdf_or_ser)
            else:
                return pdf_or_ser
        sdf = GroupBy._spark_group_map_apply(kdf, pandas_groupby_apply, [kdf._internal.spark_column_for(label) for label in groupkey_labels], return_schema, retain_index=should_infer_schema)
        if should_infer_schema:
            internal = kdf_from_pandas._internal.with_new_sdf(sdf)
        else:
            internal = InternalFrame(spark_frame=sdf, index_spark_columns=None, data_dtypes=data_dtypes)
        if should_return_series:
            kser = first_series(DataFrame(internal))
            if is_series_groupby:
                kser = kser.rename(cast(SeriesGroupBy