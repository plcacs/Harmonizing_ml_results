#
# Copyright (C) 2019 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from distutils.version import LooseVersion
from functools import partial
from typing import Any, Optional, Tuple, Union, cast, List, Sequence, Dict, Set, Callable, TYPE_CHECKING
import warnings

import pandas as pd
from pandas.api.types import is_list_like
from pandas.api.types import is_hashable

import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F, Window

# For running doctests and reference resolution in PyCharm.
from databricks import koalas as ks  # noqa: F401
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.frame import DataFrame
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeMultiIndex
from databricks.koalas.series import Series, first_series
from databricks.koalas.utils import (
    compare_disallow_null,
    default_session,
    is_name_like_tuple,
    name_like_string,
    scol_for,
    verify_temp_column_name,
)
from databricks.koalas.internal import (
    InternalFrame,
    NATURAL_ORDER_COLUMN_NAME,
    SPARK_INDEX_NAME_FORMAT,
)
from databricks.koalas.typedef import Scalar


class MultiIndex(Index):
    def __new__(
        cls,
        levels: Optional[Sequence[Sequence[Any]] = None,
        codes: Optional[Sequence[Sequence[int]] = None,
        sortorder: Optional[int] = None,
        names: Optional[Sequence[Any]] = None,
        dtype: Optional[Any] = None,
        copy: bool = False,
        name: Optional[Any] = None,
        verify_integrity: bool = True,
    ) -> "MultiIndex":
        if LooseVersion(pd.__version__) < LooseVersion("0.24"):
            if levels is None or codes is None:
                raise TypeError("Must pass both levels and codes")

            pidx = pd.MultiIndex(
                levels=levels,
                labels=codes,
                sortorder=sortorder,
                names=names,
                dtype=dtype,
                copy=copy,
                name=name,
                verify_integrity=verify_integrity,
            )
        else:
            pidx = pd.MultiIndex(
                levels=levels,
                codes=codes,
                sortorder=sortorder,
                names=names,
                dtype=dtype,
                copy=copy,
                name=name,
                verify_integrity=verify_integrity,
            )
        return ks.from_pandas(pidx)

    @property
    def _internal(self) -> InternalFrame:
        internal = self._kdf._internal
        scol = F.struct(internal.index_spark_columns)
        return internal.copy(
            column_labels=[None],
            data_spark_columns=[scol],
            data_dtypes=[None],
            column_label_names=None,
        )

    @property
    def _column_label(self) -> None:
        return None

    def __abs__(self) -> None:
        raise TypeError("TypeError: cannot perform __abs__ with this index type: MultiIndex")

    def _with_new_scol(self, scol: spark.Column, *, dtype: Optional[Any] = None) -> "MultiIndex":
        raise NotImplementedError("Not supported for type MultiIndex")

    def _align_and_column_op(self, f: Callable[..., Any], *args: Any) -> Index:
        raise NotImplementedError("Not supported for type MultiIndex")

    def any(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("cannot perform any with this index type: MultiIndex")

    def all(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("cannot perform all with this index type: MultiIndex")

    @staticmethod
    def from_tuples(
        tuples: Sequence[Tuple[Any, ...]], 
        sortorder: Optional[int] = None, 
        names: Optional[Sequence[Any]] = None
    ) -> "MultiIndex":
        return cast(
            MultiIndex,
            ks.from_pandas(
                pd.MultiIndex.from_tuples(tuples=tuples, sortorder=sortorder, names=names)
            ),
        )

    @staticmethod
    def from_arrays(
        arrays: Sequence[Sequence[Any]], 
        sortorder: Optional[int] = None, 
        names: Optional[Sequence[Any]] = None
    ) -> "MultiIndex":
        return cast(
            MultiIndex,
            ks.from_pandas(
                pd.MultiIndex.from_arrays(arrays=arrays, sortorder=sortorder, names=names)
            ),
        )

    @staticmethod
    def from_product(
        iterables: Sequence[Sequence[Any]], 
        sortorder: Optional[int] = None, 
        names: Optional[Sequence[Any]] = None
    ) -> "MultiIndex":
        return cast(
            MultiIndex,
            ks.from_pandas(
                pd.MultiIndex.from_product(iterables=iterables, sortorder=sortorder, names=names)
            ),
        )

    @staticmethod
    def from_frame(df: DataFrame, names: Optional[Sequence[Any]] = None) -> "MultiIndex":
        if not isinstance(df, DataFrame):
            raise TypeError("Input must be a DataFrame")
        sdf = df.to_spark()

        if names is None:
            names = df._internal.column_labels
        elif not is_list_like(names):
            raise ValueError("Names should be list-like for a MultiIndex")
        else:
            names = [name if is_name_like_tuple(name) else (name,) for name in names]

        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in sdf.columns],
            index_names=names,
        )
        return cast(MultiIndex, DataFrame(internal).index)

    @property
    def name(self) -> str:
        raise PandasNotImplementedError(class_name="pd.MultiIndex", property_name="name")

    @name.setter
    def name(self, name: str) -> None:
        raise PandasNotImplementedError(class_name="pd.MultiIndex", property_name="name")

    def _verify_for_rename(self, name: Any) -> Sequence[Tuple[str, ...]]:
        if is_list_like(name):
            if self._internal.index_level != len(name):
                raise ValueError(
                    "Length of new names must be {}, got {}".format(
                        self._internal.index_level, len(name)
                    )
                )
            if any(not is_hashable(n) for n in name):
                raise TypeError("MultiIndex.name must be a hashable type")
            return [n if is_name_like_tuple(n) else (n,) for n in name]
        else:
            raise TypeError("Must pass list-like as `names`.")

    def swaplevel(self, i: Union[int, str] = -2, j: Union[int, str] = -1) -> "MultiIndex":
        for index in (i, j):
            if not isinstance(index, int) and index not in self.names:
                raise KeyError("Level %s not found" % index)

        i = i if isinstance(i, int) else self.names.index(i)
        j = j if isinstance(j, int) else self.names.index(j)

        for index in (i, j):
            if index >= len(self.names) or index < -len(self.names):
                raise IndexError(
                    "Too many levels: Index has only %s levels, "
                    "%s is not a valid level number" % (len(self.names), index)
                )

        index_map = list(
            zip(
                self._internal.index_spark_columns,
                self._internal.index_names,
                self._internal.index_dtypes,
            )
        )
        index_map[i], index_map[j], = index_map[j], index_map[i]
        index_spark_columns, index_names, index_dtypes = zip(*index_map)
        internal = self._internal.copy(
            index_spark_columns=list(index_spark_columns),
            index_names=list(index_names),
            index_dtypes=list(index_dtypes),
            column_labels=[],
            data_spark_columns=[],
            data_dtypes=[],
        )
        return cast(MultiIndex, DataFrame(internal).index)

    @property
    def levshape(self) -> Tuple[int, ...]:
        result = self._internal.spark_frame.agg(
            *(F.countDistinct(c) for c in self._internal.index_spark_columns)
        ).collect()[0]
        return tuple(result)

    @staticmethod
    def _comparator_for_monotonic_increasing(data_type: Any) -> Callable[..., Any]:
        return compare_disallow_null

    def _is_monotonic(self, order: str) -> bool:
        if order == "increasing":
            return self._is_monotonic_increasing().all()
        else:
            return self._is_monotonic_decreasing().all()

    def _is_monotonic_increasing(self) -> Series:
        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-1, -1)

        cond = F.lit(True)
        has_not_null = F.lit(True)
        for scol in self._internal.index_spark_columns[::-1]:
            data_type = self._internal.spark_type_for(scol)
            prev = F.lag(scol, 1).over(window)
            compare = MultiIndex._comparator_for_monotonic_increasing(data_type)
            has_not_null = has_not_null & scol.isNotNull()
            cond = F.when(scol.eqNullSafe(prev), cond).otherwise(
                compare(scol, prev, spark.Column.__gt__)
            )

        cond = has_not_null & (prev.isNull() | cond)

        cond_name = verify_temp_column_name(
            self._internal.spark_frame.select(self._internal.index_spark_columns),
            "__is_monotonic_increasing_cond__",
        )

        sdf = self._internal.spark_frame.select(
            self._internal.index_spark_columns + [cond.alias(cond_name)]
        )

        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in self._internal.index_spark_column_names
            ],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )

        return first_series(DataFrame(internal))

    @staticmethod
    def _comparator_for_monotonic_decreasing(data_type: Any) -> Callable[..., Any]:
        return compare_disallow_null

    def _is_monotonic_decreasing(self) -> Series:
        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-1, -1)

        cond = F.lit(True)
        has_not_null = F.lit(True)
        for scol in self._internal.index_spark_columns[::-1]:
            data_type = self._internal.spark_type_for(scol)
            prev = F.lag(scol, 1).over(window)
            compare = MultiIndex._comparator_for_monotonic_increasing(data_type)
            has_not_null = has_not_null & scol.isNotNull()
            cond = F.when(scol.eqNullSafe(prev), cond).otherwise(
                compare(scol, prev, spark.Column.__lt__)
            )

        cond = has_not_null & (prev.isNull() | cond)

        cond_name = verify_temp_column_name(
            self._internal.spark_frame.select(self._internal.index_spark_columns),
            "__is_monotonic_decreasing_cond__",
        )

        sdf = self._internal.spark_frame.select(
            self._internal.index_spark_columns + [cond.alias(cond_name)]
        )

        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in self._internal.index_spark_column_names
            ],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )

        return first_series(DataFrame(internal))

    def to_frame(self, index: bool = True, name: Optional[Union[str, Sequence[str]]] = None) -> DataFrame:
        if name is None:
            name = [
                name if name is not None else (i,)
                for i, name in enumerate(self._internal.index_names)
            ]
        elif is_list_like(name):
            if len(name) != self._internal.index_level:
                raise ValueError("'name' should have same length as number of levels on index.")
            name = [n if is_name_like_tuple(n) else (n,) for n in name]
        else:
            raise TypeError("'name' must be a list / sequence of column names.")

        return self._to_frame(index=index, names=name)

    def to_pandas(self) -> pd.MultiIndex:
        return super().to_pandas()

    def toPandas(self) -> pd.MultiIndex:
        warnings.warn(
            "MultiIndex.toPandas is deprecated as of MultiIndex.to_pandas. "
            "Please use the API instead.",
            FutureWarning,
        )
        return self.to_pandas()

    def nunique(self, dropna: bool = True) -> None:
        raise NotImplementedError("nunique is not defined for MultiIndex")

    def copy(self, deep: Optional[bool] = None) -> "MultiIndex":
        return super().copy(deep=deep)

    def symmetric_difference(
        self, 
        other: Union["MultiIndex", Sequence[Tuple[Any, ...]]], 
        result_name: Optional[Sequence[str]] = None, 
        sort: Optional[bool] = None
    ) -> "MultiIndex":
        if type(self) != type(other):
            raise NotImplementedError(
                "Doesn't support symmetric_difference between Index & MultiIndex for now"
            )

        sdf_self = self._kdf._internal.spark_frame.select(self._internal.index_spark_columns)
        sdf_other = other._kdf._internal.spark_frame.select(other._internal.index_spark_columns)

        sdf_symdiff = sdf_self.union(sdf_other).subtract(sdf_self.intersect(sdf_other))

        if sort:
            sdf_symdiff = sdf_symdiff.sort(self._internal.index_spark_columns)

        internal = InternalFrame(
            spark_frame=sdf_symdiff,
            index_spark_columns=[
                scol_for(sdf_symdiff, col) for col in self._internal.index_spark_column_names
            ],
            index_names=self._internal.index_names,
        )
        result = cast(MultiIndex, DataFrame(internal).index)

        if result_name:
            result.names = result_name

        return result

    def drop(self, codes: Sequence[Any], level: Optional[Union[int, str]] = None) -> "MultiIndex":
        internal = self._internal.resolved_copy
        sdf = internal.spark_frame
        index_scols = internal.index_spark_columns
        if level is None:
            scol = index_scols[0]
        elif isinstance(level, int):
            scol = index_scols[level]
        else:
            scol = None
            for index_spark_column, index_name in zip(
                internal.index_spark_columns, internal.index_names
            ):
                if not isinstance(level, tuple):
                    level = (level,)
                if level == index_name:
                    if scol is not None:
                        raise ValueError(
                            "The name {} occurs multiple times, use a level number".format(
                                name_like_string(level)
                            )
                        )
                    scol = index_spark_column
            if scol is None:
                raise KeyError("Level {} not found".format(name_like_string(level)))
        sdf = sdf[~scol.isin(codes)]

        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in internal.index_spark_column_names],
            index_names=internal.index_names,
            index_dtypes=internal.index_dtypes,
            column_labels=[],
            data_spark_columns=[],
            data_dtypes=[],
        )
        return cast(MultiIndex, DataFrame(internal).index)

    def value_counts(
        self, 
        normalize: bool = False, 
        sort: bool = True, 
        ascending: bool = False, 
        bins: Optional[int] = None, 
        dropna: bool = True
    ) -> Series:
        if (
            LooseVersion(pyspark.__version__) < LooseVersion("2.4")
            and default_session().conf.get("spark.sql.execution.arrow.enabled") == "true"
            and isinstance(self, MultiIndex)
        ):
            raise RuntimeError(
                "if you're using pyspark < 2.4, set conf "
                "'spark.sql.execution.arrow.enabled' to 'false' "
                "for using this function with MultiIndex"
            )
        return super().value_counts(
            normalize=normalize, sort=sort, ascending=ascending, bins=bins, dropna=dropna
        )

    def argmax(self) -> None:
        raise TypeError("reduction operation 'argmax' not allowed for this dtype")

    def argmin(self) -> None:
        raise TypeError("reduction operation 'argmin' not allowed for this dtype")

    def asof(self, label: Any) -> None:
        raise NotImplementedError(
            "only the default get_loc method is currently supported for MultiIndex"
        )

    @property
    def is_all_dates(self) -> bool:
        return False

    @property
    def inferred_type(self) -> str:
        return "mixed"

    @property
    def asi8(self) -> None:
        return None

    def factorize(
        self, sort: bool = True, na_sentinel: Optional[int