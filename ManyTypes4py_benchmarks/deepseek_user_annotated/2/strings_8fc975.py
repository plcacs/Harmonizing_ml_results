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

"""
String functions on Koalas Series
"""
from typing import Union, TYPE_CHECKING, cast, Optional, List, Dict, Any

import numpy as np
from pandas.core.series import Series as PandasSeries

from pyspark.sql.types import StringType, BinaryType, ArrayType, LongType, MapType
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType

from databricks.koalas.spark import functions as SF

if TYPE_CHECKING:
    import databricks.koalas as ks


class StringMethods(object):
    """String methods for Koalas Series"""

    def __init__(self, series: "ks.Series") -> None:
        if not isinstance(series.spark.data_type, (StringType, BinaryType, ArrayType)):
            raise ValueError("Cannot call StringMethods on type {}".format(series.spark.data_type))
        self._data = series

    def capitalize(self) -> "ks.Series":
        def pandas_capitalize(s: PandasSeries) -> PandasSeries:
            return s.str.capitalize()
        return self._data.koalas.transform_batch(pandas_capitalize)

    def title(self) -> "ks.Series":
        def pandas_title(s: PandasSeries) -> PandasSeries:
            return s.str.title()
        return self._data.koalas.transform_batch(pandas_title)

    def lower(self) -> "ks.Series":
        return self._data.spark.transform(F.lower)

    def upper(self) -> "ks.Series":
        return self._data.spark.transform(F.upper)

    def swapcase(self) -> "ks.Series":
        def pandas_swapcase(s: PandasSeries) -> PandasSeries:
            return s.str.swapcase()
        return self._data.koalas.transform_batch(pandas_swapcase)

    def startswith(self, pattern: str, na: Optional[Any] = None) -> "ks.Series":
        def pandas_startswith(s: PandasSeries) -> PandasSeries:
            return s.str.startswith(pattern, na)
        return self._data.koalas.transform_batch(pandas_startswith)

    def endswith(self, pattern: str, na: Optional[Any] = None) -> "ks.Series":
        def pandas_endswith(s: PandasSeries) -> PandasSeries:
            return s.str.endswith(pattern, na)
        return self._data.koalas.transform_batch(pandas_endswith)

    def strip(self, to_strip: Optional[str] = None) -> "ks.Series":
        def pandas_strip(s: PandasSeries) -> PandasSeries:
            return s.str.strip(to_strip)
        return self._data.koalas.transform_batch(pandas_strip)

    def lstrip(self, to_strip: Optional[str] = None) -> "ks.Series":
        def pandas_lstrip(s: PandasSeries) -> PandasSeries:
            return s.str.lstrip(to_strip)
        return self._data.koalas.transform_batch(pandas_lstrip)

    def rstrip(self, to_strip: Optional[str] = None) -> "ks.Series":
        def pandas_rstrip(s: PandasSeries) -> PandasSeries:
            return s.str.rstrip(to_strip)
        return self._data.koalas.transform_batch(pandas_rstrip)

    def get(self, i: int) -> "ks.Series":
        def pandas_get(s: PandasSeries) -> PandasSeries:
            return s.str.get(i)
        return self._data.koalas.transform_batch(pandas_get)

    def isalnum(self) -> "ks.Series":
        def pandas_isalnum(s: PandasSeries) -> PandasSeries:
            return s.str.isalnum()
        return self._data.koalas.transform_batch(pandas_isalnum)

    def isalpha(self) -> "ks.Series":
        def pandas_isalpha(s: PandasSeries) -> PandasSeries:
            return s.str.isalpha()
        return self._data.koalas.transform_batch(pandas_isalpha)

    def isdigit(self) -> "ks.Series":
        def pandas_isdigit(s: PandasSeries) -> PandasSeries:
            return s.str.isdigit()
        return self._data.koalas.transform_batch(pandas_isdigit)

    def isspace(self) -> "ks.Series":
        def pandas_isspace(s: PandasSeries) -> PandasSeries:
            return s.str.isspace()
        return self._data.koalas.transform_batch(pandas_isspace)

    def islower(self) -> "ks.Series":
        def pandas_islower(s: PandasSeries) -> PandasSeries:
            return s.str.islower()
        return self._data.koalas.transform_batch(pandas_islower)

    def isupper(self) -> "ks.Series":
        def pandas_isupper(s: PandasSeries) -> PandasSeries:
            return s.str.isupper()
        return self._data.koalas.transform_batch(pandas_isupper)

    def istitle(self) -> "ks.Series":
        def pandas_istitle(s: PandasSeries) -> PandasSeries:
            return s.str.istitle()
        return self._data.koalas.transform_batch(pandas_istitle)

    def isnumeric(self) -> "ks.Series":
        def pandas_isnumeric(s: PandasSeries) -> PandasSeries:
            return s.str.isnumeric()
        return self._data.koalas.transform_batch(pandas_isnumeric)

    def isdecimal(self) -> "ks.Series":
        def pandas_isdecimal(s: PandasSeries) -> PandasSeries:
            return s.str.isdecimal()
        return self._data.koalas.transform_batch(pandas_isdecimal)

    def cat(self, others: Optional[Any] = None, sep: Optional[str] = None, 
            na_rep: Optional[str] = None, join: Optional[str] = None) -> "ks.Series":
        raise NotImplementedError()

    def center(self, width: int, fillchar: str = " ") -> "ks.Series":
        def pandas_center(s: PandasSeries) -> PandasSeries:
            return s.str.center(width, fillchar)
        return self._data.koalas.transform_batch(pandas_center)

    def contains(self, pat: str, case: bool = True, flags: int = 0, 
                na: Optional[Any] = None, regex: bool = True) -> "ks.Series":
        def pandas_contains(s: PandasSeries) -> PandasSeries:
            return s.str.contains(pat, case, flags, na, regex)
        return self._data.koalas.transform_batch(pandas_contains)

    def count(self, pat: str, flags: int = 0) -> "ks.Series":
        def pandas_count(s: PandasSeries) -> PandasSeries:
            return s.str.count(pat, flags)
        return self._data.koalas.transform_batch(pandas_count)

    def decode(self, encoding: str, errors: str = "strict") -> "ks.Series":
        raise NotImplementedError()

    def encode(self, encoding: str, errors: str = "strict") -> "ks.Series":
        raise NotImplementedError()

    def extract(self, pat: str, flags: int = 0, expand: bool = True) -> "ks.Series":
        raise NotImplementedError()

    def extractall(self, pat: str, flags: int = 0) -> "ks.Series":
        raise NotImplementedError()

    def find(self, sub: str, start: int = 0, end: Optional[int] = None) -> "ks.Series":
        def pandas_find(s: PandasSeries) -> PandasSeries:
            return s.str.find(sub, start, end)
        return self._data.koalas.transform_batch(pandas_find)

    def findall(self, pat: str, flags: int = 0) -> "ks.Series":
        pudf = pandas_udf(
            lambda s: s.str.findall(pat, flags),
            returnType=ArrayType(StringType(), containsNull=True),
            functionType=PandasUDFType.SCALAR,
        )
        return self._data._with_new_scol(scol=pudf(self._data.spark.column))

    def index(self, sub: str, start: int = 0, end: Optional[int] = None) -> "ks.Series":
        def pandas_index(s: PandasSeries) -> PandasSeries:
            return s.str.index(sub, start, end)
        return self._data.koalas.transform_batch(pandas_index)

    def join(self, sep: str) -> "ks.Series":
        def pandas_join(s: PandasSeries) -> PandasSeries:
            return s.str.join(sep)
        return self._data.koalas.transform_batch(pandas_join)

    def len(self) -> "ks.Series":
        if isinstance(self._data.spark.data_type, (ArrayType, MapType)):
            return self._data.spark.transform(lambda c: F.size(c).cast(LongType()))
        else:
            return self._data.spark.transform(lambda c: F.length(c).cast(LongType()))

    def ljust(self, width: int, fillchar: str = " ") -> "ks.Series":
        def pandas_ljust(s: PandasSeries) -> PandasSeries:
            return s.str.ljust(width, fillchar)
        return self._data.koalas.transform_batch(pandas_ljust)

    def match(self, pat: str, case: bool = True, flags: int = 0, 
             na: Any = np.NaN) -> "ks.Series":
        def pandas_match(s: PandasSeries) -> PandasSeries:
            return s.str.match(pat, case, flags, na)
        return self._data.koalas.transform_batch(pandas_match)

    def normalize(self, form: str) -> "ks.Series":
        def pandas_normalize(s: PandasSeries) -> PandasSeries:
            return s.str.normalize(form)
        return self._data.koalas.transform_batch(pandas_normalize)

    def pad(self, width: int, side: str = "left", fillchar: str = " ") -> "ks.Series":
        def pandas_pad(s: PandasSeries) -> PandasSeries:
            return s.str.pad(width, side, fillchar)
        return self._data.koalas.transform_batch(pandas_pad)

    def partition(self, sep: str = " ", expand: bool = True) -> "ks.Series":
        raise NotImplementedError()

    def repeat(self, repeats: int) -> "ks.Series":
        if not isinstance(repeats, int):
            raise ValueError("repeats expects an int parameter")
        return self._data.spark.transform(lambda c: SF.repeat(col=c, n=repeats))

    def replace(self, pat: str, repl: str, n: int = -1, case: Optional[bool] = None, 
               flags: int = 0, regex: bool = True) -> "ks.Series":
        def pandas_replace(s: PandasSeries) -> PandasSeries:
            return s.str.replace(pat, repl, n=n, case=case, flags=flags, regex=regex)
        return self._data.koalas.transform_batch(pandas_replace)

    def rfind(self, sub: str, start: int = 0, end: Optional[int] = None) -> "ks.Series":
        def pandas_rfind(s: PandasSeries) -> PandasSeries:
            return s.str.rfind(sub, start, end)
        return self._data.koalas.transform_batch(pandas_rfind)

    def rindex(self, sub: str, start: int = 0, end: Optional[int] = None) -> "ks.Series":
        def pandas_rindex(s: PandasSeries) -> PandasSeries:
            return s.str.rindex(sub, start, end)
        return self._data.koalas.transform_batch(pandas_rindex)

    def rjust(self, width: int, fillchar: str = " ") -> "ks.Series":
        def pandas_rjust(s: PandasSeries) -> PandasSeries:
            return s.str.rjust(width, fillchar)
        return self._data.koalas.transform_batch(pandas_rjust)

    def rpartition(self, sep: str = " ", expand: bool = True) -> "ks.Series":
        raise NotImplementedError()

    def slice(self, start: Optional[int] = None, stop: Optional[int] = None, 
             step: Optional[int] = None) -> "ks.Series":
        def pandas_slice(s: PandasSeries) -> PandasSeries:
            return s.str.slice(start, stop, step)
        return self._data.koalas.transform_batch(pandas_slice)

    def slice_replace(self, start: Optional[int] = None, stop: Optional[int] = None, 
                     repl: Optional[str] = None) -> "ks.Series":
        def pandas_slice_replace(s: PandasSeries) -> PandasSeries:
            return s.str.slice_replace(start, stop, repl)
        return self._data.koalas.transform_batch(pandas_slice_replace)

    def split(self, pat: Optional[str] = None, n: int = -1, 
             expand: bool = False) -> Union["ks.Series", "ks.DataFrame"]:
        from databricks.koalas.frame import DataFrame

        if expand and n <= 0:
            raise NotImplementedError("expand=True is currently only supported with n > 0.")

        pudf = pandas_udf(
            lambda s: s.str.split(pat, n),
            returnType=ArrayType(StringType(), containsNull=True),
            functionType=PandasUDFType.SCALAR,
        )
        kser = self._data._with_new_scol(pudf(self._data.spark.column), dtype=self._data.dtype)

        if expand:
            kdf = kser.to_frame()
            scol = kdf._internal.data_spark_columns[0]
            spark_columns = [scol[i].alias(str(i)) for i in range(n + 1)]
            column_labels = [(i,) for i in range(n + 1)]
            internal = kdf._internal.with_new_columns(
                spark_columns,
                column_labels=cast(Optional[List], column_labels),
                data_dtypes=([self._data.dtype] * len(column_labels)),
            )
            return DataFrame(internal)
        else:
            return kser

    def rsplit(self, pat: Optional[str] = None, n: int = -1, 
              expand: bool = False) -> Union["ks.Series", "ks.DataFrame"]:
        from databricks.koalas.frame import DataFrame

        if expand and n <= 0:
            raise NotImplementedError("expand=True is currently only supported with n > 0.")

        pudf = pandas_udf(
            lambda s: s.str.rsplit(pat, n),
            returnType=ArrayType(StringType(), containsNull=True),
            functionType=PandasUDFType.SCALAR,
        )
        kser = self._data._with_new_scol(pudf(self._data.spark.column), dtype=self._data.dtype)

        if expand:
            kdf = kser.to_frame()
            scol = kdf._internal.data_spark_columns[0]
            spark_columns = [scol[i].alias(str(i)) for i in range(n + 1)]
            column_labels = [(i,) for i in range(n + 1)]
            internal = kdf._internal.with_new_columns(
                spark_columns,
                column_labels=cast(Optional[List], column_labels),
                data_dtypes=([self._data.dtype] * len(column_labels)),
            )
            return DataFrame(internal)
        else:
            return kser

    def translate(self, table: Dict[int, Any]) -> "ks.Series":
        def pandas_translate(s: PandasSeries) -> PandasSeries:
            return s.str.translate(table)
        return self._data.koalas.transform_batch(pandas_translate)

    def wrap(self, width: int, **kwargs: Any) -> "ks.Series":
        def pandas_wrap(s: PandasSeries) -> PandasSeries:
            return s.str.wrap(width, **kwargs)
        return self._data.koalas.transform_batch(pandas_wrap)

    def zfill(self, width: int) -> "ks.Series":
        def pandas_zfill(s: PandasSeries) -> PandasSeries:
            return s.str.zfill(width)
        return self._data.koalas.transform_batch(pandas_zfill)

    def get_dummies(self, sep: str = "|") -> None:
        raise NotImplementedError()
