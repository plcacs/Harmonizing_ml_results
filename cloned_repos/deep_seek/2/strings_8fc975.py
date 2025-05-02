"""
String functions on Koalas Series
"""
from typing import Union, TYPE_CHECKING, cast, Optional, List, Dict, Any, Callable
import numpy as np
from pyspark.sql.types import StringType, BinaryType, ArrayType, LongType, MapType
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from databricks.koalas.spark import functions as SF
if TYPE_CHECKING:
    import databricks.koalas as ks
    from databricks.koalas.series import Series

class StringMethods(object):
    """String methods for Koalas Series"""

    def __init__(self, series: "Series") -> None:
        if not isinstance(series.spark.data_type, (StringType, BinaryType, ArrayType)):
            raise ValueError('Cannot call StringMethods on type {}'.format(series.spark.data_type))
        self._data = series

    def capitalize(self) -> "Series":
        """
        Convert Strings in the series to be capitalized.
        """
        def pandas_capitalize(s: "Series") -> "Series":
            return s.str.capitalize()
        return self._data.koalas.transform_batch(pandas_capitalize)

    def title(self) -> "Series":
        """
        Convert Strings in the series to be titlecase.
        """
        def pandas_title(s: "Series") -> "Series":
            return s.str.title()
        return self._data.koalas.transform_batch(pandas_title)

    def lower(self) -> "Series":
        """
        Convert strings in the Series/Index to all lowercase.
        """
        return self._data.spark.transform(F.lower)

    def upper(self) -> "Series":
        """
        Convert strings in the Series/Index to all uppercase.
        """
        return self._data.spark.transform(F.upper)

    def swapcase(self) -> "Series":
        """
        Convert strings in the Series/Index to be swapcased.
        """
        def pandas_swapcase(s: "Series") -> "Series":
            return s.str.swapcase()
        return self._data.koalas.transform_batch(pandas_swapcase)

    def startswith(self, pattern: str, na: Any = None) -> "Series":
        """
        Test if the start of each string element matches a pattern.
        """
        def pandas_startswith(s: "Series") -> "Series":
            return s.str.startswith(pattern, na)
        return self._data.koalas.transform_batch(pandas_startswith)

    def endswith(self, pattern: str, na: Any = None) -> "Series":
        """
        Test if the end of each string element matches a pattern.
        """
        def pandas_endswith(s: "Series") -> "Series":
            return s.str.endswith(pattern, na)
        return self._data.koalas.transform_batch(pandas_endswith)

    def strip(self, to_strip: Optional[str] = None) -> "Series":
        """
        Remove leading and trailing characters.
        """
        def pandas_strip(s: "Series") -> "Series":
            return s.str.strip(to_strip)
        return self._data.koalas.transform_batch(pandas_strip)

    def lstrip(self, to_strip: Optional[str] = None) -> "Series":
        """
        Remove leading characters.
        """
        def pandas_lstrip(s: "Series") -> "Series":
            return s.str.lstrip(to_strip)
        return self._data.koalas.transform_batch(pandas_lstrip)

    def rstrip(self, to_strip: Optional[str] = None) -> "Series":
        """
        Remove trailing characters.
        """
        def pandas_rstrip(s: "Series") -> "Series":
            return s.str.rstrip(to_strip)
        return self._data.koalas.transform_batch(pandas_rstrip)

    def get(self, i: int) -> "Series":
        """
        Extract element from each string or string list/tuple in the Series.
        """
        def pandas_get(s: "Series") -> "Series":
            return s.str.get(i)
        return self._data.koalas.transform_batch(pandas_get)

    def isalnum(self) -> "Series":
        """
        Check whether all characters in each string are alphanumeric.
        """
        def pandas_isalnum(s: "Series") -> "Series":
            return s.str.isalnum()
        return self._data.koalas.transform_batch(pandas_isalnum)

    def isalpha(self) -> "Series":
        """
        Check whether all characters in each string are alphabetic.
        """
        def pandas_isalpha(s: "Series") -> "Series":
            return s.str.isalpha()
        return self._data.koalas.transform_batch(pandas_isalpha)

    def isdigit(self) -> "Series":
        """
        Check whether all characters in each string are digits.
        """
        def pandas_isdigit(s: "Series") -> "Series":
            return s.str.isdigit()
        return self._data.koalas.transform_batch(pandas_isdigit)

    def isspace(self) -> "Series":
        """
        Check whether all characters in each string are whitespaces.
        """
        def pandas_isspace(s: "Series") -> "Series":
            return s.str.isspace()
        return self._data.koalas.transform_batch(pandas_isspace)

    def islower(self) -> "Series":
        """
        Check whether all characters in each string are lowercase.
        """
        def pandas_islower(s: "Series") -> "Series":
            return s.str.islower()
        return self._data.koalas.transform_batch(pandas_islower)

    def isupper(self) -> "Series":
        """
        Check whether all characters in each string are uppercase.
        """
        def pandas_isupper(s: "Series") -> "Series":
            return s.str.isupper()
        return self._data.koalas.transform_batch(pandas_isupper)

    def istitle(self) -> "Series":
        """
        Check whether all characters in each string are titlecase.
        """
        def pandas_istitle(s: "Series") -> "Series":
            return s.str.istitle()
        return self._data.koalas.transform_batch(pandas_istitle)

    def isnumeric(self) -> "Series":
        """
        Check whether all characters in each string are numeric.
        """
        def pandas_isnumeric(s: "Series") -> "Series":
            return s.str.isnumeric()
        return self._data.koalas.transform_batch(pandas_isnumeric)

    def isdecimal(self) -> "Series":
        """
        Check whether all characters in each string are decimals.
        """
        def pandas_isdecimal(s: "Series") -> "Series":
            return s.str.isdecimal()
        return self._data.koalas.transform_batch(pandas_isdecimal)

    def cat(self, others: Optional[Any] = None, sep: Optional[str] = None, 
            na_rep: Optional[str] = None, join: Optional[str] = None) -> None:
        """
        Not supported.
        """
        raise NotImplementedError()

    def center(self, width: int, fillchar: str = ' ') -> "Series":
        """
        Filling left and right side of strings in the Series/Index.
        """
        def pandas_center(s: "Series") -> "Series":
            return s.str.center(width, fillchar)
        return self._data.koalas.transform_batch(pandas_center)

    def contains(self, pat: str, case: bool = True, flags: int = 0, 
                na: Any = None, regex: bool = True) -> "Series":
        """
        Test if pattern or regex is contained within a string of a Series.
        """
        def pandas_contains(s: "Series") -> "Series":
            return s.str.contains(pat, case, flags, na, regex)
        return self._data.koalas.transform_batch(pandas_contains)

    def count(self, pat: str, flags: int = 0) -> "Series":
        """
        Count occurrences of pattern in each string of the Series.
        """
        def pandas_count(s: "Series") -> "Series":
            return s.str.count(pat, flags)
        return self._data.koalas.transform_batch(pandas_count)

    def decode(self, encoding: str, errors: str = 'strict') -> None:
        """
        Not supported.
        """
        raise NotImplementedError()

    def encode(self, encoding: str, errors: str = 'strict') -> None:
        """
        Not supported.
        """
        raise NotImplementedError()

    def extract(self, pat: str, flags: int = 0, expand: bool = True) -> None:
        """
        Not supported.
        """
        raise NotImplementedError()

    def extractall(self, pat: str, flags: int = 0) -> None:
        """
        Not supported.
        """
        raise NotImplementedError()

    def find(self, sub: str, start: int = 0, end: Optional[int] = None) -> "Series":
        """
        Return lowest indexes in each strings in the Series.
        """
        def pandas_find(s: "Series") -> "Series":
            return s.str.find(sub, start, end)
        return self._data.koalas.transform_batch(pandas_find)

    def findall(self, pat: str, flags: int = 0) -> "Series":
        """
        Find all occurrences of pattern or regular expression in the Series.
        """
        pudf = pandas_udf(lambda s: s.str.findall(pat, flags), 
                         returnType=ArrayType(StringType(), containsNull=True), 
                         functionType=PandasUDFType.SCALAR)
        return self._data._with_new_scol(scol=pudf(self._data.spark.column))

    def index(self, sub: str, start: int = 0, end: Optional[int] = None) -> "Series":
        """
        Return lowest indexes in each strings where the substring is fully contained.
        """
        def pandas_index(s: "Series") -> "Series":
            return s.str.index(sub, start, end)
        return self._data.koalas.transform_batch(pandas_index)

    def join(self, sep: str) -> "Series":
        """
        Join lists contained as elements in the Series with passed delimiter.
        """
        def pandas_join(s: "Series") -> "Series":
            return s.str.join(sep)
        return self._data.koalas.transform_batch(pandas_join)

    def len(self) -> "Series":
        """
        Computes the length of each element in the Series.
        """
        if isinstance(self._data.spark.data_type, (ArrayType, MapType)):
            return self._data.spark.transform(lambda c: F.size(c).cast(LongType()))
        else:
            return self._data.spark.transform(lambda c: F.length(c).cast(LongType()))

    def ljust(self, width: int, fillchar: str = ' ') -> "Series":
        """
        Filling right side of strings in the Series with an additional character.
        """
        def pandas_ljust(s: "Series") -> "Series":
            return s.str.ljust(width, fillchar)
        return self._data.koalas.transform_batch(pandas_ljust)

    def match(self, pat: str, case: bool = True, flags: int = 0, 
              na: Any = np.NaN) -> "Series":
        """
        Determine if each string matches a regular expression.
        """
        def pandas_match(s: "Series") -> "Series":
            return s.str.match(pat, case, flags, na)
        return self._data.koalas.transform_batch(pandas_match)

    def normalize(self, form: str) -> "Series":
        """
        Return the Unicode normal form for the strings in the Series.
        """
        def pandas_normalize(s: "Series") -> "Series":
            return s.str.normalize(form)
        return self._data.koalas.transform_batch(pandas_normalize)

    def pad(self, width: int, side: str = 'left', fillchar: str = ' ') -> "Series":
        """
        Pad strings in the Series up to width.
        """
        def pandas_pad(s: "Series") -> "Series":
            return s.str.pad(width, side, fillchar)
        return self._data.koalas.transform_batch(pandas_pad)

    def partition(self, sep: str = ' ', expand: bool = True) -> None:
        """
        Not supported.
        """
        raise NotImplementedError()

    def repeat(self, repeats: int) -> "Series":
        """
        Duplicate each string in the Series.
        """
        if not isinstance(repeats, int):
            raise ValueError('repeats expects an int parameter')
        return self._data.spark.transform(lambda c: SF.repeat(col=c, n=repeats))

    def replace(self, pat: str, repl: Union[str, Callable], n: int = -1, 
                case: Optional[bool] = None, flags: int = 0, regex: bool = True) -> "Series":
        """
        Replace occurrences of pattern/regex in the Series with some other string.
        """
        def pandas_replace(s: "Series") -> "Series":
            return s.str.replace(pat, repl, n=n, case=case, flags=flags, regex=regex)
        return self._data.koalas.transform_batch(pandas_replace)

    def rfind(self, sub: str, start: int = 0, end: Optional[int] = None) -> "Series":
        """
        Return highest indexes in each strings in the Series.
        """
        def pandas_rfind(s: "Series") -> "Series":
            return s.str.rfind(sub, start, end)
        return self._data.koalas.transform_batch(pandas_rfind)

    def rindex(self, sub: str, start: int = 0, end: Optional[int] = None) -> "Series":
        """
        Return highest indexes in each strings where the substring is fully contained.
        """
        def pandas_rindex(s: "Series") -> "Series":
            return s.str.rindex(sub, start, end)
        return self._data.koalas.transform_batch(pandas_rindex)

    def rjust(self, width: int, fillchar: str = ' ') -> "Series":
        """
        Filling left side of strings in the Series with an additional character.
        """
        def pandas_rjust(s: "Series") -> "Series":
            return s.str.rjust(width, fillchar)
        return self._data.koalas.transform_batch(pandas_rjust)

    def rpartition(self, sep: str = ' ', expand: bool = True) -> None:
        """
        Not supported.
        """
        raise NotImplementedError()

    def slice(self, start: Optional[int] = None, stop: Optional[int] = None, 
              step: Optional[int] = None) -> "Series":
        """
        Slice substrings from each element in the Series.
        """
        def pandas_slice(s: "Series") -> "Series":
            return s.str.slice(start, stop, step)
        return self._data.koalas.transform_batch(pandas_slice)

    def slice_replace(self, start: Optional[int] = None, stop: Optional[int] = None, 
                     repl: Optional[str] = None) -> "Series":
        """
        Slice substrings from each element in the Series.
        """
        def pandas_slice_replace(s: "Series") -> "Series":
            return s.str.slice_replace(start, stop, repl)
        return self._data.koalas.transform_batch(pandas_slice_replace)

    def split(self, pat: Optional[str] = None, n: int = -1, expand: bool = False) -> "Series":
        """
        Split strings around given separator/delimiter.
        """
        from databricks.koalas.frame import DataFrame
        if expand and n <= 0:
            raise NotImplementedError('expand=True is currently only supported with n > 0.')
        pudf = pandas_udf(lambda s: s.str.split(pat, n), 
                         returnType=ArrayType(StringType(), containsNull=True), 
                         functionType=PandasUDFType.SCALAR)
        kser = self._data._with_new_scol(pudf(self._data.spark.column), dtype=self._data.dtype)
        if expand:
            kdf = kser.to_frame()
            scol = kdf._internal.data_spark_columns[0]
            spark_columns = [scol[i].alias(str(i)) for i in range(n + 1)]
            column_labels = [(i,) for i in range(n + 1)]
            internal = kdf._internal.with_new_columns(spark_columns, 
                                                     column_labels=cast(Optional[List], column_labels), 
                                                     data_dtypes=[self._data.dtype] * len(column_labels))
            return DataFrame(internal)
        else:
            return kser

    def rsplit(self, pat: Optional[str] = None, n: int = -1, expand: bool = False) -> "Series":
        """
        Split strings around given separator/delimiter.
        """
        from databricks.koalas.frame import DataFrame
        if expand and n <= 0:
            raise NotImplementedError('expand=True is currently only supported with n > 0.')
        pudf = pandas_udf(lambda s: s.str.rsplit(pat, n), 
                         returnType=ArrayType(StringType(), containsNull=True), 
                         functionType=PandasUDFType.SCALAR)
        kser = self._data._with_new_scol(pudf(self._data.spark.column), dtype=self._data.dtype)
        if expand:
            kdf = kser.to_frame()
            scol = kdf._internal.data_spark_columns[0]
            spark_columns = [scol[i].alias(str(i)) for i in range(n + 1)]
            column_labels = [(i,) for i in range(n + 1)]
            internal = kdf._internal.with_new_columns(spark_columns, 
                                                     column_labels=cast(Optional[List], column_labels), 
                                                     data_dtypes=[self._data.dtype] * len(column_labels))
            return DataFrame(internal)
        else:
            return kser

    def translate(self, table: Dict[int, Union[int, str, None]]) -> "Series":
        """
        Map all characters in the string through the given mapping table.
        """
        def pandas_translate(s: "Series") -> "Series":
            return s.str.translate(table)
        return self._data.koalas.transform_batch(pandas_translate)

    def wrap(self, width: int, **kwargs: Any) -> "Series":
        """
        Wrap long strings in the Series to be formatted in paragraphs.
        """
        def pandas_wrap(s: "Series") -> "Series":
            return s.str.w