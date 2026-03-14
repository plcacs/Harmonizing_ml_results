"""
String functions on Koalas Series
"""
from typing import Union, TYPE_CHECKING, cast, Optional, List, NoReturn, Mapping, Pattern, Callable, Any
import numpy as np
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
            raise ValueError('Cannot call StringMethods on type {}'.format(series.spark.data_type))
        self._data: "ks.Series" = series

    def capitalize(self) -> "ks.Series":
        """
        Convert Strings in the series to be capitalized.
        """

        def pandas_capitalize(s):
            return s.str.capitalize()
        return self._data.koalas.transform_batch(pandas_capitalize)

    def title(self) -> "ks.Series":
        """
        Convert Strings in the series to be titlecase.
        """

        def pandas_title(s):
            return s.str.title()
        return self._data.koalas.transform_batch(pandas_title)

    def lower(self) -> "ks.Series":
        """
        Convert strings in the Series/Index to all lowercase.
        """
        return self._data.spark.transform(F.lower)

    def upper(self) -> "ks.Series":
        """
        Convert strings in the Series/Index to all uppercase.
        """
        return self._data.spark.transform(F.upper)

    def swapcase(self) -> "ks.Series":
        """
        Convert strings in the Series/Index to be swapcased.
        """

        def pandas_swapcase(s):
            return s.str.swapcase()
        return self._data.koalas.transform_batch(pandas_swapcase)

    def startswith(self, pattern: str, na: Optional[object] = None) -> "ks.Series":
        """
        Test if the start of each string element matches a pattern.
        """

        def pandas_startswith(s):
            return s.str.startswith(pattern, na)
        return self._data.koalas.transform_batch(pandas_startswith)

    def endswith(self, pattern: str, na: Optional[object] = None) -> "ks.Series":
        """
        Test if the end of each string element matches a pattern.
        """

        def pandas_endswith(s):
            return s.str.endswith(pattern, na)
        return self._data.koalas.transform_batch(pandas_endswith)

    def strip(self, to_strip: Optional[str] = None) -> "ks.Series":
        """
        Remove leading and trailing characters.
        """

        def pandas_strip(s):
            return s.str.strip(to_strip)
        return self._data.koalas.transform_batch(pandas_strip)

    def lstrip(self, to_strip: Optional[str] = None) -> "ks.Series":
        """
        Remove leading characters.
        """

        def pandas_lstrip(s):
            return s.str.lstrip(to_strip)
        return self._data.koalas.transform_batch(pandas_lstrip)

    def rstrip(self, to_strip: Optional[str] = None) -> "ks.Series":
        """
        Remove trailing characters.
        """

        def pandas_rstrip(s):
            return s.str.rstrip(to_strip)
        return self._data.koalas.transform_batch(pandas_rstrip)

    def get(self, i: int) -> "ks.Series":
        """
        Extract element from each string or string list/tuple in the Series
        at the specified position.
        """

        def pandas_get(s):
            return s.str.get(i)
        return self._data.koalas.transform_batch(pandas_get)

    def isalnum(self) -> "ks.Series":
        """
        Check whether all characters in each string are alphanumeric.
        """

        def pandas_isalnum(s):
            return s.str.isalnum()
        return self._data.koalas.transform_batch(pandas_isalnum)

    def isalpha(self) -> "ks.Series":
        """
        Check whether all characters in each string are alphabetic.
        """

        def pandas_isalpha(s):
            return s.str.isalpha()
        return self._data.koalas.transform_batch(pandas_isalpha)

    def isdigit(self) -> "ks.Series":
        """
        Check whether all characters in each string are digits.
        """

        def pandas_isdigit(s):
            return s.str.isdigit()
        return self._data.koalas.transform_batch(pandas_isdigit)

    def isspace(self) -> "ks.Series":
        """
        Check whether all characters in each string are whitespaces.
        """

        def pandas_isspace(s):
            return s.str.isspace()
        return self._data.koalas.transform_batch(pandas_isspace)

    def islower(self) -> "ks.Series":
        """
        Check whether all characters in each string are lowercase.
        """

        def pandas_isspace(s):
            return s.str.islower()
        return self._data.koalas.transform_batch(pandas_isspace)

    def isupper(self) -> "ks.Series":
        """
        Check whether all characters in each string are uppercase.
        """

        def pandas_isspace(s):
            return s.str.isupper()
        return self._data.koalas.transform_batch(pandas_isspace)

    def istitle(self) -> "ks.Series":
        """
        Check whether all characters in each string are titlecase.
        """

        def pandas_istitle(s):
            return s.str.istitle()
        return self._data.koalas.transform_batch(pandas_istitle)

    def isnumeric(self) -> "ks.Series":
        """
        Check whether all characters in each string are numeric.
        """

        def pandas_isnumeric(s):
            return s.str.isnumeric()
        return self._data.koalas.transform_batch(pandas_isnumeric)

    def isdecimal(self) -> "ks.Series":
        """
        Check whether all characters in each string are decimals.
        """

        def pandas_isdecimal(s):
            return s.str.isdecimal()
        return self._data.koalas.transform_batch(pandas_isdecimal)

    def cat(self, others: Optional[Union["ks.Series", List["ks.Series"]]] = None, sep: Optional[str] = None, na_rep: Optional[str] = None, join: Optional[str] = None) -> NoReturn:
        """
        Not supported.
        """
        raise NotImplementedError()

    def center(self, width: int, fillchar: str = ' ') -> "ks.Series":
        """
        Filling left and right side of strings in the Series/Index with an
        additional character. Equivalent to :func:`str.center`.
        """

        def pandas_center(s):
            return s.str.center(width, fillchar)
        return self._data.koalas.transform_batch(pandas_center)

    def contains(self, pat: str, case: bool = True, flags: int = 0, na: Optional[object] = None, regex: bool = True) -> "ks.Series":
        """
        Test if pattern or regex is contained within a string of a Series.
        """

        def pandas_contains(s):
            return s.str.contains(pat, case, flags, na, regex)
        return self._data.koalas.transform_batch(pandas_contains)

    def count(self, pat: str, flags: int = 0) -> "ks.Series":
        """
        Count occurrences of pattern in each string of the Series.
        """

        def pandas_count(s):
            return s.str.count(pat, flags)
        return self._data.koalas.transform_batch(pandas_count)

    def decode(self, encoding: str, errors: str = 'strict') -> NoReturn:
        """
        Not supported.
        """
        raise NotImplementedError()

    def encode(self, encoding: str, errors: str = 'strict') -> NoReturn:
        """
        Not supported.
        """
        raise NotImplementedError()

    def extract(self, pat: str, flags: int = 0, expand: bool = True) -> NoReturn:
        """
        Not supported.
        """
        raise NotImplementedError()

    def extractall(self, pat: str, flags: int = 0) -> NoReturn:
        """
        Not supported.
        """
        raise NotImplementedError()

    def find(self, sub: str, start: int = 0, end: Optional[int] = None) -> "ks.Series":
        """
        Return lowest indexes in each strings in the Series where the
        substring is fully contained between [start:end].
        """

        def pandas_find(s):
            return s.str.find(sub, start, end)
        return self._data.koalas.transform_batch(pandas_find)

    def findall(self, pat: str, flags: int = 0) -> "ks.Series":
        """
        Find all occurrences of pattern or regular expression in the Series.
        """
        pudf = pandas_udf(lambda s: s.str.findall(pat, flags), returnType=ArrayType(StringType(), containsNull=True), functionType=PandasUDFType.SCALAR)
        return self._data._with_new_scol(scol=pudf(self._data.spark.column))

    def index(self, sub: str, start: int = 0, end: Optional[int] = None) -> "ks.Series":
        """
        Return lowest indexes in each strings where the substring is fully
        contained between [start:end].
        """

        def pandas_index(s):
            return s.str.index(sub, start, end)
        return self._data.koalas.transform_batch(pandas_index)

    def join(self, sep: str) -> "ks.Series":
        """
        Join lists contained as elements in the Series with passed delimiter.
        """

        def pandas_join(s):
            return s.str.join(sep)
        return self._data.koalas.transform_batch(pandas_join)

    def len(self) -> "ks.Series":
        """
        Computes the length of each element in the Series.
        """
        if isinstance(self._data.spark.data_type, (ArrayType, MapType)):
            return self._data.spark.transform(lambda c: F.size(c).cast(LongType()))
        else:
            return self._data.spark.transform(lambda c: F.length(c).cast(LongType()))

    def ljust(self, width: int, fillchar: str = ' ') -> "ks.Series":
        """
        Filling right side of strings in the Series with an additional
        character. Equivalent to :func:`str.ljust`.
        """

        def pandas_ljust(s):
            return s.str.ljust(width, fillchar)
        return self._data.koalas.transform_batch(pandas_ljust)

    def match(self, pat: str, case: bool = True, flags: int = 0, na: object = np.NaN) -> "ks.Series":
        """
        Determine if each string matches a regular expression.
        """

        def pandas_match(s):
            return s.str.match(pat, case, flags, na)
        return self._data.koalas.transform_batch(pandas_match)

    def normalize(self, form: str) -> "ks.Series":
        """
        Return the Unicode normal form for the strings in the Series.
        """

        def pandas_normalize(s):
            return s.str.normalize(form)
        return self._data.koalas.transform_batch(pandas_normalize)

    def pad(self, width: int, side: str = 'left', fillchar: str = ' ') -> "ks.Series":
        """
        Pad strings in the Series up to width.
        """

        def pandas_pad(s):
            return s.str.pad(width, side, fillchar)
        return self._data.koalas.transform_batch(pandas_pad)

    def partition(self, sep: str = ' ', expand: bool = True) -> NoReturn:
        """
        Not supported.
        """
        raise NotImplementedError()

    def repeat(self, repeats: int) -> "ks.Series":
        """
        Duplicate each string in the Series.
        """
        if not isinstance(repeats, int):
            raise ValueError('repeats expects an int parameter')
        return self._data.spark.transform(lambda c: SF.repeat(col=c, n=repeats))

    def replace(
        self,
        pat: Union[str, Pattern[str]],
        repl: Union[str, Callable[[Any], str]],
        n: int = -1,
        case: Optional[bool] = None,
        flags: int = 0,
        regex: bool = True
    ) -> "ks.Series":
        """
        Replace occurrences of pattern/regex in the Series with some other
        string. Equivalent to :func:`str.replace` or :func:`re.sub`.
        """

        def pandas_replace(s):
            return s.str.replace(pat, repl, n=n, case=case, flags=flags, regex=regex)
        return self._data.koalas.transform_batch(pandas_replace)

    def rfind(self, sub: str, start: int = 0, end: Optional[int] = None) -> "ks.Series":
        """
        Return highest indexes in each strings in the Series where the
        substring is fully contained between [start:end].
        """

        def pandas_rfind(s):
            return s.str.rfind(sub, start, end)
        return self._data.koalas.transform_batch(pandas_rfind)

    def rindex(self, sub: str, start: int = 0, end: Optional[int] = None) -> "ks.Series":
        """
        Return highest indexes in each strings where the substring is fully
        contained between [start:end].
        """

        def pandas_rindex(s):
            return s.str.rindex(sub, start, end)
        return self._data.koalas.transform_batch(pandas_rindex)

    def rjust(self, width: int, fillchar: str = ' ') -> "ks.Series":
        """
        Filling left side of strings in the Series with an additional
        character. Equivalent to :func:`str.rjust`.
        """

        def pandas_rjust(s):
            return s.str.rjust(width, fillchar)
        return self._data.koalas.transform_batch(pandas_rjust)

    def rpartition(self, sep: str = ' ', expand: bool = True) -> NoReturn:
        """
        Not supported.
        """
        raise NotImplementedError()

    def slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> "ks.Series":
        """
        Slice substrings from each element in the Series.
        """

        def pandas_slice(s):
            return s.str.slice(start, stop, step)
        return self._data.koalas.transform_batch(pandas_slice)

    def slice_replace(self, start: Optional[int] = None, stop: Optional[int] = None, repl: Optional[str] = None) -> "ks.Series":
        """
        Slice substrings from each element in the Series.
        """

        def pandas_slice_replace(s):
            return s.str.slice_replace(start, stop, repl)
        return self._data.koalas.transform_batch(pandas_slice_replace)

    def split(self, pat: Optional[str] = None, n: int = -1, expand: bool = False) -> Union["ks.Series", "ks.DataFrame"]:
        """
        Split strings around given separator/delimiter.
        """
        from databricks.koalas.frame import DataFrame
        if expand and n <= 0:
            raise NotImplementedError('expand=True is currently only supported with n > 0.')
        pudf = pandas_udf(lambda s: s.str.split(pat, n), returnType=ArrayType(StringType(), containsNull=True), functionType=PandasUDFType.SCALAR)
        kser = self._data._with_new_scol(pudf(self._data.spark.column), dtype=self._data.dtype)
        if expand:
            kdf = kser.to_frame()
            scol = kdf._internal.data_spark_columns[0]
            spark_columns = [scol[i].alias(str(i)) for i in range(n + 1)]
            column_labels = [(i,) for i in range(n + 1)]
            internal = kdf._internal.with_new_columns(spark_columns, column_labels=cast(Optional[List], column_labels), data_dtypes=[self._data.dtype] * len(column_labels))
            return DataFrame(internal)
        else:
            return kser

    def rsplit(self, pat: Optional[str] = None, n: int = -1, expand: bool = False) -> Union["ks.Series", "ks.DataFrame"]:
        """
        Split strings around given separator/delimiter.
        """
        from databricks.koalas.frame import DataFrame
        if expand and n <= 0:
            raise NotImplementedError('expand=True is currently only supported with n > 0.')
        pudf = pandas_udf(lambda s: s.str.rsplit(pat, n), returnType=ArrayType(StringType(), containsNull=True), functionType=PandasUDFType.SCALAR)
        kser = self._data._with_new_scol(pudf(self._data.spark.column), dtype=self._data.dtype)
        if expand:
            kdf = kser.to_frame()
            scol = kdf._internal.data_spark_columns[0]
            spark_columns = [scol[i].alias(str(i)) for i in range(n + 1)]
            column_labels = [(i,) for i in range(n + 1)]
            internal = kdf._internal.with_new_columns(spark_columns, column_labels=cast(Optional[List], column_labels), data_dtypes=[self._data.dtype] * len(column_labels))
            return DataFrame(internal)
        else:
            return kser

    def translate(self, table: Mapping[int, Optional[Union[int, str]]]) -> "ks.Series":
        """
        Map all characters in the string through the given mapping table.
        Equivalent to standard :func:`str.translate`.
        """

        def pandas_translate(s):
            return s.str.translate(table)
        return self._data.koalas.transform_batch(pandas_translate)

    def wrap(self, width: int, **kwargs: Any) -> "ks.Series":
        """
        Wrap long strings in the Series to be formatted in paragraphs with
        length less than a given width.
        """

        def pandas_wrap(s):
            return s.str.wrap(width, **kwargs)
        return self._data.koalas.transform_batch(pandas_wrap)

    def zfill(self, width: int) -> "ks.Series":
        """
        Pad strings in the Series by prepending ‘0’ characters.
        """

        def pandas_zfill(s):
            return s.str.zfill(width)
        return self._data.koalas.transform_batch(pandas_zfill)

    def get_dummies(self, sep: str = '|') -> NoReturn:
        """
        Not supported.
        """
        raise NotImplementedError()