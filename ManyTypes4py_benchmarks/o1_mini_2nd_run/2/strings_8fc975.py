"""
String functions on Koalas Series
"""
from typing import Any, Callable, Dict, List, Optional, Union, cast
import numpy as np
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    LongType,
    MapType,
    StringType,
)
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType, Column
from databricks.koalas.spark import functions as SF
from databricks.koalas.frame import DataFrame
from databricks.koalas.series import Series
import databricks.koalas as ks

if TYPE_CHECKING:
    import databricks.koalas as ks


class StringMethods:
    """String methods for Koalas Series"""

    def __init__(self, series: ks.Series[Any]) -> None:
        if not isinstance(series.spark.data_type, (StringType, BinaryType, ArrayType)):
            raise ValueError("Cannot call StringMethods on type {}".format(series.spark.data_type))
        self._data: ks.Series[Any] = series

    def capitalize(self) -> ks.Series[Any]:
        """
        Convert Strings in the series to be capitalized.
        """
        def pandas_capitalize(s: Any) -> Any:
            return s.str.capitalize()

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_capitalize))

    def title(self) -> ks.Series[Any]:
        """
        Convert Strings in the series to be titlecase.
        """
        def pandas_title(s: Any) -> Any:
            return s.str.title()

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_title))

    def lower(self) -> ks.Series[Any]:
        """
        Convert strings in the Series/Index to all lowercase.
        """
        return cast(ks.Series[Any], self._data.spark.transform(F.lower)

        )

    def upper(self) -> ks.Series[Any]:
        """
        Convert strings in the Series/Index to all uppercase.
        """
        return cast(ks.Series[Any], self._data.spark.transform(F.upper))

    def swapcase(self) -> ks.Series[Any]:
        """
        Convert strings in the Series/Index to be swapcased.
        """
        def pandas_swapcase(s: Any) -> Any:
            return s.str.swapcase()

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_swapcase))

    def startswith(self, pattern: str, na: Optional[Any] = None) -> ks.Series[Optional[bool]]:
        """
        Test if the start of each string element matches a pattern.
        """
        def pandas_startswith(s: Any) -> Any:
            return s.str.startswith(pattern, na)

        return cast(ks.Series[Optional[bool]], self._data.koalas.transform_batch(pandas_startswith))

    def endswith(self, pattern: str, na: Optional[Any] = None) -> ks.Series[Optional[bool]]:
        """
        Test if the end of each string element matches a pattern.
        """
        def pandas_endswith(s: Any) -> Any:
            return s.str.endswith(pattern, na)

        return cast(ks.Series[Optional[bool]], self._data.koalas.transform_batch(pandas_endswith))

    def strip(self, to_strip: Optional[str] = None) -> ks.Series[Any]:
        """
        Remove leading and trailing characters.
        """
        def pandas_strip(s: Any) -> Any:
            return s.str.strip(to_strip)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_strip))

    def lstrip(self, to_strip: Optional[str] = None) -> ks.Series[Any]:
        """
        Remove leading characters.
        """
        def pandas_lstrip(s: Any) -> Any:
            return s.str.lstrip(to_strip)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_lstrip))

    def rstrip(self, to_strip: Optional[str] = None) -> ks.Series[Any]:
        """
        Remove trailing characters.
        """
        def pandas_rstrip(s: Any) -> Any:
            return s.str.rstrip(to_strip)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_rstrip))

    def get(self, i: int) -> ks.Series[Any]:
        """
        Extract element from each string or string list/tuple in the Series at the specified position.
        """
        def pandas_get(s: Any) -> Any:
            return s.str.get(i)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_get))

    def isalnum(self) -> ks.Series[Any]:
        """
        Check whether all characters in each string are alphanumeric.
        """
        def pandas_isalnum(s: Any) -> Any:
            return s.str.isalnum()

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_isalnum))

    def isalpha(self) -> ks.Series[Any]:
        """
        Check whether all characters in each string are alphabetic.
        """
        def pandas_isalpha(s: Any) -> Any:
            return s.str.isalpha()

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_isalpha))

    def isdigit(self) -> ks.Series[Any]:
        """
        Check whether all characters in each string are digits.
        """
        def pandas_isdigit(s: Any) -> Any:
            return s.str.isdigit()

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_isdigit))

    def isspace(self) -> ks.Series[Any]:
        """
        Check whether all characters in each string are whitespaces.
        """
        def pandas_isspace(s: Any) -> Any:
            return s.str.isspace()

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_isspace))

    def islower(self) -> ks.Series[Any]:
        """
        Check whether all characters in each string are lowercase.
        """
        def pandas_islower(s: Any) -> Any:
            return s.str.islower()

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_islower))

    def isupper(self) -> ks.Series[Any]:
        """
        Check whether all characters in each string are uppercase.
        """
        def pandas_isupper(s: Any) -> Any:
            return s.str.isupper()

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_isupper))

    def istitle(self) -> ks.Series[Any]:
        """
        Check whether all characters in each string are titlecase.
        """
        def pandas_istitle(s: Any) -> Any:
            return s.str.istitle()

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_istitle))

    def isnumeric(self) -> ks.Series[Any]:
        """
        Check whether all characters in each string are numeric.
        """
        def pandas_isnumeric(s: Any) -> Any:
            return s.str.isnumeric()

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_isnumeric))

    def isdecimal(self) -> ks.Series[Any]:
        """
        Check whether all characters in each string are decimals.
        """
        def pandas_isdecimal(s: Any) -> Any:
            return s.str.isdecimal()

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_isdecimal))

    def cat(
        self,
        others: Optional[Union[str, ks.Series[Any], List[Any]]] = None,
        sep: Optional[str] = None,
        na_rep: Optional[str] = None,
        join: Optional[str] = None,
    ) -> ks.Series[Any]:
        """
        Not supported.
        """
        raise NotImplementedError()

    def center(self, width: int, fillchar: str = " ") -> ks.Series[Any]:
        """
        Filling left and right side of strings in the Series/Index with an additional character.
        """
        def pandas_center(s: Any) -> Any:
            return s.str.center(width, fillchar)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_center))

    def contains(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Optional[Any] = None,
        regex: bool = True,
    ) -> ks.Series[Optional[bool]]:
        """
        Test if pattern or regex is contained within a string of a Series.
        """
        def pandas_contains(s: Any) -> Any:
            return s.str.contains(pat, case, flags, na, regex)

        return cast(ks.Series[Optional[bool]], self._data.koalas.transform_batch(pandas_contains))

    def count(self, pat: str, flags: int = 0) -> ks.Series[float]:
        """
        Count occurrences of pattern in each string of the Series.
        """
        def pandas_count(s: Any) -> Any:
            return s.str.count(pat, flags)

        return cast(ks.Series[float], self._data.koalas.transform_batch(pandas_count))

    def decode(self, encoding: str, errors: str = "strict") -> ks.Series[Any]:
        """
        Not supported.
        """
        raise NotImplementedError()

    def encode(self, encoding: str, errors: str = "strict") -> ks.Series[Any]:
        """
        Not supported.
        """
        raise NotImplementedError()

    def extract(
        self, pat: str, flags: int = 0, expand: bool = True
    ) -> Union[ks.Series[Any], DataFrame]:
        """
        Not supported.
        """
        raise NotImplementedError()

    def extractall(self, pat: str, flags: int = 0) -> ks.Series[Any]:
        """
        Not supported.
        """
        raise NotImplementedError()

    def find(
        self, sub: str, start: int = 0, end: Optional[int] = None
    ) -> ks.Series[int]:
        """
        Return lowest indexes in each strings in the Series where the substring is fully contained between [start:end].
        """
        def pandas_find(s: Any) -> Any:
            return s.str.find(sub, start, end)

        return cast(ks.Series[int], self._data.koalas.transform_batch(pandas_find))

    def findall(self, pat: str, flags: int = 0) -> ks.Series[List[str]]:
        """
        Find all occurrences of pattern or regular expression in the Series.
        """
        pudf: Callable[[Any], Any] = pandas_udf(
            lambda s: s.str.findall(pat, flags),
            returnType=ArrayType(StringType(), containsNull=True),
            functionType=PandasUDFType.SCALAR,
        )
        return cast(ks.Series[List[str]], self._data._with_new_scol(pudf(self._data.spark.column)))

    def index(
        self, sub: str, start: int = 0, end: Optional[int] = None
    ) -> ks.Series[int]:
        """
        Return lowest indexes in each strings where the substring is fully contained between [start:end].
        """
        def pandas_index(s: Any) -> Any:
            return s.str.index(sub, start, end)

        return cast(ks.Series[int], self._data.koalas.transform_batch(pandas_index))

    def join(self, sep: str) -> ks.Series[Any]:
        """
        Join lists contained as elements in the Series with passed delimiter.
        """
        def pandas_join(s: Any) -> Any:
            return s.str.join(sep)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_join))

    def len(self) -> ks.Series[int]:
        """
        Computes the length of each element in the Series.
        """
        if isinstance(self._data.spark.data_type, (ArrayType, MapType)):
            return cast(
                ks.Series[int],
                self._data.spark.transform(lambda c: F.size(c).cast(LongType())),
            )
        else:
            return cast(
                ks.Series[int],
                self._data.spark.transform(lambda c: F.length(c).cast(LongType())),
            )

    def ljust(self, width: int, fillchar: str = " ") -> ks.Series[Any]:
        """
        Filling right side of strings in the Series with an additional character.
        """
        def pandas_ljust(s: Any) -> Any:
            return s.str.ljust(width, fillchar)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_ljust))

    def match(
        self, pat: str, case: bool = True, flags: int = 0, na: Any = np.NaN
    ) -> ks.Series[Optional[bool]]:
        """
        Determine if each string matches a regular expression.
        """
        def pandas_match(s: Any) -> Any:
            return s.str.match(pat, case, flags, na)

        return cast(ks.Series[Optional[bool]], self._data.koalas.transform_batch(pandas_match))

    def normalize(self, form: str) -> ks.Series[Any]:
        """
        Return the Unicode normal form for the strings in the Series.
        """
        def pandas_normalize(s: Any) -> Any:
            return s.str.normalize(form)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_normalize))

    def pad(
        self, width: int, side: str = "left", fillchar: str = " "
    ) -> ks.Series[Any]:
        """
        Pad strings in the Series up to width.
        """
        def pandas_pad(s: Any) -> Any:
            return s.str.pad(width, side, fillchar)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_pad))

    def partition(self, sep: str = " ", expand: bool = True) -> Any:
        """
        Not supported.
        """
        raise NotImplementedError()

    def repeat(self, repeats: int) -> ks.Series[Any]:
        """
        Duplicate each string in the Series.
        """
        if not isinstance(repeats, int):
            raise ValueError("repeats expects an int parameter")
        return cast(
            ks.Series[Any],
            self._data.spark.transform(lambda c: SF.repeat(col=c, n=repeats)),
        )

    def replace(
        self,
        pat: Union[str, re.Pattern],
        repl: Union[str, Callable[[re.Match], str]],
        n: int = -1,
        case: Optional[bool] = None,
        flags: int = 0,
        regex: bool = True,
    ) -> ks.Series[Any]:
        """
        Replace occurrences of pattern/regex in the Series with some other string.
        """
        def pandas_replace(s: Any) -> Any:
            return s.str.replace(pat, repl, n=n, case=case, flags=flags, regex=regex)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_replace))

    def rfind(
        self, sub: str, start: int = 0, end: Optional[int] = None
    ) -> ks.Series[int]:
        """
        Return highest indexes in each strings in the Series where the substring is fully contained between [start:end].
        """
        def pandas_rfind(s: Any) -> Any:
            return s.str.rfind(sub, start, end)

        return cast(ks.Series[int], self._data.koalas.transform_batch(pandas_rfind))

    def rindex(
        self, sub: str, start: int = 0, end: Optional[int] = None
    ) -> ks.Series[int]:
        """
        Return highest indexes in each strings where the substring is fully contained between [start:end].
        """
        def pandas_rindex(s: Any) -> Any:
            return s.str.rindex(sub, start, end)

        return cast(ks.Series[int], self._data.koalas.transform_batch(pandas_rindex))

    def rjust(self, width: int, fillchar: str = " ") -> ks.Series[Any]:
        """
        Filling left side of strings in the Series with an additional character.
        """
        def pandas_rjust(s: Any) -> Any:
            return s.str.rjust(width, fillchar)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_rjust))

    def rpartition(self, sep: str = " ", expand: bool = True) -> Any:
        """
        Not supported.
        """
        raise NotImplementedError()

    def slice(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> ks.Series[Any]:
        """
        Slice substrings from each element in the Series.
        """
        def pandas_slice(s: Any) -> Any:
            return s.str.slice(start, stop, step)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_slice))

    def slice_replace(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        repl: Optional[str] = None,
    ) -> ks.Series[Any]:
        """
        Slice substrings from each element in the Series.
        """
        def pandas_slice_replace(s: Any) -> Any:
            return s.str.slice_replace(start, stop, repl)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_slice_replace))

    def split(
        self, pat: Optional[str] = None, n: int = -1, expand: bool = False
    ) -> Union[ks.Series[List[str]], DataFrame]:
        """
        Split strings around given separator/delimiter.
        """
        from databricks.koalas.frame import DataFrame

        if expand and n <= 0:
            raise NotImplementedError("expand=True is currently only supported with n > 0.")
        pudf: Callable[[Any], Any] = pandas_udf(
            lambda s: s.str.split(pat, n),
            returnType=ArrayType(StringType(), containsNull=True),
            functionType=PandasUDFType.SCALAR,
        )
        kser: ks.Series[List[str]] = self._data._with_new_scol(
            pudf(self._data.spark.column), dtype=self._data.dtype
        )
        if expand:
            kdf: ks.DataFrame = kser.to_frame()
            scol: Column = kdf._internal.data_spark_columns[0]
            spark_columns: List[Column] = [scol[i].alias(str(i)) for i in range(n + 1)]
            column_labels: List[tuple] = [(i,) for i in range(n + 1)]
            internal = kdf._internal.with_new_columns(
                spark_columns,
                column_labels=cast(Optional[List], column_labels),
                data_dtypes=[self._data.dtype] * len(column_labels),
            )
            return DataFrame(internal)
        else:
            return kser

    def rsplit(
        self, pat: Optional[str] = None, n: int = -1, expand: bool = False
    ) -> Union[ks.Series[List[str]], DataFrame]:
        """
        Split strings around given separator/delimiter.
        """
        from databricks.koalas.frame import DataFrame

        if expand and n <= 0:
            raise NotImplementedError("expand=True is currently only supported with n > 0.")
        pudf: Callable[[Any], Any] = pandas_udf(
            lambda s: s.str.rsplit(pat, n),
            returnType=ArrayType(StringType(), containsNull=True),
            functionType=PandasUDFType.SCALAR,
        )
        kser: ks.Series[List[str]] = self._data._with_new_scol(
            pudf(self._data.spark.column), dtype=self._data.dtype
        )
        if expand:
            kdf: ks.DataFrame = kser.to_frame()
            scol: Column = kdf._internal.data_spark_columns[0]
            spark_columns: List[Column] = [scol[i].alias(str(i)) for i in range(n + 1)]
            column_labels: List[tuple] = [(i,) for i in range(n + 1)]
            internal = kdf._internal.with_new_columns(
                spark_columns,
                column_labels=cast(Optional[List], column_labels),
                data_dtypes=[self._data.dtype] * len(column_labels),
            )
            return DataFrame(internal)
        else:
            return kser

    def translate(self, table: Dict[int, Union[int, str, None]]) -> ks.Series[Any]:
        """
        Map all characters in the string through the given mapping table.
        """
        def pandas_translate(s: Any) -> Any:
            return s.str.translate(table)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_translate))

    def wrap(
        self,
        width: int,
        expand_tabs: bool = True,
        replace_whitespace: bool = True,
        drop_whitespace: bool = True,
        break_long_words: bool = True,
        break_on_hyphens: bool = True,
    ) -> ks.Series[Any]:
        """
        Wrap long strings in the Series to be formatted in paragraphs with length less than a given width.
        """
        def pandas_wrap(s: Any) -> Any:
            return s.str.wrap(
                width,
                expand_tabs=expand_tabs,
                replace_whitespace=replace_whitespace,
                drop_whitespace=drop_whitespace,
                break_long_words=break_long_words,
                break_on_hyphens=break_on_hyphens,
            )

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_wrap))

    def zfill(self, width: int) -> ks.Series[Any]:
        """
        Pad strings in the Series by prepending ‘0’ characters.
        """
        def pandas_zfill(s: Any) -> Any:
            return s.str.zfill(width)

        return cast(ks.Series[Any], self._data.koalas.transform_batch(pandas_zfill))

    def get_dummies(self, sep: str = "|") -> Any:
        """
        Not supported.
        """
        raise NotImplementedError()
