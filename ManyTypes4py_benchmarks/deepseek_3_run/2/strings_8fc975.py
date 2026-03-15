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

        Examples
        --------
        >>> s = ks.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        dtype: object

        >>> s.str.capitalize()
        0                 Lower
        1              Capitals
        2    This is a sentence
        3              Swapcase
        dtype: object
        """

        def pandas_capitalize(s):
            return s.str.capitalize()
        return self._data.koalas.transform_batch(pandas_capitalize)

    def title(self) -> "Series":
        """
        Convert Strings in the series to be titlecase.

        Examples
        --------
        >>> s = ks.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        dtype: object

        >>> s.str.title()
        0                 Lower
        1              Capitals
        2    This Is A Sentence
        3              Swapcase
        dtype: object
        """

        def pandas_title(s):
            return s.str.title()
        return self._data.koalas.transform_batch(pandas_title)

    def lower(self) -> "Series":
        """
        Convert strings in the Series/Index to all lowercase.

        Examples
        --------
        >>> s = ks.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        dtype: object

        >>> s.str.lower()
        0                 lower
        1              capitals
        2    this is a sentence
        3              swapcase
        dtype: object
        """
        return self._data.spark.transform(F.lower)

    def upper(self) -> "Series":
        """
        Convert strings in the Series/Index to all uppercase.

        Examples
        --------
        >>> s = ks.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        dtype: object

        >>> s.str.upper()
        0                 LOWER
        1              CAPITALS
        2    THIS IS A SENTENCE
        3              SWAPCASE
        dtype: object
        """
        return self._data.spark.transform(F.upper)

    def swapcase(self) -> "Series":
        """
        Convert strings in the Series/Index to be swapcased.

        Examples
        --------
        >>> s = ks.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        dtype: object

        >>> s.str.swapcase()
        0                 LOWER
        1              capitals
        2    THIS IS A SENTENCE
        3              sWaPcAsE
        dtype: object
        """

        def pandas_swapcase(s):
            return s.str.swapcase()
        return self._data.koalas.transform_batch(pandas_swapcase)

    def startswith(self, pattern: str, na: Optional[Any] = None) -> "Series":
        """
        Test if the start of each string element matches a pattern.

        Equivalent to :func:`str.startswith`.

        Parameters
        ----------
        pattern : str
            Character sequence. Regular expressions are not accepted.
        na : object, default None
            Object shown if element is not a string. NaN converted to None.

        Returns
        -------
        Series of bool or object
            Koalas Series of booleans indicating whether the given pattern
            matches the start of each string element.

        Examples
        --------
        >>> s = ks.Series(['bat', 'Bear', 'cat', np.nan])
        >>> s
        0     bat
        1    Bear
        2     cat
        3    None
        dtype: object

        >>> s.str.startswith('b')
        0     True
        1    False
        2    False
        3     None
        dtype: object

        Specifying na to be False instead of None.

        >>> s.str.startswith('b', na=False)
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """

        def pandas_startswith(s):
            return s.str.startswith(pattern, na)
        return self._data.koalas.transform_batch(pandas_startswith)

    def endswith(self, pattern: str, na: Optional[Any] = None) -> "Series":
        """
        Test if the end of each string element matches a pattern.

        Equivalent to :func:`str.endswith`.

        Parameters
        ----------
        pattern : str
            Character sequence. Regular expressions are not accepted.
        na : object, default None
            Object shown if element is not a string. NaN converted to None.

        Returns
        -------
        Series of bool or object
            Koalas Series of booleans indicating whether the given pattern
            matches the end of each string element.

        Examples
        --------
        >>> s = ks.Series(['bat', 'Bear', 'cat', np.nan])
        >>> s
        0     bat
        1    Bear
        2     cat
        3    None
        dtype: object

        >>> s.str.endswith('t')
        0     True
        1    False
        2     True
        3     None
        dtype: object

        Specifying na to be False instead of None.

        >>> s.str.endswith('t', na=False)
        0     True
        1    False
        2     True
        3    False
        dtype: bool
        """

        def pandas_endswith(s):
            return s.str.endswith(pattern, na)
        return self._data.koalas.transform_batch(pandas_endswith)

    def strip(self, to_strip: Optional[str] = None) -> "Series":
        """
        Remove leading and trailing characters.

        Strip whitespaces (including newlines) or a set of specified
        characters from each string in the Series/Index from left and
        right sides. Equivalent to :func:`str.strip`.

        Parameters
        ----------
        to_strip : str
            Specifying the set of characters to be removed. All combinations
            of this set of characters will be stripped. If None then
            whitespaces are removed.

        Returns
        -------
        Series of objects

        Examples
        --------
        >>> s = ks.Series(['1. Ant.', '2. Bee!\\t', None])
        >>> s
        0      1. Ant.
        1    2. Bee!\\t
        2         None
        dtype: object

        >>> s.str.strip()
        0    1. Ant.
        1    2. Bee!
        2       None
        dtype: object

        >>> s.str.strip('12.')
        0        Ant
        1     Bee!\\t
        2       None
        dtype: object

        >>> s.str.strip('.!\\t')
        0    1. Ant
        1    2. Bee
        2      None
        dtype: object
        """

        def pandas_strip(s):
            return s.str.strip(to_strip)
        return self._data.koalas.transform_batch(pandas_strip)

    def lstrip(self, to_strip: Optional[str] = None) -> "Series":
        """
        Remove leading characters.

        Strip whitespaces (including newlines) or a set of specified
        characters from each string in the Series/Index from left side.
        Equivalent to :func:`str.lstrip`.

        Parameters
        ----------
        to_strip : str
            Specifying the set of characters to be removed. All combinations
            of this set of characters will be stripped. If None then
            whitespaces are removed.

        Returns
        -------
        Series of object

        Examples
        --------
        >>> s = ks.Series(['1. Ant.', '2. Bee!\\t', None])
        >>> s
        0      1. Ant.
        1    2. Bee!\\t
        2         None
        dtype: object

        >>> s.str.lstrip('12.')
        0       Ant.
        1     Bee!\\t
        2       None
        dtype: object
        """

        def pandas_lstrip(s):
            return s.str.lstrip(to_strip)
        return self._data.koalas.transform_batch(pandas_lstrip)

    def rstrip(self, to_strip: Optional[str] = None) -> "Series":
        """
        Remove trailing characters.

        Strip whitespaces (including newlines) or a set of specified
        characters from each string in the Series/Index from right side.
        Equivalent to :func:`str.rstrip`.

        Parameters
        ----------
        to_strip : str
            Specifying the set of characters to be removed. All combinations
            of this set of characters will be stripped. If None then
            whitespaces are removed.

        Returns
        -------
        Series of object

        Examples
        --------
        >>> s = ks.Series(['1. Ant., '2. Bee!\\t', None])
        >>> s
        0      1. Ant.
        1    2. Bee!\\t
        2         None
        dtype: object

        >>> s.str.rstrip('.!\\t')
        0    1. Ant
        1    2. Bee
        2      None
        dtype: object
        """

        def pandas_rstrip(s):
            return s.str.rstrip(to_strip)
        return self._data.koalas.transform_batch(pandas_rstrip)

    def get(self, i: int) -> "Series":
        """
        Extract element from each string or string list/tuple in the Series
        at the specified position.

        Parameters
        ----------
        i : int
            Position of element to extract.

        Returns
        -------
        Series of objects

        Examples
        --------
        >>> s1 = ks.Series(["String", "123"])
        >>> s1
        0    String
        1       123
        dtype: object

        >>> s1.str.get(1)
        0    t
        1    2
        dtype: object

        >>> s1.str.get(-1)
        0    g
        1    3
        dtype: object

        >>> s2 = ks.Series([["a", "b", "c"], ["x", "y"]])
        >>> s2
        0    [a, b, c]
        1       [x, y]
        dtype: object

        >>> s2.str.get(0)
        0    a
        1    x
        dtype: object

        >>> s2.str.get(2)
        0       c
        1    None
        dtype: object
        """

        def pandas_get(s):
            return s.str.get(i)
        return self._data.koalas.transform_batch(pandas_get)

    def isalnum(self) -> "Series":
        """
        Check whether all characters in each string are alphanumeric.

        This is equivalent to running the Python string method
        :func:`str.isalnum` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s1 = ks.Series(['one', 'one1', '1', ''])

        >>> s1.str.isalnum()
        0     True
        1     True
        2     True
        3    False
        dtype: bool

        Note that checks against characters mixed with any additional
        punctuation or whitespace will evaluate to false for an alphanumeric
        check.

        >>> s2 = ks.Series(['A B', '1.5', '3,000'])
        >>> s2.str.isalnum()
        0    False
        1    False
        2    False
        dtype: bool
        """

        def pandas_isalnum(s):
            return s.str.isalnum()
        return self._data.koalas.transform_batch(pandas_isalnum)

    def isalpha(self) -> "Series":
        """
        Check whether all characters in each string are alphabetic.

        This is equivalent to running the Python string method
        :func:`str.isalpha` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s1 = ks.Series(['one', 'one1', '1', ''])

        >>> s1.str.isalpha()
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """

        def pandas_isalpha(s):
            return s.str.isalpha()
        return self._data.koalas.transform_batch(pandas_isalpha)

    def isdigit(self) -> "Series":
        """
        Check whether all characters in each string are digits.

        This is equivalent to running the Python string method
        :func:`str.isdigit` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s = ks.Series(['23', '³', '⅕', ''])

        The s.str.isdecimal method checks for characters used to form numbers
        in base 10.

        >>> s.str.isdecimal()
        0     True
        1    False
        2    False
        3    False
        dtype: bool

        The s.str.isdigit method is the same as s.str.isdecimal but also
        includes special digits, like superscripted and subscripted digits in
        unicode.

        >>> s.str.isdigit()
        0     True
        1     True
        2    False
        3    False
        dtype: bool

        The s.str.isnumeric method is the same as s.str.isdigit but also
        includes other characters that can represent quantities such as unicode
        fractions.

        >>> s.str.isnumeric()
        0     True
        1     True
        2     True
        3    False
        dtype: bool
        """

        def pandas_isdigit(s):
            return s.str.isdigit()
        return self._data.koalas.transform_batch(pandas_isdigit)

    def isspace(self) -> "Series":
        """
        Check whether all characters in each string are whitespaces.

        This is equivalent to running the Python string method
        :func:`str.isspace` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s = ks.Series([' ', '\\t\\r\\n ', ''])
        >>> s.str.isspace()
        0     True
        1     True
        2    False
        dtype: bool
        """

        def pandas_isspace(s):
            return s.str.isspace()
        return self._data.koalas.transform_batch(pandas_isspace)

    def islower(self) -> "Series":
        """
        Check whether all characters in each string are lowercase.

        This is equivalent to running the Python string method
        :func:`str.islower` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s = ks.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])
        >>> s.str.islower()
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """

        def pandas_isspace(s):
            return s.str.islower()
        return self._data.koalas.transform_batch(pandas_isspace)

    def isupper(self) -> "Series":
        """
        Check whether all characters in each string are uppercase.

        This is equivalent to running the Python string method
        :func:`str.isupper` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s = ks.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])
        >>> s.str.isupper()
        0    False
        1    False
        2     True
        3    False
        dtype: bool
        """

        def pandas_isspace(s):
            return s.str.isupper()
        return self._data.koalas.transform_batch(pandas_isspace)

    def istitle(self) -> "Series":
        """
        Check whether all characters in each string are titlecase.

        This is equivalent to running the Python string method
        :func:`str.istitle` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s = ks.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])

        The s.str.istitle method checks for whether all words are in title
        case (whether only the first letter of each word is capitalized).
        Words are assumed to be as any sequence of non-numeric characters
        separated by whitespace characters.

        >>> s.str.istitle()
        0    False
        1     True
        2    False
        3    False
        dtype: bool
        """

        def pandas_istitle(s):
            return s.str.istitle()
