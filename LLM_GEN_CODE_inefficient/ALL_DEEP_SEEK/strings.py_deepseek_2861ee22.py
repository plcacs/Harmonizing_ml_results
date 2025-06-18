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
from typing import Union, TYPE_CHECKING, cast, Optional, List

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
            raise ValueError("Cannot call StringMethods on type {}".format(series.spark.data_type))
        self._data = series

    # Methods
    def capitalize(self) -> "ks.Series":
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

        def pandas_capitalize(s) -> "ks.Series[str]":
            return s.str.capitalize()

        return self._data.koalas.transform_batch(pandas_capitalize)

    def title(self) -> "ks.Series":
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

        def pandas_title(s) -> "ks.Series[str]":
            return s.str.title()

        return self._data.koalas.transform_batch(pandas_title)

    def lower(self) -> "ks.Series":
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

    def upper(self) -> "ks.Series":
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

    def swapcase(self) -> "ks.Series":
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

        def pandas_swapcase(s) -> "ks.Series[str]":
            return s.str.swapcase()

        return self._data.koalas.transform_batch(pandas_swapcase)

    def startswith(self, pattern: str, na: Optional[object] = None) -> "ks.Series":
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

        def pandas_startswith(s) -> "ks.Series[bool]":
            return s.str.startswith(pattern, na)

        return self._data.koalas.transform_batch(pandas_startswith)

    def endswith(self, pattern: str, na: Optional[object] = None) -> "ks.Series":
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

        def pandas_endswith(s) -> "ks.Series[bool]":
            return s.str.endswith(pattern, na)

        return self._data.koalas.transform_batch(pandas_endswith)

    def strip(self, to_strip: Optional[str] = None) -> "ks.Series":
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

        def pandas_strip(s) -> "ks.Series[str]":
            return s.str.strip(to_strip)

        return self._data.koalas.transform_batch(pandas_strip)

    def lstrip(self, to_strip: Optional[str] = None) -> "ks.Series":
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

        def pandas_lstrip(s) -> "ks.Series[str]":
            return s.str.lstrip(to_strip)

        return self._data.koalas.transform_batch(pandas_lstrip)

    def rstrip(self, to_strip: Optional[str] = None) -> "ks.Series":
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
        >>> s = ks.Series(['1. Ant.', '2. Bee!\\t', None])
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

        def pandas_rstrip(s) -> "ks.Series[str]":
            return s.str.rstrip(to_strip)

        return self._data.koalas.transform_batch(pandas_rstrip)

    def get(self, i: int) -> "ks.Series":
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

        def pandas_get(s) -> "ks.Series[str]":
            return s.str.get(i)

        return self._data.koalas.transform_batch(pandas_get)

    def isalnum(self) -> "ks.Series":
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

        def pandas_isalnum(s) -> "ks.Series[bool]":
            return s.str.isalnum()

        return self._data.koalas.transform_batch(pandas_isalnum)

    def isalpha(self) -> "ks.Series":
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

        def pandas_isalpha(s) -> "ks.Series[bool]":
            return s.str.isalpha()

        return self._data.koalas.transform_batch(pandas_isalpha)

    def isdigit(self) -> "ks.Series":
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

        def pandas_isdigit(s) -> "ks.Series[bool]":
            return s.str.isdigit()

        return self._data.koalas.transform_batch(pandas_isdigit)

    def isspace(self) -> "ks.Series":
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

        def pandas_isspace(s) -> "ks.Series[bool]":
            return s.str.isspace()

        return self._data.koalas.transform_batch(pandas_isspace)

    def islower(self) -> "ks.Series":
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

        def pandas_islower(s) -> "ks.Series[bool]":
            return s.str.islower()

        return self._data.koalas.transform_batch(pandas_islower)

    def isupper(self) -> "ks.Series":
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

        def pandas_isupper(s) -> "ks.Series[bool]":
            return s.str.isupper()

        return self._data.koalas.transform_batch(pandas_isupper)

    def istitle(self) -> "ks.Series":
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

        def pandas_istitle(s) -> "ks.Series[bool]":
            return s.str.istitle()

        return self._data.koalas.transform_batch(pandas_istitle)

    def isnumeric(self) -> "ks.Series":
        """
        Check whether all characters in each string are numeric.

        This is equivalent to running the Python string method
        :func:`str.isnumeric` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s1 = ks.Series(['one', 'one1', '1', ''])
        >>> s1.str.isnumeric()
        0    False
        1    False
        2     True
        3    False
        dtype: bool

        >>> s2 = ks.Series(['23', '³', '⅕', ''])

        The s2.str.isdecimal method checks for characters used to form numbers
        in base 10.

        >>> s2.str.isdecimal()
        0     True
        1    False
        2    False
        3    False
        dtype: bool

        The s2.str.isdigit method is the same as s2.str.isdecimal but also
        includes special digits, like superscripted and subscripted digits in
        unicode.

        >>> s2.str.isdigit()
        0     True
        1     True
        2    False
        3    False
        dtype: bool

        The s2.str.isnumeric method is the same as s2.str.isdigit but also
        includes other characters that can represent quantities such as unicode
        fractions.

        >>> s2.str.isnumeric()
        0     True
        1     True
        2     True
        3    False
        dtype: bool
        """

        def pandas_isnumeric(s) -> "ks.Series[bool]":
            return s.str.isnumeric()

        return self._data.koalas.transform_batch(pandas_isnumeric)

    def isdecimal(self) -> "ks.Series":
        """
        Check whether all characters in each string are decimals.

        This is equivalent to running the Python string method
        :func:`str.isdecimal` for each element of the Series/Index.
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

        def pandas_isdecimal(s) -> "ks.Series[bool]":
            return s.str.isdecimal()

        return self._data.koalas.transform_batch(pandas_isdecimal)

    def cat(self, others=None, sep=None, na_rep=None, join=None) -> "ks.Series":
        """
        Not supported.
        """
        raise NotImplementedError()

    def center(self, width: int, fillchar: str = " ") -> "ks.Series":
        """
        Filling left and right side of strings in the Series/Index with an
        additional character. Equivalent to :func:`str.center`.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be
            filled with fillchar.
        fillchar : str
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series of objects

        Examples
        --------
        >>> s = ks.Series(["caribou", "tiger"])
        >>> s
        0    caribou
        1      tiger
        dtype: object

        >>> s.str.center(width=10, fillchar='-')
        0    -caribou--
        1    --tiger---
        dtype: object
        """

        def pandas_center(s) -> "ks.Series[str]":
            return s.str.center(width, fillchar)

        return self._data.koalas.transform_batch(pandas_center)

    def contains(self, pat: str, case: bool = True, flags: int = 0, na=None, regex: bool = True) -> "ks.Series":
        """
        Test if pattern or regex is contained within a string of a Series.

        Return boolean Series based on whether a given pattern or regex is
        contained within a string of a Series.

        Analogous to :func:`match`, but less strict, relying on
        :func:`re.search` instead of :func:`re.match`.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Flags to pass through to the re module, e.g. re.IGNORECASE.
        na : default None
            Fill value for missing values. NaN converted to None.
        regex : bool, default True
            If True, assumes the pat is a regular expression.
            If False, treats the pat as a literal string.


        Returns
        -------
        Series of boolean values or object
            A Series of boolean values indicating whether the given pattern is
            contained within the string of each element of the Series.

        Examples
        --------
        Returning a Series of booleans using only a literal pattern.

        >>> s1 = ks.Series(['Mouse', 'dog', 'house and parrot', '23', np.NaN])
        >>> s1.str.contains('og', regex=False)
        0    False
        1     True
        2    False
        3    False
        4     None
        dtype: object

        Specifying case sensitivity using case.

        >>> s1.str.contains('oG', case=True, regex=True)
        0    False
        1    False
        2    False
        3    False
        4     None
        dtype: object

        Specifying na to be False instead of NaN replaces NaN values with
        False. If Series does not contain NaN values the resultant dtype will
        be bool, otherwise, an object dtype.

        >>> s1.str.contains('og', na=False, regex=True)
        0    False
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        Returning ‘house’ or ‘dog’ when either expression occurs in a string.

        >>> s1.str.contains('house|dog', regex=True)
        0    False
        1     True
        2     True
        3    False
        4     None
        dtype: object

        Ignoring case sensitivity using flags with regex.

        >>> import re
        >>> s1.str.contains('PARROT', flags=re.IGNORECASE, regex=True)
        0    False
        1    False
        2     True
        3    False
        4     None
        dtype: object

        Returning any digit using regular expression.

        >>> s1.str.contains('[0-9]', regex=True)
        0    False
        1    False
        2    False
        3     True
        4     None
        dtype: object

        Ensure pat is a not a literal pattern when regex is set to True.
        Note in the following example one might expect only s2[1] and s2[3]
        to return True. However, ‘.0’ as a regex matches any character followed
        by a 0.

        >>> s2 = ks.Series(['40','40.0','41','41.0','35'])
        >>> s2.str.contains('.0', regex=True)
        0     True
        1     True
        2    False
        3     True
        4    False
        dtype: bool
        """

        def pandas_contains(s) -> "ks.Series[bool]":
            return s.str.contains(pat, case, flags, na, regex)

        return self._data.koalas.transform_batch(pandas_contains)

    def count(self, pat: str, flags: int = 0) -> "ks.Series":
        """
        Count occurrences of pattern in each string of the Series.

        This function is used to count the number of times a particular regex
        pattern is repeated in each of the string elements of the Series.

        Parameters
        ----------
        pat : str
            Valid regular expression.
        flags : int, default 0 (no flags)
            Flags for the re module.

        Returns
        -------
        Series of int
            A Series containing the integer counts of pattern matches.

        Examples
        --------
        >>> s = ks.Series(['A', 'B', 'Aaba', 'Baca', np.NaN, 'CABA', 'cat'])
        >>> s.str.count('a')
        0    0.0
        1    0.0
        2    2.0
        3    2.0
        4    NaN
        5    0.0
        6    1.0
        dtype: float64

        Escape '$' to find the literal dollar sign.

        >>> s = ks.Series(['$', 'B', 'Aab$', '$$ca', 'C$B$', 'cat'])
        >>> s.str.count('\\$')
        0    1
        1    0
        2    1
        3    2
        4    2
        5    0
        dtype: int64
        """

        def pandas_count(s) -> "ks.Series[int]":
            return s.str.count(pat, flags)

        return self._data.koalas.transform_batch(pandas_count)

    def decode(self, encoding: str, errors: str = "strict") -> "ks.Series":
        """
        Not supported.
        """
        raise NotImplementedError()

    def encode(self, encoding: str, errors: str = "strict") -> "ks.Series":
        """
        Not supported.
        """
        raise NotImplementedError()

    def extract(self, pat: str, flags: int = 0, expand: bool = True) -> "ks.Series":
        """
        Not supported.
        """
        raise NotImplementedError()

    def extractall(self, pat: str, flags: int = 0) -> "ks.Series":
        """
        Not supported.
        """
        raise NotImplementedError()

    def find(self, sub: str, start: int = 0, end: Optional[int] = None) -> "ks.Series":
        """
        Return lowest indexes in each strings in the Series where the
        substring is fully contained between [start:end].

        Return -1 on failure. Equivalent to standard :func:`str.find`.

        Parameters
        ----------
        sub : str
            Substring being searched.
        start : int
            Left edge index.
        end : int
            Right edge index.

        Returns
        -------
        Series of int
            Series of lowest matching indexes.

        Examples
        --------
        >>> s = ks.Series(['apple', 'oranges', 'bananas'])

        >>> s.str.find('a')
        0    0
        1    2
        2    1
        dtype: int64

        >>> s.str.find('a', start=2)
        0   -1
        1    2
        2    3
        dtype: int64

        >>> s.str.find('a', end=1)
        0    0
        1   -1
        2   -1
        dtype: int64

        >>> s.str.find('a', start=2, end=2)
        0   -1
        1   -1
        2   -1
        dtype: int64
        """

        def pandas_find(s) -> "ks.Series[int]":
            return s.str.find(sub, start, end)

        return self._data.koalas.transform_batch(pandas_find)

    def findall(self, pat: str, flags: int = 0) -> "ks.Series":
        """
        Find all occurrences of pattern or regular expression in the Series.

        Equivalent to applying :func:`re.findall` to all the elements in
        the Series.

        Parameters
        ----------
        pat : str
            Pattern or regular expression.
        flags : int, default 0 (no flags)
            `re` module flags, e.g. `re.IGNORECASE`.

        Returns
        -------
        Series of object
            All non-overlapping matches of pattern or regular expression in
            each string of this Series.

        Examples
        --------
        >>> s = ks.Series(['Lion', 'Monkey', 'Rabbit'])

        The search for the pattern ‘Monkey’ returns one match:

        >>> s.str.findall('Monkey')
        0          []
        1    [Monkey]
        2          []
        dtype: object

        On the other hand, the search for the pattern ‘MONKEY’ doesn’t return
        any match:

        >>> s.str.findall('MONKEY')
        0    []
        1    []
        2    []
        dtype: object

        Flags can be added to the pattern or regular expression. For instance,
        to find the pattern ‘MONKEY’ ignoring the case:

        >>> import re
        >>> s.str.findall('MONKEY', flags=re.IGNORECASE)
        0          []
        1    [Monkey]
        2          []
        dtype: object

        When the pattern matches more than one string in the Series, all
        matches are returned:

        >>> s.str.findall('on')
        0    [on]
        1    [on]
        2      []
        dtype: object

        Regular expressions are supported too. For instance, the search for all
        the strings ending with the word ‘on’ is shown next:

        >>> s.str.findall('on$')
        0    [on]
        1      []
        2      []
        dtype: object

        If the pattern is found more than once in the same string, then a list
        of multiple strings is returned:

        >>> s.str.findall('b')
        0        []
        1        []
        2    [b, b]
        dtype: object
        """
        # type hint does not support to specify array type yet.
        pudf = pandas_udf(
            lambda s: s.str.findall(pat, flags),
            returnType=ArrayType(StringType(), containsNull=True),
            functionType=PandasUDFType.SCALAR,
        )
        return self._data._with_new_scol(scol=pudf(self._data.spark.column))

    def index(self, sub: str, start: int = 0, end: Optional[int] = None) -> "ks.Series":
        """
        Return lowest indexes in each strings where the substring is fully
        contained between [start:end].

        This is the same as :func:`str.find` except instead of returning -1,
        it raises a ValueError when the substring is not found. Equivalent to
        standard :func:`str.index`.

        Parameters
        ----------
        sub : str
            Substring being searched.
        start : int
            Left edge index.
        end : int
            Right edge index.

        Returns
        -------
        Series of int
            Series of lowest matching indexes.

        Examples
        --------
        >>> s = ks.Series(['apple', 'oranges', 'bananas'])

        >>> s.str.index('a')
        0    0
        1    2
        2    1
        dtype: int64

        The following expression throws an exception:

        >>> s.str.index('a', start=2) # doctest: +SKIP
        """

        def pandas_index(s) -> "ks.Series[np.int64]":
            return s.str.index(sub, start, end)

        return self._data.koalas.transform_batch(pandas_index)

    def join(self, sep: str) -> "ks.Series":
        """
        Join lists contained as elements in the Series with passed delimiter.

        If the elements of a Series are lists themselves, join the content of
        these lists using the delimiter passed to the function. This function
        is an equivalent to calling :func:`str.join` on the lists.

        Parameters
        ----------
