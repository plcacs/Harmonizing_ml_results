from typing import TYPE_CHECKING
import pandas as pd
from pandas.api.types import CategoricalDtype
if TYPE_CHECKING:
    import databricks.koalas as ks

class CategoricalAccessor(object):
    """
    Accessor object for categorical properties of the Series values.

    Examples
    --------
    >>> s = ks.Series(list("abbccc"), dtype="category")
    >>> s  # doctest: +SKIP
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['a', 'b', 'c']

    >>> s.cat.categories
    Index(['a', 'b', 'c'], dtype='object')

    >>> s.cat.codes
    0    0
    1    1
    2    1
    3    2
    4    2
    5    2
    dtype: int8
    """

    def __init__(self, series):
        if not isinstance(series.dtype, CategoricalDtype):
            raise ValueError('Cannot call CategoricalAccessor on type {}'.format(series.dtype))
        self._data = series

    @property
    def categories(self):
        """
        The categories of this categorical.

        Examples
        --------
        >>> s = ks.Series(list("abbccc"), dtype="category")
        >>> s  # doctest: +SKIP
        0    a
        1    b
        2    b
        3    c
        4    c
        5    c
        dtype: category
        Categories (3, object): ['a', 'b', 'c']

        >>> s.cat.categories
        Index(['a', 'b', 'c'], dtype='object')
        """
        return self._data.dtype.categories

    @categories.setter
    def categories(self, categories):
        raise NotImplementedError()

    @property
    def ordered(self):
        """
        Whether the categories have an ordered relationship.

        Examples
        --------
        >>> s = ks.Series(list("abbccc"), dtype="category")
        >>> s  # doctest: +SKIP
        0    a
        1    b
        2    b
        3    c
        4    c
        5    c
        dtype: category
        Categories (3, object): ['a', 'b', 'c']

        >>> s.cat.ordered
        False
        """
        return self._data.dtype.ordered

    @property
    def codes(self):
        """
        Return Series of codes as well as the index.

        Examples
        --------
        >>> s = ks.Series(list("abbccc"), dtype="category")
        >>> s  # doctest: +SKIP
        0    a
        1    b
        2    b
        3    c
        4    c
        5    c
        dtype: category
        Categories (3, object): ['a', 'b', 'c']

        >>> s.cat.codes
        0    0
        1    1
        2    1
        3    2
        4    2
        5    2
        dtype: int8
        """
        return self._data._with_new_scol(self._data.spark.column).rename()

    def add_categories(self, new_categories, inplace=False):
        raise NotImplementedError()

    def as_ordered(self, inplace=False):
        raise NotImplementedError()

    def as_unordered(self, inplace=False):
        raise NotImplementedError()

    def remove_categories(self, removals, inplace=False):
        raise NotImplementedError()

    def remove_unused_categories(self):
        raise NotImplementedError()

    def rename_categories(self, new_categories, inplace=False):
        raise NotImplementedError()

    def reorder_categories(self, new_categories, ordered=None, inplace=False):
        raise NotImplementedError()

    def set_categories(self, new_categories, ordered=None, rename=False, inplace=False):
        raise NotImplementedError()