from typing import TYPE_CHECKING, Optional
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

    def __init__(self, series: pd.Series) -> None:
        if not isinstance(series.dtype, CategoricalDtype):
            raise ValueError('Cannot call CategoricalAccessor on type {}'.format(series.dtype))
        self._data = series

    @property
    def categories(self) -> pd.Index:
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
    def categories(self, categories: pd.Index) -> None:
        raise NotImplementedError()

    @property
    def ordered(self) -> bool:
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
    def codes(self) -> pd.Series:
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

    def add_categories(self, new_categories: pd.Index, inplace: bool = False) -> pd.Series:
        raise NotImplementedError()

    def as_ordered(self, inplace: bool = False) -> pd.Series:
        raise NotImplementedError()

    def as_unordered(self, inplace: bool = False) -> pd.Series:
        raise NotImplementedError()

    def remove_categories(self, removals: pd.Index, inplace: bool = False) -> pd.Series:
        raise NotImplementedError()

    def remove_unused_categories(self) -> pd.Series:
        raise NotImplementedError()

    def rename_categories(self, new_categories: pd.Index, inplace: bool = False) -> pd.Series:
        raise NotImplementedError()

    def reorder_categories(self, new_categories: pd.Index, ordered: Optional[bool] = None, inplace: bool = False) -> pd.Series:
        raise NotImplementedError()

    def set_categories(self, new_categories: pd.Index, ordered: Optional[bool] = None, rename: bool = False, inplace: bool = False) -> pd.Series:
        raise NotImplementedError()
