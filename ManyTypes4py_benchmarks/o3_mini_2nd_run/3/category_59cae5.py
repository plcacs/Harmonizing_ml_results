from functools import partial
from typing import Any, Optional, Type, Union, Callable
import pandas as pd
from pandas.api.types import is_hashable
from databricks import koalas as ks
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeCategoricalIndex
from databricks.koalas.series import Series


class CategoricalIndex(Index):
    """
    Index based on an underlying `Categorical`.

    CategoricalIndex can only take on a limited,
    and usually fixed, number of possible values (`categories`). Also,
    it might have an order, but numerical operations
    (additions, divisions, ...) are not possible.

    Parameters
    ----------
    data : array-like (1-dimensional) or Series or Index, default None
        The values of the categorical. If `categories` are given, values not in
        `categories` will be replaced with NaN.
    categories : index-like, optional
        The categories for the categorical. Items need to be unique.
        If the categories are not given here (and also not in `dtype`), they
        will be inferred from the `data`.
    ordered : bool, optional
        Whether or not this categorical is treated as an ordered
        categorical. If not given here or in `dtype`, the resulting
        categorical will be unordered.
    dtype : CategoricalDtype or "category", optional
        If :class:`CategoricalDtype`, cannot be used together with
        `categories` or `ordered`.
    copy : bool, default False
        Make a copy of input ndarray.
    name : object, optional
        Name to be stored in the index.
    """

    def __new__(
        cls: Type["CategoricalIndex"],
        data: Optional[Any] = None,
        categories: Optional[Any] = None,
        ordered: Optional[bool] = None,
        dtype: Optional[Any] = None,
        copy: bool = False,
        name: Optional[Any] = None,
    ) -> Union["CategoricalIndex", Index]:
        if not is_hashable(name):
            raise TypeError("Index.name must be a hashable type")
        if isinstance(data, (Series, Index)):
            if dtype is None:
                dtype = "category"
            return Index(data, dtype=dtype, copy=copy, name=name)
        return ks.from_pandas(
            pd.CategoricalIndex(
                data=data, categories=categories, ordered=ordered, dtype=dtype, name=name
            )
        )

    @property
    def codes(self) -> Index:
        """
        The category codes of this categorical.

        Codes are an Index of integers which are the positions of the actual
        values in the categories Index.

        There is no setter, use the other categorical methods and the normal item
        setter to change values in the categorical.

        Returns
        -------
        Index
            A non-writable view of the `codes` Index.

        Examples
        --------
        >>> idx = ks.CategoricalIndex(list("abbccc"))
        >>> idx
        CategoricalIndex(['a', 'b', 'b', 'c', 'c', 'c'],
                         categories=['a', 'b', 'c'], ordered=False, dtype='category')
        >>> idx.codes
        Int64Index([0, 1, 1, 2, 2, 2], dtype='int64')
        """
        return self._with_new_scol(self.spark.column).rename(None)

    @property
    def categories(self) -> Index:
        """
        The categories of this categorical.

        Examples
        --------
        >>> idx = ks.CategoricalIndex(list("abbccc"))
        >>> idx
        CategoricalIndex(['a', 'b', 'b', 'c', 'c', 'c'],
                         categories=['a', 'b', 'c'], ordered=False, dtype='category')
        >>> idx.categories
        Index(['a', 'b', 'c'], dtype='object')
        """
        return self.dtype.categories

    @categories.setter
    def categories(self, categories: Any) -> None:
        raise NotImplementedError()

    @property
    def ordered(self) -> bool:
        """
        Whether the categories have an ordered relationship.

        Examples
        --------
        >>> idx = ks.CategoricalIndex(list("abbccc"))
        >>> idx
        CategoricalIndex(['a', 'b', 'b', 'c', 'c', 'c'],
                         categories=['a', 'b', 'c'], ordered=False, dtype='category')
        >>> idx.ordered
        False
        """
        return self.dtype.ordered

    def __getattr__(self, item: str) -> Union[Any, Callable]:
        if hasattr(MissingPandasLikeCategoricalIndex, item):
            property_or_func = getattr(MissingPandasLikeCategoricalIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)
            else:
                return partial(property_or_func, self)
        raise AttributeError("'CategoricalIndex' object has no attribute '{}'".format(item))