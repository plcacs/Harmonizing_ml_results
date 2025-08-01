"""
Methods that can be shared by many array-like classes or subclasses:
    Series
    Index
    ExtensionArray
"""
from __future__ import annotations
import operator
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, cast
import numpy as np
from pandas._libs import lib
from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op
from pandas.core.dtypes.generic import ABCNDFrame
from pandas.core import roperator
from pandas.core.construction import extract_array
from pandas.core.ops.common import unpack_zerodim_and_defer
REDUCTION_ALIASES: Dict[str, str] = {'maximum': 'max', 'minimum': 'min', 'add': 'sum', 'multiply': 'prod'}

T = TypeVar('T')

class OpsMixin:

    def _cmp_method(self, other: Any, op: Callable[[Any, Any], Any]) -> Any:
        return NotImplemented

    @unpack_zerodim_and_defer('__eq__')
    def __eq__(self, other: Any) -> Any:
        return self._cmp_method(other, operator.eq)

    @unpack_zerodim_and_defer('__ne__')
    def __ne__(self, other: Any) -> Any:
        return self._cmp_method(other, operator.ne)

    @unpack_zerodim_and_defer('__lt__')
    def __lt__(self, other: Any) -> Any:
        return self._cmp_method(other, operator.lt)

    @unpack_zerodim_and_defer('__le__')
    def __le__(self, other: Any) -> Any:
        return self._cmp_method(other, operator.le)

    @unpack_zerodim_and_defer('__gt__')
    def __gt__(self, other: Any) -> Any:
        return self._cmp_method(other, operator.gt)

    @unpack_zerodim_and_defer('__ge__')
    def __ge__(self, other: Any) -> Any:
        return self._cmp_method(other, operator.ge)

    def _logical_method(self, other: Any, op: Callable[[Any, Any], Any]) -> Any:
        return NotImplemented

    @unpack_zerodim_and_defer('__and__')
    def __and__(self, other: Any) -> Any:
        return self._logical_method(other, operator.and_)

    @unpack_zerodim_and_defer('__rand__')
    def __rand__(self, other: Any) -> Any:
        return self._logical_method(other, roperator.rand_)

    @unpack_zerodim_and_defer('__or__')
    def __or__(self, other: Any) -> Any:
        return self._logical_method(other, operator.or_)

    @unpack_zerodim_and_defer('__ror__')
    def __ror__(self, other: Any) -> Any:
        return self._logical_method(other, roperator.ror_)

    @unpack_zerodim_and_defer('__xor__')
    def __xor__(self, other: Any) -> Any:
        return self._logical_method(other, operator.xor)

    @unpack_zerodim_and_defer('__rxor__')
    def __rxor__(self, other: Any) -> Any:
        return self._logical_method(other, roperator.rxor)

    def _arith_method(self, other: Any, op: Callable[[Any, Any], Any]) -> Any:
        return NotImplemented

    @unpack_zerodim_and_defer('__add__')
    def __add__(self, other: Any) -> Any:
        """
        Get Addition of DataFrame and other, column-wise.

        Equivalent to ``DataFrame.add(other)``.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Object to be added to the DataFrame.

        Returns
        -------
        DataFrame
            The result of adding ``other`` to DataFrame.

        See Also
        --------
        DataFrame.add : Add a DataFrame and another object, with option for index-
            or column-oriented addition.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"height": [1.5, 2.6], "weight": [500, 800]}, index=["elk", "moose"]
        ... )
        >>> df
               height  weight
        elk       1.5     500
        moose     2.6     800

        Adding a scalar affects all rows and columns.

        >>> df[["height", "weight"]] + 1.5
               height  weight
        elk       3.0   501.5
        moose     4.1   801.5

        Each element of a list is added to a column of the DataFrame, in order.

        >>> df[["height", "weight"]] + [0.5, 1.5]
               height  weight
        elk       2.0   501.5
        moose     3.1   801.5

        Keys of a dictionary are aligned to the DataFrame, based on column names;
        each value in the dictionary is added to the corresponding column.

        >>> df[["height", "weight"]] + {"height": 0.5, "weight": 1.5}
               height  weight
        elk       2.0   501.5
        moose     3.1   801.5

        When `other` is a :class:`Series`, the index of `other` is aligned with the
        columns of the DataFrame.

        >>> s1 = pd.Series([0.5, 1.5], index=["weight", "height"])
        >>> df[["height", "weight"]] + s1
               height  weight
        elk       3.0   500.5
        moose     4.1   800.5

        Even when the index of `other` is the same as the index of the DataFrame,
        the :class:`Series` will not be reoriented. If index-wise alignment is desired,
        :meth:`DataFrame.add` should be used with `axis='index'`.

        >>> s2 = pd.Series([0.5, 1.5], index=["elk", "moose"])
        >>> df[["height", "weight"]] + s2
               elk  height  moose  weight
        elk    NaN     NaN    NaN     NaN
        moose  NaN     NaN    NaN     NaN

        >>> df[["height", "weight"]].add(s2, axis="index")
               height  weight
        elk       2.0   500.5
        moose     4.1   801.5

        When `other` is a :class:`DataFrame`, both columns names and the
        index are aligned.

        >>> other = pd.DataFrame(
        ...     {"height": [0.2, 0.4, 0.6]}, index=["elk", "moose", "deer"]
        ... )
        >>> df[["height", "weight"]] + other
               height  weight
        deer      NaN     NaN
        elk       1.7     NaN
        moose     3.0     NaN
        """
        return self._arith_method(other, operator.add)

    @unpack_zerodim_and_defer('__radd__')
    def __radd__(self, other: Any) -> Any:
        return self._arith_method(other, roperator.radd)

    @unpack_zerodim_and_defer('__sub__')
    def __sub__(self, other: Any) -> Any:
        return self._arith_method(other, operator.sub)

    @unpack_zerodim_and_defer('__rsub__')
    def __rsub__(self, other: Any) -> Any:
        return self._arith_method(other, roperator.rsub)

    @unpack_zerodim_and_defer('__mul__')
    def __mul__(self, other: Any) -> Any:
        return self._arith_method(other, operator.mul)

    @unpack_zerodim_and_defer('__rmul__')
    def __rmul__(self, other: Any) -> Any:
        return self._arith_method(other, roperator.rmul)

    @unpack_zerodim_and_defer('__truediv__')
    def __truediv__(self, other: Any) -> Any:
        return self._arith_method(other, operator.truediv)

    @unpack_zerodim_and_defer('__rtruediv__')
    def __rtruediv__(self, other: Any) -> Any:
        return self._arith_method(other, roperator.rtruediv)

    @unpack_zerodim_and_defer('__floordiv__')
    def __floordiv__(self, other: Any) -> Any:
        return self._arith_method(other, operator.floordiv)

    @unpack_zerodim_and_defer('__rfloordiv')
    def __rfloordiv__(self, other: Any) -> Any:
        return self._arith_method(other, roperator.rfloordiv)

    @unpack_zerodim_and_defer('__mod__')
    def __mod__(self, other: Any) -> Any:
        return self._arith_method(other, operator.mod)

    @unpack_zerodim_and_defer('__rmod__')
    def __rmod__(self, other: Any) -> Any:
        return self._arith_method(other, roperator.rmod)

    @unpack_zerodim_and_defer('__divmod__')
    def __divmod__(self, other: Any) -> Any:
        return self._arith_method(other, divmod)

    @unpack_zerodim_and_defer('__rdivmod__')
    def __rdivmod__(self, other: Any) -> Any:
        return self._arith_method(other, roperator.rdivmod)

    @unpack_zerodim_and_defer('__pow__')
    def __pow__(self, other: Any) -> Any:
        return self._arith_method(other, operator.pow)

    @unpack_zerodim_and_defer('__rpow__')
    def __rpow__(self, other: Any) -> Any:
        return self._arith_method(other, roperator.rpow)

def array_ufunc(self: Any, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
    """
    Compatibility with numpy ufuncs.

    See also
    --------
    numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
    """
    from pandas.core.frame import DataFrame, Series
    from pandas.core.generic import NDFrame
    from pandas.core.internals import BlockManager
    cls = type(self)
    kwargs = _standardize_out_kwarg(**kwargs)
    result = maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
    if result is not NotImplemented:
        return result
    no_defer = (np.ndarray.__array_ufunc__, cls.__array_ufunc__)
    for item in inputs:
        higher_priority = hasattr(item, '__array_priority__') and item.__array_priority__ > self.__array_priority__
        has_array_ufunc = hasattr(item, '__array_ufunc__') and type(item).__array_ufunc__ not in no_defer and (not isinstance(item, self._HANDLED_TYPES))
        if higher_priority or has_array_ufunc:
            return NotImplemented
    types: Tuple[Type, ...] = tuple((type(x) for x in inputs))
    alignable: List[Any] = [x for x, t in zip(inputs, types) if issubclass(t, NDFrame)]
    if len(alignable) > 1:
        set_types: Set[Type] = set(types)
        if len(set_types) > 1 and {DataFrame, Series}.issubset(set_types):
            raise NotImplementedError(f'Cannot apply ufunc {ufunc} to mixed DataFrame and Series inputs.')
        axes = self.axes
        for obj in alignable[1:]:
            for i, (ax1, ax2) in enumerate(zip(axes, obj.axes)):
                axes[i] = ax1.union(ax2)
        reconstruct_axes: Dict[str, Any] = dict(zip(self._AXIS_ORDERS, axes))
        inputs = tuple((x.reindex(**reconstruct_axes) if issubclass(t, NDFrame) else x for x, t in zip(inputs, types)))
    else:
        reconstruct_axes = dict(zip(self._AXIS_ORDERS, self.axes))
    if self.ndim == 1:
        names: Set[Any] = {x.name for x in inputs if hasattr(x, 'name')}
        name: Optional[Any] = names.pop() if len(names) == 1 else None
        reconstruct_kwargs: Dict[str, Any] = {'name': name}
    else:
        reconstruct_kwargs = {}

    def reconstruct(result: Any) -> Any:
        if ufunc.nout > 1:
            return tuple((_reconstruct(x) for x in result))
        return _reconstruct(result)

    def _reconstruct(result: Any) -> Any:
        if lib.is_scalar(result):
            return result
        if result.ndim != self.ndim:
            if method == 'outer':
                raise NotImplementedError
            return result
        if isinstance(result, BlockManager):
            result = self._constructor_from_mgr(result, axes=result.axes)
        else:
            result = self._constructor(result, **reconstruct_axes, **reconstruct_kwargs, copy=False)
        if len(alignable) == 1:
            result = result.__finalize__(self)
        return result
    if 'out' in kwargs:
        result = dispatch_ufunc_with_out(self, ufunc, method, *inputs, **kwargs)
        return reconstruct(result)
    if method == 'reduce':
        result = dispatch_reduction_ufunc(self, ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result
    if self.ndim > 1 and (len(inputs) > 1 or ufunc.nout > 1):
        inputs = tuple((np.asarray(x) for x in inputs))
        result = getattr(ufunc, method)(*inputs, **kwargs)
    elif self.ndim == 1:
        inputs = tuple((extract_array(x, extract_numpy=True) for x in inputs))
        result = getattr(ufunc, method)(*inputs, **kwargs)
    elif method == '__call__' and (not kwargs):
        mgr = inputs[0]._mgr
        result = mgr.apply(getattr(ufunc, method))
    else:
        result = default_array_ufunc(inputs[0], ufunc, method, *inputs, **kwargs)
    result = reconstruct(result)
    return result

def _standardize_out_kwarg(**kwargs: Any) -> Dict[str, Any]:
    """
    If kwargs contain "out1" and "out2", replace that with a tuple "out"

    np.divmod, np.modf, np.frexp can have either `out=(out1, out2)` or
    `out1=out1, out2=out2)`
    """
    if 'out' not in kwargs and 'out1' in kwargs and ('out2' in kwargs):
        out1 = kwargs.pop('out1')
        out2 = kwargs.pop('out2')
        out = (out1, out2)
        kwargs['out'] = out
    return kwargs

def dispatch_ufunc_with_out(self: Any, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
    """
    If we have an `out` keyword, then call the ufunc without `out` and then
    set the result into the given `out`.
    """
    out: Any = kwargs.pop('out')
    where: Optional[np.ndarray] = kwargs.pop('where', None)
    result = getattr(ufunc, method)(*inputs, **kwargs)
    if result is NotImplemented:
        return NotImplemented
    if isinstance(result, tuple):
        if not isinstance(out, tuple) or len(out) != len(result):
            raise NotImplementedError
        for arr, res in zip(out, result):
            _assign_where(arr, res, where)
        return out
    if isinstance(out, tuple):
        if len(out) == 1:
            out = out[0]
        else:
            raise NotImplementedError
    _assign_where(out, result, where)
    return out

def _assign_where(out: Any, result: Any, where: Optional[np.ndarray]) -> None:
    """
    Set a ufunc result into 'out', masking with a 'where' argument if necessary.
    """
    if where is None:
        out[:] = result
    else:
        np.putmask(out, where, result)

def default_array_ufunc(self: Any, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
    """
    Fallback to the behavior we would get if we did not define __array_ufunc__.

    Notes
    -----
    We are assuming that `self` is among `inputs`.
    """
    if not any((x is self for x in inputs)):
        raise NotImplementedError
    new_inputs = [x if x is not self else np.asarray(x) for x in inputs]
    return getattr(ufunc, method)(*new_inputs, **kwargs)

def dispatch_reduction_ufunc(self: Any, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
    """
    Dispatch ufunc reductions to self's reduction methods.
    """
    assert method == 'reduce'
    if len(inputs) != 1 or inputs[0] is not self:
        return NotImplemented
    if ufunc.__name__ not in REDUCTION_ALIASES:
        return NotImplemented
    method_name: str = REDUCTION_ALIASES[ufunc.__name__]
    if not hasattr(self, method_name):
        return NotImplemented
    if self.ndim > 1:
        if isinstance(self, ABCNDFrame):
            kwargs['numeric_only'] = False
        if 'axis' not in kwargs:
            kwargs['axis'] = 0
    return getattr(self, method_name)(skipna=False, **kwargs)
