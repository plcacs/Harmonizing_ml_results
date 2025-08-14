from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable, Dict, List, Mapping as MappingAlias, Optional, Sequence as SequenceAlias, Union, overload
import functools
import operator
import numpy as np

# ... (other necessary imports)

class Series(NDFrame):
    # ...
    def eq(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.eq, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("ne", "series"))
    def ne(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.ne, level=level, fill_value=fill_value, axis=axis
        )

    def le(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.le, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("lt", "series"))
    def lt(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.lt, level=level, fill_value=fill_value, axis=axis
        )

    def ge(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.ge, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("gt", "series"))
    def gt(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.gt, level=level, fill_value=fill_value, axis=axis
        )

    def add(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.add, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("radd", "series"))
    def radd(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, roperator.radd, level=level, fill_value=fill_value, axis=axis
        )

    def sub(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.sub, level=level, fill_value=fill_value, axis=axis
        )

    subtract = sub

    @Appender(ops.make_flex_doc("rsub", "series"))
    def rsub(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, roperator.rsub, level=level, fill_value=fill_value, axis=axis
        )

    def mul(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.mul, level=level, fill_value=fill_value, axis=axis
        )

    multiply = mul

    @Appender(ops.make_flex_doc("rmul", "series"))
    def rmul(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, roperator.rmul, level=level, fill_value=fill_value, axis=axis
        )

    def truediv(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.truediv, level=level, fill_value=fill_value, axis=axis
        )

    div = truediv
    divide = truediv

    @Appender(ops.make_flex_doc("rtruediv", "series"))
    def rtruediv(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, roperator.rtruediv, level=level, fill_value=fill_value, axis=axis
        )

    rdiv = rtruediv

    @Appender(ops.make_flex_doc("floordiv", "series"))
    def floordiv(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.floordiv, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("rfloordiv", "series"))
    def rfloordiv(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, roperator.rfloordiv, level=level, fill_value=fill_value, axis=axis
        )

    def mod(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.mod, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("rmod", "series"))
    def rmod(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, roperator.rmod, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("pow", "series"))
    def pow(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, operator.pow, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("rpow", "series"))
    def rpow(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, roperator.rpow, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("divmod", "series"))
    def divmod(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, divmod, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("rdivmod", "series"))
    def rdivmod(
        self,
        other: Any,
        level: Optional[Level] = None,
        fill_value: Optional[float] = None,
        axis: Axis = 0,
    ) -> Series:
        return self._flex_method(
            other, roperator.rdivmod, level=level, fill_value=fill_value, axis=axis
        )

    def _gotitem(self, key: Any, ndim: int, subset: Optional[Any] = None) -> Self:
        # Sub-classes to define. Return a sliced object.
        return self

    def map(
        self,
        arg: Union[Callable, Mapping[Any, Any], Series],
        na_action: Optional[Literal["ignore"]] = None,
        **kwargs: Any,
    ) -> Series:
        if callable(arg):
            arg = functools.partial(arg, **kwargs)
        new_values = self._map_values(arg, na_action=na_action)
        return self._constructor(new_values, index=self.index, copy=False).__finalize__(
            self, method="map"
        )

    def _reindex_indexer(
        self,
        new_index: Optional[Index],
        indexer: Optional[np.ndarray]
    ) -> Series:
        if indexer is None and (new_index is None or new_index.names == self.index.names):
            return self.copy(deep=False)
        new_values = algorithms.take_nd(self._values, indexer, allow_fill=True, fill_value=None)
        return self._constructor(new_values, index=new_index, copy=False)

    def _needs_reindex_multi(self, axes: Any, method: Any, level: Any) -> bool:
        return False

    @overload
    def rename(
        self,
        index: Union[Renamer, Hashable, None] = ...,
        *,
        axis: Optional[Axis] = ...,
        copy: Union[bool, lib.NoDefault] = ...,
        inplace: Literal[True],
        level: Optional[Level] = ...,
        errors: IgnoreRaise = ...,
    ) -> None:
        ...

    @overload
    def rename(
        self,
        index: Union[Renamer, Hashable, None] = ...,
        *,
        axis: Optional[Axis] = ...,
        copy: Union[bool, lib.NoDefault] = ...,
        inplace: Literal[False],
        level: Optional[Level] = ...,
        errors: IgnoreRaise = ...,
    ) -> Series:
        ...

    @overload
    def rename(
        self,
        index: Union[Renamer, Hashable, None] = ...,
        *,
        axis: Optional[Axis] = ...,
        copy: Union[bool, lib.NoDefault] = ...,
        inplace: bool = ...,
        level: Optional[Level] = ...,
        errors: IgnoreRaise = ...,
    ) -> Union[Series, None]:
        ...

    def rename(
        self,
        index: Union[Renamer, Hashable, None] = None,
        *,
        axis: Optional[Axis] = None,
        copy: Union[bool, lib.NoDefault] = lib.no_default,
        inplace: bool = False,
        level: Optional[Level] = None,
        errors: IgnoreRaise = "ignore",
    ) -> Union[Series, None]:
        if axis is not None:
            axis = self._get_axis_number(axis)
        if callable(index) or is_dict_like(index):
            return super()._rename(index, inplace=inplace, level=level, errors=errors)
        else:
            return self._set_name(index, inplace=inplace)

    @Appender(NDFrame.set_axis.__doc__)
    def set_axis(
        self,
        labels: Any,
        *,
        axis: Axis = 0,
        copy: Union[bool, lib.NoDefault] = lib.no_default,
    ) -> Series:
        return super().set_axis(labels, axis=axis, copy=copy)

    def reindex(
        self,
        index: Optional[Sequence[Any]] = None,
        *,
        axis: Optional[Axis] = None,
        method: Optional[ReindexMethod] = None,
        copy: Union[bool, lib.NoDefault] = lib.no_default,
        level: Optional[Level] = None,
        fill_value: Optional[Scalar] = None,
        limit: Optional[int] = None,
        tolerance: Any = None,
    ) -> Series:
        return super().reindex(
            index=index,
            method=method,
            level=level,
            fill_value=fill_value,
            limit=limit,
            tolerance=tolerance,
            copy=copy,
        )

    @overload
    def rename_axis(
        self,
        mapper: Union[IndexLabel, lib.NoDefault] = ...,
        *,
        index: Any = ...,
        axis: Axis = ...,
        copy: Union[bool, lib.NoDefault] = ...,
        inplace: Literal[True],
    ) -> None:
        ...

    @overload
    def rename_axis(
        self,
        mapper: Union[IndexLabel, lib.NoDefault] = ...,
        *,
        index: Any = ...,
        axis: Axis = ...,
        copy: Union[bool, lib.NoDefault] = ...,
        inplace: Literal[False],
    ) -> Self:
        ...

    @overload
    def rename_axis(
        self,
        mapper: Union[IndexLabel, lib.NoDefault] = ...,
        *,
        index: Any = ...,
        axis: Axis = ...,
        copy: Union[bool, lib.NoDefault] = ...,
        inplace: bool = ...,
    ) -> Union[Self, None]:
        ...

    def rename_axis(
        self,
        mapper: Union[IndexLabel, lib.NoDefault] = lib.no_default,
        *,
        index: Any = lib.no_default,
        axis: Axis = 0,
        copy: Union[bool, lib.NoDefault] = lib.no_default,
        inplace: bool = False,
    ) -> Union[Self, None]:
        return super().rename_axis(
            mapper=mapper,
            index=index,
            axis=axis,
            inplace=inplace,
            copy=copy,
        )

    @overload
    def drop(
        self,
        labels: Union[IndexLabel, Sequence[Any]] = ...,
        *,
        axis: Axis = ...,
        index: Union[IndexLabel, Sequence[Any]] = ...,
        columns: Union[IndexLabel, Sequence[Any]] = ...,
        level: Optional[Level] = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None:
        ...

    @overload
    def drop(
        self,
        labels: Union[IndexLabel, Sequence[Any]] = ...,
        *,
        axis: Axis = ...,
        index: Union[IndexLabel, Sequence[Any]] = ...,
        columns: Union[IndexLabel, Sequence[Any]] = ...,
        level: Optional[Level] = ...,
        inplace: Literal[False],
        errors: IgnoreRaise = ...,
    ) -> Series:
        ...

    @overload
    def drop(
        self,
        labels: Union[IndexLabel, Sequence[Any]] = ...,
        *,
        axis: Axis = ...,
        index: Union[IndexLabel, Sequence[Any]] = ...,
        columns: Union[IndexLabel, Sequence[Any]] = ...,
        level: Optional[Level] = ...,
        inplace: bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Union[Series, None]:
        ...

    def drop(
        self,
        labels: Union[IndexLabel, Sequence[Any]] = None,
        *,
        axis: Axis = 0,
        index: Union[IndexLabel, Sequence[Any]] = None,
        columns: Union[IndexLabel, Sequence[Any]] = None,
        level: Optional[Level] = None,
        inplace: bool = False,
        errors: IgnoreRaise = "raise",
    ) -> Union[Series, None]:
        return super().drop(
            labels=labels,
            axis=axis,
            index=index,
            columns=columns,
            level=level,
            inplace=inplace,
            errors=errors,
        )

    def pop(self, item: Hashable) -> Any:
        return super().pop(item=item)

    def info(
        self,
        verbose: Optional[bool] = None,
        buf: Optional[Any] = None,
        max_cols: Optional[int] = None,
        memory_usage: Union[bool, str, None] = None,
        show_counts: bool = True,
    ) -> None:
        return SeriesInfo(self, memory_usage).render(
            buf=buf, max_cols=max_cols, verbose=verbose, show_counts=show_counts
        )

    def memory_usage(self, index: bool = True, deep: bool = False) -> int:
        v: int = self._memory_usage(deep=deep)
        if index:
            v += self.index.memory_usage(deep=deep)
        return v

    def isin(self, values: Any) -> Series:
        result = algorithms.isin(self._values, values)
        return self._constructor(result, index=self.index, copy=False).__finalize__(
            self, method="isin"
        )

    def between(
        self,
        left: Any,
        right: Any,
        inclusive: Literal["both", "neither", "left", "right"] = "both",
    ) -> Series:
        if inclusive == "both":
            lmask = self >= left
            rmask = self <= right
        elif inclusive == "left":
            lmask = self >= left
            rmask = self < right
        elif inclusive == "right":
            lmask = self > left
            rmask = self <= right
        elif inclusive == "neither":
            lmask = self > left
            rmask = self < right
        else:
            raise ValueError(
                "Inclusive has to be either string of 'both', 'left', 'right', or 'neither'."
            )
        return lmask & rmask

    def case_when(
        self,
        caselist: List[
            Tuple[
                Union[ArrayLike, Callable[[Series], Union[Series, np.ndarray, Sequence[bool]]]],
                Union[ArrayLike, Scalar, Callable[[Series], Union[Series, np.ndarray]]],
            ]
        ],
    ) -> Series:
        if not isinstance(caselist, list):
            raise TypeError(f"The caselist argument should be a list; instead got {type(caselist)}")
        if not caselist:
            raise ValueError(
                "provide at least one boolean condition, with a corresponding replacement."
            )
        caselist = [
            (
                com.apply_if_callable(condition, self),
                com.apply_if_callable(replacement, self),
            )
            for condition, replacement in caselist
        ]
        default = self.copy(deep=False)
        conditions, replacements = zip(*caselist)
        common_dtypes = [infer_dtype_from(arg)[0] for arg in [*replacements, default]]
        if len(set(common_dtypes)) > 1:
            common_dtype = find_common_type(common_dtypes)
            updated_replacements = []
            for condition, replacement in zip(conditions, replacements):
                if is_scalar(replacement):
                    replacement = construct_1d_arraylike_from_scalar(
                        value=replacement, length=len(condition), dtype=common_dtype
                    )
                elif isinstance(replacement, ABCSeries):
                    replacement = replacement.astype(common_dtype)
                else:
                    replacement = pd_array(replacement, dtype=common_dtype)
                updated_replacements.append(replacement)
            replacements = updated_replacements
            default = default.astype(common_dtype)
        counter = range(len(conditions) - 1, -1, -1)
        for position, condition, replacement in zip(
            counter, reversed(conditions), reversed(replacements)
        ):
            try:
                default = default.mask(
                    condition, other=replacement, axis=0, inplace=False, level=None
                )
            except Exception as error:
                raise ValueError(
                    f"Failed to apply condition{position} and replacement{position}."
                ) from error
        return default

    @doc(NDFrame.isna, klass=_shared_doc_kwargs["klass"])
    def isna(self) -> Series:
        return NDFrame.isna(self)

    @doc(NDFrame.isna, klass=_shared_doc_kwargs["klass"])
    def isnull(self) -> Series:
        return super().isnull()

    @doc(NDFrame.notna, klass=_shared_doc_kwargs["klass"])
    def notna(self) -> Series:
        return super().notna()

    @doc(NDFrame.notna, klass=_shared_doc_kwargs["klass"])
    def notnull(self) -> Series:
        return super().notnull()

    @overload
    def dropna(
        self,
        *,
        axis: Axis = ...,
        inplace: Literal[False] = ...,
        how: Optional[AnyAll] = ...,
        ignore_index: bool = ...,
    ) -> Series:
        ...

    @overload
    def dropna(
        self,
        *,
        axis: Axis = ...,
        inplace: Literal[True],
        how: Optional[AnyAll] = ...,
        ignore_index: bool = ...,
    ) -> None:
        ...

    def dropna(
        self,
        *,
        axis: Axis = 0,
        inplace: bool = False,
        how: Optional[AnyAll] = None,
        ignore_index: bool = False,
    ) -> Union[Series, None]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        ignore_index = validate_bool_kwarg(ignore_index, "ignore_index")
        self._get_axis_number(axis or 0)
        if self._can_hold_na:
            result: Series = remove_na_arraylike(self)
        else:
            result = self.copy(deep=False) if not inplace else self
        if ignore_index:
            result.index = default_index(len(result))
        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    def to_timestamp(
        self,
        freq: Optional[Frequency] = None,
        how: Literal["s", "e", "start", "end"] = "start",
        copy: Union[bool, lib.NoDefault] = lib.no_default,
    ) -> Series:
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, PeriodIndex):
            raise TypeError(f"unsupported Type {type(self.index).__name__}")
        new_obj = self.copy(deep=False)
        new_index = self.index.to_timestamp(freq=freq, how=how)
        setattr(new_obj, "index", new_index)
        return new_obj

    def to_period(
        self,
        freq: Optional[str] = None,
        copy: Union[bool, lib.NoDefault] = lib.no_default,
    ) -> Series:
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, DatetimeIndex):
            raise TypeError(f"unsupported Type {type(self.index).__name__}")
        new_obj = self.copy(deep=False)
        new_index = self.index.to_period(freq=freq)
        setattr(new_obj, "index", new_index)
        return new_obj

    # Accessor properties
    index = properties.AxisProperty(
        axis=0,
        doc="""
        The index (axis labels) of the Series.
        ... (docstring continues) ...
        """
    )

    str = Accessor("str", StringMethods)
    dt = Accessor("dt", CombinedDatetimelikeProperties)
    cat = Accessor("cat", CategoricalAccessor)
    plot = Accessor("plot", pandas.plotting.PlotAccessor)
    sparse = Accessor("sparse", SparseAccessor)
    struct = Accessor("struct", StructAccessor)
    list = Accessor("list", ListAccessor)

    hist = pandas.plotting.hist_series

    def _cmp_method(self, other: Any, op: Callable[[Any, Any], Any]) -> Series:
        res_name: Hashable = ops.get_op_result_name(self, other)
        if isinstance(other, Series) and not self._indexed_same(other):
            raise ValueError("Can only compare identically-labeled Series objects")
        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        res_values = ops.comparison_op(lvalues, rvalues, op)
        return self._construct_result(res_values, name=res_name)

    def _logical_method(self, other: Any, op: Callable[[Any, Any], Any]) -> Series:
        res_name: Hashable = ops.get_op_result_name(self, other)
        self, other = self._align_for_op(other, align_asobject=True)
        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        res_values = ops.logical_op(lvalues, rvalues, op)
        return self._construct_result(res_values, name=res_name)

    def _arith_method(self, other: Any, op: Callable[[Any, Any], Any]) -> Series:
        self, other = self._align_for_op(other)
        return base.IndexOpsMixin._arith_method(self, other, op)

    def _align_for_op(self, right: Any, align_asobject: bool = False) -> Tuple[Series, Series]:
        left: Series = self
        if isinstance(right, Series):
            if not left.index.equals(right.index):
                if align_asobject:
                    if left.dtype not in (object, np.bool_) or right.dtype not in (object, np.bool_):
                        pass
                    else:
                        left = left.astype(object)
                        right = right.astype(object)
                left, right = left.align(right)
        return left, right

    def _binop(
        self, other: Series, func: Callable[[Any, Any], Any], level: Optional[Any] = None, fill_value: Optional[Any] = None
    ) -> Series:
        this: Series = self
        if not self.index.equals(other.index):
            this, other = self.align(other, level=level, join="outer")
        this_vals, other_vals = ops.fill_binop(this._values, other._values, fill_value)
        with np.errstate(all="ignore"):
            result = func(this_vals, other_vals)
        name: Hashable = ops.get_op_result_name(self, other)
        out = this._construct_result(result, name)
        return cast(Series, out)

    def _construct_result(
        self, result: Union[ArrayLike, Tuple[ArrayLike, ArrayLike]], name: Hashable
    ) -> Union[Series, Tuple[Series, Series]]:
        if isinstance(result, tuple):
            res1: Series = self._construct_result(result[0], name=name)
            res2: Series = self._construct_result(result[1], name=name)
            return (res1, res2)
        dtype = getattr(result, "dtype", None)
        out = self._constructor(result, index=self.index, dtype=dtype, copy=False)
        out = out.__finalize__(self)
        out.name = name
        return out

    def _flex_method(
        self, other: Any, op: Callable[[Any, Any], Any], *, level: Optional[Any] = None, fill_value: Optional[Any] = None, axis: Axis = 0
    ) -> Series:
        if axis is not None:
            self._get_axis_number(axis)
        res_name: Hashable = ops.get_op_result_name(self, other)
        if isinstance(other, Series):
            return self._binop(other, op, level=level, fill_value=fill_value)
        elif isinstance(other, (np.ndarray, list, tuple)):
            if len(other) != len(self):
                raise ValueError("Lengths must be equal")
            other_series: Series = self._constructor(other, self.index, copy=False)
            result = self._binop(other_series, op, level=level, fill_value=fill_value)
            result._name = res_name
            return result
        else:
            if fill_value is not None:
                if isna(other):
                    return op(self, fill_value)
                self = self.fillna(fill_value)
            return op(self, other)

    def cummin(self, axis: Axis = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Self:
        return NDFrame.cummin(self, axis, skipna, *args, **kwargs)

    def cummax(self, axis: Axis = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Self:
        return NDFrame.cummax(self, axis, skipna, *args, **kwargs)

    def cumsum(self, axis: Axis = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Self:
        return NDFrame.cumsum(self, axis, skipna, *args, **kwargs)

    def cumprod(self, axis: Axis = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Self:
        return NDFrame.cumprod(self, axis, skipna, *args, **kwargs)

    @overload
    def any(self, *, axis: Axis = ..., bool_only: bool = ..., skipna: bool = ..., **kwargs: Any) -> bool:
        ...

    def any(self, *, axis: Axis = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> bool:
        nv.validate_logical_func((), kwargs, fname="any")
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._reduce(nanops.nanany, name="any", axis=axis, numeric_only=bool_only, skipna=skipna, filter_type="bool")

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="all")
    @Appender(make_doc("all", ndim=1))
    def all(self, axis: Axis = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> bool:
        nv.validate_logical_func((), kwargs, fname="all")
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._reduce(nanops.nanall, name="all", axis=axis, numeric_only=bool_only, skipna=skipna, filter_type="bool")

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="min")
    def min(self, axis: Optional[Axis] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any):
        return NDFrame.min(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="max")
    def max(self, axis: Optional[Axis] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any):
        return NDFrame.max(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="sum")
    def sum(
        self,
        axis: Optional[Axis] = None,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ):
        return NDFrame.sum(self, axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="prod")
    @doc(make_doc("prod", ndim=1))
    def prod(self, axis: Optional[Axis] = None, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs: Any):
        return NDFrame.prod(self, axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="mean")
    def mean(self, axis: Optional[Axis] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.mean(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="median")
    def median(self, axis: Optional[Axis] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.median(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="sem")
    @doc(make_doc("sem", ndim=1))
    def sem(self, axis: Optional[Axis] = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any):
        return NDFrame.sem(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="var")
    def var(self, axis: Optional[Axis] = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any):
        return NDFrame.var(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="std")
    @doc(make_doc("std", ndim=1))
    def std(self, axis: Optional[Axis] = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any):
        return NDFrame.std(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="skew")
    @doc(make_doc("skew", ndim=1))
    def skew(self, axis: Optional[Axis] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any):
        return NDFrame.skew(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="kurt")
    def kurt(self, axis: Optional[Axis] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any):
        return NDFrame.kurt(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    kurtosis = kurt
    product = prod

    @doc(make_doc("cummin", ndim=1))
    def cummin(self, axis: Axis = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Self:
        return NDFrame.cummin(self, axis, skipna, *args, **kwargs)

    @doc(make_doc("cummax", ndim=1))
    def cummax(self, axis: Axis = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Self:
        return NDFrame.cummax(self, axis, skipna, *args, **kwargs)

    @doc(make_doc("cumsum", ndim=1))
    def cumsum(self, axis: Axis = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Self:
        return NDFrame.cumsum(self, axis, skipna, *args, **kwargs)

    @doc(make_doc("cumprod", 1))
    def cumprod(self, axis: Axis = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Self:
        return NDFrame.cumprod(self, axis, skipna, *args, **kwargs)
