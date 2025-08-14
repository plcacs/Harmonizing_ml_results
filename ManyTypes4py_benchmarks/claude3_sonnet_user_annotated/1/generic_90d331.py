@final
def _where(
    self,
    cond,
    other=lib.no_default,
    *,
    inplace: bool = False,
    axis: Axis | None = None,
    level=None,
) -> Self | None:
    """
    Equivalent to public method `where`, except that `other` is not
    applied as a function even if callable. Used in __setitem__.
    """
    inplace = validate_bool_kwarg(inplace, "inplace")

    if axis is not None:
        axis = self._get_axis_number(axis)

    # align the cond to same shape as myself
    cond = common.apply_if_callable(cond, self)
    if isinstance(cond, NDFrame):
        # CoW: Make sure reference is not kept alive
        if cond.ndim == 1 and self.ndim == 2:
            cond = cond._constructor_expanddim(
                {i: cond for i in range(len(self.columns))},
                copy=False,
            )
            cond.columns = self.columns
        cond = cond.align(self, join="right")[0]
    else:
        if not hasattr(cond, "shape"):
            cond = np.asanyarray(cond)
        if cond.shape != self.shape:
            raise ValueError("Array conditional must be same shape as self")
        cond = self._constructor(cond, **self._construct_axes_dict(), copy=False)

    # make sure we are boolean
    fill_value = bool(inplace)
    cond = cond.fillna(fill_value)
    cond = cond.infer_objects()

    msg = "Boolean array expected for the condition, not {dtype}"

    if not cond.empty:
        if not isinstance(cond, ABCDataFrame):
            # This is a single-dimensional object.
            if not is_bool_dtype(cond):
                raise TypeError(msg.format(dtype=cond.dtype))
        else:
            for _dt in cond.dtypes:
                if not is_bool_dtype(_dt):
                    raise TypeError(msg.format(dtype=_dt))
            if cond._mgr.any_extension_types:
                # GH51574: avoid object ndarray conversion later on
                cond = cond._constructor(
                    cond.to_numpy(dtype=bool, na_value=fill_value),
                    **cond._construct_axes_dict(),
                )
    else:
        # GH#21947 we have an empty DataFrame/Series, could be object-dtype
        cond = cond.astype(bool)

    cond = -cond if inplace else cond
    cond = cond.reindex(self._info_axis, axis=self._info_axis_number)

    # try to align with other
    if isinstance(other, NDFrame):
        # align with me
        if other.ndim <= self.ndim:
            # CoW: Make sure reference is not kept alive
            other = self.align(
                other,
                join="left",
                axis=axis,
                level=level,
                fill_value=None,
            )[1]

            # if we are NOT aligned, raise as we cannot where index
            if axis is None and not other._indexed_same(self):
                raise InvalidIndexError

            if other.ndim < self.ndim:
                # TODO(EA2D): avoid object-dtype cast in EA case GH#38729
                other = other._values
                if axis == 0:
                    other = np.reshape(other, (-1, 1))
                elif axis == 1:
                    other = np.reshape(other, (1, -1))

                other = np.broadcast_to(other, self.shape)

        # slice me out of the other
        else:
            raise NotImplementedError(
                "cannot align with a higher dimensional NDFrame"
            )

    elif not isinstance(other, (MultiIndex, NDFrame)):
        # mainly just catching Index here
        other = extract_array(other, extract_numpy=True)

    if isinstance(other, (np.ndarray, ExtensionArray)):
        if other.shape != self.shape:
            if self.ndim != 1:
                # In the ndim == 1 case we may have
                #  other length 1, which we treat as scalar (GH#2745, GH#4192)
                #  or len(other) == icond.sum(), which we treat like
                #  __setitem__ (GH#3235)
                raise ValueError(
                    "other must be the same shape as self when an ndarray"
                )

        # we are the same shape, so create an actual object for alignment
        else:
            other = self._constructor(
                other, **self._construct_axes_dict(), copy=False
            )

    if axis is None:
        axis = 0

    if self.ndim == getattr(other, "ndim", 0):
        align = True
    else:
        align = self._get_axis_number(axis) == 1

    if inplace:
        # we may have different type blocks come out of putmask, so
        # reconstruct the block manager

        new_data = self._mgr.putmask(mask=cond, new=other, align=align)
        result = self._constructor_from_mgr(new_data, axes=new_data.axes)
        return self._update_inplace(result)

    else:
        new_data = self._mgr.where(
            other=other,
            cond=cond,
            align=align,
        )
        result = self._constructor_from_mgr(new_data, axes=new_data.axes)
        return result.__finalize__(self)
