#!/usr/bin/env python3
# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import re
from typing import Any, NamedTuple, Optional, Union, Tuple, List, Callable

from hypothesis import assume, strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture.utils import _calc_p_continue
from hypothesis.internal.coverage import check_function
from hypothesis.internal.validation import check_type, check_valid_interval
from hypothesis.strategies._internal.utils import defines_strategy
from hypothesis.utils.conventions import UniqueIdentifier, not_set

__all__ = [
    "NDIM_MAX",
    "BasicIndex",
    "BasicIndexStrategy",
    "BroadcastableShapes",
    "MutuallyBroadcastableShapesStrategy",
    "Shape",
    "array_shapes",
    "broadcastable_shapes",
    "check_argument",
    "check_valid_dims",
    "mutually_broadcastable_shapes",
    "order_check",
    "valid_tuple_axes",
]

Shape = Tuple[int, ...]
# BasicIndex is a tuple of indices, where each index can be an int,
# a slice, None, or Ellipsis.
IndexEntry = Union[int, slice, None, type(Ellipsis)]
BasicIndex = Tuple[IndexEntry, ...]


class BroadcastableShapes(NamedTuple):
    input_shapes: Tuple[Shape, ...]
    result_shape: Shape


@check_function
def check_argument(condition: bool, fail_message: str, *f_args: Any, **f_kwargs: Any) -> None:
    if not condition:
        raise InvalidArgument(fail_message.format(*f_args, **f_kwargs))


@check_function
def order_check(name: str, floor: int, min_: int, max_: int) -> None:
    if floor > min_:
        raise InvalidArgument(f"min_{name} must be at least {floor} but was {min_}")
    if min_ > max_:
        raise InvalidArgument(f"min_{name}={min_} is larger than max_{name}={max_}")


NDIM_MAX: int = 32


@check_function
def check_valid_dims(dims: int, name: str) -> None:
    if dims > NDIM_MAX:
        raise InvalidArgument(
            f"{name}={dims}, but Hypothesis does not support arrays with "
            f"more than {NDIM_MAX} dimensions"
        )


@defines_strategy()
def array_shapes(
    *,
    min_dims: int = 1,
    max_dims: Optional[int] = None,
    min_side: int = 1,
    max_side: Optional[int] = None,
) -> st.SearchStrategy[Shape]:
    """Return a strategy for array shapes (tuples of int >= 1).

    * ``min_dims`` is the smallest length that the generated shape can possess.
    * ``max_dims`` is the largest length that the generated shape can possess,
      defaulting to ``min_dims + 2``.
    * ``min_side`` is the smallest size that a dimension can possess.
    * ``max_side`` is the largest size that a dimension can possess,
      defaulting to ``min_side + 5``.
    """
    check_type(int, min_dims, "min_dims")
    check_type(int, min_side, "min_side")
    check_valid_dims(min_dims, "min_dims")

    if max_dims is None:
        max_dims = min(min_dims + 2, NDIM_MAX)
    check_type(int, max_dims, "max_dims")
    check_valid_dims(max_dims, "max_dims")

    if max_side is None:
        max_side = min_side + 5
    check_type(int, max_side, "max_side")

    order_check("dims", 0, min_dims, max_dims)
    order_check("side", 0, min_side, max_side)

    return st.lists(
        st.integers(min_side, max_side), min_size=min_dims, max_size=max_dims
    ).map(tuple)


@defines_strategy()
def valid_tuple_axes(
    ndim: int,
    *,
    min_size: int = 0,
    max_size: Optional[int] = None,
) -> st.SearchStrategy[Tuple[int, ...]]:
    """Return a strategy for valid tuple axes."""
    check_type(int, ndim, "ndim")
    check_type(int, min_size, "min_size")
    if max_size is None:
        max_size = ndim
    check_type(int, max_size, "max_size")
    order_check("size", 0, min_size, max_size)
    check_valid_interval(max_size, ndim, "max_size", "ndim")

    axes = st.integers(0, max(0, 2 * ndim - 1)).map(
        lambda x: x if x < ndim else x - 2 * ndim
    )

    return st.lists(
        axes, min_size=min_size, max_size=max_size, unique_by=lambda x: x % ndim
    ).map(tuple)


@defines_strategy()
def broadcastable_shapes(
    shape: Shape,
    *,
    min_dims: int = 0,
    max_dims: Optional[int] = None,
    min_side: int = 1,
    max_side: Optional[int] = None,
) -> st.SearchStrategy[Shape]:
    """Return a strategy for shapes that are broadcast-compatible with the
    provided shape.
    """
    check_type(tuple, shape, "shape")
    check_type(int, min_side, "min_side")
    check_type(int, min_dims, "min_dims")
    check_valid_dims(min_dims, "min_dims")

    strict_check: bool = max_side is None or max_dims is None

    if max_dims is None:
        max_dims = min(max(len(shape), min_dims) + 2, NDIM_MAX)
    check_type(int, max_dims, "max_dims")
    check_valid_dims(max_dims, "max_dims")

    if max_side is None:
        max_side = max(shape[-max_dims:] + (min_side,)) + 2
    check_type(int, max_side, "max_side")

    order_check("dims", 0, min_dims, max_dims)
    order_check("side", 0, min_side, max_side)

    if strict_check:
        dims: int = max_dims
        bound_name: str = "max_dims"
    else:
        dims = min_dims
        bound_name = "min_dims"

    if not all(min_side <= s for s in shape[::-1][:dims] if s != 1):
        raise InvalidArgument(
            f"Given shape={shape}, there are no broadcast-compatible "
            f"shapes that satisfy: {bound_name}={dims} and min_side={min_side}"
        )

    if not (
        min_side <= 1 <= max_side or all(s <= max_side for s in shape[::-1][:dims])
    ):
        raise InvalidArgument(
            f"Given base_shape={shape}, there are no broadcast-compatible "
            f"shapes that satisfy all of {bound_name}={dims}, "
            f"min_side={min_side}, and max_side={max_side}"
        )

    if not strict_check:
        for n, s in zip(range(max_dims), shape[::-1]):
            if s < min_side and s != 1:
                max_dims = n
                break
            elif not (min_side <= 1 <= max_side or s <= max_side):
                max_dims = n
                break

    return MutuallyBroadcastableShapesStrategy(
        num_shapes=1,
        base_shape=shape,
        min_dims=min_dims,
        max_dims=max_dims,
        min_side=min_side,
        max_side=max_side,
    ).map(lambda x: x.input_shapes[0])


_DIMENSION: str = r"\w+\??"  # Note that \w permits digits too!
_SHAPE: str = rf"\((?:{_DIMENSION}(?:,{_DIMENSION}){{0,31}})?\)"
_ARGUMENT_LIST: str = f"{_SHAPE}(?:,{_SHAPE})*"
_SIGNATURE: str = rf"^{_ARGUMENT_LIST}->{_SHAPE}$"
_SIGNATURE_MULTIPLE_OUTPUT: str = rf"^{_ARGUMENT_LIST}->{_ARGUMENT_LIST}$"


class _GUfuncSig(NamedTuple):
    input_shapes: Tuple[Shape, ...]
    result_shape: Shape


def _hypothesis_parse_gufunc_signature(signature: str) -> _GUfuncSig:
    if not re.match(_SIGNATURE, signature):
        if re.match(_SIGNATURE_MULTIPLE_OUTPUT, signature):
            raise InvalidArgument(
                "Hypothesis does not yet support generalised ufunc signatures "
                "with multiple output arrays - mostly because we don't know of "
                "anyone who uses them!  Please get in touch with us to fix that."
                f"\n ({signature=})"
            )
        if re.match(
            (
                r"^\((?:\w+(?:,\w+)*)?\)(?:,\((?:\w+(?:,\w+)*)?\))*->"
                r"\((?:\w+(?:,\w+)*)?\)(?:,\((?:\w+(?:,\w+)*)?\))*$"
            ),
            signature,
        ):
            raise InvalidArgument(
                f"{signature=} matches Numpy's regex for gufunc signatures, "
                f"but contains shapes with more than {NDIM_MAX} dimensions and is thus invalid."
            )
        raise InvalidArgument(f"{signature!r} is not a valid gufunc signature")
    input_shapes_raw, output_shapes_raw = (
        tuple(tuple(re.findall(_DIMENSION, a)) for a in re.findall(_SHAPE, arg_list))
        for arg_list in signature.split("->")
    )
    assert len(output_shapes_raw) == 1
    result_shape_raw: Tuple[str, ...] = output_shapes_raw[0]
    for shape in (*input_shapes_raw, result_shape_raw):
        for name in shape:
            try:
                int(name.strip("?"))
                if "?" in name:
                    raise InvalidArgument(
                        f"Got dimension {name!r}, but handling of frozen optional dimensions "
                        "is ambiguous.  If you known how this should work, please "
                        "contact us to get this fixed and documented ({signature=})."
                    )
            except ValueError:
                names_in = {n.strip("?") for shp in input_shapes_raw for n in shp}
                names_out = {n.strip("?") for n in result_shape_raw}
                if name.strip("?") in (names_out - names_in):
                    raise InvalidArgument(
                        "The {name!r} dimension only appears in the output shape, and is "
                        "not frozen, so the size is not determined ({signature=})."
                    ) from None
    return _GUfuncSig(input_shapes=tuple(tuple(int(n) if n.isdigit() else n for n in shp) for shp in input_shapes_raw),
                      result_shape=tuple(int(n) if n.isdigit() else n for n in result_shape_raw))  # type: ignore


@defines_strategy()
def mutually_broadcastable_shapes(
    *,
    num_shapes: Union[UniqueIdentifier, int] = not_set,
    signature: Union[UniqueIdentifier, str] = not_set,
    base_shape: Shape = (),
    min_dims: int = 0,
    max_dims: Optional[int] = None,
    min_side: int = 1,
    max_side: Optional[int] = None,
) -> st.SearchStrategy[BroadcastableShapes]:
    """Return a strategy for a specified number of shapes N that are
    mutually-broadcastable with one another and with the provided base shape.
    """
    arg_msg: str = "Pass either the `num_shapes` or the `signature` argument, but not both."
    if num_shapes is not not_set:
        check_argument(signature is not not_set, arg_msg)
        check_type(int, num_shapes, "num_shapes")
        parsed_signature: Optional[_GUfuncSig] = None
        sig_dims: int = 0
    else:
        check_argument(signature is not not_set, arg_msg)
        if signature is None:
            raise InvalidArgument(
                "Expected a string, but got invalid signature=None.  "
                "(maybe .signature attribute of an element-wise ufunc?)"
            )
        check_type(str, signature, "signature")
        parsed_signature = _hypothesis_parse_gufunc_signature(signature)
        all_shapes: Tuple[Shape, ...] = (*parsed_signature.input_shapes, parsed_signature.result_shape)
        sig_dims = min(len(s) for s in all_shapes)
        num_shapes = len(parsed_signature.input_shapes)

    if num_shapes < 1:
        raise InvalidArgument(f"num_shapes={num_shapes} must be at least 1")

    check_type(tuple, base_shape, "base_shape")
    check_type(int, min_side, "min_side")
    check_type(int, min_dims, "min_dims")
    check_valid_dims(min_dims, "min_dims")

    strict_check: bool = max_dims is not None

    if max_dims is None:
        max_dims = min(max(len(base_shape), min_dims) + 2, NDIM_MAX - sig_dims)
    check_type(int, max_dims, "max_dims")
    check_valid_dims(max_dims, "max_dims")

    if max_side is None:
        max_side = max(base_shape[-max_dims:] + (min_side,)) + 2
    check_type(int, max_side, "max_side")

    order_check("dims", 0, min_dims, max_dims)
    order_check("side", 0, min_side, max_side)

    if signature is not None and max_dims > NDIM_MAX - sig_dims:
        raise InvalidArgument(
            f"max_dims={signature!r} would exceed the {NDIM_MAX}-dimension"
            "limit Hypothesis imposes on array shapes, "
            f"given signature={parsed_signature!r}"
        )

    if strict_check:
        dims: int = max_dims
        bound_name: str = "max_dims"
    else:
        dims = min_dims
        bound_name = "min_dims"

    if not all(min_side <= s for s in base_shape[::-1][:dims] if s != 1):
        raise InvalidArgument(
            f"Given base_shape={base_shape}, there are no broadcast-compatible "
            f"shapes that satisfy: {bound_name}={dims} and min_side={min_side}"
        )

    if not (
        min_side <= 1 <= max_side or all(s <= max_side for s in base_shape[::-1][:dims])
    ):
        raise InvalidArgument(
            f"Given base_shape={base_shape}, there are no broadcast-compatible "
            f"shapes that satisfy all of {bound_name}={dims}, "
            f"min_side={min_side}, and max_side={max_side}"
        )

    if not strict_check:
        for n, s in zip(range(max_dims), base_shape[::-1]):
            if s < min_side and s != 1:
                max_dims = n
                break
            elif not (min_side <= 1 <= max_side or s <= max_side):
                max_dims = n
                break

    return MutuallyBroadcastableShapesStrategy(
        num_shapes=num_shapes,
        signature=parsed_signature,
        base_shape=base_shape,
        min_dims=min_dims,
        max_dims=max_dims,
        min_side=min_side,
        max_side=max_side,
    )


class MutuallyBroadcastableShapesStrategy(st.SearchStrategy[BroadcastableShapes]):
    def __init__(
        self,
        num_shapes: int,
        signature: Optional[_GUfuncSig] = None,
        base_shape: Shape = (),
        min_dims: int = 0,
        max_dims: Optional[int] = None,
        min_side: int = 1,
        max_side: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.base_shape: Shape = base_shape
        self.side_strat = st.integers(min_side, max_side)  # type: st.SearchStrategy[int]
        self.num_shapes: int = num_shapes
        self.signature: Optional[_GUfuncSig] = signature
        self.min_dims: int = min_dims
        self.max_dims: int = max_dims  # type: ignore
        self.min_side: int = min_side
        self.max_side: int = max_side  # type: ignore

        self.size_one_allowed: bool = self.min_side <= 1 <= self.max_side  # type: ignore

    def do_draw(self, data: Any) -> BroadcastableShapes:
        if self.signature is None:
            return self._draw_loop_dimensions(data)
        core_in, core_res = self._draw_core_dimensions(data)
        use: List[bool] = [None not in shp for shp in core_in]
        loop_in, loop_res = self._draw_loop_dimensions(data, use=use)
        def add_shape(loop: Tuple[int, ...], core: Tuple[int, ...]) -> Shape:
            return tuple(x for x in (loop + core)[-NDIM_MAX:] if x is not None)
        return BroadcastableShapes(
            input_shapes=tuple(add_shape(l_in, c) for l_in, c in zip(loop_in, core_in)),
            result_shape=add_shape(loop_res, core_res),
        )

    def _draw_core_dimensions(self, data: Any) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        dims: dict[str, int] = {}
        shapes: List[List[Optional[int]]] = []
        total_shapes: List[Tuple[str, ...]] = list(self.signature.input_shapes) + [self.signature.result_shape]  # type: ignore
        for _ in total_shapes:
            shapes.append([])
        for shape_idx, shape in enumerate(self.signature.input_shapes + (self.signature.result_shape,)):  # type: ignore
            for name in shape:
                if name.isdigit():
                    shapes[shape_idx].append(int(name))
                    continue
                if name not in dims:
                    dim: int = data.draw(self.side_strat)
                    dims[name] = dim
                    if self.min_dims == 0 and not data.draw(st.booleans()):
                        dims[name + "?"] = None  # type: ignore
                    else:
                        dims[name + "?"] = dim
                shapes[shape_idx].append(dims[name])
        core_input: Tuple[Tuple[int, ...], ...] = tuple(tuple(s) for s in shapes[:-1])
        core_result: Tuple[int, ...] = tuple(shapes[-1])  # type: ignore
        return core_input, core_result

    def _draw_loop_dimensions(self, data: Any, use: Optional[List[bool]] = None) -> BroadcastableShapes:
        base_shape_rev: Tuple[int, ...] = self.base_shape[::-1]
        result_shape: List[int] = list(base_shape_rev)
        shapes: List[List[int]] = [[] for _ in range(self.num_shapes)]
        if use is None:
            use = [True for _ in range(self.num_shapes)]
        _gap: int = self.max_dims - self.min_dims  # type: ignore
        p_keep_extending_shape: float = _calc_p_continue(desired_avg=_gap / 2, max_size=_gap)
        for dim_count in range(1, self.max_dims + 1):  # type: ignore
            dim: int = dim_count - 1
            if len(base_shape_rev) < dim_count or base_shape_rev[dim] == 1:
                dim_side: int = data.draw(self.side_strat)
            elif base_shape_rev[dim] <= self.max_side:  # type: ignore
                dim_side = base_shape_rev[dim]
            else:
                dim_side = 1
            allowed_sides: List[int] = sorted([1, dim_side])
            for shape_id, shape in enumerate(shapes):
                if dim < len(result_shape) and self.size_one_allowed:
                    side: int = data.draw(st.sampled_from(allowed_sides))
                else:
                    side = dim_side
                if self.min_dims < dim_count:
                    use[shape_id] &= data.draw(st.booleans())
                if use[shape_id]:
                    shape.append(side)
                    if len(result_shape) < len(shape):
                        result_shape.append(shape[-1])
                    elif shape[-1] != 1 and result_shape[dim] == 1:
                        result_shape[dim] = shape[-1]
            if not any(use):
                break
        max_len: int = max(len(self.base_shape), *(len(s) for s in shapes))
        result_shape = result_shape[:max_len]
        for s in shapes:
            assert self.min_dims <= len(s) <= self.max_dims  # type: ignore
            assert all(self.min_side <= x <= self.max_side for x in s)  # type: ignore
        return BroadcastableShapes(
            input_shapes=tuple(tuple(reversed(shape)) for shape in shapes),
            result_shape=tuple(reversed(result_shape)),
        )


class BasicIndexStrategy(st.SearchStrategy[Union[BasicIndex, IndexEntry]]):
    def __init__(
        self,
        shape: Shape,
        min_dims: int,
        max_dims: int,
        allow_ellipsis: bool,
        allow_newaxis: bool,
        allow_fewer_indices_than_dims: bool,
    ) -> None:
        self.shape: Shape = shape
        self.min_dims: int = min_dims
        self.max_dims: int = max_dims
        self.allow_ellipsis: bool = allow_ellipsis
        self.allow_newaxis: bool = allow_newaxis
        self.allow_fewer_indices_than_dims: bool = allow_fewer_indices_than_dims

    def do_draw(self, data: Any) -> Union[BasicIndex, IndexEntry]:
        result: List[Any] = []
        for dim_size in self.shape:
            if dim_size == 0:
                result.append(slice(None))
                continue
            strategy: st.SearchStrategy[Union[int, slice]] = st.integers(-dim_size, dim_size - 1) | st.slices(dim_size)
            result.append(data.draw(strategy))
        result_dims: int = sum(1 for idx in result if isinstance(idx, slice))
        while (
            self.allow_newaxis
            and result_dims < self.max_dims
            and (result_dims < self.min_dims or data.draw(st.booleans()))
        ):
            i: int = data.draw(st.integers(0, len(result)))
            result.insert(i, None)  # Note: np.newaxis is None
            result_dims += 1
        assume(self.min_dims <= result_dims <= self.max_dims)
        if self.allow_ellipsis and data.draw(st.booleans()):
            i: int = data.draw(st.integers(0, len(result)))
            j: int = data.draw(st.integers(0, len(result)))
            while i > 0 and result[i - 1] == slice(None):
                i -= 1
            while j < len(result) and result[j] == slice(None):
                j += 1
            result[i:j] = [Ellipsis]
        elif self.allow_fewer_indices_than_dims:
            while result[-1:] == [slice(None, None)] and data.draw(st.integers(0, 7)):
                result.pop()
        if len(result) == 1 and data.draw(st.booleans()):
            return result[0]
        return tuple(result)  # type: ignore
