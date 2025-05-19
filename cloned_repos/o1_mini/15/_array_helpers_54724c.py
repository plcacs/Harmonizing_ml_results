import re
from typing import NamedTuple, Optional, Union, Tuple, Any, Dict, List
from hypothesis import assume, strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture.utils import _calc_p_continue
from hypothesis.internal.coverage import check_function
from hypothesis.internal.validation import check_type, check_valid_interval
from hypothesis.strategies._internal.utils import defines_strategy
from hypothesis.utils.conventions import UniqueIdentifier, not_set

__all__ = [
    'NDIM_MAX', 'BasicIndex', 'BasicIndexStrategy', 'BroadcastableShapes',
    'MutuallyBroadcastableShapesStrategy', 'Shape', 'array_shapes',
    'broadcastable_shapes', 'check_argument', 'check_valid_dims',
    'mutually_broadcastable_shapes', 'order_check', 'valid_tuple_axes'
]

Shape = Tuple[int, ...]
BasicIndex = Tuple[Union[int, slice, None, Ellipsis], ...]

class BroadcastableShapes(NamedTuple):
    input_shapes: Tuple[Tuple[int, ...], ...]
    result_shape: Tuple[int, ...]

@check_function
def check_argument(condition: bool, fail_message: str, *f_args: Any, **f_kwargs: Any) -> None:
    if not condition:
        raise InvalidArgument(fail_message.format(*f_args, **f_kwargs))

@check_function
def order_check(name: str, floor: int, min_: int, max_: int) -> None:
    if floor > min_:
        raise InvalidArgument(f'min_{name} must be at least {floor} but was {min_}')
    if min_ > max_:
        raise InvalidArgument(f'min_{name}={min_} is larger than max_{name}={max_}')

NDIM_MAX: int = 32

@check_function
def check_valid_dims(dims: int, name: str) -> None:
    if dims > NDIM_MAX:
        raise InvalidArgument(f'{name}={dims}, but Hypothesis does not support arrays with more than {NDIM_MAX} dimensions')

@defines_strategy()
def array_shapes(*, min_dims: int = 1, max_dims: Optional[int] = None,
                min_side: int = 1, max_side: Optional[int] = None) -> st.SearchStrategy[Shape]:
    """Return a strategy for array shapes (tuples of int >= 1).

    * ``min_dims`` is the smallest length that the generated shape can possess.
    * ``max_dims`` is the largest length that the generated shape can possess,
      defaulting to ``min_dims + 2``.
    * ``min_side`` is the smallest size that a dimension can possess.
    * ``max_side`` is the largest size that a dimension can possess,
      defaulting to ``min_side + 5``.
    """
    check_type(int, min_dims, 'min_dims')
    check_type(int, min_side, 'min_side')
    check_valid_dims(min_dims, 'min_dims')
    if max_dims is None:
        max_dims = min(min_dims + 2, NDIM_MAX)
    check_type(int, max_dims, 'max_dims')
    check_valid_dims(max_dims, 'max_dims')
    if max_side is None:
        max_side = min_side + 5
    check_type(int, max_side, 'max_side')
    order_check('dims', 0, min_dims, max_dims)
    order_check('side', 0, min_side, max_side)
    return st.lists(
        st.integers(min_side, max_side),
        min_size=min_dims,
        max_size=max_dims
    ).map(tuple)

@defines_strategy()
def valid_tuple_axes(ndim: int, *, min_size: int = 0, max_size: Optional[int] = None) -> st.SearchStrategy[Tuple[int, ...]]:
    """All tuples will have a length >= ``min_size`` and <= ``max_size``. The default
    value for ``max_size`` is ``ndim``.

    Examples from this strategy shrink towards an empty tuple, which render most
    sequential functions as no-ops.

    The following are some examples drawn from this strategy.

    .. code-block:: pycon

      >>> [valid_tuple_axes(3).example() for i in range(4)]
      [(-3, 1), (0, 1, -1), (0, 2), (0, -2, 2)]

    ``valid_tuple_axes`` can be joined with other strategies to generate
    any type of valid axis object, i.e. integers, tuples, and ``None``:

    .. code-block:: python

      any_axis_strategy = none() | integers(-ndim, ndim - 1) | valid_tuple_axes(ndim)

    """
    check_type(int, ndim, 'ndim')
    check_type(int, min_size, 'min_size')
    if max_size is None:
        max_size = ndim
    check_type(int, max_size, 'max_size')
    order_check('size', 0, min_size, max_size)
    check_valid_interval(max_size, ndim, 'max_size', 'ndim')
    axes: st.SearchStrategy[int] = st.integers(0, max(0, 2 * ndim - 1)).map(
        lambda x: x if x < ndim else x - 2 * ndim
    )
    return st.lists(
        axes,
        min_size=min_size,
        max_size=max_size,
        unique_by=lambda x: x % ndim
    ).map(tuple)

@defines_strategy()
def broadcastable_shapes(shape: Shape, *, min_dims: int = 0, max_dims: Optional[int] = None,
                        min_side: int = 1, max_side: Optional[int] = None) -> st.SearchStrategy[Shape]:
    """Return a strategy for shapes that are broadcast-compatible with the
    provided shape.

    Examples from this strategy shrink towards a shape with length ``min_dims``.
    The size of an aligned dimension shrinks towards size ``1``. The size of an
    unaligned dimension shrink towards ``min_side``.

    * ``shape`` is a tuple of integers.
    * ``min_dims`` is the smallest length that the generated shape can possess.
    * ``max_dims`` is the largest length that the generated shape can possess,
      defaulting to ``max(len(shape), min_dims) + 2``.
    * ``min_side`` is the smallest size that an unaligned dimension can possess.
    * ``max_side`` is the largest size that an unaligned dimension can possess,
      defaulting to 2 plus the size of the largest aligned dimension.

    The following are some examples drawn from this strategy.

    .. code-block:: pycon

        >>> [broadcastable_shapes(shape=(2, 3)).example() for i in range(5)]
        [(1, 3), (), (2, 3), (2, 1), (4, 1, 3), (3, )]

    """
    check_type(tuple, shape, 'shape')
    check_type(int, min_side, 'min_side')
    check_type(int, min_dims, 'min_dims')
    check_valid_dims(min_dims, 'min_dims')
    strict_check: bool = max_side is None or max_dims is None
    if max_dims is None:
        max_dims = min(max(len(shape), min_dims) + 2, NDIM_MAX)
    check_type(int, max_dims, 'max_dims')
    check_valid_dims(max_dims, 'max_dims')
    if max_side is None:
        max_side = max(shape[-max_dims:] + (min_side,)) + 2
    check_type(int, max_side, 'max_side')
    order_check('dims', 0, min_dims, max_dims)
    order_check('side', 0, min_side, max_side)
    if strict_check:
        dims = max_dims
        bound_name = 'max_dims'
    else:
        dims = min_dims
        bound_name = 'min_dims'
    if not all((min_side <= s for s in shape[::-1][:dims] if s != 1)):
        raise InvalidArgument(
            f'Given shape={shape}, there are no broadcast-compatible shapes that satisfy: {bound_name}={dims} and min_side={min_side}'
        )
    if not (min_side <= 1 <= max_side or all((s <= max_side for s in shape[::-1][:dims]))):
        raise InvalidArgument(
            f'Given base_shape={shape}, there are no broadcast-compatible shapes that satisfy all of {bound_name}={dims}, min_side={min_side}, and max_side={max_side}'
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
        max_side=max_side
    ).map(lambda x: x.input_shapes[0])

_DIMENSION: str = '\\w+\\??'
_SHAPE: str = f'\\((?:{_DIMENSION}(?:,{_DIMENSION}){{0,31}})?\\)'
_ARGUMENT_LIST: str = f'{_SHAPE}(?:,{_SHAPE})*'
_SIGNATURE: str = f'^{_ARGUMENT_LIST}->{_SHAPE}$'
_SIGNATURE_MULTIPLE_OUTPUT: str = f'^{_ARGUMENT_LIST}->{_ARGUMENT_LIST}$'

class _GUfuncSig(NamedTuple):
    input_shapes: Tuple[Tuple[str, ...], ...]
    result_shape: Tuple[str, ...]

def _hypothesis_parse_gufunc_signature(signature: str) -> _GUfuncSig:
    if not re.match(_SIGNATURE, signature):
        if re.match(_SIGNATURE_MULTIPLE_OUTPUT, signature):
            raise InvalidArgument(
                f"Hypothesis does not yet support generalised ufunc signatures with multiple output arrays - mostly because we don't know of anyone who uses them!  Please get in touch with us to fix that.\n (signature={signature!r})"
            )
        if re.match(
            '^\\((?:\\w+(?:,\\w+)*)?\\)(?:,\\((?:\\w+(?:,\\w+)*)?\\))*->\\((?:\\w+(?:,\\w+)*)?\\)(?:,\\((?:\\w+(?:,\\w+)*)?\\))*$',
            signature
        ):
            raise InvalidArgument(
                f"signature={signature!r} matches Numpy's regex for gufunc signatures, but contains shapes with more than {NDIM_MAX} dimensions and is thus invalid."
            )
        raise InvalidArgument(f'{signature!r} is not a valid gufunc signature')
    input_shapes_str, output_shapes_str = signature.split('->')
    input_shapes: Tuple[Tuple[str, ...], ...] = tuple(
        tuple(re.findall(_DIMENSION, a)) for a in re.findall(_SHAPE, input_shapes_str)
    )
    output_shapes: Tuple[Tuple[str, ...], ...] = tuple(
        tuple(re.findall(_DIMENSION, a)) for a in re.findall(_SHAPE, output_shapes_str)
    )
    if len(output_shapes) != 1:
        raise InvalidArgument("Hypothesis does not yet support multiple output shapes.")
    result_shape = output_shapes[0]
    for shape in (*input_shapes, result_shape):
        for name in shape:
            try:
                int(name.strip('?'))
                if '?' in name:
                    raise InvalidArgument(
                        f'Got dimension {name!r}, but handling of frozen optional dimensions is ambiguous.  If you know how this should work, please contact us to get this fixed and documented ({{signature=}}).'
                    )
            except ValueError:
                names_in = {n.strip('?') for shp in input_shapes for n in shp}
                names_out = {n.strip('?') for n in result_shape}
                if name.strip('?') in names_out - names_in:
                    raise InvalidArgument(
                        f'The {name!r} dimension only appears in the output shape, and is not frozen, so the size is not determined (signature={signature!r}).'
                    ) from None
    return _GUfuncSig(input_shapes=input_shapes, result_shape=result_shape)

@defines_strategy()
def mutually_broadcastable_shapes(*, num_shapes: Optional[int] = not_set,
                                 signature: Optional[str] = not_set,
                                 base_shape: Shape = (),
                                 min_dims: int = 0,
                                 max_dims: Optional[int] = None,
                                 min_side: int = 1,
                                 max_side: Optional[int] = None
                                 ) -> st.SearchStrategy[BroadcastableShapes]:
    """Return a strategy for a specified number of shapes N that are
    mutually-broadcastable with one another and with the provided base shape.

    * ``num_shapes`` is the number of mutually broadcast-compatible shapes to generate.
    * ``base_shape`` is the shape against which all generated shapes can broadcast.
      The default shape is empty, which corresponds to a scalar and thus does
      not constrain broadcasting at all.
    * ``min_dims`` is the smallest length that the generated shape can possess.
    * ``max_dims`` is the largest length that the generated shape can possess,
      defaulting to ``max(len(shape), min_dims) + 2``.
    * ``min_side`` is the smallest size that an unaligned dimension can possess.
    * ``max_side`` is the largest size that an unaligned dimension can possess,
      defaulting to 2 plus the size of the largest aligned dimension.

    The strategy will generate a :obj:`python:typing.NamedTuple` containing:

    * ``input_shapes`` as a tuple of the N generated shapes.
    * ``result_shape`` as the resulting shape produced by broadcasting the N shapes
      with the base shape.

    The following are some examples drawn from this strategy.

    .. code-block:: pycon

        >>> # Draw three shapes where each shape is broadcast-compatible with (2, 3)
        ... strat = mutually_broadcastable_shapes(num_shapes=3, base_shape=(2, 3))
        >>> for _ in range(5):
        ...     print(strat.example())
        BroadcastableShapes(input_shapes=((4, 1, 3), (4, 2, 3), ()), result_shape=(4, 2, 3))
        BroadcastableShapes(input_shapes=((3,), (1, 3), (2, 3)), result_shape=(2, 3))
        BroadcastableShapes(input_shapes=((), (), ()), result_shape=())
        BroadcastableShapes(input_shapes=((3,), (), (3,)), result_shape=(3,))
        BroadcastableShapes(input_shapes=((1, 2, 3), (3,), ()), result_shape=(1, 2, 3))

    """
    arg_msg = 'Pass either the `num_shapes` or the `signature` argument, but not both.'
    if num_shapes is not not_set:
        check_argument(signature is not_set, arg_msg)
        check_type(int, num_shapes, 'num_shapes')
        parsed_signature: Optional[_GUfuncSig] = None
        sig_dims: int = 0
    else:
        check_argument(signature is not not_set, arg_msg)
        if signature is None:
            raise InvalidArgument(
                'Expected a string, but got invalid signature=None.  (maybe .signature attribute of an element-wise ufunc?)'
            )
        check_type(str, signature, 'signature')
        parsed_signature = _hypothesis_parse_gufunc_signature(signature)
        all_shapes = (*parsed_signature.input_shapes, parsed_signature.result_shape)
        sig_dims = min(len(s) for s in all_shapes)
        num_shapes = len(parsed_signature.input_shapes)
    if num_shapes is not not_set and num_shapes < 1:
        raise InvalidArgument(f'num_shapes={num_shapes} must be at least 1')
    check_type(tuple, base_shape, 'base_shape')
    check_type(int, min_side, 'min_side')
    check_type(int, min_dims, 'min_dims')
    check_valid_dims(min_dims, 'min_dims')
    strict_check: bool = max_dims is not None
    if max_dims is None:
        max_dims = min(max(len(base_shape), min_dims) + 2, NDIM_MAX)
    check_type(int, max_dims, 'max_dims')
    check_valid_dims(max_dims, 'max_dims')
    if max_side is None:
        max_side = max((*base_shape[-max_dims:], min_side)) + 2
    check_type(int, max_side, 'max_side')
    order_check('dims', 0, min_dims, max_dims)
    order_check('side', 0, min_side, max_side)
    if signature is not_set:
        signature = None
    if signature is not None and max_dims > NDIM_MAX:
        raise InvalidArgument(
            f'max_dims={max_dims} would exceed the {NDIM_MAX}-dimension limit Hypothesis imposes on array shapes, given signature={parsed_signature!r}'
        )
    if strict_check:
        dims = max_dims
        bound_name = 'max_dims'
    else:
        dims = min_dims
        bound_name = 'min_dims'
    if not all((min_side <= s for s in base_shape[::-1][:dims] if s != 1)):
        raise InvalidArgument(
            f'Given base_shape={base_shape}, there are no broadcast-compatible shapes that satisfy: {bound_name}={dims} and min_side={min_side}'
        )
    if not (min_side <= 1 <= max_side or all((s <= max_side for s in base_shape[::-1][:dims]))):
        raise InvalidArgument(
            f'Given base_shape={base_shape}, there are no broadcast-compatible shapes that satisfy all of {bound_name}={dims}, min_side={min_side}, and max_side={max_side}'
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
        num_shapes=num_shapes if num_shapes is not not_set else 1,
        signature=parsed_signature,
        base_shape=base_shape,
        min_dims=min_dims,
        max_dims=max_dims,
        min_side=min_side,
        max_side=max_side
    )

class MutuallyBroadcastableShapesStrategy(st.SearchStrategy[BroadcastableShapes]):
    
    def __init__(self, num_shapes: int, signature: Optional[_GUfuncSig] = None, base_shape: Shape = (),
                 min_dims: int = 0, max_dims: Optional[int] = None,
                 min_side: int = 1, max_side: Optional[int] = None) -> None:
        super().__init__()
        self.base_shape: Shape = base_shape
        self.side_strat: st.SearchStrategy[int] = st.integers(min_side, max_side)
        self.num_shapes: int = num_shapes
        self.signature: Optional[_GUfuncSig] = signature
        self.min_dims: int = min_dims
        self.max_dims: Optional[int] = max_dims
        self.min_side: int = min_side
        self.max_side: Optional[int] = max_side
        self.size_one_allowed: bool = self.min_side <= 1 <= self.max_side  # type: ignore

    def do_draw(self, data: st.DataObject) -> BroadcastableShapes:
        if self.signature is None:
            return self._draw_loop_dimensions(data)
        core_in, core_res = self._draw_core_dimensions(data)
        use: List[bool] = [None not in shp for shp in core_in]
        loop_in, loop_res = self._draw_loop_dimensions(data, use=use)

        def add_shape(loop: Tuple[int, ...], core: Tuple[int, ...]) -> Tuple[int, ...]:
            return tuple(x for x in (loop + core)[-NDIM_MAX:] if x is not None)
        
        input_shapes = tuple(add_shape(l_in, c) for l_in, c in zip(loop_in, core_in))
        result_shape = add_shape(loop_res, core_res)
        return BroadcastableShapes(input_shapes=input_shapes, result_shape=result_shape)

    def _draw_core_dimensions(self, data: st.DataObject) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...]]:
        dims: Dict[str, Optional[int]] = {}
        shapes: List[List[Optional[int]]] = []
        for shape in (*self.signature.input_shapes, self.signature.result_shape):
            shapes.append([])
            for name in shape:
                if name.isdigit():
                    shapes[-1].append(int(name))
                    continue
                dim = name.rstrip('?')
                if dim not in dims:
                    dims[dim] = data.draw(self.side_strat)
                    if self.min_dims == 0 and (not data.draw(st.booleans())):
                        dims[dim + '?'] = None
                    else:
                        dims[dim + '?'] = dims[dim]
                shapes[-1].append(dims.get(name, None))
        input_shapes = tuple(tuple(s) for s in shapes[:-1])
        result_shape = tuple(shapes[-1])
        return (input_shapes, result_shape)

    def _draw_loop_dimensions(self, data: st.DataObject, use: List[bool]) -> Tuple[List[Tuple[int, ...]], Tuple[int, ...]]:
        base_shape_reversed = self.base_shape[::-1]
        result_shape: List[int] = list(base_shape_reversed)
        shapes: List[List[int]] = [[] for _ in range(self.num_shapes)]
        _gap = self.max_dims - self.min_dims if self.max_dims is not None else NDIM_MAX
        p_keep_extending_shape = _calc_p_continue(desired_avg=_gap / 2, max_size=_gap)
        for dim_count in range(1, self.max_dims + 1 if self.max_dims else NDIM_MAX + 1):
            dim = dim_count - 1
            if dim >= len(base_shape_reversed) or base_shape_reversed[dim] == 1:
                dim_side = data.draw(self.side_strat)
            elif base_shape_reversed[dim] <= self.max_side:
                dim_side = base_shape_reversed[dim]
            else:
                dim_side = 1
            allowed_sides = sorted([1, dim_side])
            for shape_id, shape in enumerate(shapes):
                if dim < len(result_shape) and self.size_one_allowed:
                    side = data.draw(st.sampled_from(allowed_sides))
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
        max_len = max(len(self.base_shape), *(len(s) for s in shapes))
        result_shape = result_shape[:max_len]
        assert len(shapes) == self.num_shapes
        assert all(self.min_dims <= len(s) <= self.max_dims for s in shapes)
        assert all(self.min_side <= s <= self.max_side for shape in shapes for s in shape)
        return (
            [tuple(reversed(shape)) for shape in shapes],
            tuple(reversed(result_shape))
        )

class BasicIndexStrategy(st.SearchStrategy[BasicIndex]):
    
    def __init__(self, shape: Shape, min_dims: int, max_dims: int,
                 allow_ellipsis: bool, allow_newaxis: bool,
                 allow_fewer_indices_than_dims: bool) -> None:
        super().__init__()
        self.shape: Shape = shape
        self.min_dims: int = min_dims
        self.max_dims: int = max_dims
        self.allow_ellipsis: bool = allow_ellipsis
        self.allow_newaxis: bool = allow_newaxis
        self.allow_fewer_indices_than_dims: bool = allow_fewer_indices_than_dims

    def do_draw(self, data: st.DataObject) -> BasicIndex:
        result: List[Union[int, slice, None, Ellipsis]] = []
        for dim_size in self.shape:
            if dim_size == 0:
                result.append(slice(None))
                continue
            strategy: st.SearchStrategy[Union[int, slice]] = st.integers(-dim_size, dim_size - 1) | st.slices(dim_size)
            result.append(data.draw(strategy))
        result_dims = sum(isinstance(idx, slice) for idx in result)
        while self.allow_newaxis and result_dims < self.max_dims and (result_dims < self.min_dims or data.draw(st.booleans())):
            i = data.draw(st.integers(0, len(result)))
            result.insert(i, None)
            result_dims += 1
        assume(self.min_dims <= result_dims <= self.max_dims)
        if self.allow_ellipsis and data.draw(st.booleans()):
            i = data.draw(st.integers(0, len(result)))
            j = data.draw(st.integers(i, len(result)))
            while i > 0 and result[i - 1] == slice(None):
                i -= 1
            while j < len(result) and result[j] == slice(None):
                j += 1
            result[i:j] = [Ellipsis]
        elif self.allow_fewer_indices_than_dims:
            while len(result) > 0 and result[-1] == slice(None) and data.draw(st.integers(0, 7)) != 0:
                result.pop()
        if len(result) == 1 and data.draw(st.booleans()):
            return result[0]  # type: ignore
        return tuple(result)  # type: ignore
