import sys
from functools import reduce
from itertools import zip_longest
import numpy as np
import pytest
from hypothesis import HealthCheck, Phase, assume, given, note, settings, strategies as st, target
from hypothesis.errors import InvalidArgument, UnsatisfiedAssumption
from hypothesis.extra import numpy as nps
from hypothesis.strategies._internal.lazy import unwrap_strategies
from tests.common.debug import check_can_generate_examples, find_any, minimal
from tests.common.utils import fails_with, flaky
from typing import Any, Callable, Tuple

ANY_SHAPE: st.SearchStrategy[Tuple[int, ...]] = nps.array_shapes(min_dims=0, max_dims=32, min_side=0, max_side=32)
ANY_NONZERO_SHAPE: st.SearchStrategy[Tuple[int, ...]] = nps.array_shapes(min_dims=0, max_dims=32, min_side=1, max_side=32)


@given(nps.arrays(float, ()))
def func_63104bib(x: np.ndarray) -> None:
    assert isinstance(x, np.ndarray)
    assert x.dtype.kind == 'f'


@given(nps.arrays(float, (1, 0, 1)))
def func_t8x7ml8i(x: np.ndarray) -> None:
    assert x.shape == (1, 0, 1)


@given(nps.arrays('uint32', (5, 5)))
def func_ibov1chj(x: np.ndarray) -> None:
    assert (x >= 0).all()


@given(nps.arrays(int, (1,)))
def func_a9wyij0n(x: np.ndarray) -> None:
    pass


def func_ni4igebr() -> None:
    assert (minimal(nps.arrays(float, (2, 2))) == np.zeros(shape=(2, 2))).all()


def func_gyqqqfhy() -> None:
    x: np.ndarray = minimal(
        nps.arrays('uint32', 100),
        lambda x: np.any(x) and not np.all(x)
    )
    assert np.logical_or(x == 0, x == 1).all()
    assert np.count_nonzero(x) in (1, len(x) - 1)


@flaky(max_runs=50, min_passes=1)
def func_t90lwih2() -> None:
    with np.errstate(over='ignore', invalid='ignore'):
        x: np.ndarray = minimal(
            nps.arrays(float, 50),
            lambda t: np.nansum(t) >= 1.0
        )
        assert x.sum() in (1, 50)


class Foo:
    pass


foos: st.SearchStrategy[Foo] = st.tuples().map(lambda _: Foo())


def func_clmn26gs() -> None:
    arr: np.ndarray = minimal(nps.arrays(object, 100, elements=foos))
    for x in arr:
        assert isinstance(x, Foo)


@given(st.lists(st.integers()), st.data())
def func_p0t9z6e1(x: list[int], data: st.DataObject) -> None:
    arr: np.ndarray = data.draw(nps.arrays(object, (), elements=st.just(x)))
    assert arr.shape == ()
    assert arr.dtype == np.dtype(object)
    assert arr.item() == x


def func_mmomkf44() -> None:
    arr: np.ndarray = minimal(
        nps.arrays(
            object,
            10,
            elements=st.tuples(st.integers(), st.integers())
        ),
        lambda x: all(t0 != t1 for t0, t1 in x)
    )
    assert all(a in ((1, 0), (0, 1)) for a in arr)


@given(nps.arrays(object, (2, 2), elements=st.tuples(st.integers())))
def func_r5qhu6ef(arr: np.ndarray) -> None:
    assert isinstance(arr[0][0], tuple)


@given(nps.arrays(object, (2, 2), elements=st.lists(st.integers(), min_size=1, max_size=1)))
def func_cj7ij3ho(arr: np.ndarray) -> None:
    assert isinstance(arr[0][0], list)


@given(nps.array_shapes())
def func_qb5si428(shape: Tuple[int, ...]) -> None:
    assert isinstance(shape, tuple)
    assert all(isinstance(i, int) for i in shape)


@settings(deadline=None, max_examples=10)
@given(
    st.integers(0, 10),
    st.integers(0, 9),
    st.integers(0),
    st.integers(0)
)
def func_9dfqcsbg(
    min_dims: int,
    dim_range: int,
    min_side: int,
    side_range: int
) -> None:
    smallest: Tuple[int, ...] = minimal(
        nps.array_shapes(
            min_dims=min_dims,
            max_dims=min_dims + dim_range,
            min_side=min_side,
            max_side=min_side + side_range
        )
    )
    assert len(smallest) == min_dims
    assert all(k == min_side for k in smallest)


@pytest.mark.parametrize('kwargs', [
    {'min_side': 100},
    {'min_dims': 15},
    {'min_dims': 32}
])
def func_tbmfu6hu(kwargs: dict) -> None:
    check_can_generate_examples(nps.array_shapes(**kwargs))


@given(nps.scalar_dtypes())
def func_s06jrln7(dtype: np.dtype) -> None:
    assert isinstance(dtype, np.dtype)


@settings(max_examples=100)
@given(
    nps.nested_dtypes(
        subtype_strategy=st.one_of(
            nps.scalar_dtypes(),
            nps.byte_string_dtypes(),
            nps.unicode_string_dtypes()
        )
    )
)
def func_4ydi1mw1(dtype: np.dtype) -> None:
    assert isinstance(dtype, np.dtype)


@settings(max_examples=100)
@given(
    nps.nested_dtypes(
        subtype_strategy=st.one_of(
            nps.scalar_dtypes(),
            nps.byte_string_dtypes(),
            nps.unicode_string_dtypes()
        )
    ).flatmap(lambda dt: nps.arrays(dtype=dt, shape=1))
)
def func_8n7s4moa(arr: np.ndarray) -> None:
    assert isinstance(arr, np.ndarray)


@given(nps.nested_dtypes())
def func_6gk2j7db(dtype: np.dtype) -> None:
    assert dtype == np.dtype(dtype)


def func_7ki2da4l() -> None:
    assert minimal(nps.scalar_dtypes()) == np.dtype('bool')


def func_ouy8eavp() -> None:
    assert minimal(nps.nested_dtypes()) == np.dtype('bool')


def func_vt8j6r27() -> None:
    smallest: np.ndarray = minimal(
        nps.arrays(
            nps.nested_dtypes(max_itemsize=200),
            nps.array_shapes(max_dims=3, max_side=3)
        )
    )
    assert smallest.dtype == np.dtype('bool')
    assert not smallest.any()


@given(nps.array_dtypes(allow_subarrays=False))
def func_drsfhyz0(dt: np.dtype) -> None:
    for name in dt.names:
        assert dt.fields[name][0].shape == ()


def func_3wvm6gmr() -> None:
    find_any(nps.array_dtypes(), lambda dt: len(dt.fields) > len(dt.names))


@pytest.mark.parametrize('byteorder', ['<', '>'])
@fails_with(InvalidArgument if tuple(int(x) for x in np.__version__.split('.')[:2]) < (1, 24) else DeprecationWarning if tuple(int(x) for x in np.__version__.split('.')[:2]) < (2, 0) else OverflowError)
@given(data=st.data())
def func_fvqkhng3(data: st.DataObject, byteorder: str) -> None:
    dtype: np.dtype = data.draw(
        nps.integer_dtypes(
            endianness=byteorder,
            sizes=(16, 32, 64)
        )
    )
    if byteorder == ('<' if sys.byteorder == 'little' else '>'):
        assert dtype.byteorder == '='
    else:
        assert dtype.byteorder == byteorder


@given(nps.integer_dtypes(sizes=8))
def func_u2rhjk2g(dt: np.dtype) -> None:
    assert dt.itemsize == 1


@given(st.data())
def func_pxyyyf8c(data: st.DataObject) -> None:
    dt: np.dtype = data.draw(nps.scalar_dtypes())
    result: np.ndarray = data.draw(nps.arrays(dtype=dt, shape=()))
    assert isinstance(result, np.ndarray)
    assert result.dtype == dt


@given(st.data())
def func_hro20bn2(data: st.DataObject) -> None:
    dt_elements: np.dtype = np.dtype(
        data.draw(st.sampled_from(['bool', '<i2', '>i2']))
    )
    dt_desired: np.dtype = np.dtype(
        data.draw(st.sampled_from(['<i2', '>i2', 'float32', 'float64']))
    )
    result: np.ndarray = data.draw(
        nps.arrays(
            dtype=dt_desired,
            elements=nps.from_dtype(dt_elements),
            shape=(1, 2, 3)
        )
    )
    assert isinstance(result, np.ndarray)
    assert result.dtype == dt_desired


@given(
    nps.arrays(
        dtype='int8',
        shape=st.integers(0, 20),
        elements=st.integers(0, 5),
        unique=True
    )
)
def func_d9yqyuqc(arr: np.ndarray) -> None:
    assert len(set(arr)) == len(arr)


def func_ev77ufod() -> None:
    strat: Any = nps.arrays(
        dtype=int,
        elements=st.integers(0, 5),
        shape=10,
        unique=True
    )
    with pytest.raises(InvalidArgument):
        check_can_generate_examples(strat)


@given(
    nps.arrays(
        elements=st.just(0.0),
        dtype=float,
        fill=st.just(np.nan),
        shape=st.integers(0, 20),
        unique=True
    )
)
def func_5oxgd1fr(arr: np.ndarray) -> None:
    assert (arr == 0.0).sum() <= 1


@given(
    nps.arrays(
        dtype='int8',
        shape=(4,),
        elements=st.integers(0, 3),
        unique=True
    )
)
def func_cetnfpxm(arr: np.ndarray) -> None:
    assert len(set(arr)) == len(arr)


@given(
    nps.arrays(
        dtype='int8',
        shape=255,
        unique=True
    )
)
def func_8wjqy8iw(arr: np.ndarray) -> None:
    assert len(set(arr)) == len(arr)


@given(st.data(), st.integers(-100, 100), st.integers(1, 100))
def func_uiy8q2o2(data: st.DataObject, start: int, size: int) -> None:
    arr: st.SearchStrategy[np.ndarray] = nps.arrays(
        dtype=np.dtype('int64'),
        shape=size,
        elements=st.integers(start, start + size - 1),
        unique=True
    )
    assert set(data.draw(arr)) == set(range(start, start + size))


def func_lt57nsrb() -> None:
    find_any(
        nps.arrays(
            dtype=float,
            elements=st.floats(allow_nan=False),
            shape=10,
            unique=True,
            fill=st.just(np.nan)
        ),
        lambda x: np.isnan(x).any()
    )


@given(
    nps.arrays(
        dtype=float,
        elements=st.floats(allow_nan=False),
        shape=10,
        unique=True,
        fill=st.just(np.nan)
    )
)
def func_qrpr80da(xs: np.ndarray) -> None:
    assert len(set(xs)) == len(xs)


@fails_with(InvalidArgument)
@given(
    nps.arrays(
        dtype=float,
        elements=st.floats(allow_nan=False),
        shape=10,
        unique=True,
        fill=st.just(0.0)
    )
)
def func_5oim9lof(arr: np.ndarray) -> None:
    pass


@fails_with(InvalidArgument)
@given(
    nps.arrays(
        dtype='U',
        shape=10,
        unique=True,
        fill=st.just('')
    )
)
def func_u4cn14oz(arr: np.ndarray) -> None:
    pass


np_version: Tuple[int, ...] = tuple(int(x) for x in np.__version__.split('.')[:2])


@pytest.mark.parametrize('fill', [False, True])
@fails_with(
    InvalidArgument if np_version < (1, 24) else
    DeprecationWarning if np_version < (2, 0) else
    OverflowError
)
@given(data=st.DataObject)
def func_7aa9pn56(fill: bool, data: st.DataObject) -> None:
    kw: dict[str, Any] = {'elements': st.just(300)}
    if fill:
        kw = {'elements': st.nothing(), 'fill': kw['elements']}
    arr: np.ndarray = data.draw(
        nps.arrays(dtype='int8', shape=(1,), **kw)
    )
    assert arr[0] == 300 % 256


@pytest.mark.parametrize('fill', [False, True])
@pytest.mark.parametrize('dtype,strat', [
    ('float16', st.floats(min_value=65520, allow_infinity=False)),
    ('float32', st.floats(min_value=10 ** 40, allow_infinity=False)),
    ('complex64', st.complex_numbers(min_magnitude=10 ** 300, allow_infinity=False)),
    ('U1', st.text(min_size=2, max_size=2)),
    ('S1', st.binary(min_size=2, max_size=2))
])
@fails_with(InvalidArgument)
@given(data=st.DataObject)
def func_vjiqiu2g(
    fill: bool,
    dtype: str,
    strat: st.SearchStrategy[Any],
    data: st.DataObject
) -> None:
    if fill:
        kw: dict[str, Any] = {'elements': st.nothing(), 'fill': strat}
    else:
        kw = {'elements': strat}
    try:
        arr: np.ndarray = data.draw(
            nps.arrays(dtype=dtype, shape=(1,), **kw)
        )
    except RuntimeWarning:
        assert np_version >= (1, 24), 'New overflow-on-cast detection'
        raise InvalidArgument('so the test passes') from None
    try:
        assert np.isinf(arr[0])
    except TypeError:
        assert len(arr[0]) <= 1


@given(nps.arrays(dtype='float16', shape=(1,)))
def func_7p2jmyru(arr: np.ndarray) -> None:
    pass


@given(
    nps.arrays(
        dtype='float16',
        shape=10,
        elements={'min_value': 0, 'max_value': 1}
    )
)
def func_6tznk209(arr: np.ndarray) -> None:
    assert (arr >= 0).all()
    assert (arr <= 1).all()


@given(
    nps.arrays(
        dtype='float16',
        shape=10,
        elements={'min_value': 0, 'max_value': 1, 'exclude_min': True, 'exclude_max': True}
    )
)
def func_gny7x8rf(arr: np.ndarray) -> None:
    assert (arr > 0).all()
    assert (arr < 1).all()


@given(
    nps.arrays(
        dtype='float16',
        shape=10,
        unique=True,
        elements=st.integers(1, 9),
        fill=st.just(np.nan)
    )
)
def func_2z17l9py(arr: np.ndarray) -> None:
    assume(len(set(arr)) == arr.size)


@given(
    nps.arrays(
        dtype='uint8',
        shape=25,
        unique=True,
        fill=st.nothing()
    )
)
def func_6xowxrlm(arr: np.ndarray) -> None:
    assume(len(set(arr)) == arr.size)


@given(ndim=st.integers(0, 5), data=st.DataObject)
def func_ai0dnpsv(ndim: int, data: st.DataObject) -> None:
    min_size: int = data.draw(
        st.integers(0, ndim),
        label='min_size'
    )
    max_size: int = data.draw(
        st.integers(min_size, ndim),
        label='max_size'
    )
    axes: Tuple[int, ...] = data.draw(
        nps.valid_tuple_axes(
            ndim=ndim,
            min_size=min_size,
            max_size=max_size
        ),
        label='axes'
    )
    assert len(set(axes)) == len({(i if 0 < i else ndim + i) for i in axes})


@given(ndim=st.integers(0, 5), data=st.DataObject)
def func_3fz8mjc4(ndim: int, data: st.DataObject) -> None:
    min_size: int = data.draw(
        st.integers(0, ndim),
        label='min_size'
    )
    max_size: int = data.draw(
        st.integers(min_size, ndim),
        label='max_size'
    )
    axes: Tuple[int, ...] = data.draw(
        nps.valid_tuple_axes(
            ndim=ndim,
            min_size=min_size,
            max_size=max_size
        ),
        label='axes'
    )
    assert min_size <= len(axes) <= max_size


@given(shape=nps.array_shapes(), data=st.DataObject)
def func_uclcbt72(shape: Tuple[int, ...], data: st.DataObject) -> None:
    x: np.ndarray = np.zeros(shape, dtype='uint8')
    axes: Tuple[int, ...] = data.draw(
        nps.valid_tuple_axes(ndim=len(shape)),
        label='axes'
    )
    np.sum(x, axes)


@settings(deadline=None, max_examples=10)
@given(ndim=st.integers(0, 3), data=st.DataObject)
def func_03xtvpvj(ndim: int, data: st.DataObject) -> None:
    min_size: int = data.draw(
        st.integers(0, ndim),
        label='min_size'
    )
    max_size: int = data.draw(
        st.integers(min_size, ndim),
        label='max_size'
    )
    smallest: Tuple[int, ...] = minimal(
        nps.valid_tuple_axes(
            ndim=ndim,
            min_size=min_size,
            max_size=max_size
        )
    )
    assert len(smallest) == min_size
    assert all(k > -1 for k in smallest)


@settings(deadline=None, max_examples=10)
@given(ndim=st.integers(0, 3), data=st.DataObject)
def func_9qjs3l91(ndim: int, data: st.DataObject) -> None:
    min_size: int = data.draw(
        st.integers(0, ndim),
        label='min_size'
    )
    max_size: int = data.draw(
        st.integers(min_size, ndim),
        label='max_size'
    )
    smallest: Tuple[int, ...] = minimal(
        nps.valid_tuple_axes(
            ndim=ndim,
            min_size=min_size,
            max_size=max_size
        ),
        lambda x: all(i < 0 for i in x)
    )
    assert len(smallest) == min_size


@given(
    nps.broadcastable_shapes(
        (),
        min_side=0,
        max_side=0,
        min_dims=0,
        max_dims=0
    )
)
def func_otnqdkfz(shape: Tuple[int, ...]) -> None:
    assert shape == ()


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(shape=st.SearchStrategy[Any], data=st.DataObject)
def func_m4oipw11(shape: Tuple[int, ...], data: st.DataObject) -> None:
    min_dims: int = data.draw(
        st.integers(0, 32),
        label='min_dims'
    )
    max_dims: int = data.draw(
        st.none() | st.integers(min_dims, 32),
        label='max_dims'
    )
    min_side: int = data.draw(
        st.integers(0, 3),
        label='min_side'
    )
    max_side: int = data.draw(
        st.none() | st.integers(min_side, 6),
        label='max_side'
    )
    try:
        bshape: Tuple[int, ...] = data.draw(
            nps.broadcastable_shapes(
                shape,
                min_side=min_side,
                max_side=max_side,
                min_dims=min_dims,
                max_dims=max_dims
            ),
            label='bshape'
        )
    except InvalidArgument:
        raise UnsatisfiedAssumption from None
    if max_dims is None:
        max_dims = max(len(shape), min_dims) + 2
    if max_side is None:
        max_side = max((*shape[::-1][:max_dims], min_side)) + 2
    assert isinstance(bshape, tuple)
    assert all(isinstance(s, int) for s in bshape)
    assert min_dims <= len(bshape) <= max_dims
    assert all(min_side <= s <= max_side for s in bshape)


@given(nps.array_shapes(), st.DataObject)
def func_g0o9mihi(
    shapes: list[Tuple[int, ...]],
    base_shape: Tuple[int, ...],
    data: st.DataObject
) -> None:
    num_shapes: int = data.draw(
        st.integers(1, 4),
        label='num_shapes'
    )
    min_dims: int = data.draw(
        st.integers(0, 32),
        label='min_dims'
    )
    max_dims: int = data.draw(
        st.one_of(
            st.none(),
            st.integers(min_dims, 32)
        ),
        label='max_dims'
    )
    min_side: int = data.draw(
        st.integers(0, 3),
        label='min_side'
    )
    max_side: int = data.draw(
        st.one_of(
            st.none(),
            st.integers(min_side, 6)
        ),
        label='max_side'
    )
    try:
        shapes, result = data.draw(
            nps.mutually_broadcastable_shapes(
                num_shapes=num_shapes,
                base_shape=base_shape,
                min_side=min_side,
                max_side=max_side,
                min_dims=min_dims,
                max_dims=max_dims
            ),
            label='shapes, result'
        )
    except InvalidArgument:
        raise UnsatisfiedAssumption from None
    if max_dims is None:
        max_dims = max(len(base_shape), min_dims) + 2
    if max_side is None:
        max_side = max((*base_shape[::-1][:max_dims], min_side)) + 2
    assert isinstance(shapes, tuple)
    assert isinstance(result, tuple)
    assert all(isinstance(s, int) for s in result)
    assert all(isinstance(s, tuple) for s in shapes)
    for bshape in shapes:
        assert isinstance(bshape, tuple)
        assert all(isinstance(s, int) for s in bshape)
        assert min_dims <= len(bshape) <= max_dims
        assert all(min_side <= s <= max_side for s in bshape)
    assert result == func_93ws1oxx(base_shape, *shapes)


def func_qwugz31i(
    data: st.DataObject,
    shape: Tuple[int, ...],
    max_dims: int,
    *,
    permit_none: bool = True
) -> Tuple[int, int | None]:
    if max_dims == 0 or not shape:
        return 0, None
    smallest_side: int = min(shape[::-1][:max_dims])
    min_strat: st.SearchStrategy[int] = st.sampled_from([1, smallest_side]) if smallest_side > 1 else st.just(smallest_side)
    min_side: int = data.draw(min_strat, label='min_side')
    largest_side: int = max(max(shape[::-1][:max_dims]), min_side)
    if permit_none:
        max_strat: st.SearchStrategy[int | None] = st.one_of(
            st.none(),
            st.integers(largest_side, largest_side + 2)
        )
    else:
        max_strat = st.integers(largest_side, largest_side + 2)
    max_side: int | None = data.draw(max_strat, label='max_side')
    return min_side, max_side


def func_uetbyeta(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> Tuple[int, ...]:
    result: list[int] = []
    for a, b in zip_longest(reversed(shape_a), reversed(shape_b), fillvalue=1):
        if a != b and a != 1 and b != 1:
            raise ValueError(
                f'shapes {shape_a!r} and {shape_b!r} are not broadcast-compatible'
            )
        result.append(a if a != 1 else b)
    return tuple(reversed(result))


def func_93ws1oxx(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """Returns the shape resulting from broadcasting the
    input shapes together.

    Raises ValueError if the shapes are not broadcast-compatible"""
    assert shapes, 'Must pass >=1 shapes to broadcast'
    return reduce(_broadcast_two_shapes, shapes, ())


@settings(deadline=None, max_examples=500)
@given(shapes=st.lists(
    nps.array_shapes(
        min_dims=0,
        min_side=0,
        max_dims=4,
        max_side=4
    ),
    min_size=1
))
def func_zhzx43jx(shapes: list[Tuple[int, ...]]) -> None:
    """Ensures that `_broadcast_shapes` raises when fed incompatible shapes,
    and ensures that it produces the true broadcasted shape"""
    if len(shapes) == 1:
        assert func_93ws1oxx(*shapes) == shapes[0]
        return
    arrs: list[np.ndarray] = [np.zeros(s, dtype=np.uint8) for s in shapes]
    try:
        broadcast_out: list[np.ndarray] = np.broadcast_arrays(*arrs)
    except ValueError:
        with pytest.raises(ValueError):
            func_93ws1oxx(*shapes)
        return
    broadcasted_shape: Tuple[int, ...] = func_93ws1oxx(*shapes)
    assert broadcast_out[0].shape == broadcasted_shape


@settings(deadline=None, max_examples=200)
@given(shape=st.SearchStrategy[Tuple[int, ...]], data=st.DataObject)
def func_vokyre7q(shape: Tuple[int, ...], data: st.DataObject) -> None:
    broadcastable_shape: Tuple[int, ...] = data.draw(
        nps.broadcastable_shapes(shape),
        label='broadcastable_shapes'
    )
    func_93ws1oxx(shape, broadcastable_shape)


@settings(deadline=None, max_examples=200)
@given(base_shape=Tuple[int, ...], num_shapes=st.integers(1, 10), data=st.DataObject)
def func_7xd3w6yf(num_shapes: int, base_shape: Tuple[int, ...], data: st.DataObject) -> None:
    shapes, result = data.draw(
        nps.mutually_broadcastable_shapes(
            num_shapes=num_shapes,
            base_shape=base_shape
        ),
        label='shapes, result'
    )
    assert len(shapes) == num_shapes
    assert result == func_93ws1oxx(base_shape, *shapes)


@settings(deadline=None)
@given(min_dims=st.integers(0, 32), shape=Tuple[int, ...], data=st.DataObject)
def func_cwlftpfk(
    min_dims: int,
    shape: Tuple[int, ...],
    data: st.DataObject
) -> None:
    max_dims: int = data.draw(
        st.none() | st.integers(min_dims, 32),
        label='max_dims'
    )
    min_side, max_side = func_qwugz31i(data, shape, max_dims)
    broadcastable_shape: Tuple[int, ...] = data.draw(
        nps.broadcastable_shapes(
            shape,
            min_side=min_side,
            max_side=max_side,
            min_dims=min_dims,
            max_dims=max_dims
        ),
        label='broadcastable_shapes'
    )
    func_93ws1oxx(shape, broadcastable_shape)


@settings(deadline=None)
@given(
    num_shapes=st.integers(1, 10),
    min_dims=st.integers(0, 32),
    min_side=st.integers(0, 3),
    dtype=st.one_of(nps.unsigned_integer_dtypes(), nps.integer_dtypes()),
    data=st.DataObject
)
def func_p75fpe7k(
    shape: Tuple[int, ...],
    min_dims: int,
    min_side: int,
    dtype: np.dtype,
    data: st.DataObject
) -> None:
    index: tuple = data.draw(nps.integer_array_indices(shape, dtype=dtype))
    x: np.ndarray = np.zeros(shape)
    out: np.ndarray = x[index]
    assert not np.shares_memory(x, out)
    assert all(dtype == x.dtype for x in index)


@settings(deadline=None)
@given(
    shape=nps.array_shapes(min_dims=1, max_dims=2, min_side=1, max_side=3),
    data=st.DataObject
)
def func_rvev2mre(shape: Tuple[int, ...], data: st.DataObject) -> None:
    x: np.ndarray = np.arange(np.prod(shape)).reshape(shape)
    target_array: np.ndarray = data.draw(
        nps.arrays(
            shape=nps.array_shapes(min_dims=1, max_dims=2, min_side=1, max_side=2),
            elements=st.sampled_from(x.flatten()),
            dtype=x.dtype
        ),
        label='target'
    )

    def func_qkoq4coo(index: Any) -> bool:
        selected: np.ndarray = x[index]
        target(len(set(selected.flatten())), label='unique indices')
        target(float(np.sum(target_array == selected)), label='elements correct')
        return np.all(target_array == selected)

    minimal(
        nps.integer_array_indices(
            shape,
            result_shape=st.just(target_array.shape)
        ),
        func_qkoq4coo,
        settings=settings(
            max_examples=10 ** 6,
            phases=[Phase.generate, Phase.target]
        )
    )


@pytest.mark.parametrize('condition', [
    lambda ix: isinstance(ix, tuple) and Ellipsis in ix,
    lambda ix: isinstance(ix, tuple) and Ellipsis not in ix,
    lambda ix: isinstance(ix, tuple) and np.newaxis in ix,
    lambda ix: isinstance(ix, tuple) and np.newaxis not in ix,
    lambda ix: ix is Ellipsis,
    lambda ix: ix == np.newaxis
])
def func_eygkuz38(condition: Callable[[Any], bool]) -> None:
    indexers: st.SearchStrategy[Any] = nps.array_shapes(min_dims=0, max_dims=32).flatmap(
        lambda shape: nps.basic_indices(shape, allow_newaxis=True)
    )
    find_any(indexers, condition)


def func_4fvfa1gs() -> None:
    find_any(
        nps.basic_indices(shape=(0, 0), allow_ellipsis=True),
        lambda ix: ix == ()
    )


def func_pz5cvjqp() -> None:
    find_any(
        nps.basic_indices(shape=(0, 0), allow_ellipsis=True),
        lambda ix: not isinstance(ix, tuple)
    )


def func_0cn5iqfh() -> None:
    find_any(
        nps.basic_indices(shape=(1, 0, 0, 0, 1), allow_ellipsis=True),
        lambda ix: len(ix) == 3 and ix[1] == Ellipsis
    )


@given(
    nps.basic_indices(shape=(0, 0, 0, 0, 0), allow_ellipsis=True).filter(
        lambda idx: isinstance(idx, tuple) and Ellipsis in idx
    )
)
def func_hq8gfbnv(idx: Any) -> None:
    assert slice(None) not in idx


def func_tdifqpx5() -> None:
    find_any(
        nps.basic_indices(shape=(3, 3, 3)),
        lambda ix: (
            not isinstance(ix, tuple) and ix != Ellipsis or
            isinstance(ix, tuple) and Ellipsis not in ix and len(ix) < 3
        ),
        settings=settings(max_examples=5000)
    )


@given(
    shape=nps.array_shapes(min_dims=0, max_side=4) | 
          nps.array_shapes(min_dims=0, min_side=0, max_side=10),
    allow_newaxis=st.booleans(),
    allow_ellipsis=st.booleans(),
    data=st.DataObject
)
def func_w21xatql(
    shape: Tuple[int, ...],
    allow_newaxis: bool,
    allow_ellipsis: bool,
    data: st.DataObject
) -> None:
    min_dims: int = data.draw(
        st.integers(0, 5 if allow_newaxis else len(shape)),
        label='min_dims'
    )
    max_dims: int = data.draw(
        st.none() | st.integers(min_dims, 32 if allow_newaxis else len(shape)),
        label='max_dims'
    )
    indexer: Any = data.draw(
        nps.basic_indices(
            shape,
            min_dims=min_dims,
            max_dims=max_dims,
            allow_ellipsis=allow_ellipsis,
            allow_newaxis=allow_newaxis
        ),
        label='indexer'
    )
    if not allow_newaxis:
        if isinstance(indexer, tuple):
            assert 0 <= len(indexer) <= len(shape) + int(allow_ellipsis)
        else:
            assert 1 <= len(shape) + int(allow_ellipsis)
        assert np.newaxis not in shape
    if not allow_ellipsis:
        assert Ellipsis not in shape
    if 0 in shape:
        array: np.ndarray = np.zeros(shape)
        assert array.size == 0
    elif np.prod(shape) <= 10 ** 5:
        array = np.arange(np.prod(shape)).reshape(shape)
    else:
        assume(False)
    view: Any = array[indexer]
    if not np.isscalar(view):
        assert min_dims <= view.ndim <= (32 if max_dims is None else max_dims)
        if view.size:
            assert np.shares_memory(view, array)


@given(
    num_shapes=st.integers(1, 4),
    base_shape=ANY_SHAPE,
    data=st.DataObject
)
def func_g0o9mihi(num_shapes: int, base_shape: Tuple[int, ...], data: st.DataObject) -> None:
    min_dims: int = data.draw(
        st.integers(0, 32),
        label='min_dims'
    )
    max_dims: int = data.draw(
        st.one_of(st.none(), st.integers(min_dims, 32)),
        label='max_dims'
    )
    min_side: int = data.draw(
        st.integers(0, 3),
        label='min_side'
    )
    max_side: int = data.draw(
        st.one_of(st.none(), st.integers(min_side, 6)),
        label='max_side'
    )
    try:
        shapes, result = data.draw(
            nps.mutually_broadcastable_shapes(
                num_shapes=num_shapes,
                base_shape=base_shape,
                min_side=min_side,
                max_side=max_side,
                min_dims=min_dims,
                max_dims=max_dims
            ),
            label='shapes, result'
        )
    except InvalidArgument:
        raise UnsatisfiedAssumption from None
    if max_dims is None:
        max_dims = max(len(base_shape), min_dims) + 2
    if max_side is None:
        max_side = max((*base_shape[::-1][:max_dims], min_side)) + 2
    assert isinstance(shapes, tuple)
    assert len(shapes) == num_shapes
    assert result == func_93ws1oxx(base_shape, *shapes)


def func_qwugz31i(
    data: st.DataObject,
    shape: Tuple[int, ...],
    max_dims: int,
    *,
    permit_none: bool = True
) -> Tuple[int, int | None]:
    if max_dims == 0 or not shape:
        return 0, None
    smallest_side: int = min(shape[::-1][:max_dims])
    min_strat: st.SearchStrategy[int] = st.sampled_from([1, smallest_side]) if smallest_side > 1 else st.just(smallest_side)
    min_side: int = data.draw(min_strat, label='min_side')
    largest_side: int = max(max(shape[::-1][:max_dims]), min_side)
    if permit_none:
        max_strat: st.SearchStrategy[int | None] = st.one_of(
            st.none(),
            st.integers(largest_side, largest_side + 2)
        )
    else:
        max_strat = st.integers(largest_side, largest_side + 2)
    max_side: int | None = data.draw(max_strat, label='max_side')
    return min_side, max_side


def func_uetbyeta(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> Tuple[int, ...]:
    result: list[int] = []
    for a, b in zip_longest(reversed(shape_a), reversed(shape_b), fillvalue=1):
        if a != b and a != 1 and b != 1:
            raise ValueError(
                f'shapes {shape_a!r} and {shape_b!r} are not broadcast-compatible'
            )
        result.append(a if a != 1 else b)
    return tuple(reversed(result))


def func_93ws1oxx(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """Returns the shape resulting from broadcasting the
    input shapes together.

    Raises ValueError if the shapes are not broadcast-compatible"""
    assert shapes, 'Must pass >=1 shapes to broadcast'
    return reduce(func_uetbyeta, shapes, ())


@settings(deadline=None, max_examples=500)
@given(shapes=st.lists(
    nps.array_shapes(
        min_dims=0,
        min_side=0,
        max_dims=4,
        max_side=4
    ),
    min_size=1
))
def func_zhzx43jx(shapes: list[Tuple[int, ...]]) -> None:
    """Ensures that `_broadcast_shapes` raises when fed incompatible shapes,
    and ensures that it produces the true broadcasted shape"""
    if len(shapes) == 1:
        assert func_93ws1oxx(*shapes) == shapes[0]
        return
    arrs: list[np.ndarray] = [np.zeros(s, dtype=np.uint8) for s in shapes]
    try:
        broadcast_out: list[np.ndarray] = np.broadcast_arrays(*arrs)
    except ValueError:
        with pytest.raises(ValueError):
            func_93ws1oxx(*shapes)
        return
    broadcasted_shape: Tuple[int, ...] = func_93ws1oxx(*shapes)
    assert broadcast_out[0].shape == broadcasted_shape


@settings(deadline=None, max_examples=200)
@given(shape=st.SearchStrategy[Tuple[int, ...]], data=st.DataObject)
def func_vokyre7q(shape: Tuple[int, ...], data: st.DataObject) -> None:
    broadcastable_shape: Tuple[int, ...] = data.draw(
        nps.broadcastable_shapes(shape),
        label='broadcastable_shapes'
    )
    func_93ws1oxx(shape, broadcastable_shape)


@settings(deadline=None, max_examples=200)
@given(base_shape=Tuple[int, ...], num_shapes=st.integers(1, 10), data=st.DataObject)
def func_7xd3w6yf(num_shapes: int, base_shape: Tuple[int, ...], data: st.DataObject) -> None:
    shapes, result = data.draw(
        nps.mutually_broadcastable_shapes(
            num_shapes=num_shapes,
            base_shape=base_shape
        ),
        label='shapes, result'
    )
    assert len(shapes) == num_shapes
    assert result == func_93ws1oxx(base_shape, *shapes)


@settings(deadline=None)
@given(min_dims=st.integers(0, 32), shape=Tuple[int, ...], data=st.DataObject)
def func_cwlftpfk(
    min_dims: int,
    shape: Tuple[int, ...],
    data: st.DataObject
) -> None:
    max_dims: int = data.draw(
        st.none() | st.integers(min_dims, 32),
        label='max_dims'
    )
    min_side, max_side = func_qwugz31i(data, shape, max_dims)
    broadcastable_shape: Tuple[int, ...] = data.draw(
        nps.broadcastable_shapes(
            shape,
            min_side=min_side,
            max_side=max_side,
            min_dims=min_dims,
            max_dims=max_dims
        ),
        label='broadcastable_shapes'
    )
    func_93ws1oxx(shape, broadcastable_shape)
