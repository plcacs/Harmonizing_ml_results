import math
import pytest
from hypothesis.extra.array_api import find_castable_builtin_for_dtype
from hypothesis.internal.floats import width_smallest_normals
from tests.array_api.common import dtype_name_params, flushes_to_zero
from tests.common.debug import assert_all_examples, assert_no_examples, find_any, minimal
from typing import Any, Callable, Dict, Tuple

@pytest.mark.parametrize('dtype_name', dtype_name_params)
def test_strategies_have_reusable_values(
    xp: Any, 
    xps: Any, 
    dtype_name: str
) -> None:
    """Inferred strategies have reusable values."""
    strat = xps.from_dtype(dtype_name)
    assert strat.has_reusable_values

@pytest.mark.parametrize('dtype_name', dtype_name_params)
def test_produces_castable_instances_from_dtype(
    xp: Any, 
    xps: Any, 
    dtype_name: str
) -> None:
    """Strategies inferred by dtype generate values of a builtin type castable
    to the dtype."""
    dtype = getattr(xp, dtype_name)
    builtin = find_castable_builtin_for_dtype(xp, xps.api_version, dtype)
    assert_all_examples(xps.from_dtype(dtype), lambda v: isinstance(v, builtin))

@pytest.mark.parametrize('dtype_name', dtype_name_params)
def test_produces_castable_instances_from_name(
    xp: Any, 
    xps: Any, 
    dtype_name: str
) -> None:
    """Strategies inferred by dtype name generate values of a builtin type
    castable to the dtype."""
    dtype = getattr(xp, dtype_name)
    builtin = find_castable_builtin_for_dtype(xp, xps.api_version, dtype)
    assert_all_examples(xps.from_dtype(dtype_name), lambda v: isinstance(v, builtin))

@pytest.mark.parametrize('dtype_name', dtype_name_params)
def test_passing_inferred_strategies_in_arrays(
    xp: Any, 
    xps: Any, 
    dtype_name: str
) -> None:
    """Inferred strategies usable in arrays strategy."""
    elements = xps.from_dtype(dtype_name)
    find_any(xps.arrays(dtype_name, 10, elements=elements))

@pytest.mark.parametrize(
    'dtype, kwargs, predicate', 
    [
        ('float32', {'min_value': 1, 'max_value': 2}, lambda x: 1 <= x <= 2),
        ('float32', {'min_value': 1, 'max_value': 2, 'exclude_min': True, 'exclude_max': True}, lambda x: 1 < x < 2),
        ('float32', {'allow_nan': False}, lambda x: not math.isnan(x)),
        ('float32', {'allow_infinity': False}, lambda x: not math.isinf(x)),
        ('float32', {'allow_nan': False, 'allow_infinity': False}, math.isfinite),
        ('int8', {'min_value': -1, 'max_value': 1}, lambda x: -1 <= x <= 1),
        ('uint8', {'min_value': 1, 'max_value': 2}, lambda x: 1 <= x <= 2)
    ]
)
def test_from_dtype_with_kwargs(
    xp: Any, 
    xps: Any, 
    dtype: str, 
    kwargs: Dict[str, Any], 
    predicate: Callable[[Any], bool]
) -> None:
    """Strategies inferred with kwargs generate values in bounds."""
    strat = xps.from_dtype(dtype, **kwargs)
    assert_all_examples(strat, predicate)

def test_can_minimize_floats(xp: Any, xps: Any) -> None:
    """Inferred float strategy minimizes to a good example."""
    smallest = minimal(xps.from_dtype(xp.float32), lambda n: n >= 1.0)
    assert smallest in {1, math.inf}

smallest_normal: float = width_smallest_normals[32]

@pytest.mark.parametrize(
    'kwargs', 
    [
        {}, 
        {'min_value': -1}, 
        {'max_value': 1}, 
        {'min_value': -1, 'max_value': 1}
    ]
)
def test_subnormal_generation(
    xp: Any, 
    xps: Any, 
    kwargs: Dict[str, Any]
) -> None:
    """Generation of subnormals is dependent on FTZ behaviour of array module."""
    strat = xps.from_dtype(xp.float32, **kwargs).filter(lambda n: n != 0)
    if flushes_to_zero(xp, width=32):
        assert_no_examples(strat, lambda n: -smallest_normal < n < smallest_normal)
    else:
        find_any(strat, lambda n: -smallest_normal < n < smallest_normal)
