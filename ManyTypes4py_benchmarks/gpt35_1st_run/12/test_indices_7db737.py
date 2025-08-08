from typing import Callable

def test_generate_optional_indices(xp, xps, condition: Callable[[tuple], bool]):
def test_cannot_generate_newaxis_when_disabled(xp, xps):
def test_generate_indices_for_0d_shape(xp, xps):
def test_generate_tuples_and_non_tuples_for_1d_shape(xp, xps):
def test_generate_long_ellipsis(xp, xps):
def test_indices_replaces_whole_axis_slices_with_ellipsis(xp, xps):
def test_efficiently_generate_indexers(xp, xps):
def test_generate_valid_indices(xp, xps, allow_newaxis: bool, allow_ellipsis: bool, data):
