from typing import List, Dict, Union, Any

def test_multifunc_numba_udf_frame(agg_kwargs: Dict[str, Union[str, List[Callable[[Any, Any], Any]]]], expected_func: Union[str, List[str]]):
def test_multifunc_numba_kwarg_propagation(data: Union[Series, DataFrame], agg_kwargs: Dict[str, Any]):
def test_multiindex_one_key(nogil: bool, parallel: bool, nopython: bool):
def test_multiindex_multi_key_not_supported(nogil: bool, parallel: bool, nopython: bool):
def test_multilabel_numba_vs_cython(numba_supported_reductions: Tuple[str, Dict[str, Any]]):
