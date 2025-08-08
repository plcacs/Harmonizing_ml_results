    def __new__(cls, data: Any = None, categories: Any = None, ordered: Any = None, dtype: Any = None, copy: bool = False, name: Any = None) -> CategoricalIndex:
    def _is_dtype_compat(self, other: Index) -> Categorical:
    def equals(self, other: Any) -> bool:
    def reindex(self, target: Any, method: Any = None, level: Any = None, limit: Any = None, tolerance: Any = None) -> Tuple[pd.Index, Optional[np.ndarray[np.intp]]]:
    def _maybe_cast_indexer(self, key: Any) -> Any:
    def _maybe_cast_listlike_indexer(self, values: Any) -> CategoricalIndex:
    def _is_comparable_dtype(self, dtype: Any) -> bool:
    def map(self, mapper: Union[Callable, Dict, Series], na_action: Optional[Literal['ignore']] = None) -> Union[CategoricalIndex, pd.Index]:
    def _concat(self, to_concat: List[Any], name: Any) -> Union[CategoricalIndex, pd.Index]:
