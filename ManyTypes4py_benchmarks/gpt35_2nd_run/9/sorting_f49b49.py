def get_indexer_indexer(target: Index, level: int | str | list[int] | list[str], ascending: bool | list[bool] = True, kind: str = 'quicksort', na_position: str = 'last', sort_remaining: bool, key: Callable = None) -> Optional[np.ndarray[np.intp]]:
def get_group_index(labels: Sequence[ArrayLike], shape: Shape, sort: bool, xnull: bool) -> np.ndarray[np.int64]:
def get_compressed_ids(labels: list[np.ndarray], sizes: Shape) -> Tuple[np.ndarray[np.intp], np.ndarray[np.int64]]:
def is_int64_overflow_possible(shape: Shape) -> bool:
def _decons_group_index(comp_labels: np.ndarray[np.intp], shape: Shape) -> list[np.ndarray[np.int64]]:
def decons_obs_group_ids(comp_ids: np.ndarray[np.intp], obs_ids: np.ndarray[np.intp], shape: Shape, labels: Sequence[np.ndarray[np.signedinteger]], xnull: bool) -> list[np.ndarray[np.intp]]:
def lexsort_indexer(keys: Sequence[ArrayLike | Index | Series], orders: bool | list[bool] | None = None, na_position: str = 'last', key: Callable = None, codes_given: bool = False) -> np.ndarray[np.intp]:
def nargsort(items: np.ndarray | ExtensionArray | Index | Series, kind: str = 'quicksort', ascending: bool = True, na_position: str = 'last', key: Callable = None, mask: np.ndarray[bool] = None) -> np.ndarray[np.intp]:
def nargminmax(values: ExtensionArray, method: str, axis: int = 0) -> int:
def _ensure_key_mapped_multiindex(index: MultiIndex, key: Callable, level: list[int | str] | None) -> MultiIndex:
def ensure_key_mapped(values: Series | DataFrame | Index, key: Callable, levels: list | None) -> Series | DataFrame | Index:
def get_indexer_dict(label_list: list[np.ndarray], keys: list) -> dict:
def get_group_index_sorter(group_index: np.ndarray[np.intp], ngroups: int | None = None) -> np.ndarray[np.intp]:
def compress_group_index(group_index: np.ndarray[np.intp], sort: bool = True) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]:
def _reorder_by_uniques(uniques: np.ndarray[np.int64], labels: np.ndarray[np.intp]) -> Tuple[np.ndarray[np.int64], np.ndarray[np.intp]]:
