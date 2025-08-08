def combine_hash_arrays(arrays: Iterator[np.ndarray], num_items: int) -> np.ndarray:
def hash_pandas_object(obj: Union[Index, Series, DataFrame], index: bool = True, encoding: str = 'utf8', hash_key: str = _default_hash_key, categorize: bool = True) -> Series:
def hash_tuples(vals: Union[MultiIndex, Iterable[Tuple]], encoding: str = 'utf8', hash_key: str = _default_hash_key) -> np.ndarray:
def hash_array(vals: Union[np.ndarray, ExtensionArray], encoding: str = 'utf8', hash_key: str = _default_hash_key, categorize: bool = True) -> np.ndarray:
def _hash_ndarray(vals: np.ndarray, encoding: str = 'utf8', hash_key: str = _default_hash_key, categorize: bool = True) -> np.ndarray:
