    tuples: List[List[Union[int, str]]
    multi_index: pd.MultiIndex
    datetime_index: pd.DatetimeIndex
    timedelta_index: pd.TimedeltaIndex
    period_index: pd.PeriodIndex
    categorical: pd.Categorical
    categorical_df: pd.DataFrame
    df: pd.DataFrame
    sparse_array: pd.arrays.SparseArray
    datetime_array: np.ndarray
    timedelta_array: np.ndarray
    abc_pairs: List[Tuple[str, Any]]
    abc_subclasses: Dict[str, List[str]]

    def test_abc_pairs_instance_check(self, abctype1: str, abctype2: str, inst: Any, _: Any) -> None:
    def test_abc_pairs_subclass_check(self, abctype1: str, abctype2: str, inst: Any, _: Any) -> None:
    def test_abc_hierarchy(self, parent: str, subs: List[str], abctype: str, inst: Any) -> None:
    def test_abc_coverage(self, abctype: str) -> None:

def test_setattr_warnings() -> None:
