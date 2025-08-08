    def test_unique(self, data: List[Union[int, str]], categories: List[Union[int, str]], expected_data: List[Union[int, str]], ordered: bool):
    def test_drop_duplicates(self, data: List[Union[int, str]], categories: List[Union[int, str]], expected: Dict[str, np.ndarray]):
    def test_isin(self):
    def test_isin_overlapping_intervals(self):
    def test_view_i8(self):
    def test_engine_type(self, dtype: np.dtype, engine_type: Type[libindex.CategoricalEngine]):
    def test_disallow_addsub_ops(self, func: Callable, op_name: str):
