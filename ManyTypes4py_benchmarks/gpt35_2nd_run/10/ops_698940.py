    def _get_expected_exception(self, op_name: str, obj: pd.Series, other: pd.Series) -> type:
    def _cast_pointwise_result(self, op_name: str, obj: pd.Series, other: pd.Series, pointwise_result: pd.Series) -> pd.Series:
    def check_opname(self, ser: pd.Series, op_name: str, other: pd.Series) -> None:
    def _combine(self, obj: Union[pd.DataFrame, pd.Series], other: pd.Series, op: Callable) -> pd.DataFrame:
    def _check_op(self, ser: pd.Series, op: Callable, other: pd.Series, op_name: str, exc: Optional[type] = NotImplementedError) -> None:
    def _check_divmod_op(self, ser: pd.Series, op: Callable, other: int) -> None:
    def test_arith_series_with_scalar(self, data: pd.Series, all_arithmetic_operators: str) -> None:
    def test_arith_frame_with_scalar(self, data: pd.Series, all_arithmetic_operators: str) -> None:
    def test_arith_series_with_array(self, data: pd.Series, all_arithmetic_operators: str) -> None:
    def test_divmod(self, data: pd.Series) -> None:
    def test_divmod_series_array(self, data: pd.Series, data_for_twos: pd.Series) -> None:
    def test_add_series_with_extension_array(self, data: pd.Series) -> None:
    def test_direct_arith_with_ndframe_returns_not_implemented(self, data: pd.Series, box: Union[pd.Series, pd.DataFrame, pd.Index], op_name: str) -> None:
    def _compare_other(self, ser: pd.Series, data: pd.Series, op: Callable, other: Union[int, pd.Series]) -> None:
    def test_compare_scalar(self, data: pd.Series, comparison_op: Callable) -> None:
    def test_compare_array(self, data: pd.Series, comparison_op: Callable) -> None:
    def test_invert(self, data: pd.Series) -> None:
    def test_unary_ufunc_dunder_equivalence(self, data: pd.Series, ufunc: Callable) -> None:
