def test_convert_dttm(target_type: str, dttm: datetime, expected_result: Optional[str]):
def test_get_column_spec(native_type: str, sqla_type: Any, attrs: Optional[dict], generic_type: GenericDataType, is_dttm: bool):
def test_where_latest_partition(mock_latest_partition: mock.MagicMock, column_type: str, column_value: Any, expected_value: str):
def test_adjust_engine_params_fully_qualified():
def test_adjust_engine_params_catalog_only():
def test_get_default_catalog():
def test_timegrain_expressions(time_grain: str, expected_result: str):
