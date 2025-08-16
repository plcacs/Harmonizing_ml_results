def test_convert_dttm(target_type: str, expected_result: Optional[str], dttm: datetime) -> None:

def test_epoch_to_dttm(dttm: datetime) -> None:

def test_get_column_spec(native_type: str, sqla_type: Any, attrs: Any, generic_type: GenericDataType, is_dttm: bool) -> None:

def test_get_schema_from_engine_params() -> None:

def test_get_default_catalog() -> None:

def test_timegrain_expressions(time_grain: Optional[str], expected_result: str) -> None:
