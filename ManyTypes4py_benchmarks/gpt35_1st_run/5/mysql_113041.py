from typing import Any, Callable, Optional, Pattern

class MySQLEngineSpec(BasicParametersMixin, BaseEngineSpec):
    engine: str = 'mysql'
    engine_name: str = 'MySQL'
    max_column_name_length: int = 64
    default_driver: str = 'mysqldb'
    sqlalchemy_uri_placeholder: str = 'mysql://user:password@host:port/dbname[?key=value&key=value...]'
    encryption_parameters: dict = {'ssl': '1'}
    supports_dynamic_schema: bool = True
    column_type_mappings: tuple = ((Pattern, types.INTEGER, GenericDataType.NUMERIC), (Pattern, types.TINYINT, GenericDataType.NUMERIC), (Pattern, types.MEDIUMINT, GenericDataType.NUMERIC), (Pattern, types.DECIMAL, GenericDataType.NUMERIC), (Pattern, types.FLOAT, GenericDataType.NUMERIC), (Pattern, types.DOUBLE, GenericDataType.NUMERIC), (Pattern, types.BIT, GenericDataType.NUMERIC), (Pattern, types.TINYTEXT, GenericDataType.STRING), (Pattern, types.MEDIUMTEXT, GenericDataType.STRING), (Pattern, types.LONGTEXT, GenericDataType.STRING))
    column_type_mutators: dict = {types.DECIMAL: Callable[[Any], Any]}
    _time_grain_expressions: dict = {None: str, TimeGrain.SECOND: str, TimeGrain.MINUTE: str, TimeGrain.HOUR: str, TimeGrain.DAY: str, TimeGrain.WEEK: str, TimeGrain.MONTH: str, TimeGrain.QUARTER: str, TimeGrain.YEAR: str, TimeGrain.WEEK_STARTING_MONDAY: str}
    type_code_map: dict = {}
    custom_errors: dict = {Pattern: tuple, Pattern: tuple, Pattern: tuple, Pattern: tuple, Pattern: tuple}
    disallow_uri_query_params: dict = {'mysqldb': set, 'mysqlconnector': set}
    enforce_uri_query_params: dict = {'mysqldb': dict, 'mysqlconnector': dict}

    @classmethod
    def convert_dttm(cls, target_type: Any, dttm: datetime, db_extra: Optional[str] = None) -> Optional[str]:
        ...

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: dict, catalog: Optional[str] = None, schema: Optional[str] = None) -> tuple:
        ...

    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: dict) -> str:
        ...

    @classmethod
    def get_datatype(cls, type_code: Any) -> Optional[str]:
        ...

    @classmethod
    def epoch_to_dttm(cls) -> str:
        ...

    @classmethod
    def _extract_error_message(cls, ex: Exception) -> str:
        ...

    @classmethod
    def get_cancel_query_id(cls, cursor: Any, query: Query) -> int:
        ...

    @classmethod
    def cancel_query(cls, cursor: Any, query: Query, cancel_query_id: int) -> bool:
        ...
