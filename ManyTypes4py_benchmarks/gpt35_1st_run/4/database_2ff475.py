from snowflake.connector.cursor import SnowflakeCursor

class SnowflakeConnector(DatabaseBlock):
    credentials: SnowflakeCredentials
    database: str
    warehouse: str
    schema_: str
    fetch_size: int
    poll_frequency_s: int
    _connection: Optional[SnowflakeConnection] = None
    _unique_cursors: Optional[Dict[str, SnowflakeCursor]] = None

    def get_connection(self, **connect_kwargs) -> SnowflakeConnection:
        ...

    def _start_connection(self) -> None:
        ...

    def _get_cursor(self, inputs, cursor_type=Type[SnowflakeCursor]) -> Tuple[bool, SnowflakeCursor]:
        ...

    async def _execute_async(self, cursor, inputs) -> None:
        ...

    def reset_cursors(self) -> None:
        ...

    def fetch_one(self, operation, parameters=None, cursor_type=Type[SnowflakeCursor], **execute_kwargs) -> Tuple:
        ...

    async def fetch_one_async(self, operation, parameters=None, cursor_type=Type[SnowflakeCursor], **execute_kwargs) -> Tuple:
        ...

    def fetch_many(self, operation, parameters=None, size=None, cursor_type=Type[SnowflakeCursor], **execute_kwargs) -> List[Tuple]:
        ...

    async def fetch_many_async(self, operation, parameters=None, size=None, cursor_type=Type[SnowflakeCursor], **execute_kwargs) -> List[Tuple]:
        ...

    def fetch_all(self, operation, parameters=None, cursor_type=Type[SnowflakeCursor], **execute_kwargs) -> List[Tuple]:
        ...

    async def fetch_all_async(self, operation, parameters=None, cursor_type=Type[SnowflakeCursor], **execute_kwargs) -> List[Tuple]:
        ...

    def execute(self, operation, parameters=None, cursor_type=Type[SnowflakeCursor], **execute_kwargs) -> None:
        ...

    async def execute_async(self, operation, parameters=None, cursor_type=Type[SnowflakeCursor], **execute_kwargs) -> None:
        ...

    def execute_many(self, operation, seq_of_parameters) -> None:
        ...

    async def execute_many_async(self, operation, seq_of_parameters) -> None:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> 'SnowflakeConnector':
        ...

    def __exit__(self, *args) -> None:
        ...

    def __getstate__(self) -> Dict:
        ...

    def __setstate__(self, data: Dict) -> None:
        ...

@task
def snowflake_query(query: str, snowflake_connector: SnowflakeConnector, params: Optional[Dict] = None, cursor_type: Type[SnowflakeCursor] = SnowflakeCursor, poll_frequency_seconds: int = 1) -> List[Tuple]:
    ...

@task
async def snowflake_query_async(query: str, snowflake_connector: SnowflakeConnector, params: Optional[Dict] = None, cursor_type: Type[SnowflakeCursor] = SnowflakeCursor, poll_frequency_seconds: int = 1) -> List[Tuple]:
    ...

@task
def snowflake_multiquery(queries: List[str], snowflake_connector: SnowflakeConnector, params: Optional[Dict] = None, cursor_type: Type[SnowflakeCursor] = SnowflakeCursor, as_transaction: bool = False, return_transaction_control_results: bool = False, poll_frequency_seconds: int = 1) -> List[List[Tuple]]:
    ...

@task
async def snowflake_multiquery_async(queries: List[str], snowflake_connector: SnowflakeConnector, params: Optional[Dict] = None, cursor_type: Type[SnowflakeCursor] = SnowflakeCursor, as_transaction: bool = False, return_transaction_control_results: bool = False, poll_frequency_seconds: int = 1) -> List[List[Tuple]]:
    ...

@task
def snowflake_query_sync(query: str, snowflake_connector: SnowflakeConnector, params: Optional[Dict] = None, cursor_type: Type[SnowflakeCursor] = SnowflakeCursor) -> List[Tuple]:
    ...
