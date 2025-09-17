"""Module for querying against Snowflake databases."""
import asyncio
from time import sleep
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
from pydantic import AliasChoices, Field
from snowflake.connector.connection import SnowflakeConnection
from snowflake.connector.cursor import SnowflakeCursor
from prefect import task
from prefect.blocks.abstract import DatabaseBlock
from prefect.utilities.asyncutils import run_coro_as_sync, run_sync_in_worker_thread
from prefect.utilities.hashing import hash_objects
from prefect_snowflake import SnowflakeCredentials

BEGIN_TRANSACTION_STATEMENT: str = 'BEGIN TRANSACTION'
END_TRANSACTION_STATEMENT: str = 'COMMIT'


class SnowflakeConnector(DatabaseBlock):
    """
    Block used to manage connections with Snowflake.

    Upon instantiating, a connection is created and maintained for the life of
    the object until the close method is called.

    It is recommended to use this block as a context manager, which will automatically
    close the engine and its connections when the context is exited.

    It is also recommended that this block is loaded and consumed within a single task
    or flow because if the block is passed across separate tasks and flows,
    the state of the block's connection and cursor will be lost.
    """
    _block_type_name: str = 'Snowflake Connector'
    _logo_url: str = 'https://cdn.sanity.io/images/3ugk85nk/production/bd359de0b4be76c2254bd329fe3a267a1a3879c2-250x250.png'
    _documentation_url: str = 'https://docs.prefect.io/integrations/prefect-snowflake'
    _description: str = 'Perform data operations against a Snowflake database.'
    credentials: Any = Field(default=..., description='The credentials to authenticate with Snowflake.')
    database: Any = Field(default=..., description='The name of the default database to use.')
    warehouse: Any = Field(default=..., description='The name of the default warehouse to use.')
    schema_: Any = Field(
        default=...,
        serialization_alias='schema',
        validation_alias=AliasChoices('schema_', 'schema'),
        description='The name of the default schema to use.'
    )
    fetch_size: int = Field(default=1, description='The default number of rows to fetch at a time.')
    poll_frequency_s: int = Field(
        default=1,
        title='Poll Frequency [seconds]',
        description='The number of seconds between checking query status for long running queries.'
    )
    _connection: Optional[SnowflakeConnection] = None
    _unique_cursors: Optional[Dict[Any, SnowflakeCursor]] = None

    def get_connection(self, **connect_kwargs: Any) -> SnowflakeConnection:
        """
        Returns an authenticated connection that can be
        used to query from Snowflake databases.
        """
        if self._connection is not None:
            return self._connection
        connect_params: Dict[str, Any] = {'database': self.database, 'warehouse': self.warehouse, 'schema': self.schema_}
        connection: SnowflakeConnection = self.credentials.get_client(**connect_kwargs, **connect_params)
        self._connection = connection
        self.logger.info('Started a new connection to Snowflake.')
        return connection

    def _start_connection(self) -> None:
        """
        Starts Snowflake database connection.
        """
        self.get_connection()
        if self._unique_cursors is None:
            self._unique_cursors = {}

    def _get_cursor(
        self,
        inputs: Any,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor
    ) -> Tuple[bool, SnowflakeCursor]:
        """
        Get a Snowflake cursor.
        """
        self._start_connection()
        input_hash: Any = hash_objects(inputs)
        if input_hash is None:
            raise RuntimeError(
                f'We were not able to hash your inputs, {inputs!r}, which resulted in an unexpected data return; please open an issue with a reproducible example.'
            )
        if input_hash not in self._unique_cursors.keys():
            new_cursor: SnowflakeCursor = self._connection.cursor(cursor_type)
            self._unique_cursors[input_hash] = new_cursor
            return (True, new_cursor)
        else:
            existing_cursor: SnowflakeCursor = self._unique_cursors[input_hash]
            return (False, existing_cursor)

    async def _execute_async(self, cursor: SnowflakeCursor, inputs: Dict[str, Any]) -> None:
        """Helper method to execute operations asynchronously."""
        response: Dict[str, Any] = await run_sync_in_worker_thread(cursor.execute_async, **inputs)
        self.logger.info(
            f"Executing the operation, {inputs['command']!r}, asynchronously; polling for the result every {self.poll_frequency_s} seconds."
        )
        query_id: str = response['queryId']
        while self._connection.is_still_running(
            await run_sync_in_worker_thread(self._connection.get_query_status_throw_if_error, query_id)
        ):
            await asyncio.sleep(self.poll_frequency_s)
        await run_sync_in_worker_thread(cursor.get_results_from_sfqid, query_id)

    def reset_cursors(self) -> None:
        """
        Tries to close all opened cursors.
        """
        if not self._unique_cursors:
            self.logger.info('There were no cursors to reset.')
            return
        input_hashes: Tuple[Any, ...] = tuple(self._unique_cursors.keys())
        for input_hash in input_hashes:
            cursor: SnowflakeCursor = self._unique_cursors.pop(input_hash)
            try:
                cursor.close()
            except Exception as exc:
                self.logger.warning(f'Failed to close cursor for input hash {input_hash!r}: {exc}')
        self.logger.info('Successfully reset the cursors.')

    def fetch_one(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any
    ) -> Optional[Any]:
        """
        Fetch a single result from the database.
        """
        inputs: Dict[str, Any] = dict(command=operation, params=parameters, **execute_kwargs)
        new, cursor = self._get_cursor(inputs, cursor_type=cursor_type)
        if new:
            cursor.execute(operation, params=parameters, **execute_kwargs)
        self.logger.debug('Preparing to fetch a row.')
        return cursor.fetchone()

    async def fetch_one_async(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any
    ) -> Optional[Any]:
        """
        Fetch a single result from the database asynchronously.
        """
        inputs: Dict[str, Any] = dict(command=operation, params=parameters, **execute_kwargs)
        new, cursor = self._get_cursor(inputs, cursor_type=cursor_type)
        if new:
            await self._execute_async(cursor, inputs)
        self.logger.debug('Preparing to fetch a row.')
        result: Optional[Any] = await run_sync_in_worker_thread(cursor.fetchone)
        return result

    def fetch_many(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        size: Optional[int] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any
    ) -> List[Any]:
        """
        Fetch a limited number of results from the database.
        """
        inputs: Dict[str, Any] = dict(command=operation, params=parameters, **execute_kwargs)
        new, cursor = self._get_cursor(inputs, cursor_type)
        if new:
            cursor.execute(operation, params=parameters, **execute_kwargs)
        size = size or self.fetch_size
        self.logger.debug(f'Preparing to fetch {size} rows.')
        return cursor.fetchmany(size=size)

    async def fetch_many_async(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        size: Optional[int] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any
    ) -> List[Any]:
        """
        Fetch a limited number of results from the database asynchronously.
        """
        inputs: Dict[str, Any] = dict(command=operation, params=parameters, **execute_kwargs)
        new, cursor = self._get_cursor(inputs, cursor_type)
        if new:
            await self._execute_async(cursor, inputs)
        size = size or self.fetch_size
        self.logger.debug(f'Preparing to fetch {size} rows.')
        result: List[Any] = await run_sync_in_worker_thread(cursor.fetchmany, size=size)
        return result

    def fetch_all(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any
    ) -> List[Any]:
        """
        Fetch all results from the database.
        """
        inputs: Dict[str, Any] = dict(command=operation, params=parameters, **execute_kwargs)
        new, cursor = self._get_cursor(inputs, cursor_type)
        if new:
            cursor.execute(operation, params=parameters, **execute_kwargs)
        self.logger.debug('Preparing to fetch all rows.')
        return cursor.fetchall()

    async def fetch_all_async(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any
    ) -> List[Any]:
        """
        Fetch all results from the database asynchronously.
        """
        inputs: Dict[str, Any] = dict(command=operation, params=parameters, **execute_kwargs)
        new, cursor = self._get_cursor(inputs, cursor_type)
        if new:
            await self._execute_async(cursor, inputs)
        self.logger.debug('Preparing to fetch all rows.')
        result: List[Any] = await run_sync_in_worker_thread(cursor.fetchall)
        return result

    def execute(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any
    ) -> None:
        """
        Executes an operation on the database.
        """
        self._start_connection()
        inputs: Dict[str, Any] = dict(command=operation, params=parameters, **execute_kwargs)
        with self._connection.cursor(cursor_type) as cursor:
            run_coro_as_sync(self._execute_async(cursor, inputs))
        self.logger.info(f'Executed the operation, {operation!r}.')

    async def execute_async(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any
    ) -> None:
        """
        Executes an operation on the database asynchronously.
        """
        self._start_connection()
        inputs: Dict[str, Any] = dict(command=operation, params=parameters, **execute_kwargs)
        with self._connection.cursor(cursor_type) as cursor:
            await run_sync_in_worker_thread(cursor.execute, **inputs)
        self.logger.info(f'Executed the operation, {operation!r}.')

    def execute_many(self, operation: str, seq_of_parameters: Sequence[Dict[str, Any]]) -> None:
        """
        Executes many operations on the database.
        """
        self._start_connection()
        inputs: Dict[str, Any] = dict(command=operation, seqparams=seq_of_parameters)
        with self._connection.cursor() as cursor:
            cursor.executemany(**inputs)
        self.logger.info(f'Executed {len(seq_of_parameters)} operations off {operation!r}.')

    async def execute_many_async(self, operation: str, seq_of_parameters: Sequence[Dict[str, Any]]) -> None:
        """
        Executes many operations on the database asynchronously.
        """
        self._start_connection()
        inputs: Dict[str, Any] = dict(command=operation, seqparams=seq_of_parameters)
        with self._connection.cursor() as cursor:
            await run_sync_in_worker_thread(cursor.executemany, **inputs)
        self.logger.info(f'Executed {len(seq_of_parameters)} operations off {operation!r}.')

    def close(self) -> None:
        """
        Closes connection and its cursors.
        """
        try:
            self.reset_cursors()
        finally:
            if self._connection is None:
                self.logger.info('There was no connection open to be closed.')
                return
            self._connection.close()
            self._connection = None
            self.logger.info('Successfully closed the Snowflake connection.')

    def __enter__(self) -> "SnowflakeConnector":
        """
        Start a connection upon entry.
        """
        return self

    def __exit__(self, *args: Any) -> None:
        """
        Closes connection and its cursors upon exit.
        """
        self.close()

    def __getstate__(self) -> Dict[str, Any]:
        """Allows block to be pickled and dumped."""
        data: Dict[str, Any] = self.__dict__.copy()
        data.update({k: None for k in {'_connection', '_unique_cursors'}})
        return data

    def __setstate__(self, data: Dict[str, Any]) -> None:
        """Reset connection and cursors upon loading."""
        self.__dict__.update(data)
        self._start_connection()


@task
def snowflake_query(
    query: str,
    snowflake_connector: SnowflakeConnector,
    params: Optional[Dict[str, Any]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
    poll_frequency_seconds: int = 1
) -> List[Any]:
    """
    Executes a query against a Snowflake database.
    """
    with snowflake_connector.get_connection() as connection:
        with connection.cursor(cursor_type) as cursor:
            response: Dict[str, Any] = cursor.execute_async(query, params=params)
            query_id: str = response['queryId']
            while connection.is_still_running(connection.get_query_status_throw_if_error(query_id)):
                sleep(poll_frequency_seconds)
            cursor.get_results_from_sfqid(query_id)
            result: List[Any] = cursor.fetchall()
    return result


@task
async def snowflake_query_async(
    query: str,
    snowflake_connector: SnowflakeConnector,
    params: Optional[Dict[str, Any]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
    poll_frequency_seconds: int = 1
) -> List[Any]:
    """
    Executes a query against a Snowflake database asynchronously.
    """
    with snowflake_connector.get_connection() as connection:
        with connection.cursor(cursor_type) as cursor:
            response: Dict[str, Any] = cursor.execute_async(query, params=params)
            query_id: str = response['queryId']
            while connection.is_still_running(connection.get_query_status_throw_if_error(query_id)):
                await asyncio.sleep(poll_frequency_seconds)
            cursor.get_results_from_sfqid(query_id)
            result: List[Any] = cursor.fetchall()
    return result


@task
def snowflake_multiquery(
    queries: List[str],
    snowflake_connector: SnowflakeConnector,
    params: Optional[Dict[str, Any]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
    as_transaction: bool = False,
    return_transaction_control_results: bool = False,
    poll_frequency_seconds: int = 1
) -> List[Any]:
    """
    Executes multiple queries against a Snowflake database in a shared session.
    """
    with snowflake_connector.get_connection() as connection:
        if as_transaction:
            queries.insert(0, BEGIN_TRANSACTION_STATEMENT)
            queries.append(END_TRANSACTION_STATEMENT)
        with connection.cursor(cursor_type) as cursor:
            results: List[Any] = []
            for query in queries:
                response: Dict[str, Any] = cursor.execute_async(query, params=params)
                query_id: str = response['queryId']
                while connection.is_still_running(connection.get_query_status_throw_if_error(query_id)):
                    sleep(poll_frequency_seconds)
                cursor.get_results_from_sfqid(query_id)
                result: List[Any] = cursor.fetchall()
                results.append(result)
    if as_transaction and (not return_transaction_control_results):
        return results[1:-1]
    else:
        return results


@task
async def snowflake_multiquery_async(
    queries: List[str],
    snowflake_connector: SnowflakeConnector,
    params: Optional[Dict[str, Any]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
    as_transaction: bool = False,
    return_transaction_control_results: bool = False,
    poll_frequency_seconds: int = 1
) -> List[Any]:
    """
    Executes multiple queries against a Snowflake database in a shared session asynchronously.
    """
    with snowflake_connector.get_connection() as connection:
        if as_transaction:
            queries.insert(0, BEGIN_TRANSACTION_STATEMENT)
            queries.append(END_TRANSACTION_STATEMENT)
        with connection.cursor(cursor_type) as cursor:
            results: List[Any] = []
            for query in queries:
                response: Dict[str, Any] = cursor.execute_async(query, params=params)
                query_id: str = response['queryId']
                while connection.is_still_running(connection.get_query_status_throw_if_error(query_id)):
                    await asyncio.sleep(poll_frequency_seconds)
                cursor.get_results_from_sfqid(query_id)
                result: List[Any] = cursor.fetchall()
                results.append(result)
    if as_transaction and (not return_transaction_control_results):
        return results[1:-1]
    else:
        return results


@task
def snowflake_query_sync(
    query: str,
    snowflake_connector: SnowflakeConnector,
    params: Optional[Dict[str, Any]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor
) -> List[Any]:
    """
    Executes a query in sync mode against a Snowflake database.
    """
    with snowflake_connector.get_connection() as connection:
        with connection.cursor(cursor_type) as cursor:
            cursor.execute(query, params=params)
            result: List[Any] = cursor.fetchall()
    return result
