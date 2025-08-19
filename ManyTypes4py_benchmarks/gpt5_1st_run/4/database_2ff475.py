"""Module for querying against Snowflake databases."""
import asyncio
from time import sleep
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
from types import TracebackType

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

    Args:
        credentials: The credentials to authenticate with Snowflake.
        database: The name of the default database to use.
        warehouse: The name of the default warehouse to use.
        schema: The name of the default schema to use;
            this attribute is accessible through `SnowflakeConnector(...).schema_`.
        fetch_size: The number of rows to fetch at a time.
        poll_frequency_s: The number of seconds before checking query.

    Examples:
        Load stored Snowflake connector as a context manager:
        ```python
        from prefect_snowflake.database import SnowflakeConnector

        snowflake_connector = SnowflakeConnector.load("BLOCK_NAME")
        ```

        Insert data into database and fetch results.
        ```python
        from prefect_snowflake.database import SnowflakeConnector

        with SnowflakeConnector.load("BLOCK_NAME") as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS customers (name varchar, address varchar);"
            )
            conn.execute_many(
                "INSERT INTO customers (name, address) VALUES (%(name)s, %(address)s);",
                seq_of_parameters=[
                    {"name": "Ford", "address": "Highway 42"},
                    {"name": "Unknown", "address": "Space"},
                    {"name": "Me", "address": "Myway 88"},
                ],
            )
            results = conn.fetch_all(
                "SELECT * FROM customers WHERE address = %(address)s",
                parameters={"address": "Space"}
            )
            print(results)
        ```
    """
    _block_type_name: str = 'Snowflake Connector'
    _logo_url: str = 'https://cdn.sanity.io/images/3ugk85nk/production/bd359de0b4be76c2254bd329fe3a267a1a3879c2-250x250.png'
    _documentation_url: str = 'https://docs.prefect.io/integrations/prefect-snowflake'
    _description: str = 'Perform data operations against a Snowflake database.'

    credentials: SnowflakeCredentials = Field(default=..., description='The credentials to authenticate with Snowflake.')
    database: str = Field(default=..., description='The name of the default database to use.')
    warehouse: str = Field(default=..., description='The name of the default warehouse to use.')
    schema_: str = Field(
        default=...,
        serialization_alias='schema',
        validation_alias=AliasChoices('schema_', 'schema'),
        description='The name of the default schema to use.',
    )
    fetch_size: int = Field(default=1, description='The default number of rows to fetch at a time.')
    poll_frequency_s: int = Field(
        default=1,
        title='Poll Frequency [seconds]',
        description='The number of seconds between checking query status for long running queries.',
    )
    _connection: Optional[SnowflakeConnection] = None
    _unique_cursors: Optional[Dict[str, SnowflakeCursor]] = None

    def get_connection(self, **connect_kwargs: Any) -> SnowflakeConnection:
        """
        Returns an authenticated connection that can be
        used to query from Snowflake databases.

        Args:
            **connect_kwargs: Additional arguments to pass to
                `snowflake.connector.connect`.

        Returns:
            The authenticated SnowflakeConnection.

        Examples:
            ```python
            from prefect_snowflake.credentials import SnowflakeCredentials
            from prefect_snowflake.database import SnowflakeConnector

            snowflake_credentials = SnowflakeCredentials(
                account="account",
                user="user",
                password="password",
            )
            snowflake_connector = SnowflakeConnector(
                database="database",
                warehouse="warehouse",
                schema="schema",
                credentials=snowflake_credentials
            )
            with snowflake_connector.get_connection() as connection:
                ...
            ```
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
        inputs: Dict[str, Any],
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
    ) -> Tuple[bool, SnowflakeCursor]:
        """
        Get a Snowflake cursor.

        Args:
            inputs: The inputs to generate a unique hash, used to decide
                whether a new cursor should be used.
            cursor_type: The class of the cursor to use when creating a
                Snowflake cursor.

        Returns:
            Whether a cursor is new and a Snowflake cursor.
        """
        self._start_connection()
        input_hash: Optional[str] = hash_objects(inputs)
        if input_hash is None:
            raise RuntimeError(
                f'We were not able to hash your inputs, {inputs!r}, which resulted in an unexpected data return; please open an issue with a reproducible example.'
            )
        if input_hash not in self._unique_cursors.keys():
            assert self._connection is not None  # for type checkers
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
        assert self._connection is not None  # for type checkers
        query_id: str = response['queryId']
        while self._connection.is_still_running(
            await run_sync_in_worker_thread(self._connection.get_query_status_throw_if_error, query_id)
        ):
            await asyncio.sleep(self.poll_frequency_s)
        await run_sync_in_worker_thread(cursor.get_results_from_sfqid, query_id)

    def reset_cursors(self) -> None:
        """
        Tries to close all opened cursors.

        Examples:
            Reset the cursors to refresh cursor position.
            ```python
            from prefect_snowflake.database import SnowflakeConnector

            with SnowflakeConnector.load("BLOCK_NAME") as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS customers (name varchar, address varchar);"
                )
                conn.execute_many(
                    "INSERT INTO customers (name, address) VALUES (%(name)s, %(address)s);",
                    seq_of_parameters=[
                        {"name": "Ford", "address": "Highway 42"},
                        {"name": "Unknown", "address": "Space"},
                        {"name": "Me", "address": "Myway 88"},
                    ],
                )
                print(conn.fetch_one("SELECT * FROM customers"))  # Ford
                conn.reset_cursors()
                print(conn.fetch_one("SELECT * FROM customers"))  # should be Ford again
            ```
        """
        if not self._unique_cursors:
            self.logger.info('There were no cursors to reset.')
            return
        input_hashes: Tuple[str, ...] = tuple(self._unique_cursors.keys())
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
        parameters: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> Optional[Tuple[Any, ...]]:
        """
        Fetch a single result from the database.
        Repeated calls using the same inputs to *any* of the fetch methods of this
        block will skip executing the operation again, and instead,
        return the next set of results from the previous execution,
        until the reset_cursors method is called.

        Args:
            operation: The SQL query or other operation to be executed.
            parameters: The parameters for the operation.
            cursor_type: The class of the cursor to use when creating a Snowflake cursor.
            **execute_kwargs: Additional options to pass to `cursor.execute_async`.

        Returns:
            A tuple containing the data returned by the database,
                where each row is a tuple and each column is a value in the tuple.

        Examples:
            Fetch one row from the database where address is Space.
            ```python
            from prefect_snowflake.database import SnowflakeConnector

            with SnowflakeConnector.load("BLOCK_NAME") as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS customers (name varchar, address varchar);"
                )
                conn.execute_many(
                    "INSERT INTO customers (name, address) VALUES (%(name)s, %(address)s);",
                    seq_of_parameters=[
                        {"name": "Ford", "address": "Highway 42"},
                        {"name": "Unknown", "address": "Space"},
                        {"name": "Me", "address": "Myway 88"},
                    ],
                )
                result = conn.fetch_one(
                    "SELECT * FROM customers WHERE address = %(address)s",
                    parameters={"address": "Space"}
                )
                print(result)
            ```
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
        parameters: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> Optional[Tuple[Any, ...]]:
        """
        Fetch a single result from the database asynchronously.
        Repeated calls using the same inputs to *any* of the fetch methods of this
        block will skip executing the operation again, and instead,
        return the next set of results from the previous execution,
        until the reset_cursors method is called.

        Args:
            operation: The SQL query or other operation to be executed.
            parameters: The parameters for the operation.
            cursor_type: The class of the cursor to use when creating a Snowflake cursor.
            **execute_kwargs: Additional options to pass to `cursor.execute_async`.

        Returns:
            A tuple containing the data returned by the database,
                where each row is a tuple and each column is a value in the tuple.

        Examples:
            Fetch one row from the database where address is Space.
            ```python
            from prefect_snowflake.database import SnowflakeConnector

            with SnowflakeConnector.load("BLOCK_NAME") as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS customers (name varchar, address varchar);"
                )
                conn.execute_many(
                    "INSERT INTO customers (name, address) VALUES (%(name)s, %(address)s);",
                    seq_of_parameters=[
                        {"name": "Ford", "address": "Highway 42"},
                        {"name": "Unknown", "address": "Space"},
                        {"name": "Me", "address": "Myway 88"},
                    ],
                )
                result = await conn.fetch_one_async(
                    "SELECT * FROM customers WHERE address = %(address)s",
                    parameters={"address": "Space"}
                )
                print(result)
            ```
        """
        inputs: Dict[str, Any] = dict(command=operation, params=parameters, **execute_kwargs)
        new, cursor = self._get_cursor(inputs, cursor_type=cursor_type)
        if new:
            await self._execute_async(cursor, inputs)
        self.logger.debug('Preparing to fetch a row.')
        result: Optional[Tuple[Any, ...]] = await run_sync_in_worker_thread(cursor.fetchone)
        return result

    def fetch_many(
        self,
        operation: str,
        parameters: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
        size: Optional[int] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> List[Tuple[Any, ...]]:
        """
        Fetch a limited number of results from the database.
        Repeated calls using the same inputs to *any* of the fetch methods of this
        block will skip executing the operation again, and instead,
        return the next set of results from the previous execution,
        until the reset_cursors method is called.

        Args:
            operation: The SQL query or other operation to be executed.
            parameters: The parameters for the operation.
            size: The number of results to return; if None or 0, uses the value of
                `fetch_size` configured on the block.
            cursor_type: The class of the cursor to use when creating a Snowflake cursor.
            **execute_kwargs: Additional options to pass to `cursor.execute_async`.

        Returns:
            A list of tuples containing the data returned by the database,
                where each row is a tuple and each column is a value in the tuple.

        Examples:
            Repeatedly fetch two rows from the database where address is Highway 42.
            ```python
            from prefect_snowflake.database import SnowflakeConnector

            with SnowflakeConnector.load("BLOCK_NAME") as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS customers (name varchar, address varchar);"
                )
                conn.execute_many(
                    "INSERT INTO customers (name, address) VALUES (%(name)s, %(address)s);",
                    seq_of_parameters=[
                        {"name": "Marvin", "address": "Highway 42"},
                        {"name": "Ford", "address": "Highway 42"},
                        {"name": "Unknown", "address": "Highway 42"},
                        {"name": "Me", "address": "Highway 42"},
                    ],
                )
                result = conn.fetch_many(
                    "SELECT * FROM customers WHERE address = %(address)s",
                    parameters={"address": "Highway 42"},
                    size=2
                )
                print(result)  # Marvin, Ford
                result = conn.fetch_many(
                    "SELECT * FROM customers WHERE address = %(address)s",
                    parameters={"address": "Highway 42"},
                    size=2
                )
                print(result)  # Unknown, Me
            ```
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
        parameters: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
        size: Optional[int] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> List[Tuple[Any, ...]]:
        """
        Fetch a limited number of results from the database asynchronously.
        Repeated calls using the same inputs to *any* of the fetch methods of this
        block will skip executing the operation again, and instead,
        return the next set of results from the previous execution,
        until the reset_cursors method is called.

        Args:
            operation: The SQL query or other operation to be executed.
            parameters: The parameters for the operation.
            size: The number of results to return; if None or 0, uses the value of
                `fetch_size` configured on the block.
            cursor_type: The class of the cursor to use when creating a Snowflake cursor.
            **execute_kwargs: Additional options to pass to `cursor.execute_async`.

        Returns:
            A list of tuples containing the data returned by the database,
                where each row is a tuple and each column is a value in the tuple.

        Examples:
            Repeatedly fetch two rows from the database where address is Highway 42.
            ```python
            from prefect_snowflake.database import SnowflakeConnector

            with SnowflakeConnector.load("BLOCK_NAME") as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS customers (name varchar, address varchar);"
                )
                conn.execute_many(
                    "INSERT INTO customers (name, address) VALUES (%(name)s, %(address)s);",
                    seq_of_parameters=[
                        {"name": "Marvin", "address": "Highway 42"},
                        {"name": "Ford", "address": "Highway 42"},
                        {"name": "Unknown", "address": "Highway 42"},
                        {"name": "Me", "address": "Highway 42"},
                    ],
                )
                result = conn.fetch_many(
                    "SELECT * FROM customers WHERE address = %(address)s",
                    parameters={"address": "Highway 42"},
                    size=2
                )
                print(result)  # Marvin, Ford
                result = conn.fetch_many(
                    "SELECT * FROM customers WHERE address = %(address)s",
                    parameters={"address": "Highway 42"},
                    size=2
                )
                print(result)  # Unknown, Me
            ```
        """
        inputs: Dict[str, Any] = dict(command=operation, params=parameters, **execute_kwargs)
        new, cursor = self._get_cursor(inputs, cursor_type)
        if new:
            await self._execute_async(cursor, inputs)
        size = size or self.fetch_size
        self.logger.debug(f'Preparing to fetch {size} rows.')
        result: List[Tuple[Any, ...]] = await run_sync_in_worker_thread(cursor.fetchmany, size=size)
        return result

    def fetch_all(
        self,
        operation: str,
        parameters: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> List[Tuple[Any, ...]]:
        """
        Fetch all results from the database.
        Repeated calls using the same inputs to *any* of the fetch methods of this
        block will skip executing the operation again, and instead,
        return the next set of results from the previous execution,
        until the reset_cursors method is called.

        Args:
            operation: The SQL query or other operation to be executed.
            parameters: The parameters for the operation.
            cursor_type: The class of the cursor to use when creating a Snowflake cursor.
            **execute_kwargs: Additional options to pass to `cursor.execute_async`.

        Returns:
            A list of tuples containing the data returned by the database,
                where each row is a tuple and each column is a value in the tuple.

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
        parameters: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> List[Tuple[Any, ...]]:
        """
        Fetch all results from the database.
        Repeated calls using the same inputs to *any* of the fetch methods of this
        block will skip executing the operation again, and instead,
        return the next set of results from the previous execution,
        until the reset_cursors method is called.

        Args:
            operation: The SQL query or other operation to be executed.
            parameters: The parameters for the operation.
            cursor_type: The class of the cursor to use when creating a Snowflake cursor.
            **execute_kwargs: Additional options to pass to `cursor.execute_async`.

        Returns:
            A list of tuples containing the data returned by the database,
                where each row is a tuple and each column is a value in the tuple.

        Examples:
            Fetch all rows from the database where address is Highway 42.
            ```python
            from prefect_snowflake.database import SnowflakeConnector

            with SnowflakeConnector.load("BLOCK_NAME") as conn:
                await conn.execute_async(
                    "CREATE TABLE IF NOT EXISTS customers (name varchar, address varchar);"
                )
                await conn.execute_many_async(
                    "INSERT INTO customers (name, address) VALUES (%(name)s, %(address)s);",
                    seq_of_parameters=[
                        {"name": "Marvin", "address": "Highway 42"},
                        {"name": "Ford", "address": "Highway 42"},
                        {"name": "Unknown", "address": "Highway 42"},
                        {"name": "Me", "address": "Myway 88"},
                    ],
                )
                result = await conn.fetch_all_async(
                    "SELECT * FROM customers WHERE address = %(address)s",
                    parameters={"address": "Highway 42"},
                )
                print(result)  # Marvin, Ford, Unknown
            ```
        """
        inputs: Dict[str, Any] = dict(command=operation, params=parameters, **execute_kwargs)
        new, cursor = self._get_cursor(inputs, cursor_type)
        if new:
            await self._execute_async(cursor, inputs)
        self.logger.debug('Preparing to fetch all rows.')
        result: List[Tuple[Any, ...]] = await run_sync_in_worker_thread(cursor.fetchall)
        return result

    def execute(
        self,
        operation: str,
        parameters: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> None:
        """
        Executes an operation on the database. This method is intended to be used
        for operations that do not return data, such as INSERT, UPDATE, or DELETE.
        Unlike the fetch methods, this method will always execute the operation
        upon calling.

        Args:
            operation: The SQL query or other operation to be executed.
            parameters: The parameters for the operation.
            cursor_type: The class of the cursor to use when creating a Snowflake cursor.
            **execute_kwargs: Additional options to pass to `cursor.execute_async`.

        Examples:
            Create table named customers with two columns, name and address.
            ```python
            from prefect_snowflake.database import SnowflakeConnector

            with SnowflakeConnector.load("BLOCK_NAME") as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS customers (name varchar, address varchar);"
                )
            ```
        """
        self._start_connection()
        inputs: Dict[str, Any] = dict(command=operation, params=parameters, **execute_kwargs)
        assert self._connection is not None  # for type checkers
        with self._connection.cursor(cursor_type) as cursor:
            run_coro_as_sync(self._execute_async(cursor, inputs))
        self.logger.info(f'Executed the operation, {operation!r}.')

    async def execute_async(
        self,
        operation: str,
        parameters: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> None:
        """
        Executes an operation on the database. This method is intended to be used
        for operations that do not return data, such as INSERT, UPDATE, or DELETE.
        Unlike the fetch methods, this method will always execute the operation
        upon calling.

        Args:
            operation: The SQL query or other operation to be executed.
            parameters: The parameters for the operation.
            cursor_type: The class of the cursor to use when creating a Snowflake cursor.
            **execute_kwargs: Additional options to pass to `cursor.execute_async`.

        Examples:
            Create table named customers with two columns, name and address.
            ```python
            from prefect_snowflake.database import SnowflakeConnector

            with SnowflakeConnector.load("BLOCK_NAME") as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS customers (name varchar, address varchar);"
                )
            ```
        """
        self._start_connection()
        inputs: Dict[str, Any] = dict(command=operation, params=parameters, **execute_kwargs)
        assert self._connection is not None  # for type checkers
        with self._connection.cursor(cursor_type) as cursor:
            await run_sync_in_worker_thread(cursor.execute, **inputs)
        self.logger.info(f'Executed the operation, {operation!r}.')

    def execute_many(self, operation: str, seq_of_parameters: Sequence[Union[Dict[str, Any], Sequence[Any]]]) -> None:
        """
        Executes many operations on the database. This method is intended to be used
        for operations that do not return data, such as INSERT, UPDATE, or DELETE.
        Unlike the fetch methods, this method will always execute the operations
        upon calling.

        Args:
            operation: The SQL query or other operation to be executed.
            seq_of_parameters: The sequence of parameters for the operation.

        Examples:
            Create table and insert three rows into it.
            ```python
            from prefect_snowflake.database import SnowflakeConnector

            with SnowflakeConnector.load("BLOCK_NAME") as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS customers (name varchar, address varchar);"
                )
                conn.execute_many(
                    "INSERT INTO customers (name, address) VALUES (%(name)s, %(address)s);",
                    seq_of_parameters=[
                        {"name": "Marvin", "address": "Highway 42"},
                        {"name": "Ford", "address": "Highway 42"},
                        {"name": "Unknown", "address": "Space"},
                    ],
                )
            ```
        """
        self._start_connection()
        inputs: Dict[str, Any] = dict(command=operation, seqparams=seq_of_parameters)
        assert self._connection is not None  # for type checkers
        with self._connection.cursor() as cursor:
            cursor.executemany(**inputs)
        self.logger.info(f'Executed {len(seq_of_parameters)} operations off {operation!r}.')

    async def execute_many_async(self, operation: str, seq_of_parameters: Sequence[Union[Dict[str, Any], Sequence[Any]]]) -> None:
        """
        Executes many operations on the database. This method is intended to be used
        for operations that do not return data, such as INSERT, UPDATE, or DELETE.
        Unlike the fetch methods, this method will always execute the operations
        upon calling.

        Args:
            operation: The SQL query or other operation to be executed.
            seq_of_parameters: The sequence of parameters for the operation.

        Examples:
            Create table and insert three rows into it.
            ```python
            from prefect_snowflake.database import SnowflakeConnector

            with SnowflakeConnector.load("BLOCK_NAME") as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS customers (name varchar, address varchar);"
                )
                conn.execute_many(
                    "INSERT INTO customers (name, address) VALUES (%(name)s, %(address)s);",
                    seq_of_parameters=[
                        {"name": "Marvin", "address": "Highway 42"},
                        {"name": "Ford", "address": "Highway 42"},
                        {"name": "Unknown", "address": "Space"},
                    ],
                )
            ```
        """
        self._start_connection()
        inputs: Dict[str, Any] = dict(command=operation, seqparams=seq_of_parameters)
        assert self._connection is not None  # for type checkers
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

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
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
    params: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
    poll_frequency_seconds: int = 1,
) -> List[Tuple[Any, ...]]:
    """
    Executes a query against a Snowflake database.

    Args:
        query: The query to execute against the database.
        params: The params to replace the placeholders in the query.
        snowflake_connector: The credentials to use to authenticate.
        cursor_type: The type of database cursor to use for the query.
        poll_frequency_seconds: Number of seconds to wait in between checks for
            run completion.

    Returns:
        The output of `response.fetchall()`.

    Examples:
        Query Snowflake table with the ID value parameterized.
        ```python
        from prefect import flow
        from prefect_snowflake.credentials import SnowflakeCredentials
        from prefect_snowflake.database import SnowflakeConnector, snowflake_query


        @flow
        def snowflake_query_flow():
            snowflake_credentials = SnowflakeCredentials(
                account="account",
                user="user",
                password="password",
            )
            snowflake_connector = SnowflakeConnector(
                database="database",
                warehouse="warehouse",
                schema="schema",
                credentials=snowflake_credentials
            )
            result = snowflake_query(
                "SELECT * FROM table WHERE id=%{id_param}s LIMIT 8;",
                snowflake_connector,
                params={"id_param": 1}
            )
            return result

        snowflake_query_flow()
        ```
    """
    with snowflake_connector.get_connection() as connection:
        with connection.cursor(cursor_type) as cursor:
            response: Dict[str, Any] = cursor.execute_async(query, params=params)
            query_id: str = response['queryId']
            while connection.is_still_running(connection.get_query_status_throw_if_error(query_id)):
                sleep(poll_frequency_seconds)
            cursor.get_results_from_sfqid(query_id)
            result: List[Tuple[Any, ...]] = cursor.fetchall()
    return result


@task
async def snowflake_query_async(
    query: str,
    snowflake_connector: SnowflakeConnector,
    params: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
    poll_frequency_seconds: int = 1,
) -> List[Tuple[Any, ...]]:
    """
    Executes a query against a Snowflake database.

    Args:
        query: The query to execute against the database.
        params: The params to replace the placeholders in the query.
        snowflake_connector: The credentials to use to authenticate.
        cursor_type: The type of database cursor to use for the query.
        poll_frequency_seconds: Number of seconds to wait in between checks for
            run completion.

    Returns:
        The output of `response.fetchall()`.

    Examples:
        Query Snowflake table with the ID value parameterized.
        ```python
        from prefect import flow
        from prefect_snowflake.credentials import SnowflakeCredentials
        from prefect_snowflake.database import SnowflakeConnector, snowflake_query


        @flow
        def snowflake_query_flow():
            snowflake_credentials = SnowflakeCredentials(
                account="account",
                user="user",
                password="password",
            )
            snowflake_connector = SnowflakeConnector(
                database="database",
                warehouse="warehouse",
                schema="schema",
                credentials=snowflake_credentials
            )
            result = snowflake_query(
                "SELECT * FROM table WHERE id=%{id_param}s LIMIT 8;",
                snowflake_connector,
                params={"id_param": 1}
            )
            return result

        snowflake_query_flow()
        ```
    """
    with snowflake_connector.get_connection() as connection:
        with connection.cursor(cursor_type) as cursor:
            response: Dict[str, Any] = cursor.execute_async(query, params=params)
            query_id: str = response['queryId']
            while connection.is_still_running(connection.get_query_status_throw_if_error(query_id)):
                await asyncio.sleep(poll_frequency_seconds)
            cursor.get_results_from_sfqid(query_id)
            result: List[Tuple[Any, ...]] = cursor.fetchall()
    return result


@task
def snowflake_multiquery(
    queries: List[str],
    snowflake_connector: SnowflakeConnector,
    params: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
    as_transaction: bool = False,
    return_transaction_control_results: bool = False,
    poll_frequency_seconds: int = 1,
) -> List[List[Tuple[Any, ...]]]:
    """
    Executes multiple queries against a Snowflake database in a shared session.
    Allows execution in a transaction.

    Args:
        queries: The list of queries to execute against the database.
        params: The params to replace the placeholders in the query.
        snowflake_connector: The credentials to use to authenticate.
        cursor_type: The type of database cursor to use for the query.
        as_transaction: If True, queries are executed in a transaction.
        return_transaction_control_results: Determines if the results of queries
            controlling the transaction (BEGIN/COMMIT) should be returned.
        poll_frequency_seconds: Number of seconds to wait in between checks for
            run completion.

    Returns:
        List of the outputs of `response.fetchall()` for each query.

    Examples:
        Query Snowflake table with the ID value parameterized.
        ```python
        from prefect import flow
        from prefect_snowflake.credentials import SnowflakeCredentials
        from prefect_snowflake.database import SnowflakeConnector, snowflake_multiquery


        @flow
        def snowflake_multiquery_flow():
            snowflake_credentials = SnowflakeCredentials(
                account="account",
                user="user",
                password="password",
            )
            snowflake_connector = SnowflakeConnector(
                database="database",
                warehouse="warehouse",
                schema="schema",
                credentials=snowflake_credentials
            )
            result = snowflake_multiquery(
                ["SELECT * FROM table WHERE id=%{id_param}s LIMIT 8;", "SELECT 1,2"],
                snowflake_connector,
                params={"id_param": 1},
                as_transaction=True
            )
            return result

        snowflake_multiquery_flow()
        ```
    """
    with snowflake_connector.get_connection() as connection:
        if as_transaction:
            queries.insert(0, BEGIN_TRANSACTION_STATEMENT)
            queries.append(END_TRANSACTION_STATEMENT)
        with connection.cursor(cursor_type) as cursor:
            results: List[List[Tuple[Any, ...]]] = []
            for query in queries:
                response: Dict[str, Any] = cursor.execute_async(query, params=params)
                query_id: str = response['queryId']
                while connection.is_still_running(connection.get_query_status_throw_if_error(query_id)):
                    sleep(poll_frequency_seconds)
                cursor.get_results_from_sfqid(query_id)
                result: List[Tuple[Any, ...]] = cursor.fetchall()
                results.append(result)
    if as_transaction and (not return_transaction_control_results):
        return results[1:-1]
    else:
        return results


@task
async def snowflake_multiquery_async(
    queries: List[str],
    snowflake_connector: SnowflakeConnector,
    params: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
    as_transaction: bool = False,
    return_transaction_control_results: bool = False,
    poll_frequency_seconds: int = 1,
) -> List[List[Tuple[Any, ...]]]:
    """
    Executes multiple queries against a Snowflake database in a shared session.
    Allows execution in a transaction.

    Args:
        queries: The list of queries to execute against the database.
        params: The params to replace the placeholders in the query.
        snowflake_connector: The credentials to use to authenticate.
        cursor_type: The type of database cursor to use for the query.
        as_transaction: If True, queries are executed in a transaction.
        return_transaction_control_results: Determines if the results of queries
            controlling the transaction (BEGIN/COMMIT) should be returned.
        poll_frequency_seconds: Number of seconds to wait in between checks for
            run completion.

    Returns:
        List of the outputs of `response.fetchall()` for each query.

    Examples:
        Query Snowflake table with the ID value parameterized.
        ```python
        from prefect import flow
        from prefect_snowflake.credentials import SnowflakeCredentials
        from prefect_snowflake.database import SnowflakeConnector, snowflake_multiquery


        @flow
        def snowflake_multiquery_flow():
            snowflake_credentials = SnowflakeCredentials(
                account="account",
                user="user",
                password="password",
            )
            snowflake_connector = SnowflakeConnector(
                database="database",
                warehouse="warehouse",
                schema="schema",
                credentials=snowflake_credentials
            )
            result = snowflake_multiquery(
                ["SELECT * FROM table WHERE id=%{id_param}s LIMIT 8;", "SELECT 1,2"],
                snowflake_connector,
                params={"id_param": 1},
                as_transaction=True
            )
            return result

        snowflake_multiquery_flow()
        ```
    """
    with snowflake_connector.get_connection() as connection:
        if as_transaction:
            queries.insert(0, BEGIN_TRANSACTION_STATEMENT)
            queries.append(END_TRANSACTION_STATEMENT)
        with connection.cursor(cursor_type) as cursor:
            results: List[List[Tuple[Any, ...]]] = []
            for query in queries:
                response: Dict[str, Any] = cursor.execute_async(query, params=params)
                query_id: str = response['queryId']
                while connection.is_still_running(connection.get_query_status_throw_if_error(query_id)):
                    await asyncio.sleep(poll_frequency_seconds)
                cursor.get_results_from_sfqid(query_id)
                result: List[Tuple[Any, ...]] = cursor.fetchall()
                results.append(result)
    if as_transaction and (not return_transaction_control_results):
        return results[1:-1]
    else:
        return results


@task
def snowflake_query_sync(
    query: str,
    snowflake_connector: SnowflakeConnector,
    params: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
) -> List[Tuple[Any, ...]]:
    """
    Executes a query in sync mode against a Snowflake database.

    Args:
        query: The query to execute against the database.
        params: The params to replace the placeholders in the query.
        snowflake_connector: The credentials to use to authenticate.
        cursor_type: The type of database cursor to use for the query.

    Returns:
        The output of `response.fetchall()`.

    Examples:
        Execute a put statement.
        ```python
        from prefect import flow
        from prefect_snowflake.credentials import SnowflakeCredentials
        from prefect_snowflake.database import SnowflakeConnector, snowflake_query


        @flow
        def snowflake_query_sync_flow():
            snowflake_credentials = SnowflakeCredentials(
                account="account",
                user="user",
                password="password",
            )
            snowflake_connector = SnowflakeConnector(
                database="database",
                warehouse="warehouse",
                schema="schema",
                credentials=snowflake_credentials
            )
            result = snowflake_query_sync(
                "put file://a_file.csv @mystage;",
                snowflake_connector,
            )
            return result

        snowflake_query_sync_flow()
        ```
    """
    with snowflake_connector.get_connection() as connection:
        with connection.cursor(cursor_type) as cursor:
            cursor.execute(query, params=params)
            result: List[Tuple[Any, ...]] = cursor.fetchall()
    return result