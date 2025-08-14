"""Module for querying against Snowflake databases."""

import asyncio
from time import sleep
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, Literal

from pydantic import AliasChoices, Field
from snowflake.connector.connection import SnowflakeConnection
from snowflake.connector.cursor import SnowflakeCursor

from prefect import task
from prefect.blocks.abstract import DatabaseBlock
from prefect.utilities.asyncutils import run_coro_as_sync, run_sync_in_worker_thread
from prefect.utilities.hashing import hash_objects
from prefect_snowflake import SnowflakeCredentials

BEGIN_TRANSACTION_STATEMENT: str = "BEGIN TRANSACTION"
END_TRANSACTION_STATEMENT: str = "COMMIT"


class SnowflakeConnector(DatabaseBlock):
    _block_type_name: str = "Snowflake Connector"
    _logo_url: str = "https://cdn.sanity.io/images/3ugk85nk/production/bd359de0b4be76c2254bd329fe3a267a1a3879c2-250x250.png"  # noqa
    _documentation_url: str = "https://docs.prefect.io/integrations/prefect-snowflake"  # noqa
    _description: str = "Perform data operations against a Snowflake database."

    credentials: SnowflakeCredentials = Field(
        default=..., description="The credentials to authenticate with Snowflake."
    )
    database: str = Field(
        default=..., description="The name of the default database to use."
    )
    warehouse: str = Field(
        default=..., description="The name of the default warehouse to use."
    )
    schema_: str = Field(
        default=...,
        serialization_alias="schema",
        validation_alias=AliasChoices("schema_", "schema"),
        description="The name of the default schema to use.",
    )
    fetch_size: int = Field(
        default=1, description="The default number of rows to fetch at a time."
    )
    poll_frequency_s: int = Field(
        default=1,
        title="Poll Frequency [seconds]",
        description=(
            "The number of seconds between checking query "
            "status for long running queries."
        ),
    )

    _connection: Optional[SnowflakeConnection] = None
    _unique_cursors: Optional[Dict[str, SnowflakeCursor]] = None

    def get_connection(self, **connect_kwargs: Any) -> SnowflakeConnection:
        if self._connection is not None:
            return self._connection

        connect_params = {
            "database": self.database,
            "warehouse": self.warehouse,
            "schema": self.schema_,
        }
        connection = self.credentials.get_client(**connect_kwargs, **connect_params)
        self._connection = connection
        self.logger.info("Started a new connection to Snowflake.")
        return connection

    def _start_connection(self) -> None:
        self.get_connection()
        if self._unique_cursors is None:
            self._unique_cursors = {}

    def _get_cursor(
        self,
        inputs: Dict[str, Any],
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
    ) -> Tuple[bool, SnowflakeCursor]:
        self._start_connection()

        input_hash = hash_objects(inputs)
        if input_hash is None:
            raise RuntimeError(
                f"We were not able to hash your inputs, {inputs!r}, "
                "which resulted in an unexpected data return; "
                "please open an issue with a reproducible example."
            )
        if input_hash not in self._unique_cursors.keys():
            new_cursor = self._connection.cursor(cursor_type)
            self._unique_cursors[input_hash] = new_cursor
            return True, new_cursor
        else:
            existing_cursor = self._unique_cursors[input_hash]
            return False, existing_cursor

    async def _execute_async(self, cursor: SnowflakeCursor, inputs: Dict[str, Any]) -> None:
        response = await run_sync_in_worker_thread(cursor.execute_async, **inputs)
        self.logger.info(
            f"Executing the operation, {inputs['command']!r}, asynchronously; "
            f"polling for the result every {self.poll_frequency_s} seconds."
        )

        query_id = response["queryId"]
        while self._connection.is_still_running(
            await run_sync_in_worker_thread(
                self._connection.get_query_status_throw_if_error, query_id
            )
        ):
            await asyncio.sleep(self.poll_frequency_s)
        await run_sync_in_worker_thread(cursor.get_results_from_sfqid, query_id)

    def reset_cursors(self) -> None:
        if not self._unique_cursors:
            self.logger.info("There were no cursors to reset.")
            return

        input_hashes = tuple(self._unique_cursors.keys())
        for input_hash in input_hashes:
            cursor = self._unique_cursors.pop(input_hash)
            try:
                cursor.close()
            except Exception as exc:
                self.logger.warning(
                    f"Failed to close cursor for input hash {input_hash!r}: {exc}"
                )
        self.logger.info("Successfully reset the cursors.")

    def fetch_one(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> Optional[Tuple[Any, ...]]:
        inputs = dict(
            command=operation,
            params=parameters,
            **execute_kwargs,
        )
        new, cursor = self._get_cursor(inputs, cursor_type=cursor_type)
        if new:
            cursor.execute(operation, params=parameters, **execute_kwargs)
        self.logger.debug("Preparing to fetch a row.")
        return cursor.fetchone()

    async def fetch_one_async(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> Optional[Tuple[Any, ...]]:
        inputs = dict(
            command=operation,
            params=parameters,
            **execute_kwargs,
        )
        new, cursor = self._get_cursor(inputs, cursor_type=cursor_type)
        if new:
            await self._execute_async(cursor, inputs)
        self.logger.debug("Preparing to fetch a row.")
        result = await run_sync_in_worker_thread(cursor.fetchone)
        return result

    def fetch_many(
        self,
        operation: str,
        parameters: Optional[Sequence[Dict[str, Any]]] = None,
        size: Optional[int] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> List[Tuple[Any, ...]]:
        inputs = dict(
            command=operation,
            params=parameters,
            **execute_kwargs,
        )
        new, cursor = self._get_cursor(inputs, cursor_type)
        if new:
            cursor.execute(operation, params=parameters, **execute_kwargs)
        size = size or self.fetch_size
        self.logger.debug(f"Preparing to fetch {size} rows.")
        return cursor.fetchmany(size=size)

    async def fetch_many_async(
        self,
        operation: str,
        parameters: Optional[Sequence[Dict[str, Any]]] = None,
        size: Optional[int] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> List[Tuple[Any, ...]]:
        inputs = dict(
            command=operation,
            params=parameters,
            **execute_kwargs,
        )
        new, cursor = self._get_cursor(inputs, cursor_type)
        if new:
            await self._execute_async(cursor, inputs)
        size = size or self.fetch_size
        self.logger.debug(f"Preparing to fetch {size} rows.")
        result = await run_sync_in_worker_thread(cursor.fetchmany, size=size)
        return result

    def fetch_all(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> List[Tuple[Any, ...]]:
        inputs = dict(
            command=operation,
            params=parameters,
            **execute_kwargs,
        )
        new, cursor = self._get_cursor(inputs, cursor_type)
        if new:
            cursor.execute(operation, params=parameters, **execute_kwargs)
        self.logger.debug("Preparing to fetch all rows.")
        return cursor.fetchall()

    async def fetch_all_async(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> List[Tuple[Any, ...]]:
        inputs = dict(
            command=operation,
            params=parameters,
            **execute_kwargs,
        )
        new, cursor = self._get_cursor(inputs, cursor_type)
        if new:
            await self._execute_async(cursor, inputs)
        self.logger.debug("Preparing to fetch all rows.")
        result = await run_sync_in_worker_thread(cursor.fetchall)
        return result

    def execute(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> None:
        self._start_connection()

        inputs = dict(
            command=operation,
            params=parameters,
            **execute_kwargs,
        )
        with self._connection.cursor(cursor_type) as cursor:
            run_coro_as_sync(self._execute_async(cursor, inputs))
        self.logger.info(f"Executed the operation, {operation!r}.")

    async def execute_async(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
        **execute_kwargs: Any,
    ) -> None:
        self._start_connection()

        inputs = dict(
            command=operation,
            params=parameters,
            **execute_kwargs,
        )
        with self._connection.cursor(cursor_type) as cursor:
            await run_sync_in_worker_thread(cursor.execute, **inputs)
        self.logger.info(f"Executed the operation, {operation!r}.")

    def execute_many(
        self,
        operation: str,
        seq_of_parameters: List[Dict[str, Any]],
    ) -> None:
        self._start_connection()

        inputs = dict(
            command=operation,
            seqparams=seq_of_parameters,
        )
        with self._connection.cursor() as cursor:
            cursor.executemany(**inputs)
        self.logger.info(
            f"Executed {len(seq_of_parameters)} operations off {operation!r}."
        )

    async def execute_many_async(
        self,
        operation: str,
        seq_of_parameters: List[Dict[str, Any]],
    ) -> None:
        self._start_connection()

        inputs = dict(
            command=operation,
            seqparams=seq_of_parameters,
        )
        with self._connection.cursor() as cursor:
            await run_sync_in_worker_thread(cursor.executemany, **inputs)
        self.logger.info(
            f"Executed {len(seq_of_parameters)} operations off {operation!r}."
        )

    def close(self) -> None:
        try:
            self.reset_cursors()
        finally:
            if self._connection is None:
                self.logger.info("There was no connection open to be closed.")
                return
            self._connection.close()
            self._connection = None
            self.logger.info("Successfully closed the Snowflake connection.")

    def __enter__(self) -> "SnowflakeConnector":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __getstate__(self) -> dict:
        data = self.__dict__.copy()
        data.update({k: None for k in {"_connection", "_unique_cursors"}})
        return data

    def __setstate__(self, data: dict) -> None:
        self.__dict__.update(data)
        self._start_connection()


@task
def snowflake_query(
    query: str,
    snowflake_connector: SnowflakeConnector,
    params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
    poll_frequency_seconds: int = 1,
) -> List[Tuple[Any, ...]]:
    with snowflake_connector.get_connection() as connection:
        with connection.cursor(cursor_type) as cursor:
            response = cursor.execute_async(query, params=params)
            query_id = response["queryId"]
            while connection.is_still_running(
                connection.get_query_status_throw_if_error(query_id)
            ):
                sleep(poll_frequency_seconds)
            cursor.get_results_from_sfqid(query_id)
            result = cursor.fetchall()
    return result


@task
async def snowflake_query_async(
    query: str,
    snowflake_connector: SnowflakeConnector,
    params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
    poll_frequency_seconds: int = 1,
) -> List[Tuple[Any, ...]]:
    with snowflake_connector.get_connection() as connection:
        with connection.cursor(cursor_type) as cursor:
            response = cursor.execute_async(query, params=params)
            query_id = response["queryId"]
            while connection.is_still_running(
                connection.get_query_status_throw_if_error(query_id)
            ):
                await asyncio.sleep(poll_frequency_seconds)
            cursor.get_results_from_sfqid(query_id)
            result = cursor.fetchall()
    return result


@task
def snowflake_multiquery(
    queries: List[str],
    snowflake_connector: SnowflakeConnector,
    params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
    as_transaction: bool = False,
    return_transaction_control_results: bool = False,
    poll_frequency_seconds: int = 1,
) -> List[List[Tuple[Any, ...]]]:
    with snowflake_connector.get_connection() as connection:
        if as_transaction:
            queries.insert(0, BEGIN_TRANSACTION_STATEMENT)
            queries.append(END_TRANSACTION_STATEMENT)

        with connection.cursor(cursor_type) as cursor:
            results = []
            for query in queries:
                response = cursor.execute_async(query, params=params)
                query_id = response["queryId"]
                while connection.is_still_running(
                    connection.get_query_status_throw_if_error(query_id)
                ):
                    sleep(poll_frequency_seconds)
                cursor.get_results_from_sfqid(query_id)
                result = cursor.fetchall()
                results.append(result)

    if as_transaction and not return_transaction_control_results:
        return results[1:-1]
    else:
        return results


@task
async def snowflake_multiquery_async(
    queries: List[str],
    snowflake_connector: SnowflakeConnector,
    params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
    as_transaction: bool = False,
    return_transaction_control_results: bool = False,
    poll_frequency_seconds: int = 1,
) -> List[List[Tuple[Any, ...]]]:
    with snowflake_connector.get_connection() as connection:
        if as_transaction:
            queries.insert(0, BEGIN_TRANSACTION_STATEMENT)
            queries.append(END_TRANSACTION_STATEMENT)

        with connection.cursor(cursor_type) as cursor:
            results = []
            for query in queries:
                response = cursor.execute_async(query, params=params)
                query_id = response["queryId"]
                while connection.is_still_running(
                    connection.get_query_status_throw_if_error(query_id)
                ):
                    await asyncio.sleep(poll_frequency_seconds)
                cursor.get_results_from_sfqid(query_id)
                result = cursor.fetchall()
                results.append(result)

    if as_transaction and not return_transaction_control_results:
        return results[1:-1]
    else:
        return results


@task
def snowflake_query_sync(
    query: str,
    snowflake_connector: SnowflakeConnector,
    params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None,
    cursor_type: Type[SnowflakeCursor] = SnowflakeCursor,
) -> List[Tuple[Any, ...]]:
    with snowflake_connector.get_connection() as connection:
        with connection.cursor(cursor_type) as cursor:
            cursor.execute(query, params=params)
            result = cursor.fetchall()
    return result
