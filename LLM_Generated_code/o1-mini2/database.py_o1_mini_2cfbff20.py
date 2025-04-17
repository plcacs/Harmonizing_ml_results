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

BEGIN_TRANSACTION_STATEMENT: str = "BEGIN TRANSACTION"
END_TRANSACTION_STATEMENT: str = "COMMIT"


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
        