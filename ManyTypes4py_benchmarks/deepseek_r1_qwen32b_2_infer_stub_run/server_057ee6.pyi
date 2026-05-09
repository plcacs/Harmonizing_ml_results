"""
Command line interface for working with the Prefect API and server.
"""

from __future__ import annotations
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import typer
import uvicorn
from prefect.settings import (
    PREFECT_API_SERVICES_LATE_RUNS_ENABLED,
    PREFECT_API_SERVICES_SCHEDULER_ENABLED,
    PREFECT_API_URL,
    PREFECT_HOME,
    PREFECT_SERVER_ANALYTICS_ENABLED,
    PREFECT_SERVER_API_HOST,
    PREFECT_SERVER_API_KEEPALIVE_TIMEOUT,
    PREFECT_SERVER_API_PORT,
    PREFECT_SERVER_LOGGING_LEVEL,
    PREFECT_UI_ENABLED,
)
from prefect.cli._types import PrefectTyper

__all__: List[str] = [
    "server_app",
    "database_app",
    "services_app",
    "start",
    "stop",
    "reset",
    "upgrade",
    "downgrade",
    "revision",
    "stamp",
    "list_services",
    "start_services",
    "stop_services",
]

server_app: PrefectTyper = ...
database_app: PrefectTyper = ...
services_app: PrefectTyper = ...

def start(
    host: str = PREFECT_SERVER_API_HOST.value(),
    port: int = PREFECT_SERVER_API_PORT.value(),
    keep_alive_timeout: int = PREFECT_SERVER_API_KEEPALIVE_TIMEOUT.value(),
    log_level: str = PREFECT_SERVER_LOGGING_LEVEL.value(),
    scheduler: bool = PREFECT_API_SERVICES_SCHEDULER_ENABLED.value(),
    analytics: bool = PREFECT_SERVER_ANALYTICS_ENABLED.value(),
    late_runs: bool = PREFECT_API_SERVICES_LATE_RUNS_ENABLED.value(),
    ui: bool = PREFECT_UI_ENABLED.value(),
    no_services: bool = False,
    background: bool = False,
) -> None:
    ...

async def stop() -> None:
    ...

async def reset(yes: bool = False) -> None:
    ...

async def upgrade(
    yes: bool = False,
    revision: str = "head",
    dry_run: bool = False,
) -> None:
    ...

async def downgrade(
    yes: bool = False,
    revision: str = "-1",
    dry_run: bool = False,
) -> None:
    ...

async def revision(
    message: Optional[str] = None,
    autogenerate: bool = False,
) -> None:
    ...

async def stamp(revision: str) -> None:
    ...

def list_services() -> None:
    ...

def start_services(
    background: bool = False,
) -> Awaitable[None]:
    ...

async def stop_services() -> None:
    ...