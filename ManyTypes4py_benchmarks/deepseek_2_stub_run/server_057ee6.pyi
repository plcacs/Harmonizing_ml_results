```python
"""
Command line interface for working with the Prefect API and server.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
import os
import pathlib
import signal
import socket
import subprocess
import sys
import typer
import uvicorn
from rich.table import Table
from rich.text import Text
import prefect
from prefect.cli._types import PrefectTyper
from prefect.settings import PREFECT_API_SERVICES_LATE_RUNS_ENABLED, PREFECT_API_SERVICES_SCHEDULER_ENABLED, PREFECT_API_URL, PREFECT_HOME, PREFECT_SERVER_ANALYTICS_ENABLED, PREFECT_SERVER_API_HOST, PREFECT_SERVER_API_KEEPALIVE_TIMEOUT, PREFECT_SERVER_API_PORT, PREFECT_SERVER_LOGGING_LEVEL, PREFECT_UI_ENABLED, Profile

if TYPE_CHECKING:
    import logging

server_app: PrefectTyper = ...
database_app: PrefectTyper = ...
services_app: PrefectTyper = ...
logger: logging.Logger = ...
SERVER_PID_FILE_NAME: str = ...
SERVICES_PID_FILE: pathlib.Path = ...

def generate_welcome_blurb(base_url: str, ui_enabled: Any) -> str: ...
def prestart_check(base_url: str) -> None: ...

@server_app.command()
def start(
    host: Any = ...,
    port: Any = ...,
    keep_alive_timeout: Any = ...,
    log_level: Any = ...,
    scheduler: Any = ...,
    analytics: Any = ...,
    late_runs: Any = ...,
    ui: Any = ...,
    no_services: bool = ...,
    background: bool = ...
) -> None: ...

def _run_in_background(
    pid_file: pathlib.Path,
    server_settings: dict[str, str],
    host: Any,
    port: Any,
    keep_alive_timeout: Any,
    no_services: bool
) -> None: ...

def _run_in_foreground(
    server_settings: dict[str, str],
    host: Any,
    port: Any,
    keep_alive_timeout: Any,
    no_services: bool
) -> None: ...

@server_app.command()
async def stop() -> None: ...

@database_app.command()
async def reset(yes: bool = ...) -> None: ...

@database_app.command()
async def upgrade(
    yes: bool = ...,
    revision: str = ...,
    dry_run: bool = ...
) -> None: ...

@database_app.command()
async def downgrade(
    yes: bool = ...,
    revision: str = ...,
    dry_run: bool = ...
) -> None: ...

@database_app.command()
async def revision(
    message: Optional[str] = ...,
    autogenerate: bool = ...
) -> None: ...

@database_app.command()
async def stamp(revision: Any) -> None: ...

def _is_process_running(pid: int) -> bool: ...
def _read_pid_file(path: pathlib.Path) -> Optional[int]: ...
def _write_pid_file(path: pathlib.Path, pid: int) -> None: ...
def _cleanup_pid_file(path: pathlib.Path) -> None: ...

@services_app.command(hidden=True, name='manager')
def run_manager_process() -> None: ...

@services_app.command(aliases=['ls'])
def list_services() -> None: ...

@services_app.command(aliases=['start'])
def start_services(background: bool = ...) -> None: ...

@services_app.command(aliases=['stop'])
async def stop_services() -> None: ...
```