```python
from __future__ import annotations
import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING
import typer
import uvicorn
from rich.table import Table
from rich.text import Text
import prefect
from prefect.cli._types import PrefectTyper
from prefect.logging import Logger
from prefect.server.services.base import Service
from prefect.settings import Setting

if TYPE_CHECKING:
    import logging

server_app: PrefectTyper = ...
database_app: PrefectTyper = ...
services_app: PrefectTyper = ...
app: Any = ...
logger: Logger = ...
SERVER_PID_FILE_NAME: str = ...
SERVICES_PID_FILE: Path = ...

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
    pid_file: Path,
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
    message: str | None = ...,
    autogenerate: bool = ...
) -> None: ...

@database_app.command()
async def stamp(revision: str) -> None: ...

def _is_process_running(pid: int) -> bool: ...

def _read_pid_file(path: Path) -> int | None: ...

def _write_pid_file(path: Path, pid: int) -> None: ...

def _cleanup_pid_file(path: Path) -> None: ...

@services_app.command(hidden=True, name='manager')
def run_manager_process() -> None: ...

@services_app.command(aliases=['ls'])
def list_services() -> None: ...

@services_app.command(aliases=['start'])
def start_services(background: bool = ...) -> None: ...

@services_app.command(aliases=['stop'])
async def stop_services() -> None: ...
```