from __future__ import annotations
import asyncio
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    import logging
    from prefect.cli._types import PrefectTyper
    from prefect.settings import Setting
    import typer
    from rich.console import Console
    from rich.table import Table

app: PrefectTyper
server_app: PrefectTyper
database_app: PrefectTyper
services_app: PrefectTyper
logger: logging.Logger
SERVER_PID_FILE_NAME: str
SERVICES_PID_FILE: Path

def generate_welcome_blurb(base_url: str, ui_enabled: bool) -> str: ...

def prestart_check(base_url: str) -> None: ...

@server_app.command()
def start(
    host: str = ...,
    port: int = ...,
    keep_alive_timeout: int = ...,
    log_level: str = ...,
    scheduler: bool = ...,
    analytics: bool = ...,
    late_runs: bool = ...,
    ui: bool = ...,
    no_services: bool = ...,
    background: bool = ...
) -> None: ...

def _run_in_background(
    pid_file: Path,
    server_settings: dict[str, str],
    host: str,
    port: int,
    keep_alive_timeout: int,
    no_services: bool
) -> None: ...

def _run_in_foreground(
    server_settings: dict[str, str],
    host: str,
    port: int,
    keep_alive_timeout: int,
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
async def stamp(revision: str) -> None: ...

def _is_process_running(pid: int) -> bool: ...

def _read_pid_file(path: Path) -> Optional[int]: ...

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