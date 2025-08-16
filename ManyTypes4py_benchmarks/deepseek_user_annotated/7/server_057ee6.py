from __future__ import annotations

import asyncio
import inspect
import os
import shlex
import signal
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import typer
import uvicorn
from rich.table import Table
from rich.text import Text

import prefect
import prefect.settings
from prefect.cli._prompts import prompt
from prefect.cli._types import PrefectTyper, SettingsOption
from prefect.cli._utilities import exit_with_error, exit_with_success
from prefect.cli.cloud import prompt_select_from_list
from prefect.cli.root import app, is_interactive
from prefect.logging import get_logger
from prefect.server.services.base import Service
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
    Profile,
    load_current_profile,
    load_profiles,
    save_profiles,
    update_current_profile,
)
from prefect.settings.context import temporary_settings
from prefect.utilities.asyncutils import run_sync_in_worker_thread

if TYPE_CHECKING:
    import logging

server_app: PrefectTyper = PrefectTyper(
    name="server",
    help="Start a Prefect server instance and interact with the database",
)
database_app: PrefectTyper = PrefectTyper(
    name="database", help="Interact with the database."
)
services_app: PrefectTyper = PrefectTyper(
    name="services", help="Interact with server loop services."
)
server_app.add_typer(database_app)
server_app.add_typer(services_app)
app.add_typer(server_app)

logger: logging.Logger = get_logger(__name__)

SERVER_PID_FILE_NAME: str = "server.pid"
SERVICES_PID_FILE: Path = Path(PREFECT_HOME.value()) / "services.pid"


def generate_welcome_blurb(base_url: str, ui_enabled: bool) -> str:
    blurb = textwrap.dedent(
        r"""
         ___ ___ ___ ___ ___ ___ _____
        | _ \ _ \ __| __| __/ __|_   _|
        |  _/   / _|| _|| _| (__  | |
        |_| |_|_\___|_| |___\___| |_|

        Configure Prefect to communicate with the server with:

            prefect config set PREFECT_API_URL={api_url}

        View the API reference documentation at {docs_url}
        """
    ).format(api_url=base_url + "/api", docs_url=base_url + "/docs")

    visit_dashboard = textwrap.dedent(
        f"""
        Check out the dashboard at {base_url}
        """
    )

    dashboard_not_built = textwrap.dedent(
        """
        The dashboard is not built. It looks like you're on a development version.
        See `prefect dev` for development commands.
        """
    )

    dashboard_disabled = textwrap.dedent(
        """
        The dashboard is disabled. Set `PREFECT_UI_ENABLED=1` to re-enable it.
        """
    )

    if not os.path.exists(prefect.__ui_static_path__):
        blurb += dashboard_not_built
    elif not ui_enabled:
        blurb += dashboard_disabled
    else:
        blurb += visit_dashboard

    return blurb


def prestart_check(base_url: str) -> None:
    current_profile: Optional[Profile] = load_current_profile()
    profiles: Dict[str, Profile] = load_profiles()
    if current_profile and PREFECT_API_URL not in current_profile.settings:
        profiles_with_matching_url: List[str] = [
            name
            for name, profile in profiles.items()
            if profile.settings.get(PREFECT_API_URL) == f"{base_url}/api"
        ]
        if len(profiles_with_matching_url) == 1:
            profiles.set_active(profiles_with_matching_url[0])
            save_profiles(profiles)
            app.console.print(
                f"Switched to profile {profiles_with_matching_url[0]!r}",
                style="green",
            )
            return
        elif len(profiles_with_matching_url) > 1:
            app.console.print(
                "Your current profile doesn't have `PREFECT_API_URL` set to the address"
                " of the server that's running. Some of your other profiles do."
            )
            selected_profile: str = prompt_select_from_list(
                app.console,
                "Which profile would you like to switch to?",
                sorted(
                    [profile for profile in profiles_with_matching_url],
                ),
            )
            profiles.set_active(selected_profile)
            save_profiles(profiles)
            app.console.print(
                f"Switched to profile {selected_profile!r}", style="green"
            )
            return

        app.console.print(
            "The `PREFECT_API_URL` setting for your current profile doesn't match the"
            " address of the server that's running. You need to set it to communicate"
            " with the server.",
            style="yellow",
        )

        choice: str = prompt_select_from_list(
            app.console,
            "How would you like to proceed?",
            [
                (
                    "create",
                    "Create a new profile with `PREFECT_API_URL` set and switch to it",
                ),
                (
                    "set",
                    f"Set `PREFECT_API_URL` in the current profile: {current_profile.name!r}",
                ),
            ],
        )

        if choice == "create":
            while True:
                profile_name: str = prompt("Enter a new profile name")
                if profile_name in profiles:
                    app.console.print(
                        f"Profile {profile_name!r} already exists. Please choose a different name.",
                        style="red",
                    )
                else:
                    break

            profiles.add_profile(
                Profile(
                    name=profile_name, settings={PREFECT_API_URL: f"{base_url}/api"}
                )
            )
            profiles.set_active(profile_name)
            save_profiles(profiles)

            app.console.print(
                f"Switched to new profile {profile_name!r}", style="green"
            )
        elif choice == "set":
            api_url: str = prompt(
                "Enter the `PREFECT_API_URL` value", default="http://127.0.0.1:4200/api"
            )
            update_current_profile({PREFECT_API_URL: api_url})
            app.console.print(
                f"Set `PREFECT_API_URL` to {api_url!r} in the current profile {current_profile.name!r}",
                style="green",
            )


@server_app.command()
def start(
    host: str = SettingsOption(PREFECT_SERVER_API_HOST),
    port: int = SettingsOption(PREFECT_SERVER_API_PORT),
    keep_alive_timeout: int = SettingsOption(PREFECT_SERVER_API_KEEPALIVE_TIMEOUT),
    log_level: str = SettingsOption(PREFECT_SERVER_LOGGING_LEVEL),
    scheduler: bool = SettingsOption(PREFECT_API_SERVICES_SCHEDULER_ENABLED),
    analytics: bool = SettingsOption(
        PREFECT_SERVER_ANALYTICS_ENABLED, "--analytics-on/--analytics-off"
    ),
    late_runs: bool = SettingsOption(PREFECT_API_SERVICES_LATE_RUNS_ENABLED),
    ui: bool = SettingsOption(PREFECT_UI_ENABLED),
    no_services: bool = typer.Option(
        False, "--no-services", help="Only run the webserver API and UI"
    ),
    background: bool = typer.Option(
        False, "--background", "-b", help="Run the server in the background"
    ),
) -> None:
    base_url: str = f"http://{host}:{port}"
    if is_interactive():
        try:
            prestart_check(base_url)
        except Exception:
            pass

    server_settings: Dict[str, str] = {
        "PREFECT_API_SERVICES_SCHEDULER_ENABLED": str(scheduler),
        "PREFECT_SERVER_ANALYTICS_ENABLED": str(analytics),
        "PREFECT_API_SERVICES_LATE_RUNS_ENABLED": str(late_runs),
        "PREFECT_UI_ENABLED": str(ui),
        "PREFECT_SERVER_LOGGING_LEVEL": log_level,
    }

    if no_services:
        server_settings["PREFECT_SERVER_ANALYTICS_ENABLED"] = "False"

    pid_file: Path = Path(PREFECT_HOME.value()) / SERVER_PID_FILE_NAME
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
    except socket.gaierror:
        exit_with_error(
            f"Invalid host '{host}'. Please specify a valid hostname or IP address."
        )
    except socket.error:
        if pid_file.exists():
            exit_with_error(
                f"A background server process is already running on port {port}. "
                "Run `prefect server stop` to stop it or specify a different port "
                "with the `--port` flag."
            )
        exit_with_error(
            f"Port {port} is already in use. Please specify a different port with the "
            "`--port` flag."
        )

    if background:
        try:
            pid_file.touch(mode=0o600, exist_ok=False)
        except FileExistsError:
            exit_with_error(
                "A server is already running in the background. To stop it,"
                " run `prefect server stop`."
            )

    app.console.print(generate_welcome_blurb(base_url, ui_enabled=ui))
    app.console.print("\n")

    if background:
        _run_in_background(
            pid_file, server_settings, host, port, keep_alive_timeout, no_services
        )
    else:
        _run_in_foreground(server_settings, host, port, keep_alive_timeout, no_services)


def _run_in_background(
    pid_file: Path,
    server_settings: Dict[str, str],
    host: str,
    port: int,
    keep_alive_timeout: int,
    no_services: bool,
) -> None:
    command: List[str] = [
        sys.executable,
        "-m",
        "uvicorn",
        "--app-dir",
        str(prefect.__module_path__.parent),
        "--factory",
        "prefect.server.api.server:create_app",
        "--host",
        str(host),
        "--port",
        str(port),
        "--timeout-keep-alive",
        str(keep_alive_timeout),
    ]
    logger.debug("Opening server process with command: %s", shlex.join(command))

    env: Dict[str, str] = {**os.environ, **server_settings, "PREFECT__SERVER_FINAL": "1"}
    if no_services:
        env["PREFECT__SERVER_WEBSERVER_ONLY"] = "1"

    process: subprocess.Popen = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    process_id: int = process.pid
    pid_file.write_text(str(process_id))

    app.console.print(
        "The Prefect server is running in the background. Run `prefect"
        " server stop` to stop it."
    )


def _run_in_foreground(
    server_settings: Dict[str, str],
    host: str,
    port: int,
    keep_alive_timeout: int,
    no_services: bool,
) -> None:
    from prefect.server.api.server import create_app

    try:
        with temporary_settings(
            {getattr(prefect.settings, k): v for k, v in server_settings.items()}
        ):
            uvicorn.run(
                app=create_app(final=True, webserver_only=no_services),
                app_dir=str(prefect.__module_path__.parent),
                host=host,
                port=port,
                timeout_keep_alive=keep_alive_timeout,
                log_level=server_settings.get(
                    "PREFECT_SERVER_LOGGING_LEVEL", "info"
                ).lower(),
            )
    finally:
        app.console.print("Server stopped!")


@server_app.command()
async def stop() -> None:
    pid_file: Path = Path(PREFECT_HOME.value()) / SERVER_PID_FILE_NAME
    if not pid_file.exists():
        exit_with_success("No server running in the background.")
    pid: int = int(pid_file.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        exit_with_success(
            "The server process is not running. Cleaning up stale PID file."
        )
    finally:
        pid_file.unlink(missing_ok=True)
    app.console.print("Server stopped!")


@database_app.command()
async def reset(yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    from prefect.server.database import provide_database_interface

    db = provide_database_interface()
    engine = await db.engine()
    if not yes:
        confirm: bool = typer.confirm(
            "Are you sure you want to reset the Prefect database located "
            f'at "{engine.url!r}"? This will drop and recreate all tables.'
        )
        if not confirm:
            exit_with_error("Database reset aborted")
    app.console.print("Downgrading database...")
    await db.drop_db()
    app.console.print("Upgrading database...")
    await db.create_db()
    exit_with_success(f'Prefect database "{engine.url!r}" reset!')


@database_app.command()
async def upgrade(
    yes: bool = typer.Option(False, "--yes", "-y"),
    revision: str = typer.Option(
        "head",
        "-r",
        help=(
            "The revision to pass to `alembic upgrade`. If not provided, runs all"
            " migrations."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        help=(
            "Flag to show what migrations would be made without applying them. Will"
            " emit sql statements to stdout."
        ),
    ),
) -> None:
    from prefect.server.database import provide_database_interface
    from prefect.server.database.alembic_commands import alembic_upgrade

    db = provide_database_interface()
    engine = await db.engine()

    if not yes:
        confirm: bool = typer.confirm(
            f"Are you sure you want to upgrade the Prefect database at {engine.url!r}?"
        )
        if not confirm:
            exit_with_error("Database upgrade aborted!")

    app.console.print("Running upgrade migrations ...")
    await run_sync_in_worker_thread(alembic_upgrade, revision=revision, dry_run=dry_run)
    app.console.print("Migrations succeeded!")
    exit_with_success(f"Prefect database at {engine.url!r} upgraded!")


@database_app.command()
async def downgrade(
    yes: bool = typer.Option(False, "--yes", "-y"),
    revision: str = typer.Option(
        "-1",
        "-r",
        help=(
            "The revision to pass to `alembic downgrade`. If not provided, "
            "downgrades to the most recent revision. Use 'base' to run all "
            "migrations."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        help=(
            "Flag to show what migrations would be made without applying them. Will"
            " emit sql statements to stdout."
        ),
    ),
) -> None:
    from prefect.server.database import provide_database_interface
    from prefect.server.database.alembic_commands import alembic_downgrade

    db = provide_database_interface()

    engine = await db.engine()

    if not yes:
        confirm: bool = typer.confirm(
            "Are you sure you want to downgrade the Prefect "
            f"database at {engine.url!r}?"
        )
        if not confirm:
            exit_with_error("Database downgrade aborted!")

    app.console.print("Running downgrade migrations ...")
    await run_sync_in_worker_thread(
        alembic_downgrade, revision=revision, dry_run=dry_run
    )
    app.console.print("Migrations succeeded!")
    exit_with_success(f"Prefect database at {engine.url!r} downgraded!")


@database_app.command()
async def revision(
    message: Optional[str] = typer.Option(
        None,
        "--message",
        "-m",
        help="A message to describe the migration.",
    ),
    autogenerate: bool = False,
) -> None:
    from prefect.server.database.alembic_commands import alembic_revision

    app.console.print("Running migration file creation ...")
    await run_sync_in_worker_thread(
        alembic_revision,
        message=message,
        autogenerate=autogenerate,
    )
    exit_with_success("Creating new migration file succeeded!")


@database_app.command()
async def stamp(revision: str) -> None:
    from prefect.server.database.alembic_commands import alembic_stamp

    app.console.print("Stamping database with revision ...")
    await run_sync_in_worker_thread(alembic_stamp, revision=revision)
    exit_with_success("Stamping database with revision succeeded!")


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def