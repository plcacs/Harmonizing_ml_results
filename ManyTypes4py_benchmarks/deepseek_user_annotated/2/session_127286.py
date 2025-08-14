"""This module implements Kedro session responsible for project lifecycle."""

from __future__ import annotations

import getpass
import logging
import logging.config
import os
import subprocess
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, List, Set, Tuple

import click

from kedro import __version__ as kedro_version
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.hooks.manager import _register_hooks, _register_hooks_entry_points
from kedro.framework.project import (
    pipelines,
    settings,
    validate_settings,
)
from kedro.io.core import generate_timestamp
from kedro.runner import AbstractRunner, SequentialRunner
from kedro.utils import _find_kedro_project

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kedro.config import AbstractConfigLoader
    from kedro.framework.context import KedroContext
    from kedro.framework.session.store import BaseSessionStore


def _describe_git(project_path: Path) -> Dict[str, Dict[str, Any]]:
    path = str(project_path)
    try:
        res = subprocess.check_output(  # noqa: S603
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            cwd=path,
            stderr=subprocess.STDOUT,
        )
        git_data: Dict[str, Any] = {"commit_sha": res.decode().strip()}
        git_status_res = subprocess.check_output(  # noqa: S603
            ["git", "status", "--short"],  # noqa: S607
            cwd=path,
            stderr=subprocess.STDOUT,
        )
        git_data["dirty"] = bool(git_status_res.decode().strip())

    # `subprocess.check_output()` raises `NotADirectoryError` on Windows
    except Exception:
        logger = logging.getLogger(__name__)
        logger.debug("Unable to git describe %s", path)
        logger.debug(traceback.format_exc())
        return {}

    return {"git": git_data}


def _jsonify_cli_context(ctx: click.core.Context) -> Dict[str, Any]:
    return {
        "args": ctx.args,
        "params": ctx.params,
        "command_name": ctx.command.name,
        "command_path": " ".join(["kedro"] + sys.argv[1:]),
    }


class KedroSessionError(Exception):
    """``KedroSessionError`` raised by ``KedroSession``
    in the case that multiple runs are attempted in one session.
    """

    pass


class KedroSession:
    """``KedroSession`` is the object that is responsible for managing the lifecycle
    of a Kedro run. Use `KedroSession.create()` as
    a context manager to construct a new KedroSession with session data
    provided (see the example below).