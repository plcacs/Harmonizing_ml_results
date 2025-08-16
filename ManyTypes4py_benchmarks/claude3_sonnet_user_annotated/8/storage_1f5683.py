from __future__ import annotations

import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)
from urllib.parse import urlparse, urlsplit, urlunparse
from uuid import uuid4

import fsspec
from anyio import run_process
from pydantic import SecretStr

from prefect._internal.concurrency.api import create_call, from_async
from prefect.blocks.core import Block, BlockNotSavedError
from prefect.blocks.system import Secret
from prefect.filesystems import ReadableDeploymentStorage, WritableDeploymentStorage
from prefect.logging.loggers import get_logger
from prefect.utilities.collections import visit_collection


@runtime_checkable
class RunnerStorage(Protocol):
    """
    A storage interface for a runner to use to retrieve
    remotely stored flow code.
    """

    def set_base_path(self, path: Path) -> None:
        """
        Sets the base path to use when pulling contents from remote storage to
        local storage.
        """
        ...

    @property
    def pull_interval(self) -> Optional[int]:
        """
        The interval at which contents from remote storage should be pulled to
        local storage. If None, remote storage will perform a one-time sync.
        """
        ...

    @property
    def destination(self) -> Path:
        """
        The local file path to pull contents from remote storage to.
        """
        ...

    async def pull_code(self) -> None:
        """
        Pulls contents from remote storage to the local filesystem.
        """
        ...

    def to_pull_step(self) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Returns a dictionary representation of the storage object that can be
        used as a deployment pull step.
        """
        ...

    def __eq__(self, __value: Any) -> bool:
        """
        Equality check for runner storage objects.
        """
        ...


class GitCredentials(TypedDict, total=False):
    username: str
    access_token: Union[str, Secret[str]]


class GitRepository:
    """
    Pulls the contents of a git repository to the local filesystem.

    Parameters:
        url: The URL of the git repository to pull from
        credentials: A dictionary of credentials to use when pulling from the
            repository. If a username is provided, an access token must also be
            provided.
        name: The name of the repository. If not provided, the name will be
            inferred from the repository URL.
        branch: The branch to pull from. Defaults to "main".
        pull_interval: The interval in seconds at which to pull contents from
            remote storage to local storage. If None, remote storage will perform
            a one-time sync.
        directories: The directories to pull from the Git repository (uses git sparse-checkout)

    Examples:
        Pull the contents of a private git repository to the local filesystem:

        