from __future__ import annotations
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, TypedDict, Union, runtime_checkable
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
    def set_base_path(self, path: Path) -> None:
        ...

    @property
    def pull_interval(self) -> Optional[int]:
        ...

    @property
    def destination(self) -> Path:
        ...

    async def pull_code(self) -> None:
        ...

    def to_pull_step(self) -> Dict[str, Any]:
        ...

    def __eq__(self, __value: object) -> bool:
        ...

class GitCredentials(TypedDict, total=False):
    username: Optional[str]
    access_token: Optional[Union[Secret, SecretStr]]
    password: Optional[Union[Secret, SecretStr]]

class GitRepository:
    def __init__(self, url: str, credentials: Optional[GitCredentials] = None, name: Optional[str] = None, branch: Optional[str] = None, include_submodules: bool = False, pull_interval: int = 60, directories: Optional[list[str]] = None) -> None:
        ...

    @property
    def destination(self) -> Path:
        ...

    def set_base_path(self, path: Path) -> None:
        ...

    @property
    def pull_interval(self) -> int:
        ...

    @property
    def _formatted_credentials(self) -> Optional[str]:
        ...

    def _add_credentials_to_url(self, url: str) -> str:
        ...

    @property
    def _repository_url_with_credentials(self) -> str:
        ...

    @property
    def _git_config(self) -> list[str]:
        ...

    async def is_sparsely_checked_out(self) -> bool:
        ...

    async def pull_code(self) -> None:
        ...

    async def _clone_repo(self) -> None:
        ...

    def __eq__(self, __value: object) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def to_pull_step(self) -> Dict[str, Any]:
        ...

class RemoteStorage:
    def __init__(self, url: str, pull_interval: int = 60, **settings: Any) -> None:
        ...

    @staticmethod
    def _get_required_package_for_scheme(scheme: str) -> Optional[str]:
        ...

    @property
    def _filesystem(self) -> fsspec.core.OpenFileSystem:
        ...

    def set_base_path(self, path: Path) -> None:
        ...

    @property
    def pull_interval(self) -> int:
        ...

    @property
    def destination(self) -> Path:
        ...

    @property
    def _remote_path(self) -> Path:
        ...

    async def pull_code(self) -> None:
        ...

    def to_pull_step(self) -> Dict[str, Any]:
        ...

    def __eq__(self, __value: object) -> bool:
        ...

    def __repr__(self) -> str:
        ...

class BlockStorageAdapter:
    def __init__(self, block: Block, pull_interval: int = 60) -> None:
        ...

    def set_base_path(self, path: Path) -> None:
        ...

    @property
    def pull_interval(self) -> int:
        ...

    @property
    def destination(self) -> Path:
        ...

    async def pull_code(self) -> None:
        ...

    def to_pull_step(self) -> Dict[str, Any]:
        ...

    def __eq__(self, __value: object) -> bool:
        ...

class LocalStorage:
    def __init__(self, path: str, pull_interval: Optional[int] = None) -> None:
        ...

    @property
    def destination(self) -> Path:
        ...

    def set_base_path(self, path: Path) -> None:
        ...

    @property
    def pull_interval(self) -> Optional[int]:
        ...

    async def pull_code(self) -> None:
        ...

    def to_pull_step(self) -> Dict[str, Any]:
        ...

    def __eq__(self, __value: object) -> bool:
        ...

    def __repr__(self) -> str:
        ...

def create_storage_from_source(source: str, pull_interval: int = 60) -> RunnerStorage:
    ...

def _format_token_from_credentials(netloc: str, credentials: GitCredentials) -> str:
    ...

def _strip_auth_from_url(url: str) -> str:
    ...
