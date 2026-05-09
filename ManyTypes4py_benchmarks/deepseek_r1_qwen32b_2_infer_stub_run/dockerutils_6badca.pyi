import hashlib
import os
import shutil
import sys
import warnings
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import (
    Any,
    Optional,
    TextIO,
    Union,
    Generator,
    Iterator,
    Iterable,
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    TextIO,
    Union,
    Generator,
    Iterator,
    Iterable,
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    TextIO,
    Union,
    Generator,
    Iterator,
    Iterable,
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    TextIO,
    Union,
    Generator,
    Iterator,
    Iterable,
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    TextIO,
    Union,
)
from urllib.parse import urlsplit
from packaging.version import Version
from typing_extensions import Self

import docker
import prefect

CONTAINER_LABELS: Dict[str, str]

def python_version_minor() -> str: ...

def python_version_micro() -> str: ...

def get_prefect_image_name(
    prefect_version: Optional[str] = None,
    python_version: Optional[str] = None,
    flavor: Optional[str] = None,
) -> str: ...

@contextmanager
def silence_docker_warnings() -> Generator[None, None, None]: ...

@contextmanager
def docker_client() -> Generator[docker.DockerClient, None, None]: ...

class BuildError(Exception): ...

IMAGE_LABELS: Dict[str, str]

def build_image(
    context: Union[Path, str],
    dockerfile: str = 'Dockerfile',
    tag: Optional[str] = None,
    pull: bool = False,
    platform: Optional[str] = None,
    stream_progress_to: Optional[TextIO] = None,
    **kwargs: Any,
) -> str: ...

class ImageBuilder:
    def __init__(
        self,
        base_image: str,
        base_directory: Optional[Path] = None,
        platform: Optional[str] = None,
        context: Optional[Path] = None,
    ) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc: Optional[BaseException],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None: ...

    def add_line(self, line: str) -> None: ...

    def add_lines(self, lines: Iterable[str]) -> None: ...

    def copy(
        self,
        source: Union[Path, str],
        destination: Union[PurePosixPath, str],
    ) -> None: ...

    def write_text(
        self,
        text: str,
        destination: Union[PurePosixPath, str],
    ) -> None: ...

    def build(
        self,
        pull: bool = False,
        stream_progress_to: Optional[TextIO] = None,
    ) -> str: ...

    def assert_has_line(self, line: str) -> None: ...

    def assert_line_absent(self, line: str) -> None: ...

    def assert_line_before(self, first: str, second: str) -> None: ...

    def assert_line_after(self, second: str, first: str) -> None: ...

    def assert_has_file(
        self,
        source: Union[Path, str],
        container_path: Union[PurePosixPath, str],
    ) -> None: ...

class PushError(Exception): ...

def push_image(
    image_id: str,
    registry_url: str,
    name: str,
    tag: Optional[str] = None,
    stream_progress_to: Optional[TextIO] = None,
) -> str: ...

def to_run_command(command: List[str]) -> str: ...

def parse_image_tag(name: str) -> Tuple[str, Optional[str]]: ...

def split_repository_path(repository_path: str) -> Tuple[Optional[str], str]: ...

def format_outlier_version_name(version: str) -> str: ...

@contextmanager
def generate_default_dockerfile(
    context: Optional[Path] = None,
) -> Generator[Path, None, None]: ...