import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Any, Optional, TextIO, Union

from typing_extensions import Self

if True:  # TYPE_CHECKING
    import docker
    import docker.errors
    from docker import DockerClient
    from docker.models.images import Image

CONTAINER_LABELS: dict[str, str]

def python_version_minor() -> str: ...
def python_version_micro() -> str: ...
def get_prefect_image_name(
    prefect_version: Optional[str] = ...,
    python_version: Optional[str] = ...,
    flavor: Optional[str] = ...,
) -> str: ...

@contextmanager
def silence_docker_warnings() -> Generator[None, None, None]: ...
@contextmanager
def docker_client() -> Generator[DockerClient, None, None]: ...

class BuildError(Exception): ...

IMAGE_LABELS: dict[str, str]

def build_image(
    context: Union[str, Path],
    dockerfile: str = ...,
    tag: Optional[str] = ...,
    pull: bool = ...,
    platform: Optional[str] = ...,
    stream_progress_to: Optional[TextIO] = ...,
    **kwargs: Any,
) -> str: ...

class ImageBuilder:
    base_directory: Path
    temporary_directory: Optional[TemporaryDirectory[str]]
    context: Optional[Path]
    platform: Optional[str]
    dockerfile_lines: list[str]

    def __init__(
        self,
        base_image: str,
        base_directory: Optional[Union[str, Path]] = ...,
        platform: Optional[str] = ...,
        context: Optional[Path] = ...,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc: Optional[type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None: ...
    def add_line(self, line: str) -> None: ...
    def add_lines(self, lines: list[str]) -> None: ...
    def copy(self, source: Union[str, Path], destination: Union[str, PurePosixPath]) -> None: ...
    def write_text(self, text: str, destination: Union[str, PurePosixPath]) -> None: ...
    def build(
        self,
        pull: bool = ...,
        stream_progress_to: Optional[TextIO] = ...,
    ) -> str: ...
    def assert_has_line(self, line: str) -> None: ...
    def assert_line_absent(self, line: str) -> None: ...
    def assert_line_before(self, first: str, second: str) -> None: ...
    def assert_line_after(self, second: str, first: str) -> None: ...
    def assert_has_file(self, source: Path, container_path: Union[str, PurePosixPath]) -> None: ...

class PushError(Exception): ...

def push_image(
    image_id: str,
    registry_url: str,
    name: str,
    tag: Optional[str] = ...,
    stream_progress_to: Optional[TextIO] = ...,
) -> str: ...
def to_run_command(command: list[str]) -> str: ...
def parse_image_tag(name: str) -> tuple[str, Optional[str]]: ...
def split_repository_path(repository_path: str) -> tuple[Optional[str], str]: ...
def format_outlier_version_name(version: str) -> str: ...

@contextmanager
def generate_default_dockerfile(
    context: Optional[Path] = ...,
) -> Generator[Path, None, None]: ...