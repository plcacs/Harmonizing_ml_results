```python
import sys
from collections.abc import Generator, Iterable, Iterator
from contextlib import AbstractContextManager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, TextIO, Union
from urllib.parse import SplitResult

if TYPE_CHECKING:
    import docker
    import docker.errors
    from docker import DockerClient
    from docker.models.images import Image

CONTAINER_LABELS: dict[str, str] = ...
IMAGE_LABELS: dict[str, str] = ...

def python_version_minor() -> str: ...
def python_version_micro() -> str: ...

def get_prefect_image_name(
    prefect_version: Optional[str] = None,
    python_version: Optional[str] = None,
    flavor: Optional[str] = None
) -> str: ...

@contextmanager
def silence_docker_warnings() -> Generator[None, None, None]: ...

@contextmanager
def docker_client() -> Generator["DockerClient", None, None]: ...

class BuildError(Exception): ...

def build_image(
    context: Union[str, Path],
    dockerfile: str = "Dockerfile",
    tag: Optional[str] = None,
    pull: bool = False,
    platform: Optional[str] = None,
    stream_progress_to: Optional[TextIO] = None,
    **kwargs: Any
) -> str: ...

class ImageBuilder:
    def __init__(
        self,
        base_image: str,
        base_directory: Optional[Union[str, Path]] = None,
        platform: Optional[str] = None,
        context: Optional[Union[str, Path]] = None
    ) -> None: ...
    def __enter__(self) -> "ImageBuilder": ...
    def __exit__(
        self,
        exc: Optional[type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType]
    ) -> Optional[bool]: ...
    def add_line(self, line: str) -> None: ...
    def add_lines(self, lines: Iterable[str]) -> None: ...
    def copy(
        self,
        source: Union[str, Path],
        destination: Union[str, PurePosixPath]
    ) -> None: ...
    def write_text(self, text: str, destination: Union[str, PurePosixPath]) -> None: ...
    def build(
        self,
        pull: bool = False,
        stream_progress_to: Optional[TextIO] = None
    ) -> str: ...
    def assert_has_line(self, line: str) -> None: ...
    def assert_line_absent(self, line: str) -> None: ...
    def assert_line_before(self, first: str, second: str) -> None: ...
    def assert_line_after(self, second: str, first: str) -> None: ...
    def assert_has_file(
        self,
        source: Union[str, Path],
        container_path: Union[str, PurePosixPath]
    ) -> None: ...

class PushError(Exception): ...

def push_image(
    image_id: str,
    registry_url: str,
    name: str,
    tag: Optional[str] = None,
    stream_progress_to: Optional[TextIO] = None
) -> str: ...

def to_run_command(command: list[str]) -> str: ...

def parse_image_tag(name: str) -> tuple[str, Optional[str]]: ...

def split_repository_path(repository_path: str) -> tuple[Optional[str], str]: ...

def format_outlier_version_name(version: str) -> str: ...

@contextmanager
def generate_default_dockerfile(
    context: Optional[Union[str, Path]] = None
) -> Generator[Path, None, None]: ...
```