#!/usr/bin/env python3
import hashlib
import os
import shutil
import sys
import warnings
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, TextIO, Union, cast
from urllib.parse import urlsplit
from packaging.version import Version
from typing_extensions import Self

import prefect
from prefect.types._datetime import now
from prefect.utilities.importtools import lazy_import
from prefect.utilities.slugify import slugify

if TYPE_CHECKING:
    import docker
    import docker.errors
    from docker import DockerClient
    from docker.models.images import Image

CONTAINER_LABELS: dict[str, str] = {'io.prefect.version': prefect.__version__}


def python_version_minor() -> str:
    return f'{sys.version_info.major}.{sys.version_info.minor}'


def python_version_micro() -> str:
    return f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'


def get_prefect_image_name(
    prefect_version: Optional[str] = None,
    python_version: Optional[str] = None,
    flavor: Optional[str] = None
) -> str:
    """
    Get the Prefect image name matching the current Prefect and Python versions.
    """
    parsed_version = Version(prefect_version or prefect.__version__)
    is_prod_build = parsed_version.local is None
    prefect_version_str: str = (
        parsed_version.base_version
        if is_prod_build
        else 'sha-' + prefect.__version_info__['full-revisionid'][:7]
    )
    python_version = python_version or python_version_minor()
    tag = slugify(
        f'{prefect_version_str}-python{python_version}' + (f'-{flavor}' if flavor else ''),
        lowercase=False,
        max_length=128,
        regex_pattern='[^a-zA-Z0-9_.-]+'
    )
    image = 'prefect' if is_prod_build else 'prefect-dev'
    return f'prefecthq/{image}:{tag}'


@contextmanager
def silence_docker_warnings() -> Generator[None, None, None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='distutils Version classes are deprecated.*',
            category=DeprecationWarning
        )
        warnings.filterwarnings(
            'ignore',
            message='The distutils package is deprecated and slated for removal.*',
            category=DeprecationWarning
        )
        yield


with silence_docker_warnings():
    if not TYPE_CHECKING:
        docker = lazy_import('docker')


@contextmanager
def docker_client() -> Generator["docker.DockerClient", None, None]:
    """Get the environmentally-configured Docker client"""
    client: Optional["docker.DockerClient"] = None
    try:
        with silence_docker_warnings():
            client = docker.DockerClient.from_env()  # type: ignore
            yield client
    except docker.errors.DockerException as exc:  # type: ignore
        raise RuntimeError(
            'This error is often thrown because Docker is not running. Please ensure Docker is running.'
        ) from exc
    finally:
        if client is not None:
            client.close()


class BuildError(Exception):
    """Raised when a Docker build fails"""


IMAGE_LABELS: dict[str, str] = {'io.prefect.version': prefect.__version__}


@silence_docker_warnings()
def build_image(
    context: Union[str, Path],
    dockerfile: str = 'Dockerfile',
    tag: Optional[str] = None,
    pull: bool = False,
    platform: Optional[str] = None,
    stream_progress_to: Optional[TextIO] = None,
    **kwargs: Any
) -> str:
    """Builds a Docker image, returning the image ID

    Args:
        context: the root directory for the Docker build context
        dockerfile: the path to the Dockerfile, relative to the context
        tag: the tag to give this image
        pull: True to pull the base image during the build
        stream_progress_to: an optional stream (like sys.stdout, or an io.TextIO) that
            will collect the build output as it is reported by Docker

    Returns:
        The image ID
    """
    if not context:
        raise ValueError('context required to build an image')
    context_path = Path(context) if isinstance(context, str) else context
    if not context_path.exists():
        raise ValueError(f'Context path {context} does not exist')
    kwargs = {key: kwargs[key] for key in kwargs if key not in ['decode', 'labels']}
    image_id: Optional[str] = None
    with docker_client() as client:
        events: Iterator[dict[str, Any]] = client.api.build(
            path=str(context_path),
            tag=tag,
            dockerfile=dockerfile,
            pull=pull,
            decode=True,
            labels=IMAGE_LABELS,
            platform=platform,
            **kwargs
        )
        try:
            for event in events:
                if 'stream' in event:
                    if not stream_progress_to:
                        continue
                    stream_progress_to.write(event['stream'])
                    stream_progress_to.flush()
                elif 'aux' in event:
                    image_id = event['aux']['ID']
                elif 'error' in event:
                    raise BuildError(event['error'])
                elif 'message' in event:
                    raise BuildError(event['message'])
        except docker.errors.APIError as e:  # type: ignore
            raise BuildError(e.explanation) from e
    assert image_id, 'The Docker daemon did not return an image ID'
    return image_id


class ImageBuilder:
    """An interface for preparing Docker build contexts and building images"""

    def __init__(
        self,
        base_image: str,
        base_directory: Optional[Union[str, Path]] = None,
        platform: Optional[str] = None,
        context: Optional[Path] = None
    ) -> None:
        """Create an ImageBuilder

        Args:
            base_image: the base image to use
            base_directory: the starting point on your host for relative file locations,
                defaulting to the current directory
            context: use this path as the build context (if not provided, will create a
                temporary directory for the context)
        """
        if base_directory is None:
            self.base_directory: Path = context if context is not None else Path().absolute()
        else:
            self.base_directory = Path(base_directory)
        self.temporary_directory: Optional[TemporaryDirectory] = None
        self.context: Optional[Path] = context
        self.platform = platform
        self.dockerfile_lines: list[str] = []
        if self.context:
            dockerfile_path = self.context / 'Dockerfile'
            if dockerfile_path.exists():
                raise ValueError(f'There is already a Dockerfile at {context}')
        self.add_line(f'FROM {base_image}')

    def __enter__(self) -> Self:
        if self.context and (not self.temporary_directory):
            return self
        self.temporary_directory = TemporaryDirectory()
        self.context = Path(self.temporary_directory.__enter__())
        return self

    def __exit__(
        self,
        exc: Optional[BaseException],
        value: Optional[BaseException],
        traceback: Optional[TracebackType]
    ) -> None:
        if not self.temporary_directory:
            return
        self.temporary_directory.__exit__(exc, value, traceback)
        self.temporary_directory = None
        self.context = None

    def add_line(self, line: str) -> None:
        """Add a line to this image's Dockerfile"""
        self.add_lines([line])

    def add_lines(self, lines: Iterable[str]) -> None:
        """Add lines to this image's Dockerfile"""
        self.dockerfile_lines.extend(lines)

    def copy(self, source: Union[Path, str], destination: Union[PurePosixPath, str]) -> None:
        """Copy a file to this image"""
        if not self.context:
            raise Exception('No context available')
        dest_path: PurePosixPath = destination if isinstance(destination, PurePosixPath) else PurePosixPath(destination)
        source_path: Path = source if isinstance(source, Path) else Path(source)
        if source_path.is_absolute():
            source_path = source_path.resolve().relative_to(self.base_directory)
        if self.temporary_directory:
            os.makedirs(self.context / source_path.parent, exist_ok=True)
            if (self.base_directory / source_path).is_dir():
                shutil.copytree(self.base_directory / source_path, self.context / source_path)
            else:
                shutil.copy2(self.base_directory / source_path, self.context / source_path)
        self.add_line(f'COPY {source_path} {dest_path}')

    def write_text(self, text: str, destination: Union[PurePosixPath, str]) -> None:
        if not self.context:
            raise Exception('No context available')
        dest_path: PurePosixPath = destination if isinstance(destination, PurePosixPath) else PurePosixPath(destination)
        source_hash: str = hashlib.sha256(text.encode()).hexdigest()
        (self.context / f'.{source_hash}').write_text(text)
        self.add_line(f'COPY .{source_hash} {dest_path}')

    def build(self, pull: bool = False, stream_progress_to: Optional[TextIO] = None) -> str:
        """Build the Docker image from the current state of the ImageBuilder

        Args:
            pull: True to pull the base image during the build
            stream_progress_to: an optional stream (like sys.stdout, or an io.TextIO)
                that will collect the build output as it is reported by Docker

        Returns:
            The image ID
        """
        assert self.context is not None
        dockerfile_path = self.context / 'Dockerfile'
        with dockerfile_path.open('w') as dockerfile:
            dockerfile.writelines((line + '\n' for line in self.dockerfile_lines))
        try:
            return build_image(
                self.context,
                platform=self.platform,
                pull=pull,
                stream_progress_to=stream_progress_to
            )
        finally:
            os.unlink(dockerfile_path)

    def assert_has_line(self, line: str) -> None:
        """Asserts that the given line is in the Dockerfile"""
        all_lines = '\n'.join([f'  {i + 1:>3}: {l}' for i, l in enumerate(self.dockerfile_lines)])
        message = f'Expected {line!r} not found in Dockerfile.  Dockerfile:\n{all_lines}'
        assert line in self.dockerfile_lines, message

    def assert_line_absent(self, line: str) -> None:
        """Asserts that the given line is absent from the Dockerfile"""
        if line not in self.dockerfile_lines:
            return
        i = self.dockerfile_lines.index(line)
        surrounding_lines = '\n'.join(
            [f'  {j + 1:>3}: {l}' for j, l in enumerate(self.dockerfile_lines[max(0, i - 2): i + 2])]
        )
        message = f'Unexpected {line!r} found in Dockerfile at line {i + 1}.  Surrounding lines:\n{surrounding_lines}'
        assert line not in self.dockerfile_lines, message

    def assert_line_before(self, first: str, second: str) -> None:
        """Asserts that the first line appears before the second line"""
        self.assert_has_line(first)
        self.assert_has_line(second)
        first_index = self.dockerfile_lines.index(first)
        second_index = self.dockerfile_lines.index(second)
        surrounding_lines = '\n'.join(
            [f'  {j + 1:>3}: {l}' for j, l in enumerate(self.dockerfile_lines[min(first_index, second_index): max(first_index, second_index) + 1])]
        )
        message = (
            f'Expected {first!r} to appear before {second!r} in the Dockerfile, but '
            f'{first!r} was at line {first_index + 1} and {second!r} at line {second_index + 1}.  '
            f'Surrounding lines:\n{surrounding_lines}'
        )
        assert first_index < second_index, message

    def assert_line_after(self, second: str, first: str) -> None:
        """Asserts that the second line appears after the first line"""
        self.assert_line_before(first, second)

    def assert_has_file(self, source: Path, container_path: Union[PurePosixPath, str]) -> None:
        """Asserts that the given file or directory will be copied into the container
        at the given path"""
        source_relative: Path = source if not source.is_absolute() else source.relative_to(self.base_directory)
        container_dest: PurePosixPath = container_path if isinstance(container_path, PurePosixPath) else PurePosixPath(container_path)
        self.assert_has_line(f'COPY {source_relative} {container_dest}')


class PushError(Exception):
    """Raised when a Docker image push fails"""


@silence_docker_warnings()
def push_image(
    image_id: str,
    registry_url: str,
    name: str,
    tag: Optional[str] = None,
    stream_progress_to: Optional[TextIO] = None
) -> str:
    """Pushes a local image to a Docker registry, returning the registry-qualified tag
    for that image

    Args:
        image_id (str): a Docker image ID
        registry_url (str): the URL of a Docker registry
        name (str): the name of this image
        tag (str): the tag to give this image (defaults to a short representation of
            the image's ID)
        stream_progress_to: an optional stream (like sys.stdout, or an io.TextIO) that
            will collect the build output as it is reported by Docker

    Returns:
        A registry-qualified tag, like my-registry.example.com/my-image:abcdefg
    """
    if not tag:
        tag = slugify(now('UTC').isoformat())
    _, registry, _, _, _ = urlsplit(registry_url)
    repository: str = f'{registry}/{name}'
    with docker_client() as client:
        image = client.images.get(image_id)  # type: ignore
        image.tag(repository, tag=tag)
        events: Iterator[dict[str, Any]] = cast(
            Iterator[dict[str, Any]],
            client.api.push(repository, tag=tag, stream=True, decode=True)
        )
        try:
            for event in events:
                if 'status' in event:
                    if not stream_progress_to:
                        continue
                    stream_progress_to.write(event['status'])
                    if 'progress' in event:
                        stream_progress_to.write(' ' + event['progress'])
                    stream_progress_to.write('\n')
                    stream_progress_to.flush()
                elif 'error' in event:
                    raise PushError(event['error'])
        finally:
            client.api.remove_image(f'{repository}:{tag}', noprune=True)
    return f'{repository}:{tag}'


def to_run_command(command: list[str]) -> str:
    """
    Convert a process-style list of command arguments to a single Dockerfile RUN
    instruction.
    """
    if not command:
        return ''
    run_command: str = f'RUN {command[0]}'
    if len(command) > 1:
        run_command += ' ' + ' '.join([repr(arg) for arg in command[1:]])
    return run_command


def parse_image_tag(name: str) -> tuple[str, Optional[str]]:
    """
    Parse Docker Image String

    - If a tag exists, this function parses and returns the image registry and tag,
      separately as a tuple.
    - Example 1: 'prefecthq/prefect:latest' -> ('prefecthq/prefect', 'latest')
    - Example 2: 'hostname.io:5050/folder/subfolder:latest' -> ('hostname.io:5050/folder/subfolder', 'latest')
    """
    tag: Optional[str] = None
    name_parts = name.split('/')
    if len(name_parts) == 1:
        if ':' in name_parts[0]:
            image_name, tag = name_parts[0].split(':', 1)
        else:
            image_name = name_parts[0]
    else:
        index_name = name_parts[0]
        image_path = '/'.join(name_parts[1:])
        if ':' in image_path:
            image_path, tag = image_path.split(':', 1)
        image_name = f'{index_name}/{image_path}'
    return (image_name, tag)


def split_repository_path(repository_path: str) -> tuple[Optional[str], str]:
    """
    Splits a Docker repository path into its namespace and repository components.

    Args:
        repository_path: The Docker repository path to split.

    Returns:
        Tuple[Optional[str], str]: A tuple containing the namespace and repository components.
    """
    parts = repository_path.split('/', 2)
    if len(parts) == 3 or (len(parts) == 2 and ('.' in parts[0] or ':' in parts[0])):
        namespace = '/'.join(parts[:-1])
        repository = parts[-1]
    elif len(parts) == 2:
        namespace = parts[0]
        repository = parts[1]
    else:
        namespace = None
        repository = parts[0]
    return (namespace, repository)


def format_outlier_version_name(version: str) -> str:
    """
    Formats outlier docker version names to pass `packaging.version.parse` validation
    """
    return version.replace('-ce', '').replace('-ee', '')


@contextmanager
def generate_default_dockerfile(context: Optional[Path] = None) -> Generator[Path, None, None]:
    """
    Generates a default Dockerfile used for deploying flows.
    """
    if not context:
        context = Path.cwd()
    lines: list[str] = []
    base_image: str = get_prefect_image_name()
    lines.append(f'FROM {base_image}')
    dir_name: str = context.name
    if (context / 'requirements.txt').exists():
        lines.append(f'COPY requirements.txt /opt/prefect/{dir_name}/requirements.txt')
        lines.append(f'RUN python -m pip install -r /opt/prefect/{dir_name}/requirements.txt')
    lines.append(f'COPY . /opt/prefect/{dir_name}/')
    lines.append(f'WORKDIR /opt/prefect/{dir_name}/')
    temp_dockerfile: Path = context / 'Dockerfile'
    if temp_dockerfile.exists():
        raise RuntimeError('Failed to generate Dockerfile. Dockerfile already exists in the current directory.')
    with temp_dockerfile.open('w') as f:
        f.writelines((line + '\n' for line in lines))
    try:
        yield temp_dockerfile
    finally:
        temp_dockerfile.unlink()