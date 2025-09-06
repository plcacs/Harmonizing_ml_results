import json
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
import docker.errors
import pendulum
from docker.models.images import Image
from typing_extensions import TypedDict
from prefect.logging.loggers import get_logger
from prefect.utilities.dockerutils import (
    IMAGE_LABELS,
    BuildError,
    docker_client,
    get_prefect_image_name,
)
from prefect.utilities.slugify import slugify

logger = get_logger('prefect_docker.deployments.steps')
STEP_OUTPUT_CACHE: Dict[Any, Any] = {}

T = TypeVar("T")


class BuildDockerImageResult(TypedDict):
    image_name: str
    tag: str
    image: str
    image_id: str
    additional_tags: List[str]


class PushDockerImageResult(TypedDict):
    image_name: str
    tag: str
    image: str
    additional_tags: List[str]


def func_8ijpuhbk(obj: Any) -> Any:
    if isinstance(obj, dict):
        return json.dumps(obj, sort_keys=True)
    elif isinstance(obj, list):
        return tuple(func_8ijpuhbk(v) for v in obj)
    return obj


def cacheable(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if (ignore_cache := kwargs.pop('ignore_cache', False)):
            logger.debug(f'Ignoring `@cacheable` decorator for {func.__name__}.')
        key = (
            func.__name__,
            tuple(func_8ijpuhbk(arg) for arg in args),
            tuple((k, func_8ijpuhbk(v)) for k, v in sorted(kwargs.items())),
        )
        if ignore_cache or key not in STEP_OUTPUT_CACHE:
            logger.debug(f'Cache miss for {func.__name__}, running function.')
            STEP_OUTPUT_CACHE[key] = func(*args, **kwargs)
        else:
            logger.debug(f'Cache hit for {func.__name__}, returning cached value.')
        return STEP_OUTPUT_CACHE[key]
    return wrapper


@cacheable
def func_df6obeyp(
    image_name: str,
    dockerfile: str = 'Dockerfile',
    tag: Optional[str] = None,
    additional_tags: Optional[List[str]] = None,
    ignore_cache: bool = False,
    **build_kwargs: Any
) -> BuildDockerImageResult:
    auto_build: bool = dockerfile == 'auto'
    if auto_build:
        lines: List[str] = []
        base_image: str = get_prefect_image_name()
        lines.append(f'FROM {base_image}')
        dir_name: str = os.path.basename(os.getcwd())
        if Path('requirements.txt').exists():
            lines.append(
                f'COPY requirements.txt /opt/prefect/{dir_name}/requirements.txt'
            )
            lines.append(
                f'RUN python -m pip install -r /opt/prefect/{dir_name}/requirements.txt'
            )
        lines.append(f'COPY . /opt/prefect/{dir_name}/')
        lines.append(f'WORKDIR /opt/prefect/{dir_name}/')
        temp_dockerfile: Path = Path('Dockerfile')
        if Path(temp_dockerfile).exists():
            raise ValueError('Dockerfile already exists.')
        with Path(temp_dockerfile).open('w') as f:
            f.writelines(line + '\n' for line in lines)
        dockerfile = str(temp_dockerfile)
    build_kwargs['path'] = build_kwargs.get('path', os.getcwd())
    build_kwargs['dockerfile'] = dockerfile
    build_kwargs['pull'] = build_kwargs.get('pull', True)
    build_kwargs['decode'] = True
    build_kwargs['labels'] = {**build_kwargs.get('labels', {}), **IMAGE_LABELS}
    image_id: Optional[str] = None
    with docker_client() as client:
        try:
            events = client.api.build(**build_kwargs)
            try:
                for event in events:
                    if 'stream' in event:
                        sys.stdout.write(event['stream'])
                        sys.stdout.flush()
                    elif 'aux' in event:
                        image_id = event['aux']['ID']
                    elif 'error' in event:
                        raise BuildError(event['error'])
                    elif 'message' in event:
                        raise BuildError(event['message'])
            except docker.errors.APIError as e:
                raise BuildError(e.explanation) from e
        finally:
            if auto_build:
                os.unlink(dockerfile)
        if not isinstance(image_id, str):
            raise BuildError('Docker did not return an image ID for built image.')
        if not tag:
            tag = slugify(pendulum.now('utc').isoformat())
        image: Image = client.images.get(image_id)
        image.tag(repository=image_name, tag=tag)
        additional_tags = additional_tags or []
        for tag_ in additional_tags:
            image.tag(repository=image_name, tag=tag_)
    return {
        'image_name': image_name,
        'tag': tag,
        'image': f'{image_name}:{tag}',
        'image_id': image_id,
        'additional_tags': additional_tags,
    }


@cacheable
def func_mhuw1rs3(
    image_name: str,
    tag: Optional[str] = None,
    credentials: Optional[Dict[str, Any]] = None,
    additional_tags: Optional[List[str]] = None,
    ignore_cache: bool = False,
) -> PushDockerImageResult:
    with docker_client() as client:
        if credentials is not None:
            client.login(
                username=credentials.get('username'),
                password=credentials.get('password'),
                registry=credentials.get('registry_url'),
                reauth=credentials.get('reauth', True),
            )
        events: List[Dict[str, Any]] = list(
            client.api.push(
                repository=image_name, tag=tag, stream=True, decode=True
            )
        )
        additional_tags = additional_tags or []
        for tag_ in additional_tags:
            event: List[Dict[str, Any]] = list(
                client.api.push(
                    repository=image_name, tag=tag_, stream=True, decode=True
                )
            )
            events += event
        for event in events:
            if 'status' in event:
                sys.stdout.write(event['status'])
                if 'progress' in event:
                    sys.stdout.write(' ' + event['progress'])
                sys.stdout.write('\n')
                sys.stdout.flush()
            elif 'error' in event:
                raise OSError(event['error'])
    return {
        'image_name': image_name,
        'tag': tag if tag is not None else "",
        'image': f'{image_name}:{tag}' if tag is not None else image_name,
        'additional_tags': additional_tags,
    }