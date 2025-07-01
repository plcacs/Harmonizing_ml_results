import json
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import docker.errors
import pendulum
from docker.models.images import Image
from typing_extensions import TypedDict
from prefect.logging.loggers import get_logger
from prefect.utilities.dockerutils import IMAGE_LABELS, BuildError, docker_client, get_prefect_image_name
from prefect.utilities.slugify import slugify

logger = get_logger('prefect_docker.deployments.steps')
STEP_OUTPUT_CACHE: Dict[Any, Any] = {}

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

def _make_hashable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return json.dumps(obj, sort_keys=True)
    elif isinstance(obj, list):
        return tuple((_make_hashable(v) for v in obj))
    return obj

def cacheable(func: Any) -> Any:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if (ignore_cache := kwargs.pop('ignore_cache', False)):
            logger.debug(f'Ignoring `@cacheable` decorator for {func.__name__}.')
        key: Tuple = (func.__name__, tuple((_make_hashable(arg) for arg in args)), tuple(((k, _make_hashable(v)) for k, v in sorted(kwargs.items()))))
        if ignore_cache or key not in STEP_OUTPUT_CACHE:
            logger.debug(f'Cache miss for {func.__name__}, running function.')
            STEP_OUTPUT_CACHE[key] = func(*args, **kwargs)
        else:
            logger.debug(f'Cache hit for {func.__name__}, returning cached value.')
        return STEP_OUTPUT_CACHE[key]
    return wrapper

@cacheable
def build_docker_image(image_name: str, dockerfile: str = 'Dockerfile', tag: Optional[str] = None, additional_tags: Optional[List[str]] = None, ignore_cache: bool = False, **build_kwargs: Any) -> BuildDockerImageResult:
    auto_build = dockerfile == 'auto'
    if auto_build:
        lines: List[str] = []
        base_image = get_prefect_image_name()
        lines.append(f'FROM {base_image}')
        dir_name = os.path.basename(os.getcwd())
        if Path('requirements.txt').exists():
            lines.append(f'COPY requirements.txt /opt/prefect/{dir_name}/requirements.txt')
            lines.append(f'RUN python -m pip install -r /opt/prefect/{dir_name}/requirements.txt')
        lines.append(f'COPY . /opt/prefect/{dir_name}/')
        lines.append(f'WORKDIR /opt/prefect/{dir_name}/')
        temp_dockerfile = Path('Dockerfile')
        if Path(temp_dockerfile).exists():
            raise ValueError('Dockerfile already exists.')
        with Path(temp_dockerfile).open('w') as f:
            f.writelines((line + '\n' for line in lines))
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
        image = client.images.get(image_id)
        image.tag(repository=image_name, tag=tag)
        additional_tags = additional_tags or []
        for tag_ in additional_tags:
            image.tag(repository=image_name, tag=tag_)
    return {'image_name': image_name, 'tag': tag, 'image': f'{image_name}:{tag}', 'image_id': image_id, 'additional_tags': additional_tags}

@cacheable
def push_docker_image(image_name: str, tag: Optional[str] = None, credentials: Optional[Dict[str, str]] = None, additional_tags: Optional[List[str]] = None, ignore_cache: bool = False) -> PushDockerImageResult:
    with docker_client() as client:
        if credentials is not None:
            client.login(username=credentials.get('username'), password=credentials.get('password'), registry=credentials.get('registry_url'), reauth=credentials.get('reauth', True))
        events = list(client.api.push(repository=image_name, tag=tag, stream=True, decode=True))
        additional_tags = additional_tags or []
        for i, tag_ in enumerate(additional_tags):
            event = list(client.api.push(repository=image_name, tag=tag_, stream=True, decode=True))
            events = events + event
        for event in events:
            if 'status' in event:
                sys.stdout.write(event['status'])
                if 'progress' in event:
                    sys.stdout.write(' ' + event['progress'])
                sys.stdout.write('\n')
                sys.stdout.flush()
            elif 'error' in event:
                raise OSError(event['error'])
    return {'image_name': image_name, 'tag': tag, 'image': f'{image_name}:{tag}', 'additional_tags': additional_tags}
