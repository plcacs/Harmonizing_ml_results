import json
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional
import docker.errors
import pendulum
from docker.models.images import Image
from typing_extensions import TypedDict
from prefect.logging.loggers import get_logger
from prefect.utilities.dockerutils import IMAGE_LABELS, BuildError, docker_client, get_prefect_image_name
from prefect.utilities.slugify import slugify
logger = get_logger('prefect_docker.deployments.steps')
STEP_OUTPUT_CACHE: Dict[tuple, BuildDockerImageResult] = {}

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

def _make_hashable(obj: object) -> str:
    if isinstance(obj, dict):
        return json.dumps(obj, sort_keys=True)
    elif isinstance(obj, list):
        return tuple((_make_hashable(v) for v in obj))
    return obj

def cacheable(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if (ignore_cache := kwargs.pop('ignore_cache', False)):
            logger.debug(f'Ignoring `@cacheable` decorator for {func.__name__}.')
        key = (func.__name__, tuple((_make_hashable(arg) for arg in args)), tuple(((k, _make_hashable(v)) for k, v in sorted(kwargs.items()))))
        if ignore_cache or key not in STEP_OUTPUT_CACHE:
            logger.debug(f'Cache miss for {func.__name__}, running function.')
            STEP_OUTPUT_CACHE[key] = func(*args, **kwargs)
        else:
            logger.debug(f'Cache hit for {func.__name__}, returning cached value.')
        return STEP_OUTPUT_CACHE[key]
    return wrapper

@cacheable
def build_docker_image(image_name: str, dockerfile: str = 'Dockerfile', tag: Optional[str] = None, additional_tags: Optional[List[str]] = None, ignore_cache: bool = False, **build_kwargs: dict) -> BuildDockerImageResult:
    ...

@cacheable
def push_docker_image(image_name: str, tag: Optional[str] = None, credentials: Optional[dict] = None, additional_tags: Optional[List[str]] = None, ignore_cache: bool = False) -> PushDockerImageResult:
    ...
