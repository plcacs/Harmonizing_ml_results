from typing import Dict, List, Optional
from prefect.utilities.dockerutils import BuildError
from docker.models.images import Image
from typing_extensions import TypedDict

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

def func_df6obeyp(image_name: str, dockerfile: str = 'Dockerfile', tag: Optional[str] = None,
    additional_tags: Optional[List[str]] = None, ignore_cache: bool = False, **build_kwargs) -> BuildDockerImageResult:

def func_mhuw1rs3(image_name: str, tag: Optional[str] = None, credentials: Optional[Dict[str, str]] = None, additional_tags:
    Optional[List[str]] = None, ignore_cache: bool = False) -> PushDockerImageResult:
