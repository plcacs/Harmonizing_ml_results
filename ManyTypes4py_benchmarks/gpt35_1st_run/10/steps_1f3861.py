def build_docker_image(image_name: str, dockerfile: str = 'Dockerfile', tag: Optional[str] = None, additional_tags: Optional[List[str]] = None, ignore_cache: bool = False, **build_kwargs: Dict[str, str]) -> BuildDockerImageResult:

def push_docker_image(image_name: str, tag: Optional[str] = None, credentials: Optional[Dict[str, str]] = None, additional_tags: Optional[List[str]] = None, ignore_cache: bool = False) -> PushDockerImageResult:
