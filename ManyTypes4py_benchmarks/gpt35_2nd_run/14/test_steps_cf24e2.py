from typing import Dict, Any, List

def build_docker_image(image_name: str, dockerfile: str = 'Dockerfile', tag: str = '2022-08-31t18-01-32-00-00', additional_tags: List[str] = None, ignore_cache: bool = False) -> Dict[str, Any]:
    ...

def push_docker_image(image_name: str, tag: str, credentials: Dict[str, str] = None, additional_tags: List[str] = None, ignore_cache: bool = False) -> Dict[str, Any]:
    ...

class TestCachedSteps:

    def test_cached_build_docker_image(self, mock_docker_client: MagicMock) -> None:
        ...

    def test_uncached_build_docker_image(self, mock_docker_client: MagicMock) -> None:
        ...

    def test_cached_push_docker_image(self, mock_docker_client: MagicMock) -> None:
        ...

    def test_uncached_push_docker_image(self, mock_docker_client: MagicMock) -> None:
        ...

    def test_avoids_aggressive_caching(self, mock_docker_client: MagicMock) -> None:
        ...
