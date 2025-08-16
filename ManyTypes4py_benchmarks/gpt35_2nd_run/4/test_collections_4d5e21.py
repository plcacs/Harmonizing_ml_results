from typing import Dict, Any, Tuple, List
import pytest
import respx
from httpx import Response

class TestReadCollectionViews:

    def collection_view_url(self, view: str) -> str:
        return f'https://raw.githubusercontent.com/PrefectHQ/prefect-collection-registry/main/views/aggregate-{view}-metadata.json'

    @pytest.fixture
    def mock_flow_response(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        return {'collection-name': {'flow-name': {'name': 'flow-name'}}}

    @pytest.fixture
    def mock_block_response(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        return {'collection-name': {'block_types': {'block-name': {'name': 'block-name'}}}

    @pytest.fixture
    def mock_collection_response(self) -> Dict[str, Dict[str, str]]:
        return {'collection-name': {'name': 'collection-name'}}

    @pytest.fixture
    def mock_worker_response(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        return {'prefect': {'prefect-agent': {}, 'process': {}}}

    @respx.mock
    @pytest.fixture
    def mock_get_view(self, respx_mock, mock_flow_response, mock_block_response, mock_worker_response) -> Tuple[respx.Mock, respx.Route, respx.Route, respx.Route]:
        ...

    @respx.mock
    @pytest.fixture
    def mock_get_missing_view(self, mock_flow_response, mock_block_response, mock_collection_response) -> respx.Mock:
        ...

    @pytest.mark.parametrize('view', ['aggregate-flow-metadata', 'aggregate-block-metadata'])
    async def test_read_view(self, client, view: str, mock_get_view) -> None:
        ...

    async def test_read_collection_view_when_missing(self, client, mock_get_missing_view) -> None:
        ...

    async def test_read_collection_view_invalid(self, client) -> None:
        ...

    @pytest.mark.parametrize('view', ['aggregate-flow-metadata', 'aggregate-block-metadata'])
    async def test_collection_view_cached(self, client, mock_get_view, view: str) -> None:
        ...

    async def test_read_worker_view_failed_fetch(self, client, mock_get_missing_view) -> None:
        ...

    async def test_prefect_agent_excluded_from_worker_metadata(self, client, mock_get_view) -> None:
        ...
