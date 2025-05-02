import pytest
import respx
from httpx import Response, AsyncClient
from typing import Dict, Tuple, Any, Generator


class TestReadCollectionViews:
    def collection_view_url(self, view: str) -> str:
        return (
            "https://raw.githubusercontent.com/"
            "PrefectHQ/prefect-collection-registry/main/"
            f"views/aggregate-{view}-metadata.json"
        )

    @pytest.fixture
    def mock_flow_response(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        return {
            "collection-name": {
                "flow-name": {
                    "name": "flow-name",
                },
            }
        }

    @pytest.fixture
    def mock_block_response(self) -> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
        return {
            "collection-name": {
                "block_types": {
                    "block-name": {
                        "name": "block-name",
                    },
                },
            }
        }

    @pytest.fixture
    def mock_collection_response(self) -> Dict[str, Dict[str, str]]:
        return {
            "collection-name": {
                "name": "collection-name",
            },
        }

    @pytest.fixture
    def mock_worker_response(self) -> Dict[str, Dict[str, Dict]]:
        return {
            "prefect": {
                "prefect-agent": {},
                "process": {},
            }
        }

    @respx.mock
    @pytest.fixture
    def mock_get_view(
        self,
        respx_mock: respx.Router,
        mock_flow_response: Dict[str, Any],
        mock_block_response: Dict[str, Any],
        mock_worker_response: Dict[str, Any],
    ) -> Generator[Tuple[respx.Router, respx.Route, respx.Route, respx.Route], None, None]:
        with respx.mock(
            using="httpx", assert_all_mocked=False, assert_all_called=False
        ) as respx_mock:
            flow_route = respx_mock.get(self.collection_view_url("flow")).mock(
                return_value=Response(200, json=mock_flow_response)
            )
            block_route = respx_mock.get(self.collection_view_url("block")).mock(
                return_value=Response(200, json=mock_block_response)
            )
            worker_route = respx_mock.get(self.collection_view_url("worker")).mock(
                return_value=Response(200, json=mock_worker_response)
            )
            respx_mock.route(host="test").pass_through()

            yield respx_mock, flow_route, block_route, worker_route

    @respx.mock
    @pytest.fixture
    def mock_get_missing_view(
        self,
        mock_flow_response: Dict[str, Any],
        mock_block_response: Dict[str, Any],
        mock_collection_response: Dict[str, Any],
    ) -> Generator[respx.Router, None, None]:
        with respx.mock(
            using="httpx",
            assert_all_mocked=False,
            assert_all_called=False,
            base_url="https://raw.githubusercontent.com",
        ) as respx_mock:
            respx_mock.get(self.collection_view_url("flow")).mock(
                return_value=Response(404, json=mock_flow_response)
            )
            respx_mock.get(self.collection_view_url("block")).mock(
                return_value=Response(404, json=mock_block_response)
            )
            respx_mock.get(self.collection_view_url("worker")).mock(
                return_value=Response(404, json=mock_collection_response)
            )
            respx_mock.route(host="test").pass_through()

            yield respx_mock

    @pytest.mark.parametrize(
        "view", ["aggregate-flow-metadata", "aggregate-block-metadata"]
    )
    async def test_read_view(self, client: AsyncClient, view: str, mock_get_view: Tuple[respx.Router, respx.Route, respx.Route, respx.Route]) -> None:
        res = await client.get(f"/collections/views/{view}")

        assert res.status_code == 200
        assert isinstance(res.json(), dict)

    async def test_read_collection_view_when_missing(
        self, client: AsyncClient, mock_get_missing_view: respx.Router
    ) -> None:
        res = await client.get("/collections/views/aggregate-flow-metadata")
        detail = res.json()["detail"]

        assert res.status_code == 404
        assert detail == "Requested content missing for view aggregate-flow-metadata"

    async def test_read_collection_view_invalid(self, client: AsyncClient) -> None:
        res = await client.get("/collections/views/invalid")
        detail = res.json()["detail"]

        assert res.status_code == 404
        assert detail == "View invalid not found in registry"

    @pytest.mark.parametrize(
        "view", ["aggregate-flow-metadata", "aggregate-block-metadata"]
    )
    async def test_collection_view_cached(self, client: AsyncClient, mock_get_view: Tuple[respx.Router, respx.Route, respx.Route, respx.Route], view: str) -> None:
        respx_mock, flow_route, block_route, worker_route = mock_get_view
        res1 = await client.get(f"/collections/views/{view}")

        assert res1.status_code == 200
        assert isinstance(res1.json(), dict)

        res2 = await client.get(f"/collections/views/{view}")

        assert res2.status_code == 200
        assert isinstance(res2.json(), dict)

        assert res1.json() == res2.json()
        if view == "aggregate-flow-metadata":
            flow_route.calls.assert_called_once()
        elif view == "aggregate-block-metadata":
            block_route.calls.assert_called_once()

    async def test_read_worker_view_failed_fetch(self, client: AsyncClient, mock_get_missing_view: respx.Router) -> None:
        res = await client.get("/collections/views/aggregate-worker-metadata")

        assert res.status_code == 200
        # check for expected key to ensure it isn't an error
        assert isinstance(res.json()["prefect"], dict)

    async def test_prefect_agent_excluded_from_worker_metadata(
        self, client: AsyncClient, mock_get_view: Tuple[respx.Router, respx.Route, respx.Route, respx.Route]
    ) -> None:
        res = await client.get("/collections/views/aggregate-worker-metadata")

        assert res.status_code == 200
        assert "prefect-agent" not in res.json()["prefect"]
