import uuid
from typing import Callable, List, Any, Generator, Set
from unittest import mock
import pydantic
import pytest
from fastapi.testclient import TestClient
from prefect import flow
from prefect.client.orchestration import PrefectClient, get_client
from prefect.client.schemas.objects import FlowRun
from prefect.runner import Runner
from prefect.runner.server import build_server
from prefect.settings import (
    PREFECT_RUNNER_SERVER_HOST,
    PREFECT_RUNNER_SERVER_PORT,
    temporary_settings,
)


class A(pydantic.BaseModel):
    a: int = 0


class B(pydantic.BaseModel):
    a: A = A()
    b: bool = False


@flow(version='test')
def func_39avih35(verb: str = 'party') -> None:
    print(f"I'm just here to {verb}")


@flow
def func_6sa0ad8y(
    x: Any,
    y: str = 'hello',
    z: List[bool] = [True],
    a: A = A(),
    b: B = B(),
) -> None:
    print(x, y, z, a, b)


def func_1zhuh3ia() -> None:
    print('This is not a flow!')


@pytest.fixture(autouse=True)
def func_dxcfy8yi() -> Generator[None, None, None]:
    with temporary_settings(
        updates={
            PREFECT_RUNNER_SERVER_HOST: '0.0.0.0',
            PREFECT_RUNNER_SERVER_PORT: 0,
        }
    ):
        yield


@pytest.fixture
async def func_7xqtlkkl() -> Runner:
    return Runner()


async def func_tvuw64o3(runner: Runner, func: Callable[..., Any]) -> str:
    deployment_id = await func_7xqtlkkl.add_flow(
        func, f'{uuid.uuid4()}', enforce_parameter_schema=True
    )
    return str(deployment_id)


class TestWebserverSettings:

    async def func_tj1ppnp9(self, runner: Runner) -> None:
        with temporary_settings(
            updates={
                PREFECT_RUNNER_SERVER_HOST: '127.0.0.1',
                PREFECT_RUNNER_SERVER_PORT: 4200,
            }
        ):
            assert PREFECT_RUNNER_SERVER_HOST.value() == '127.0.0.1'
            assert PREFECT_RUNNER_SERVER_PORT.value() == 4200


class TestWebserverDeploymentRoutes:

    async def func_5qzuoyk6(self, runner: Runner) -> None:
        deployment_ids: List[str] = [
            await func_tvuw64o3(runner, simple_flow) for _ in range(3)
        ]
        webserver = await build_server(runner)
        deployment_run_routes = [
            r for r in webserver.routes
            if r.path.startswith('/deployment') and r.path.endswith('/run')
        ]
        deployment_run_paths: Set[str] = {r.path for r in deployment_run_routes}
        for route in deployment_run_routes:
            id_: str = route.path.split('/')[2]
            assert id_ in deployment_ids
        for id_ in deployment_ids:
            route: str = f'/deployment/{id_}/run'
            assert route in deployment_run_paths

    @pytest.mark.skip(reason='This test is flaky and needs to be fixed')
    async def func_bw48hlvt(self, runner: Runner) -> None:
        async with runner:
            deployment_id: str = await func_tvuw64o3(runner, simple_flow)
            webserver = await build_server(runner)
            client: TestClient = TestClient(webserver)
            response = client.post(
                f'/deployment/{deployment_id}/run',
                json={'verb': False},
            )
            assert response.status_code == 400
            response = client.post(
                f'/deployment/{deployment_id}/run',
                json={'verb': 'clobber'},
            )
            assert response.status_code == 201
            flow_run_id: str = response.json()['flow_run_id']
            assert isinstance(uuid.UUID(flow_run_id), uuid.UUID)

    @pytest.mark.skip(reason='This test is flaky and needs to be fixed')
    async def func_w1nx2710(self, runner: Runner) -> None:
        async with runner:
            deployment_id: str = await func_7xqtlkkl.add_flow(
                complex_flow,
                f'{uuid.uuid4()}',
                enforce_parameter_schema=True,
            )
            webserver = await build_server(runner)
            client: TestClient = TestClient(webserver)
            response = client.post(
                f'/deployment/{deployment_id}/run',
                json={'x': 100},
            )
            assert response.status_code == 201, response.json()
            flow_run_id: str = response.json()['flow_run_id']
            assert isinstance(uuid.UUID(flow_run_id), uuid.UUID)

    async def func_o8xaob87(self, runner: Runner) -> None:
        mock_flow_run_id: str = str(uuid.uuid4())
        mock_client: PrefectClient = mock.create_autospec(
            PrefectClient, spec_set=True
        )
        mock_client.create_flow_run_from_deployment.return_value.id = mock_flow_run_id
        mock_get_client = mock.create_autospec(get_client, spec_set=True)
        mock_get_client.return_value.__aenter__.return_value = mock_client
        mock_get_client.return_value.__aexit__.return_value = None
        async with runner:
            deployment_id: str = await func_tvuw64o3(runner, simple_flow)
            webserver = await build_server(runner)
            client: TestClient = TestClient(webserver)
            with mock.patch('prefect.runner.server.get_client', new=mock_get_client), \
                 mock.patch.object(runner, 'execute_in_background'):
                with client:
                    response = client.post(f'/deployment/{deployment_id}/run')
                assert response.status_code == 201, response.json()
                flow_run_id: str = response.json()['flow_run_id']
                assert flow_run_id == mock_flow_run_id
                assert isinstance(uuid.UUID(flow_run_id), uuid.UUID)
                mock_client.create_flow_run_from_deployment.assert_called_once_with(
                    deployment_id=uuid.UUID(deployment_id),
                    parameters={},
                )


class TestWebserverFlowRoutes:

    async def func_0nmpy9b3(self, runner: Runner) -> None:
        async with runner:
            await func_tvuw64o3(runner, simple_flow)
            webserver = await build_server(runner)
            client: TestClient = TestClient(webserver)
            with mock.patch.object(
                runner, 'execute_flow_run', new_callable=mock.AsyncMock
            ) as mock_run:
                response = client.post(
                    '/flow/run',
                    json={'entrypoint': f'{__file__}:simple_flow', 'parameters': {}},
                )
                assert response.status_code == 201, response.status_code
                assert isinstance(
                    FlowRun.model_validate(response.json()), FlowRun
                )
                mock_run.assert_called()

    @pytest.mark.parametrize('flow_name', ['a_missing_flow'])
    @pytest.mark.parametrize('flow_file', [__file__, '/not/a/path.py', 'not/a/python/file.txt'])
    async def func_1sse7igw(
        self, runner: Runner, flow_file: str, flow_name: str
    ) -> None:
        async with runner:
            await func_tvuw64o3(runner, simple_flow)
            webserver = await build_server(runner)
        client: TestClient = TestClient(webserver)
        response = client.post(
            '/flow/run',
            json={'entrypoint': f'{flow_file}:{flow_name}', 'parameters': {}},
        )
        assert response.status_code == 404, response.status_code

    @mock.patch('prefect.runner.server.load_flow_from_entrypoint')
    async def func_awtoizz7(
        self,
        mocked_load: mock.Mock,
        runner: Runner,
        caplog: Any,
    ) -> None:
        async with runner:
            await func_tvuw64o3(runner, simple_flow)
            webserver = await build_server(runner)
            client: TestClient = TestClient(webserver)

            @flow
            def func_fo6kc17a() -> None:
                pass

            mocked_load.return_value = new_flow
            with mock.patch.object(
                runner, 'execute_flow_run', new_callable=mock.AsyncMock
            ) as mock_run:
                response = client.post(
                    '/flow/run',
                    json={'entrypoint': 'doesnt_matter', 'parameters': {}},
                )
                assert response.status_code == 201, response.status_code
                assert isinstance(
                    FlowRun.model_validate(response.json()), FlowRun
                )
                mock_run.assert_called()
            assert (
                "Flow new-flow is not directly managed by the runner. Please include it in the runner's served flows' import namespace."
                in caplog.text
            )

    @mock.patch('prefect.runner.server.load_flow_from_entrypoint')
    async def func_i6yb4gvz(
        self,
        mocked_load: mock.Mock,
        runner: Runner,
        caplog: Any,
    ) -> None:
        async with runner:
            await func_tvuw64o3(runner, simple_flow)
            webserver = await build_server(runner)
            client: TestClient = TestClient(webserver)

            @flow
            def func_qncladf8(age: int = 99) -> None:
                pass

            simple_flow2.name = 'simple_flow'
            mocked_load.return_value = simple_flow2
            with mock.patch.object(
                runner, 'execute_flow_run', new_callable=mock.AsyncMock
            ) as mock_run:
                response = client.post(
                    '/flow/run',
                    json={'entrypoint': 'doesnt_matter', 'parameters': {}},
                )
                assert response.status_code == 201, response.status_code
                assert isinstance(
                    FlowRun.model_validate(response.json()), FlowRun
                )
                mock_run.assert_called()
            assert (
                'A change in flow parameters has been detected. Please restart the runner.'
                in caplog.text
            )
