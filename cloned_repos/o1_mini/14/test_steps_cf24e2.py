import os
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock
from uuid import uuid4
import docker
import docker.errors
import docker.models.containers
import docker.models.images
import pendulum
import pytest
from prefect_docker.deployments.steps import build_docker_image, push_docker_image
import prefect
import prefect.utilities.dockerutils

FAKE_CONTAINER_ID: str = 'fake-id'
FAKE_BASE_URL: str = 'my-url'
FAKE_DEFAULT_TAG: str = '2022-08-31t18-01-32-00-00'
FAKE_IMAGE_NAME: str = 'registry/repo'
FAKE_TAG: str = 'mytag'
FAKE_ADDITIONAL_TAGS: list[str] = ['addtag1', 'addtag2']
FAKE_EVENT: list[dict[str, str]] = [{'status': 'status', 'progress': 'progress'}, {'status': 'status'}]
FAKE_CREDENTIALS: dict[str, str | bool] = {
    'username': 'user',
    'password': 'pass',
    'registry_url': 'https://registry.com',
    'reauth': True
}

@pytest.fixture(autouse=True)
def reset_cachable_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('prefect_docker.deployments.steps.STEP_OUTPUT_CACHE', {})

@pytest.fixture
def mock_docker_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_client = MagicMock(name='DockerClient', spec=docker.DockerClient)
    mock_client.version.return_value = {'Version': '20.10'}
    fake_container: docker.models.containers.Container = docker.models.containers.Container()
    fake_container.client = MagicMock(name='Container.client')
    fake_container.collection = MagicMock(name='Container.collection')
    attrs: dict[str, any] = {
        'Id': FAKE_CONTAINER_ID,
        'Name': 'fake-name',
        'State': {
            'Status': 'exited',
            'Running': False,
            'Paused': False,
            'Restarting': False,
            'OOMKilled': False,
            'Dead': True,
            'Pid': 0,
            'ExitCode': 0,
            'Error': '',
            'StartedAt': '2022-08-31T18:01:32.645851548Z',
            'FinishedAt': '2022-08-31T18:01:32.657076632Z'
        }
    }
    fake_container.collection.get().attrs = attrs
    fake_container.attrs = attrs
    fake_container.stop = MagicMock()
    mock_client.containers.get.return_value = fake_container
    mock_client.containers.create.return_value = fake_container
    fake_api: MagicMock = MagicMock(name='APIClient')
    fake_api.build.return_value = [{'aux': {'ID': FAKE_CONTAINER_ID}}]
    fake_api.base_url = FAKE_BASE_URL
    mock_client.api = fake_api
    mock_docker_client_func: MagicMock = MagicMock(
        name='docker_client',
        spec=prefect.utilities.dockerutils.docker_client
    )
    mock_docker_client_func.return_value.__enter__.return_value = mock_client
    monkeypatch.setattr('prefect_docker.deployments.steps.docker_client', mock_docker_client_func)
    return mock_client

@pytest.fixture
def mock_pendulum(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_pendulum = MagicMock(name='pendulum', spec=pendulum)
    mock_pendulum.now.return_value = pendulum.datetime(2022, 8, 31, 18, 1, 32)
    monkeypatch.setattr('prefect_docker.deployments.steps.pendulum', mock_pendulum)
    return mock_pendulum

@pytest.mark.parametrize(
    'kwargs, expected_image',
    [
        ({'image_name': 'registry/repo'}, f'registry/repo:{FAKE_DEFAULT_TAG}'),
        (
            {'image_name': 'registry/repo', 'dockerfile': 'Dockerfile.dev'},
            f'registry/repo:{FAKE_DEFAULT_TAG}'
        ),
        ({'image_name': 'registry/repo', 'tag': 'mytag'}, 'registry/repo:mytag'),
        ({'image_name': 'registry/repo'}, f'registry/repo:{FAKE_DEFAULT_TAG}'),
        (
            {'image_name': 'registry/repo', 'dockerfile': 'auto'},
            f'registry/repo:{FAKE_DEFAULT_TAG}'
        ),
        (
            {'image_name': 'registry/repo', 'dockerfile': 'auto'},
            f'registry/repo:{FAKE_DEFAULT_TAG}'
        ),
        (
            {'image_name': 'registry/repo', 'dockerfile': 'auto', 'path': 'path/to/context'},
            f'registry/repo:{FAKE_DEFAULT_TAG}'
        ),
        (
            {
                'image_name': 'registry/repo',
                'tag': 'mytag',
                'additional_tags': FAKE_ADDITIONAL_TAGS
            },
            'registry/repo:mytag'
        )
    ]
)
def test_build_docker_image(
    monkeypatch: pytest.MonkeyPatch,
    mock_docker_client: MagicMock,
    mock_pendulum: MagicMock,
    kwargs: dict[str, any],
    expected_image: str
) -> None:
    auto_build: bool = False
    image_name: str = kwargs.get('image_name')
    dockerfile: str = kwargs.get('dockerfile', 'Dockerfile')
    tag: str = kwargs.get('tag', FAKE_DEFAULT_TAG)
    additional_tags: list[str] | None = kwargs.get('additional_tags', None)
    path: str = kwargs.get('path', os.getcwd())
    result: dict[str, any] = build_docker_image(**kwargs | {'ignore_cache': True})
    assert result['image'] == expected_image
    assert result['tag'] == tag
    assert result['image_name'] == image_name
    assert result['image_id'] == FAKE_CONTAINER_ID
    if additional_tags:
        assert result['additional_tags'] == FAKE_ADDITIONAL_TAGS
        mock_docker_client.images.get.return_value.tag.assert_has_calls([
            mock.call(repository=image_name, tag=tag),
            mock.call(repository=image_name, tag=FAKE_ADDITIONAL_TAGS[0]),
            mock.call(repository=image_name, tag=FAKE_ADDITIONAL_TAGS[1])
        ])
    else:
        assert result['additional_tags'] == []
        mock_docker_client.images.get.return_value.tag.assert_called_once_with(
            repository=image_name,
            tag=tag
        )
    if dockerfile == 'auto':
        auto_build = True
        dockerfile = 'Dockerfile'
    mock_docker_client.api.build.assert_called_once_with(
        path=path,
        dockerfile=dockerfile,
        decode=True,
        pull=True,
        labels=prefect.utilities.dockerutils.IMAGE_LABELS
    )
    mock_docker_client.images.get.assert_called_once_with(FAKE_CONTAINER_ID)
    if auto_build:
        assert not Path('Dockerfile').exists()

def test_build_docker_image_raises_with_auto_and_existing_dockerfile() -> None:
    try:
        Path('Dockerfile').touch()
        with pytest.raises(ValueError, match='Dockerfile already exists'):
            build_docker_image(image_name='registry/repo', dockerfile='auto', ignore_cache=True)
    finally:
        Path('Dockerfile').unlink()

def test_real_auto_dockerfile_build(docker_client_with_cleanup: docker.DockerClient) -> None:
    os.chdir(str(Path(__file__).parent.parent / 'test-project'))
    image_name: str = 'local/repo'
    tag: str = f'test-{uuid4()}'
    image_reference: str = f'{image_name}:{tag}'
    try:
        result: dict[str, any] = build_docker_image(
            image_name=image_name,
            tag=tag,
            dockerfile='auto',
            pull=False
        )
        image: docker.models.images.Image = docker_client_with_cleanup.images.get(result['image'])
        assert image
        expected_prefect_version: str = prefect.__version__
        expected_prefect_version = expected_prefect_version.replace('.dirty', '')
        expected_prefect_version = expected_prefect_version.split('+')[0]
        cases: list[dict[str, str]] = [
            {'command': 'prefect version', 'expected': expected_prefect_version},
            {'command': 'ls', 'expected': 'requirements.txt'}
        ]
        for case in cases:
            output: bytes = docker_client_with_cleanup.containers.run(
                image=result['image'],
                command=case['command'],
                labels=['prefect-docker-test'],
                remove=True
            )
            assert case['expected'] in output.decode()
        output: bytes = docker_client_with_cleanup.containers.run(
            image=result['image'],
            command="python -c 'import pandas; print(pandas.__version__)'",
            labels=['prefect-docker-test'],
            remove=True
        )
        assert '2' in output.decode()
    finally:
        docker_client_with_cleanup.containers.prune(filters={'label': 'prefect-docker-test'})
        try:
            docker_client_with_cleanup.images.remove(image=image_reference, force=True)
        except docker.errors.ImageNotFound:
            pass

def test_push_docker_image_with_additional_tags(
    mock_docker_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_stdout: MagicMock = MagicMock()
    monkeypatch.setattr(sys, 'stdout', mock_stdout)
    mock_docker_client.api.push.side_effect = [
        (e for e in FAKE_EVENT),
        (e for e in FAKE_EVENT),
        (e for e in FAKE_EVENT)
    ]
    result: dict[str, any] = push_docker_image(
        image_name=FAKE_IMAGE_NAME,
        tag=FAKE_TAG,
        credentials=FAKE_CREDENTIALS,
        additional_tags=FAKE_ADDITIONAL_TAGS
    )
    assert result['image_name'] == FAKE_IMAGE_NAME
    assert result['tag'] == FAKE_TAG
    assert result['image'] == f'{FAKE_IMAGE_NAME}:{FAKE_TAG}'
    assert result['additional_tags'] == FAKE_ADDITIONAL_TAGS
    mock_docker_client.api.push.assert_has_calls([
        mock.call(repository=FAKE_IMAGE_NAME, tag=FAKE_TAG, stream=True, decode=True),
        mock.call(repository=FAKE_IMAGE_NAME, tag=FAKE_ADDITIONAL_TAGS[0], stream=True, decode=True),
        mock.call(repository=FAKE_IMAGE_NAME, tag=FAKE_ADDITIONAL_TAGS[1], stream=True, decode=True)
    ])
    assert mock_stdout.write.call_count == 15

def test_push_docker_image_with_credentials(
    mock_docker_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_stdout: MagicMock = MagicMock()
    monkeypatch.setattr(sys, 'stdout', mock_stdout)
    mock_docker_client.api.push.return_value = FAKE_EVENT
    result: dict[str, any] = push_docker_image(
        image_name=FAKE_IMAGE_NAME,
        tag=FAKE_TAG,
        credentials=FAKE_CREDENTIALS
    )
    assert result['image_name'] == FAKE_IMAGE_NAME
    assert result['tag'] == FAKE_TAG
    assert result['image'] == f'{FAKE_IMAGE_NAME}:{FAKE_TAG}'
    mock_docker_client.login.assert_called_once_with(
        username=FAKE_CREDENTIALS['username'],
        password=FAKE_CREDENTIALS['password'],
        registry=FAKE_CREDENTIALS['registry_url'],
        reauth=FAKE_CREDENTIALS.get('reauth', True)
    )
    mock_docker_client.api.push.assert_called_once_with(
        repository=FAKE_IMAGE_NAME,
        tag=FAKE_TAG,
        stream=True,
        decode=True
    )
    assert mock_stdout.write.call_count == 5

def test_push_docker_image_without_credentials(
    mock_docker_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_stdout: MagicMock = MagicMock()
    monkeypatch.setattr(sys, 'stdout', mock_stdout)
    mock_docker_client.api.push.return_value = FAKE_EVENT
    result: dict[str, any] = push_docker_image(
        image_name=FAKE_IMAGE_NAME,
        tag=FAKE_TAG
    )
    assert result['image_name'] == FAKE_IMAGE_NAME
    assert result['tag'] == FAKE_TAG
    assert result['image'] == f'{FAKE_IMAGE_NAME}:{FAKE_TAG}'
    mock_docker_client.login.assert_not_called()
    mock_docker_client.api.push.assert_called_once_with(
        repository=FAKE_IMAGE_NAME,
        tag=FAKE_TAG,
        stream=True,
        decode=True
    )
    assert mock_stdout.write.call_count == 5

def test_push_docker_image_raises_on_event_error(mock_docker_client: MagicMock) -> None:
    error_event: list[dict[str, str]] = [{'error': 'Error'}]
    mock_docker_client.api.push.return_value = error_event
    with pytest.raises(OSError, match='Error'):
        push_docker_image(
            image_name=FAKE_IMAGE_NAME,
            tag=FAKE_TAG,
            credentials=FAKE_CREDENTIALS,
            ignore_cache=True
        )

class TestCachedSteps:

    def test_cached_build_docker_image(self, mock_docker_client: MagicMock) -> None:
        image_name: str = 'registry/repo'
        dockerfile: str = 'Dockerfile'
        tag: str = 'mytag'
        additional_tags: list[str] = ['tag1', 'tag2']
        expected_result: dict[str, any] = {
            'image': f'{image_name}:{tag}',
            'tag': tag,
            'image_name': image_name,
            'image_id': FAKE_CONTAINER_ID,
            'additional_tags': additional_tags
        }
        for _ in range(3):
            result: dict[str, any] = build_docker_image(
                image_name=image_name,
                dockerfile=dockerfile,
                tag=tag,
                additional_tags=additional_tags
            )
            assert result == expected_result
        mock_docker_client.api.build.assert_called_once()
        mock_docker_client.images.get.assert_called_once_with(FAKE_CONTAINER_ID)
        assert mock_docker_client.images.get.return_value.tag.call_count == 1 + len(additional_tags)

    def test_uncached_build_docker_image(self, mock_docker_client: MagicMock) -> None:
        image_name: str = 'registry/repo'
        dockerfile: str = 'Dockerfile'
        tag: str = 'mytag'
        additional_tags: list[str] = ['tag1', 'tag2']
        expected_result: dict[str, any] = {
            'image': f'{image_name}:{tag}',
            'tag': tag,
            'image_name': image_name,
            'image_id': FAKE_CONTAINER_ID,
            'additional_tags': additional_tags
        }
        for _ in range(3):
            result: dict[str, any] = build_docker_image(
                image_name=image_name,
                dockerfile=dockerfile,
                tag=tag,
                additional_tags=additional_tags,
                ignore_cache=True
            )
            assert result == expected_result
        assert mock_docker_client.api.build.call_count == 3
        assert mock_docker_client.images.get.call_count == 3
        expected_tag_calls: int = 1 + len(additional_tags)
        assert mock_docker_client.images.get.return_value.tag.call_count == expected_tag_calls * 3

    def test_cached_push_docker_image(self, mock_docker_client: MagicMock) -> None:
        image_name: str = FAKE_IMAGE_NAME
        tag: str = FAKE_TAG
        credentials: dict[str, str | bool] = FAKE_CREDENTIALS
        additional_tags: list[str] = FAKE_ADDITIONAL_TAGS
        expected_result: dict[str, any] = {
            'image_name': image_name,
            'tag': tag,
            'image': f'{image_name}:{tag}',
            'additional_tags': additional_tags
        }
        for _ in range(2):
            result: dict[str, any] = push_docker_image(
                image_name=image_name,
                tag=tag,
                credentials=credentials,
                additional_tags=additional_tags
            )
            assert result == expected_result
        mock_docker_client.login.assert_called_once()
        assert mock_docker_client.api.push.call_count == 1 + len(additional_tags)

    def test_uncached_push_docker_image(self, mock_docker_client: MagicMock) -> None:
        image_name: str = FAKE_IMAGE_NAME
        tag: str = FAKE_TAG
        credentials: dict[str, str | bool] = FAKE_CREDENTIALS
        additional_tags: list[str] = FAKE_ADDITIONAL_TAGS
        expected_result: dict[str, any] = {
            'image_name': image_name,
            'tag': tag,
            'image': f'{image_name}:{tag}',
            'additional_tags': additional_tags
        }
        for _ in range(3):
            result: dict[str, any] = push_docker_image(
                image_name=image_name,
                tag=tag,
                credentials=credentials,
                additional_tags=additional_tags,
                ignore_cache=True
            )
            assert result == expected_result
        assert mock_docker_client.login.call_count == 3
        expected_push_calls: int = 1 + len(additional_tags)
        assert mock_docker_client.api.push.call_count == expected_push_calls * 3

    def test_avoids_aggressive_caching(self, mock_docker_client: MagicMock) -> None:
        """this is a regression test for https://github.com/PrefectHQ/prefect/issues/15258
        where all decorated functions were sharing a cache, so dict(image=..., tag=...) passed to
        build_docker_image and push_docker_image would hit the cache for push_docker_image,
        even though the function was different and should not have been cached.

        here we test that the caches are distinct for each decorated function.
        """
        image_name: str = 'registry/repo'
        tag: str = 'latest'
        build_docker_image(image_name=image_name, tag=tag)
        push_docker_image(image_name=image_name, tag=tag)
        mock_docker_client.api.build.assert_called_once()
        mock_docker_client.api.push.assert_called_once()
