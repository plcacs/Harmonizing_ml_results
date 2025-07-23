import json
import logging
from functools import partial
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union, cast
from unittest.mock import ANY, MagicMock
from unittest.mock import patch as mock_patch
from uuid import UUID, uuid4
import anyio
import botocore
import pytest
import yaml
from exceptiongroup import ExceptionGroup, catch
from moto import mock_ec2, mock_ecs, mock_logs
from moto.ec2.utils import generate_instance_identity_document
from prefect_aws.credentials import _get_client_cached
from prefect_aws.workers.ecs_worker import (
    _TAG_REGEX,
    _TASK_DEFINITION_CACHE,
    ECS_DEFAULT_CONTAINER_NAME,
    ECS_DEFAULT_CPU,
    ECS_DEFAULT_FAMILY,
    ECS_DEFAULT_MEMORY,
    AwsCredentials,
    ECSJobConfiguration,
    ECSVariables,
    ECSWorker,
    _get_container,
    get_prefect_image_name,
    mask_sensitive_env_values,
    parse_identifier,
)
from pydantic import ValidationError
from prefect.server.schemas.core import FlowRun
from prefect.utilities.asyncutils import run_sync_in_worker_thread
from prefect.utilities.slugify import slugify

TEST_TASK_DEFINITION_YAML: str = '\ncontainerDefinitions:\n- cpu: 1024\n  image: prefecthq/prefect:2.1.0-python3.9\n  memory: 2048\n  name: prefect\nfamily: prefect\n'
TEST_TASK_DEFINITION: Dict[str, Any] = yaml.safe_load(TEST_TASK_DEFINITION_YAML)

@pytest.fixture
def flow_run() -> FlowRun:
    return FlowRun(flow_id=uuid4(), deployment_id=uuid4())

@pytest.fixture
def container_status_code() -> MagicMock:
    yield MagicMock(return_value=0)

@pytest.fixture(autouse=True)
def reset_task_definition_cache() -> None:
    _TASK_DEFINITION_CACHE.clear()
    yield

def inject_moto_patches(moto_mock: Any, patches: Dict[str, List[Callable[..., Any]]]) -> None:
    def injected_call(method: Callable[..., Any], patch_list: List[Callable[..., Any]], *args: Any, **kwargs: Any) -> Any:
        for patch in patch_list:
            result = patch(method, *args, **kwargs)
        return result
    
    for account in moto_mock.backends:
        for region in moto_mock.backends[account]:
            backend = moto_mock.backends[account][region]
            for attr, attr_patches in patches.items():
                original_method = getattr(backend, attr)
                setattr(backend, attr, partial(injected_call, original_method, attr_patches))

def patch_run_task(mock: MagicMock, run_task: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    mock(*args, **kwargs)
    return run_task(*args, **kwargs)

def patch_describe_tasks_add_containers(
    session: Any,
    container_status_code: MagicMock,
    describe_tasks: Callable[..., Any],
    *args: Any,
    **kwargs: Any
) -> Any:
    ecs_client = session.client('ecs')
    result = describe_tasks(*args, **kwargs)
    for task in result:
        if not task.containers:
            task_definition = ecs_client.describe_task_definition(taskDefinition=task.task_definition_arn)['taskDefinition']
            task.containers = [{'name': container['name'], 'exitCode': container_status_code.return_value} for container in task_definition.get('containerDefinitions', [])]
        if task.overrides.get('container_overrides'):
            for container in task.overrides['container_overrides']:
                if not _get_container(task.containers, container.name):
                    task.containers.append({'name': container.name, 'exitCode': container_status_code.return_value})
        elif not _get_container(task.containers, ECS_DEFAULT_CONTAINER_NAME):
            task.containers.append({'name': ECS_DEFAULT_CONTAINER_NAME, 'exitCode': container_status_code.return_value})
    return result

def patch_calculate_task_resource_requirements(
    _calculate_task_resource_requirements: Callable[..., Any],
    task_definition: Dict[str, Any]
) -> Any:
    for container_definition in task_definition.container_definitions:
        container_definition.setdefault('memory', 0)
    return _calculate_task_resource_requirements(task_definition)

def create_log_stream(session: Any, run_task: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    tasks = run_task(*args, **kwargs)
    if not tasks:
        return tasks
    task = tasks[0]
    ecs_client = session.client('ecs')
    logs_client = session.client('logs')
    task_definition = ecs_client.describe_task_definition(taskDefinition=task.task_definition_arn)['taskDefinition']
    for container in task_definition.get('containerDefinitions', []):
        log_config = container.get('logConfiguration', {})
        if log_config:
            if log_config.get('logDriver') != 'awslogs':
                continue
            options = log_config.get('options', {})
            if not options:
                raise ValueError('logConfiguration does not include options.')
            group_name = options.get('awslogs-group')
            if not group_name:
                raise ValueError('logConfiguration.options does not include awslogs-group')
            if options.get('awslogs-create-group') == 'true':
                logs_client.create_log_group(logGroupName=group_name)
            stream_prefix = options.get('awslogs-stream-prefix')
            if not stream_prefix:
                raise ValueError('logConfiguration.options does not include awslogs-stream-prefix')
            logs_client.create_log_stream(logGroupName=group_name, logStreamName=f'{stream_prefix}/{container["name"]}/{task.id}')
    return tasks

def add_ec2_instance_to_ecs_cluster(session: Any, cluster_name: str) -> None:
    ecs_client = session.client('ecs')
    ec2_client = session.client('ec2')
    ec2_resource = session.resource('ec2')
    ecs_client.create_cluster(clusterName=cluster_name)
    images = ec2_client.describe_images()
    image_id = images['Images'][0]['ImageId']
    test_instance = ec2_resource.create_instances(ImageId=image_id, MinCount=1, MaxCount=1)[0]
    ecs_client.register_container_instance(cluster=cluster_name, instanceIdentityDocument=json.dumps(generate_instance_identity_document(test_instance)))

def create_test_ecs_cluster(ecs_client: Any, cluster_name: str) -> str:
    return ecs_client.create_cluster(clusterName=cluster_name)['cluster']['clusterArn']

def describe_task(ecs_client: Any, task_arn: str, **kwargs: Any) -> Dict[str, Any]:
    return ecs_client.describe_tasks(tasks=[task_arn], include=['TAGS'], **kwargs)['tasks'][0]

async def stop_task(ecs_client: Any, task_arn: str, **kwargs: Any) -> None:
    task = await run_sync_in_worker_thread(describe_task, ecs_client, task_arn)
    assert task['lastStatus'] == 'RUNNING', 'Task should be RUNNING before stopping'
    print('Stopping task...')
    await run_sync_in_worker_thread(ecs_client.stop_task, task=task_arn, **kwargs)

def describe_task_definition(ecs_client: Any, task: Dict[str, Any]) -> Dict[str, Any]:
    return ecs_client.describe_task_definition(taskDefinition=task['taskDefinitionArn'])['taskDefinition']

@pytest.fixture
def ecs_mocks(aws_credentials: AwsCredentials, flow_run: FlowRun, container_status_code: MagicMock) -> Any:
    with mock_ecs() as ecs:
        with mock_ec2():
            with mock_logs():
                session = aws_credentials.get_boto3_session()
                inject_moto_patches(ecs, {
                    'describe_tasks': [partial(patch_describe_tasks_add_containers, session, container_status_code)],
                    '_calculate_task_resource_requirements': [patch_calculate_task_resource_requirements],
                    'run_task': [partial(create_log_stream, session)]
                })
                create_test_ecs_cluster(session.client('ecs'), 'default')
                add_ec2_instance_to_ecs_cluster(session, 'default')
                yield ecs

async def construct_configuration(**options: Any) -> ECSJobConfiguration:
    variables = ECSVariables(**options | {'task_watch_poll_interval': 0.03})
    print(f'Using variables: {variables.model_dump_json(indent=2, exclude_none=True)}')
    configuration = await ECSJobConfiguration.from_template_and_values(
        base_job_template=ECSWorker.get_default_base_job_template(),
        values={**variables.model_dump(exclude_none=True)}
    )
    print(f'Constructed test configuration: {configuration.model_dump_json(indent=2)}')
    return configuration

async def construct_configuration_with_job_template(
    template_overrides: Dict[str, Any],
    **variables: Any
) -> ECSJobConfiguration:
    variables = ECSVariables(**variables | {'task_watch_poll_interval': 0.03})
    print(f'Using variables: {variables.model_dump_json(indent=2)}')
    base_template = ECSWorker.get_default_base_job_template()
    for key in template_overrides:
        base_template['job_configuration'][key] = template_overrides[key]
    print(f'Using base template configuration: {json.dumps(base_template["job_configuration"], indent=2)}')
    configuration = await ECSJobConfiguration.from_template_and_values(
        base_job_template=base_template,
        values={**variables.model_dump(exclude_none=True)}
    )
    print(f'Constructed test configuration: {configuration.model_dump_json(indent=2)}')
    return configuration

async def run_then_stop_task(
    worker: ECSWorker,
    configuration: ECSJobConfiguration,
    flow_run: FlowRun,
    after_start: Optional[Callable[[str], Awaitable[None]]] = None
) -> Any:
    session = configuration.aws_credentials.get_boto3_session()
    result: Any = None

    async def run(task_status: Any) -> None:
        nonlocal result
        result = await worker.run(flow_run, configuration, task_status=task_status)
        return
    
    with anyio.fail_after(20):
        async with anyio.create_task_group() as tg:
            identifier = await tg.start(run)
            cluster, task_arn = parse_identifier(identifier)
            if after_start:
                await after_start(task_arn)
            tg.start_soon(partial(stop_task, session.client('ecs'), task_arn, cluster=cluster))
    return result

# ... (rest of the test functions with similar type annotations)
