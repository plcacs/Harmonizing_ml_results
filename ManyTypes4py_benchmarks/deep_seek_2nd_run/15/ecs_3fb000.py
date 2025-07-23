from __future__ import annotations
import base64
import contextlib
import contextvars
import importlib
import ipaddress
import json
import shlex
import sys
from copy import deepcopy
from functools import partial
from textwrap import dedent
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Union, cast
import anyio
import anyio.to_thread
from anyio import run_process
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.syntax import Syntax
from prefect.cli._prompts import prompt
from prefect.client.schemas.actions import BlockDocumentCreate
from prefect.client.utilities import inject_client
from prefect.exceptions import ObjectNotFound
from prefect.settings import PREFECT_DEFAULT_DOCKER_BUILD_NAMESPACE, update_current_profile
from prefect.utilities.collections import get_from_dict
from prefect.utilities.importtools import lazy_import
if TYPE_CHECKING:
    from prefect.client.orchestration import PrefectClient
    from boto3 import Session
    from mypy_boto3_iam import IAMClient
    from mypy_boto3_ecs import ECSClient
    from mypy_boto3_ec2 import EC2Client, EC2ServiceResource
    from mypy_boto3_ecr import ECRClient

boto3 = lazy_import('boto3')
current_console = contextvars.ContextVar('console', default=Console())

@contextlib.contextmanager
def console_context(value: Console) -> Generator[None, None, None]:
    token = current_console.set(value)
    try:
        yield
    finally:
        current_console.reset(token)

class IamPolicyResource:
    """
    Represents an IAM policy resource for managing ECS tasks.

    Args:
        policy_name: The name of the IAM policy. Defaults to "prefect-ecs-policy".
    """

    def __init__(self, policy_name: str) -> None:
        self._iam_client: IAMClient = boto3.client('iam')
        self._policy_name: str = policy_name
        self._requires_provisioning: Optional[bool] = None

    async def get_task_count(self) -> int:
        """
        Returns the number of tasks that will be executed to provision this resource.

        Returns:
            int: The number of tasks to be provisioned.
        """
        return 1 if await self.requires_provisioning() else 0

    def _get_policy_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        paginator = self._iam_client.get_paginator('list_policies')
        page_iterator = paginator.paginate(Scope='Local')
        for page in page_iterator:
            for policy in page['Policies']:
                if policy['PolicyName'] == name:
                    return policy
        return None

    async def requires_provisioning(self) -> bool:
        """
        Check if this resource requires provisioning.

        Returns:
            bool: True if provisioning is required, False otherwise.
        """
        if self._requires_provisioning is not None:
            return self._requires_provisioning
        policy = await anyio.to_thread.run_sync(partial(self._get_policy_by_name, self._policy_name))
        if policy is not None:
            self._requires_provisioning = False
            return False
        self._requires_provisioning = True
        return True

    async def get_planned_actions(self) -> List[str]:
        """
        Returns a description of the planned actions for provisioning this resource.

        Returns:
            Optional[str]: A description of the planned actions for provisioning the resource,
                or None if provisioning is not required.
        """
        if await self.requires_provisioning():
            return [f'Creating and attaching an IAM policy for managing ECS tasks: [blue]{self._policy_name}[/]']
        return []

    async def provision(self, policy_document: Dict[str, Any], advance: Callable[[], None]) -> str:
        """
        Provisions an IAM policy.

        Args:
            advance: A callback function to indicate progress.

        Returns:
            str: The ARN (Amazon Resource Name) of the created IAM policy.
        """
        if await self.requires_provisioning():
            console = current_console.get()
            console.print('Creating IAM policy')
            policy = await anyio.to_thread.run_sync(partial(self._iam_client.create_policy, PolicyName=self._policy_name, PolicyDocument=json.dumps(policy_document)))
            policy_arn = policy['Policy']['Arn']
            advance()
            return policy_arn
        else:
            policy = await anyio.to_thread.run_sync(partial(self._get_policy_by_name, self._policy_name))
            assert policy is not None, 'Could not find expected policy'
            return policy['Arn']

    @property
    def next_steps(self) -> List[str]:
        return []

class IamUserResource:
    """
    Represents an IAM user resource for managing ECS tasks.

    Args:
        user_name: The desired name of the IAM user.
    """

    def __init__(self, user_name: str) -> None:
        self._iam_client: IAMClient = boto3.client('iam')
        self._user_name: str = user_name
        self._requires_provisioning: Optional[bool] = None

    async def get_task_count(self) -> int:
        """
        Returns the number of tasks that will be executed to provision this resource.

        Returns:
            int: The number of tasks to be provisioned.
        """
        return 1 if await self.requires_provisioning() else 0

    async def requires_provisioning(self) -> bool:
        """
        Check if this resource requires provisioning.

        Returns:
            bool: True if provisioning is required, False otherwise.
        """
        if self._requires_provisioning is None:
            try:
                await anyio.to_thread.run_sync(partial(self._iam_client.get_user, UserName=self._user_name))
                self._requires_provisioning = False
            except self._iam_client.exceptions.NoSuchEntityException:
                self._requires_provisioning = True
        return self._requires_provisioning

    async def get_planned_actions(self) -> List[str]:
        """
        Returns a description of the planned actions for provisioning this resource.

        Returns:
            Optional[str]: A description of the planned actions for provisioning the resource,
                or None if provisioning is not required.
        """
        if await self.requires_provisioning():
            return [f'Creating an IAM user for managing ECS tasks: [blue]{self._user_name}[/]']
        return []

    async def provision(self, advance: Callable[[], None]) -> None:
        """
        Provisions an IAM user.

        Args:
            advance: A callback function to indicate progress.
        """
        console = current_console.get()
        if await self.requires_provisioning():
            console.print('Provisioning IAM user')
            await anyio.to_thread.run_sync(partial(self._iam_client.create_user, UserName=self._user_name))
            advance()

    @property
    def next_steps(self) -> List[str]:
        return []

class CredentialsBlockResource:

    def __init__(self, user_name: str, block_document_name: str) -> None:
        self._block_document_name: str = block_document_name
        self._user_name: str = user_name
        self._requires_provisioning: Optional[bool] = None

    async def get_task_count(self) -> int:
        """
        Returns the number of tasks that will be executed to provision this resource.

        Returns:
            int: The number of tasks to be provisioned.
        """
        return 2 if await self.requires_provisioning() else 0

    @inject_client
    async def requires_provisioning(self, client: Optional[PrefectClient] = None) -> bool:
        if self._requires_provisioning is None:
            try:
                assert client is not None
                await client.read_block_document_by_name(self._block_document_name, 'aws-credentials')
                self._requires_provisioning = False
            except ObjectNotFound:
                self._requires_provisioning = True
        return self._requires_provisioning

    async def get_planned_actions(self) -> List[str]:
        """
        Returns a description of the planned actions for provisioning this resource.

        Returns:
            Optional[str]: A description of the planned actions for provisioning the resource,
                or None if provisioning is not required.
        """
        if await self.requires_provisioning():
            return ['Storing generated AWS credentials in a block']
        return []

    @inject_client
    async def provision(self, base_job_template: Dict[str, Any], advance: Callable[[], None], client: Optional[PrefectClient] = None) -> None:
        """
        Provisions an AWS credentials block.

        Will generate new credentials if the block does not already exist. Updates
        the `aws_credentials` variable in the job template to reference the block.

        Args:
            base_job_template: The base job template.
            advance: A callback function to indicate progress.
            client: A Prefect client to use for interacting with the Prefect API.
        """
        assert client is not None, 'Client injection failed'
        if not await self.requires_provisioning():
            block_doc = await client.read_block_document_by_name(self._block_document_name, 'aws-credentials')
        else:
            console = current_console.get()
            console.print('Generating AWS credentials')
            iam_client: IAMClient = boto3.client('iam')
            access_key_data = await anyio.to_thread.run_sync(partial(iam_client.create_access_key, UserName=self._user_name))
            access_key = access_key_data['AccessKey']
            advance()
            console.print('Creating AWS credentials block')
            assert client is not None
            try:
                credentials_block_type = await client.read_block_type_by_slug('aws-credentials')
            except ObjectNotFound as exc:
                raise RuntimeError(dedent('                    Unable to find block type "aws-credentials".\n                    To register the `aws-credentials` block type, run:\n\n                            pip install prefect-aws\n                            prefect blocks register -m prefect_aws\n\n                    ')) from exc
            credentials_block_schema = await client.get_most_recent_block_schema_for_block_type(block_type_id=credentials_block_type.id)
            assert credentials_block_schema is not None, f'Unable to find schema for block type {credentials_block_type.slug}'
            block_doc = await client.create_block_document(block_document=BlockDocumentCreate(name=self._block_document_name, data={'aws_access_key_id': access_key['AccessKeyId'], 'aws_secret_access_key': access_key['SecretAccessKey'], 'region_name': boto3.session.Session().region_name}, block_type_id=credentials_block_type.id, block_schema_id=credentials_block_schema.id))
            advance()
        base_job_template['variables']['properties']['aws_credentials']['default'] = {'$ref': {'block_document_id': str(block_doc.id)}}

    @property
    def next_steps(self) -> List[str]:
        return []

class AuthenticationResource:

    def __init__(self, work_pool_name: str, user_name: str = 'prefect-ecs-user', policy_name: str = 'prefect-ecs-policy', credentials_block_name: Optional[str] = None) -> None:
        self._user_name: str = user_name
        self._credentials_block_name: str = credentials_block_name or f'{work_pool_name}-aws-credentials'
        self._policy_name: str = policy_name
        self._policy_document: Dict[str, Any] = {'Version': '2012-10-17', 'Statement': [{'Sid': 'PrefectEcsPolicy', 'Effect': 'Allow', 'Action': ['ec2:AuthorizeSecurityGroupIngress', 'ec2:CreateSecurityGroup', 'ec2:CreateTags', 'ec2:DescribeNetworkInterfaces', 'ec2:DescribeSecurityGroups', 'ec2:DescribeSubnets', 'ec2:DescribeVpcs', 'ecs:CreateCluster', 'ecs:DeregisterTaskDefinition', 'ecs:DescribeClusters', 'ecs:DescribeTaskDefinition', 'ecs:DescribeTasks', 'ecs:ListAccountSettings', 'ecs:ListClusters', 'ecs:ListTaskDefinitions', 'ecs:RegisterTaskDefinition', 'ecs:RunTask', 'ecs:StopTask', 'ecs:TagResource', 'logs:CreateLogStream', 'logs:PutLogEvents', 'logs:DescribeLogGroups', 'logs:GetLogEvents'], 'Resource': '*'}]}
        self._iam_user_resource = IamUserResource(user_name=user_name)
        self._iam_policy_resource = IamPolicyResource(policy_name=policy_name)
        self._credentials_block_resource = CredentialsBlockResource(user_name=user_name, block_document_name=self._credentials_block_name)
        self._execution_role_resource = ExecutionRoleResource()

    @property
    def resources(self) -> List[Union[IamUserResource, IamPolicyResource, CredentialsBlockResource, ExecutionRoleResource]]:
        return [self._execution_role_resource, self._iam_user_resource, self._iam_policy_resource, self._credentials_block_resource]

    async def get_task_count(self) -> int:
        """
        Returns the number of tasks that will be executed to provision this resource.

        Returns:
            int: The number of tasks to be provisioned.
        """
        return sum([await resource.get_task_count() for resource in self.resources])

    async def requires_provisioning(self) -> bool:
        """
        Check if this resource requires provisioning.

        Returns:
            bool: True if provisioning is required, False otherwise.
        """
        return any([await resource.requires_provisioning() for resource in self.resources])

    async def get_planned_actions(self) -> List[str]:
        """
        Returns a description of the planned actions for provisioning this resource.

        Returns:
            Optional[str]: A description of the planned actions for provisioning the resource,
                or None if provisioning is not required.
        """
        return [action for resource in self.resources for action in await resource.get_planned_actions()]

    async def provision(self, base_job_template: Dict[str, Any], advance: Callable[[], None]) -> None:
        """
        Provisions the authentication resources.

        Args:
            base_job_template: The base job template of the work pool to provision
                infrastructure for.
            advance: A callback function to indicate progress.
        """
        role_arn = await self._execution_role_resource.provision(base_job_template=base_job_template, advance=advance)
        self._policy_document['Statement'].append({'Sid': 'AllowPassRoleForEcs', 'Effect': 'Allow', 'Action': 'iam:PassRole', 'Resource': role_arn})
        await self._iam_user_resource.provision(advance=advance)
        policy_arn = await self._iam_policy_resource.provision(policy_document=self._policy_document, advance=advance)
        if policy_arn:
            iam_client: IAMClient = boto3.client('iam')
            await anyio.to_thread.run_sync(partial(iam_client.attach_user_policy, UserName=self._user_name, PolicyArn=policy_arn))
        await self._credentials_block_resource.provision(base_job_template=base_job_template, advance=advance)

    @property
    def next_steps(self) -> List[str]:
        return [next_step for resource in self.resources for next_step in resource.next_steps]

class ClusterResource:

    def __init__(self, cluster_name: str = 'prefect-ecs-cluster') -> None:
        self._ecs_client: ECSClient = boto3.client('ecs')
        self._cluster_name: str = cluster_name
        self._requires_provisioning: Optional[bool] = None

    async def get_task_count(self) -> int:
        """
        Returns the number of tasks that will be executed to provision this resource.

        Returns:
            int: The number of tasks to be provisioned.
        """
        return 1 if await self.requires_provisioning() else 0

    async def requires_provisioning(self) -> bool:
        """
        Check if this resource requires provisioning.

        Returns:
            bool: True if provisioning is required, False otherwise.
        """
        if self._requires_provisioning is None:
            response = await anyio.to_thread.run_sync(partial(self._ecs_client.describe_clusters, clusters=[self._cluster_name]))
            if response['clusters'] and response['clusters'][0]['status'] == 'ACTIVE':
                self._requires_provisioning = False
            else:
                self._requires_provisioning = True
        return self._requires_provisioning

    async def get_planned_actions(self) -> List[str]:
        """
        Returns a description of the planned actions for provisioning this resource.

        Returns:
            Optional[str]: A description of the planned actions for provisioning the resource,
                or None if provisioning is not required.
        """
        if await self.requires_provisioning():
            return [f'Creating an ECS cluster for running Prefect flows: [blue]{self._cluster_name}[/]']
        return []

    async def provision(self, base_job_template: Dict[str, Any], advance: Callable[[], None]) -> None:
        """
        Provisions an ECS cluster.

        Will update the `cluster` variable in the job template to reference the cluster.

        Args:
            base_job_template: The base job template of the work pool to provision
                infrastructure for.
            advance: A callback function to indicate progress.
        """
        if await self.requires_provisioning():
            console = current_console.get()
            console.print('Provisioning ECS cluster')
            await anyio.to_thread.run_sync(partial(self._ecs_client.create_cluster, clusterName=self._cluster_name))
            advance()
        base_job_template['variables']['properties']['cluster']['default'] = self._cluster_name

    @property
    def next_steps(self) -> List[str]:
        return []

class VpcResource:

    def __init__(self, vpc_name: str = 'prefect-ecs-vpc', ecs_security_group_name: str = 'prefect-ecs-security-group') -> None:
        self._ec2_client: EC2Client = boto3.client('ec2')
        self._ec2_resource: EC