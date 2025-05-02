```python
from __future__ import annotations

# pylint: disable=too-many-lines
import os
import time
import tempfile
from datetime import datetime
import zipfile
import shutil
import json
import re
import uuid
from collections import OrderedDict
from typing import (
    Any,
    Optional,
    Dict,
    Callable,
    List,
    Iterator,
    Iterable,
    Sequence,
    IO,
    Tuple,
    Union,
    cast,
)  # noqa

import botocore.session  # noqa
from botocore.loaders import create_loader
from botocore.exceptions import ClientError
from botocore.utils import datetime2timestamp
from botocore.vendored.requests import (
    ConnectionError as RequestsConnectionError,
)
from botocore.vendored.requests.exceptions import (
    ReadTimeout as RequestsReadTimeout,
)
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from chalice.constants import DEFAULT_STAGE_NAME
from chalice.constants import MAX_LAMBDA_DEPLOYMENT_SIZE
from chalice.vendored.botocore.regions import EndpointResolver

StrMap = Optional[Dict[str, str]]
StrAnyMap = Dict[str, Any]
OptStr = Optional[str]
OptInt = Optional[int]
OptStrList = Optional[List[str]]
ClientMethod = Callable[..., Dict[str, Any]]
CWLogEvent = TypedDict(
    'CWLogEvent',
    {
        'eventId': str,
        'ingestionTime': datetime,
        'logStreamName': str,
        'message': str,
        'timestamp': datetime,
        'logShortId': str,
    },
)
LogEventsResponse = TypedDict(
    'LogEventsResponse',
    {
        'events': List[CWLogEvent],
        'nextToken': str,
    },
    total=False,
)
DomainNameResponse = TypedDict(
    'DomainNameResponse',
    {
        'domain_name': str,
        'security_policy': str,
        'hosted_zone_id': str,
        'certificate_arn': str,
        'alias_domain_name': str,
    },
)

_REMOTE_CALL_ERRORS = (
    ClientError,
    RequestsConnectionError,
)


class AWSClientError(Exception):
    pass


class ReadTimeout(AWSClientError):
    def __init__(self, message: str) -> None:
        self.message = message


class ResourceDoesNotExistError(AWSClientError):
    pass


class LambdaClientError(AWSClientError):
    def __init__(
        self, original_error: Exception, context: LambdaErrorContext
    ) -> None:
        self.original_error = original_error
        self.context = context
        super(LambdaClientError, self).__init__(str(original_error))


class DeploymentPackageTooLargeError(LambdaClientError):
    pass


class LambdaErrorContext(object):
    def __init__(
        self,
        function_name: str,
        client_method_name: str,
        deployment_size: int,
    ) -> None:
        self.function_name = function_name
        self.client_method_name = client_method_name
        self.deployment_size = deployment_size


class TypedAWSClient(object):
    LAMBDA_CREATE_ATTEMPTS = 30
    DELAY_TIME = 5

    def __init__(
        self,
        session: botocore.session.Session,
        sleep: Callable[[int], None] = time.sleep,
    ) -> None:
        self._session = session
        self._sleep = sleep
        self._client_cache: Dict[str, Any] = {}
        loader = create_loader('data_loader')
        endpoints = loader.load_data('endpoints')
        self._endpoint_resolver = EndpointResolver(endpoints)

    def resolve_endpoint(
        self, service: str, region: str
    ) -> Optional[OrderedDict[str, Any]]:
        return self._endpoint_resolver.construct_endpoint(service, region)

    def endpoint_from_arn(self, arn: str) -> Optional[OrderedDict[str, Any]]:
        arn_split = arn.split(':')
        return self.resolve_endpoint(arn_split[2], arn_split[3])

    def endpoint_dns_suffix(self, service: str, region: str) -> str:
        endpoint = self.resolve_endpoint(service, region)
        return endpoint['dnsSuffix'] if endpoint else 'amazonaws.com'

    def endpoint_dns_suffix_from_arn(self, arn: str) -> str:
        endpoint = self.endpoint_from_arn(arn)
        return endpoint['dnsSuffix'] if endpoint else 'amazonaws.com'

    def service_principal(
        self,
        service: str,
        region: str = 'us-east-1',
        url_suffix: str = 'amazonaws.com',
    ) -> str:
        matches = re.match(
            (
                r'^([^.]+)'
                r'(?:(?:\.amazonaws\.com(?:\.cn)?)|'
                r'(?:\.c2s\.ic\.gov)|'
                r'(?:\.sc2s\.sgov\.gov))?$'
            ),
            service,
        )

        if matches is None:
            return service

        service_name = matches.group(1)

        us_iso_exceptions = {'cloudhsm', 'config', 'states', 'workspaces'}

        us_isob_exceptions = {'dms', 'states'}

        if region.startswith('us-iso-') and service_name in us_iso_exceptions:
            if service_name == 'states':
                return f'{service_name}.amazonaws.com'
            else:
                return f'{service_name}.{url_suffix}'

        if (
            region.startswith('us-isob-')
            and service_name in us_isob_exceptions
        ):
            if service_name == 'states':
                return f'{service_name}.amazonaws.com'
            else:
                return f'{service_name}.{url_suffix}'

        if service_name in ['codedeploy', 'logs']:
            return f'{service_name}.{region}.{url_suffix}'
        elif service_name == 'states':
            return f'{service_name}.{region}.amazonaws.com'
        elif service_name == 'ec2':
            return f'{service_name}.{url_suffix}'
        else:
            return f'{service_name}.amazonaws.com'

    def lambda_function_exists(self, name: str) -> bool:
        client = self._client('lambda')
        try:
            client.get_function(FunctionName=name)
            return True
        except client.exceptions.ResourceNotFoundException:
            return False

    def api_mapping_exists(self, domain_name: str, api_map_key: str) -> bool:
        client = self._client('apigatewayv2')
        try:
            result = client.get_api_mappings(DomainName=domain_name)
            api_map = [
                api_map
                for api_map in result['Items']
                if api_map['ApiMappingKey'] == api_map_key
            ]
            return bool(api_map)
        except client.exceptions.NotFoundException:
            return False

    def get_domain_name(self, domain_name: str) -> Dict[str, Any]:
        client = self._client('apigateway')
        try:
            domain = client.get_domain_name(domainName=domain_name)
        except client.exceptions.NotFoundException:
            raise ResourceDoesNotExistError(f"No domain name found by {domain_name} name")
        return domain

    def domain_name_exists(self, domain_name: str) -> bool:
        try:
            self.get_domain_name(domain_name)
            return True
        except ResourceDoesNotExistError:
            return False

    def domain_name_exists_v2(self, domain_name: str) -> bool:
        client = self._client('apigatewayv2')
        try:
            client.get_domain_name(DomainName=domain_name)
            return True
        except client.exceptions.NotFoundException:
            return False

    def get_function_configuration(self, name: str) -> Dict[str, Any]:
        response = self._client('lambda').get_function_configuration(
            FunctionName=name
        )
        return response

    def _create_vpc_config(
        self, security_group_ids: OptStrList, subnet_ids: OptStrList
    ) -> Dict[str, List[str]]:
        vpc_config: Dict[str, List[str]] = {
            'SubnetIds': [],
            'SecurityGroupIds': [],
        }
        if security_group_ids is not None and subnet_ids is not None:
            vpc_config['SubnetIds'] = subnet_ids
            vpc_config['SecurityGroupIds'] = security_group_ids
        return vpc_config

    def publish_layer(
        self, layer_name: str, zip_contents: bytes, runtime: str
    ) -> str:
        try:
            return self._client('lambda').publish_layer_version(
                LayerName=layer_name,
                Content={'ZipFile': zip_contents},
                CompatibleRuntimes=[runtime],
            )['LayerVersionArn']
        except _REMOTE_CALL_ERRORS as e:
            context = LambdaErrorContext(
                layer_name, 'publish_layer_version', len(zip_contents)
            )
            raise self._get_lambda_code_deployment_error(e, context)

    def delete_layer_version(self, layer_version_arn: str) -> None:
        client = self._client('lambda')
        _, layer_name, version_number = layer_version_arn.rsplit(":", 2)
        try:
            client.delete_layer_version(
                LayerName=layer_name, VersionNumber=int(version_number)
            )
        except client.exceptions.ResourceNotFoundException:
            pass

    def get_layer_version(self, layer_version_arn: str) -> Dict[str, Any]:
        client = self._client('lambda')
        try:
            return client.get_layer_version_by_arn(Arn=layer_version_arn)
        except client.exceptions.ResourceNotFoundException:
            return {}

    def create_function(
        self,
        function_name: str,
        role_arn: str,
        zip_contents: bytes,
        runtime: str,
        handler: str,
        environment_variables: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        xray: Optional[bool] = None,
        timeout: Optional[int] = None,
        memory_size: Optional[int] = None,
        security_group_ids: Optional[List[str]] = None,
        subnet_ids: Optional[List[str]] = None,
        layers: Optional[List[str]] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {
            'FunctionName': function_name,
            'Runtime': runtime,
            'Code': {'ZipFile': zip_contents},
            'Handler': handler,
            'Role': role_arn,
        }
        if environment_variables is not None:
            kwargs['Environment'] = {"Variables": environment_variables}
        if tags is not None:
            kwargs['Tags'] = tags
        if xray is True:
            kwargs['TracingConfig'] = {'Mode': 'Active'}
        if timeout is not None:
            kwargs['Timeout'] = timeout
        if memory_size is not None:
            kwargs['MemorySize'] = memory_size
        if security_group_ids is not None and subnet_ids is not None:
            kwargs['VpcConfig'] = self._create_vpc_config(
                security_group_ids=security_group_ids,
                subnet_ids=subnet_ids,
            )
        if layers is not None:
            kwargs['Layers'] = layers
        arn, state = self._create_lambda_function(kwargs)
        if state != 'Active':
            self._wait_for_active(function_name)
        return arn

    def _wait_for_active(self, function_name: str) -> None:
        client = self._client('lambda')
        waiter = client.get_waiter('function_active')
        waiter.wait(FunctionName=function_name)

    def create_api_mapping(
        self, domain_name: str, path_key: str, api_id: str, stage: str
    ) -> Dict[str, str]:
        kwargs = {
            'DomainName': domain_name,
            'ApiMappingKey': path_key,
            'ApiId': api_id,
            'Stage': stage,
        }
        return self._create_api_mapping(kwargs)

    def create_base_path_mapping(
        self, domain_name: str, path_key: str, api_id: str, stage: str
    ) -> Dict[str, str]:
        kwargs = {
            'domainName': domain_name,
            'basePath': path_key,
            'restApiId': api_id,
            'stage': stage,
        }
        return self._create_base_path_mapping(kwargs)

    def _create_base_path_mapping(
        self, base_path_args: Dict[str, Any]
    ) -> Dict[str, str]:
        result = self._client('apigateway').create_base_path_mapping(
            **base_path_args
        )
        base_path = "/" if result['basePath'] == '(none)' else f"/{result['basePath']}"
        return {'key': base_path}

    def _create_api_mapping(self, api_args: Dict[str, Any]) -> Dict[str, str]:
        result = self._client('apigatewayv2').create_api_mapping(**api_args)
        map_key = "/" if result['ApiMappingKey'] == '(none)' else f"/{result['ApiMappingKey']}"
        return {'key': map_key}

    def create_domain_name(
        self,
        protocol: str,
        domain_name: str,
        endpoint_type: str,
        certificate_arn: str,
        security_policy: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> DomainNameResponse:
        if protocol == 'HTTP':
            kwargs: Dict[str, Any] = {
                'domainName': domain_name,
                'endpointConfiguration': {'types': [endpoint_type]},
            }
            if security_policy is not None:
                kwargs['securityPolicy'] = security_policy
            if endpoint_type == 'EDGE':
                kwargs['certificateArn'] = certificate_arn
            else:
                kwargs['regionalCertificateArn'] = certificate_arn
            if tags is not None:
                kwargs['tags'] = tags
            created_domain_name = self._create_domain_name(kwargs)
        elif protocol == 'WEBSOCKET':
            kwargs = self.get_custom_domain_params_v2(
                domain_name=domain_name,
                endpoint_type=endpoint_type,
                security_policy=security_policy,
                certificate_arn=certificate_arn,
                tags=tags,
            )
            created_domain_name = self._create_domain_name_v2(kwargs)
        else:
            raise ValueError("Unsupported protocol value.")
        return created_domain_name

    def _create_domain_name(
        self, api_args: Dict[str, Any]
    ) -> DomainNameResponse:
        client = self._client('apigateway')
        result = self._call_client_method_with_retries(
            client.create_domain_name,
            api_args,
            max_attempts=6,
            should_retry=lambda x: True,
            retryable_exceptions=(client.exceptions.TooManyRequestsException,),
        )
        hosted_zone_id = result.get('regionalHostedZoneId', result['distributionHostedZoneId'])
        certificate_arn = result.get('regionalCertificateArn', result['certificateArn'])
        alias_domain_name = result.get('regionalDomainName', result['distributionDomainName'])
        return {
            'domain_name': result['domainName'],
            'security_policy': result['securityPolicy'],
            'hosted_zone_id': hosted_zone_id,
            'certificate_arn': certificate_arn,
            'alias_domain_name': alias_domain_name,
        }

    def _create_domain_name_v2(
        self, api_args: Dict[str, Any]
    ) -> DomainNameResponse:
        client = self._client('apigatewayv2')
        result = self._call_client_method_with_retries(
            client.create_domain_name,
            api_args,
            max_attempts=6,
            should_retry=lambda x: True,
            retryable_exceptions=(client.exceptions.TooManyRequestsException,),
        )
        result_data = result['DomainNameConfigurations'][0]
        return {
            'domain_name': result['DomainName'],
            'alias_domain_name': result_data['ApiGatewayDomainName'],
            'security_policy': result_data['SecurityPolicy'],
            'hosted_zone_id': result_data['HostedZoneId'],
            'certificate_arn': result_data['CertificateArn'],
        }

    def _create_lambda_function(
        self, api_args: Dict[str, Any]
    ) -> Tuple[str, str]:
        try:
            result = self._call_client_method_with_retries(
                self._client('lambda').create_function,
                api_args,
                max_attempts=self.LAMBDA_CREATE_ATTEMPTS,
            )
            return result['FunctionArn'], result['State']
        except _REMOTE_CALL_ERRORS as e:
            context = LambdaErrorContext(
                api_args['FunctionName'],
                'create_function',
                len(api_args['Code']['ZipFile']),
            )
            raise self._get_lambda_code_deployment_error(e, context)

    def _is_settling_error(
        self, error: ClientError
    ) -> bool:
        message = error.response['Error'].get('Message', '')
        return bool(re.search('event source mapping.*is in use', message))

    def invoke_function(
        self, name: str, payload: Optional[bytes] = None
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Union[str, bytes]] = {
            'FunctionName': name,
            'InvocationType': 'RequestResponse',
        }
        if payload is not None:
            kwargs['Payload'] = payload

        try:
            return self._client('lambda').invoke(**kwargs)
        except RequestsReadTimeout as e:
            raise ReadTimeout(str(e))

    def _is_iam_role_related_error(
        self, error: ClientError
    ) -> bool:
        message = error.response['Error'].get('Message', '')
        return bool(re.search('role.*cannot be assumed', message) or
                   re.search('role.*does not have permissions', message) or
                   re.search('InvalidArnException.*valid principal', message))

    def _get_lambda_code_deployment_error(
        self, error: Exception, context: LambdaErrorContext
    ) -> LambdaClientError:
        error_cls: Type[LambdaClientError] = LambdaClientError
        if isinstance(error, RequestsConnectionError) and context.deployment_size > MAX_LAMBDA_DEPLOY