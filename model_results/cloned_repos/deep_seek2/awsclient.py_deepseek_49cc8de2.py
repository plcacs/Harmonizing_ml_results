"""Simplified AWS client.

This module abstracts the botocore session and clients
to provide a simpler interface.  This interface only
contains the API calls needed to work with AWS services
used by chalice.

The interface provided can range from a direct 1-1 mapping
of a method to a method on a botocore client all the way up
to combining API calls across multiple AWS services.

As a side benefit, I can also add type annotations to
this class to get improved type checking across chalice.

"""
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
    botocore.exceptions.ClientError,
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
    # 30 * 5 == 150 seconds or 2.5 minutes for the initial lambda
    # creation + role propagation.
    LAMBDA_CREATE_ATTEMPTS: int = 30
    DELAY_TIME: int = 5

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
        """Find details of an endpoint based on the service and region.

        This utilizes the botocore EndpointResolver in order to find details on
        the given service and region combination.  If the service and region
        combination is not found the None will be returned.
        """
        return self._endpoint_resolver.construct_endpoint(service, region)

    def endpoint_from_arn(self, arn: str) -> Optional[OrderedDict[str, Any]]:
        """Find details for the endpoint associated with a resource ARN.

        This allows the an endpoint to be discerned based on an ARN.  This
        is a convenience method due to the need to parse multiple ARNs
        throughout the project. If the service and region combination
        is not found the None will be returned.
        """
        arn_split = arn.split(':')
        return self.resolve_endpoint(arn_split[2], arn_split[3])

    def endpoint_dns_suffix(self, service: str, region: str) -> str:
        """Discover the dns suffix for a given service and region combination.

        This allows the service DNS suffix to be discoverable throughout the
        framework.  If the ARN's service and region combination is not found
        then amazonaws.com is returned.

        """
        endpoint = self.resolve_endpoint(service, region)
        return endpoint['dnsSuffix'] if endpoint else 'amazonaws.com'

    def endpoint_dns_suffix_from_arn(self, arn: str) -> str:
        """Discover the dns suffix for a given ARN.

        This allows the service DNS suffix to be discoverable throughout the
        framework based on the ARN.  If the ARN's service and region
        combination is not found then amazonaws.com is returned.

        """
        endpoint = self.endpoint_from_arn(arn)
        return endpoint['dnsSuffix'] if endpoint else 'amazonaws.com'

    def service_principal(
        self,
        service: str,
        region: str = 'us-east-1',
        url_suffix: str = 'amazonaws.com',
    ) -> str:
        # Disable too-many-return-statements due to ported code
        # pylint: disable=too-many-return-statements
        """Compute a "standard" AWS Service principal for given arguments.

        Attribution: This code was ported from https://github.com/aws/aws-cdk
        and more specifically, aws-cdk/region-info/lib/default.ts

        Computes a "standard" AWS Service principal for a given service, region
        and suffix. This is useful for example when you need to compute a
        service principal name, but you do not have a synthesize-time region
        literal available (so all you have is `{ "Ref": "AWS::Region" }`). This
        way you get the same defaulting behavior that is normally used for
        built-in data.

        :param service: the name of the service (s3, s3.amazonaws.com, ...)
        :param region: the region in which the service principal is needed.
        :param url_suffix: the URL suffix for the partition in which the region
        is located.
        :return: The service principal for the given combination of arguments
        """
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
            #  Return "service" if it does not look like any of the following:
            #  - s3
            #  - s3.amazonaws.com
            #  - s3.amazonaws.com.cn
            #  - s3.c2s.ic.gov
            #  - s3.sc2s.sgov.gov
            return service

        # Simplify the service name down to something like "s3"
        service_name = matches.group(1)

        # Exceptions for Service Principals in us-iso-*
        us_iso_exceptions = {'cloudhsm', 'config', 'states', 'workspaces'}

        # Exceptions for Service Principals in us-isob-*
        us_isob_exceptions = {'dms', 'states'}

        # Account for idiosyncratic Service Principals in `us-iso-*` regions
        if region.startswith('us-iso-') and service_name in us_iso_exceptions:
            if service_name == 'states':
                # Services with universal principal
                return '{}.amazonaws.com'.format(service_name)
            else:
                # Services with a partitional principal
                return '{}.{}'.format(service_name, url_suffix)

        # Account for idiosyncratic Service Principals in `us-isob-*` regions
        if (
            region.startswith('us-isob-')
            and service_name in us_isob_exceptions
        ):
            if service_name == 'states':
                # Services with universal principal
                return '{}.amazonaws.com'.format(service_name)
            else:
                # Services with a partitional principal
                return '{}.{}'.format(service_name, url_suffix)

        if service_name in ['codedeploy', 'logs']:
            return '{}.{}.{}'.format(service_name, region, url_suffix)
        elif service_name == 'states':
            return '{}.{}.amazonaws.com'.format(service_name, region)
        elif service_name == 'ec2':
            return '{}.{}'.format(service_name, url_suffix)
        else:
            return '{}.amazonaws.com'.format(service_name)

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
            if api_map:
                return True
            return False
        except client.exceptions.NotFoundException:
            return False

    def get_domain_name(self, domain_name: str) -> Dict[str, Any]:
        client = self._client('apigateway')
        try:
            domain = client.get_domain_name(domainName=domain_name)
        except client.exceptions.NotFoundException:
            err_msg = "No domain name found by %s name" % domain_name
            raise ResourceDoesNotExistError(err_msg)
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
        # We always set the SubnetIds and SecurityGroupIds to an empty
        # list to ensure that we properly remove Vpc configuration
        # if you remove these values from your config.json.  Omitting
        # the VpcConfig key or just setting to {} won't actually remove
        # the VPC configuration.
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
            return client.delete_layer_version(
                LayerName=layer_name, VersionNumber=int(version_number)
            )
        except client.exceptions.ResourceNotFoundException:
            pass

    def get_layer_version(self, layer_version_arn: str) -> Dict[str, Any]:
        client = self._client('lambda')
        try:
            return client.get_layer_version_by_arn(Arn=layer_version_arn)
        except client.exceptions.ResourceNotFoundException:
            pass
        return {}

    def create_function(
        self,
        function_name: str,
        role_arn: str,
        zip_contents: str,
        runtime: str,
        handler: str,
        environment_variables: Optional[StrMap] = None,
        tags: Optional[StrMap] = None,
        xray: Optional[bool] = None,
        timeout: OptInt = None,
        memory_size: OptInt = None,
        security_group_ids: OptStrList = None,
        subnet_ids: OptStrList = None,
        layers: OptStrList = None,
    ) -> str:
        # pylint: disable=too-many-locals
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
        # Avoid the GetFunctionConfiguration call unless
        # we're not immediately active.
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
        if result['basePath'] == '(none)':
            base_path = "/"
        else:
            base_path = "/%s" % result['basePath']
        base_path_mapping = {'key': base_path}
        return base_path_mapping

    def _create_api_mapping(self, api_args: Dict[str, Any]) -> Dict[str, str]:
        result = self._client('apigatewayv2').create_api_mapping(**api_args)
        if result['ApiMappingKey'] == '(none)':
            map_key = "/"
        else:
            map_key = "/%s" % result['ApiMappingKey']
        api_mapping = {'key': map_key}
        return api_mapping

    def create_domain_name(
        self,
        protocol: str,
        domain_name: str,
        endpoint_type: str,
        certificate_arn: str,
        security_policy: Optional[str] = None,
        tags: Optional[StrMap] = None,
    ) -> DomainNameResponse:
        if protocol == 'HTTP':
            kwargs = {
                'domainName': domain_name,
                'endpointConfiguration': {
                    'types': [endpoint_type],
                },
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
        exceptions = (client.exceptions.TooManyRequestsException,)
        result = self._call_client_method_with_retries(
            client.create_domain_name,
            api_args,
            max_attempts=6,
            should_retry=lambda x: True,
            retryable_exceptions=exceptions,
        )
        if result.get('regionalHostedZoneId'):
            hosted_zone_id = result['regionalHostedZoneId']
        else:
            hosted_zone_id = result['distributionHostedZoneId']

        if result.get('regionalCertificateArn'):
            certificate_arn = result['regionalCertificateArn']
        else:
            certificate_arn = result['certificateArn']

        if result.get('regionalDomainName') is not None:
            alias_domain_name = result['regionalDomainName']
        else:
            alias_domain_name = result['distributionDomainName']
        domain_name: DomainNameResponse = {
            'domain_name': result['domainName'],
            'security_policy': result['securityPolicy'],
            'hosted_zone_id': hosted_zone_id,
            'certificate_arn': certificate_arn,
            'alias_domain_name': alias_domain_name,
        }
        return domain_name

    def _create_domain_name_v2(
        self, api_args: Dict[str, Any]
    ) -> DomainNameResponse:
        client = self._client('apigatewayv2')
        exceptions = (client.exceptions.TooManyRequestsException,)
        result = self._call_client_method_with_retries(
            client.create_domain_name,
            api_args,
            max_attempts=6,
            should_retry=lambda x: True,
            retryable_exceptions=exceptions,
        )
        result_data = result['DomainNameConfigurations'][0]
        domain_name: DomainNameResponse = {
            'domain_name': result['DomainName'],
            'alias_domain_name': result_data['ApiGatewayDomainName'],
            'security_policy': result_data['SecurityPolicy'],
            'hosted_zone_id': result_data['HostedZoneId'],
            'certificate_arn': result_data['CertificateArn'],
        }
        return domain_name

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
        self, error: botocore.exceptions.ClientError
    ) -> bool:
        message = error.response['Error'].get('Message', '')
        if re.search('event source mapping.*is in use', message):
            return True
        return False

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
        self, error: botocore.exceptions.ClientError
    ) -> bool:
        message = error.response['Error'].get('Message', '')
        if re.search('role.*cannot be assumed', message):
            return True
        if re.search('role.*does not have permissions', message):
            return True
        # This message is also related to IAM roles, it happens when the grant
        # used for the KMS key for encrypting env vars doesn't think the
        # principal is valid yet.
        if re.search('InvalidArnException.*valid principal', message):
            return True
        return False

    def _get_lambda_code_deployment_error(
        self, error: Any, context: LambdaErrorContext
    ) -> LambdaClientError:
        error_cls = LambdaClientError
        if (
            isinstance(error, RequestsConnectionError)
            and context.deployment_size > MAX_LAMBDA_DEPLOYMENT_SIZE
        ):
            # When the zip deployment package is too large and Lambda
            # aborts the connection as chalice is still sending it
            # data
            error_cls = DeploymentPackageTooLargeError
        elif isinstance(error, ClientError):
            code = error.response['Error'].get('Code', '')
            message = error.response['Error'].get('Message', '')
            if code == 'RequestEntityTooLargeException':
                # Happens when the zipped deployment package sent to lambda
                # is too large
                error_cls = DeploymentPackageTooLargeError
            elif (
                code == 'InvalidParameterValueException'
                and 'Unzipped size must be smaller' in message
            ):
                # Happens when the contents of the unzipped deployment
                # package sent to lambda is too large
                error_cls = DeploymentPackageTooLargeError
        return error_cls(error, context)

    def delete_function(self, function_name: str) -> None:
        lambda_client = self._client('lambda')
        try:
            lambda_client.delete_function(FunctionName=function_name)
        except lambda_client.exceptions.ResourceNotFoundException:
            raise ResourceDoesNotExistError(function_name)

    def get_custom_domain_params_v2(
        self,
        domain_name: str,
        endpoint_type: str,
        certificate_arn: str,
        security_policy: Optional[str] = None,
        tags: Optional[StrMap] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            'DomainName': domain_name,
            'DomainNameConfigurations': [
                {
                    'ApiGatewayDomainName': domain_name,
                    'CertificateArn': certificate_arn,
                    'EndpointType': endpoint_type,
                    'SecurityPolicy': security_policy,
                    'DomainNameStatus': 'AVAILABLE',
                }
            ],
        }
        if tags:
            kwargs['Tags'] = tags
        return kwargs

    def get_custom_domain_patch_operations(
        self,
        certificate_arn: str,
        endpoint_type: Optional[str],
        security_policy: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        patch_operations = []
        if security_policy is not None:
            patch_operations.append(
                {
                    'op': 'replace',
                    'path': '/securityPolicy',
                    'value': security_policy,
                }
            )
        if endpoint_type == 'EDGE':
            patch_operations.append(
                {
                    'op': 'replace',
                    'path': '/certificateArn',
                    'value': certificate_arn,
                }
            )
        else:
            patch_operations.append(
                {
                    'op': 'replace',
                    'path': '/regionalCertificateArn',
                    'value': certificate_arn,
                }
            )
        return patch_operations

    def update_domain_name(
        self,
        protocol: str,
        domain_name: str,
        endpoint_type: str,
        certificate_arn: str,
        security_policy: Optional[str] = None,
        tags: Optional[StrMap] = None,
    ) -> DomainNameResponse:
        if protocol == 'HTTP':
            patch_operations = self.get_custom_domain_patch_operations(
                certificate_arn,
                endpoint_type,
                security_policy,
            )
            updated_domain_name = self._update_domain_name(
                domain_name, patch_operations
            )
        elif protocol == 'WEBSOCKET':
            kwargs = self.get_custom_domain_params_v2(
                domain_name=domain_name,
                endpoint_type=endpoint_type,
                security_policy=security_policy,
                certificate_arn=certificate_arn,
            )
            updated_domain_name = self._update_domain_name_v2(kwargs)
        else:
            raise ValueError('Unsupported protocol value.')
        resource_arn = (
            'arn:{partition}:apigateway:{region_name}:'
            ':/domainnames/{domain_name}'.format(
                partition=self.partition_name,
                region_name=self.region_name,
                domain_name=domain_name,
            )
        )
        self._update_resource_tags(resource_arn, tags)
        return updated_domain_name

    def _update_resource_tags(
        self, resource_arn: str, requested_tags: Optional[Dict[str, str]]
    ) -> None:
        if not requested_tags:
            requested_tags = {}

        remote_tags = self._client('apigatewayv2').get_tags(
            ResourceArn=resource_arn
        )['Tags']
        self._remove_unrequested_resource_tags(
            resource_arn, requested_tags, remote_tags
        )
        self._add_missing_or_differing_value_resource_tags(
            resource_arn, requested_tags, remote_tags
        )

    def _remove_unrequested_resource_tags(
        self,
        resource_arn: str,
        requested_tags: Dict[Any, Any],
        remote_tags: Dict[Any, Any],
    ) -> None:
        tag_keys_to_remove = list(set(remote_tags) - set(requested_tags))
        if tag_keys_to_remove:
            self._client('apigatewayv2').untag_resource(
                ResourceArn=resource_arn, TagKeys=tag_keys_to_remove
            )

    def _add_missing_or_differing_value_resource_tags(
        self,
        resource_arn: str,
        requested_tags: Dict[Any, Any],
        remote_tags: Dict[Any, Any],
    ) -> None:
        tags_to_add = {
            k: v
            for k, v in requested_tags.items()
            if k not in remote_tags or v != remote_tags[k]
        }
        if tags_to_add:
            self._client('apigatewayv2').tag_resource(
                ResourceArn=resource_arn, Tags=tags_to_add
            )

    def _update_domain_name(
        self, custom_domain_name: str, patch_operations: List[Dict[str, str]]
    ) -> DomainNameResponse:
        client = self._client('apigateway')
        exceptions = (client.exceptions.TooManyRequestsException,)
        result = {}
        for patch_operation in patch_operations:
            api_args = {
                'domainName': custom_domain_name,
                'patchOperations': [patch_operation],
            }
            response = self._call_client_method_with_retries(
                client.update_domain_name,
                api_args,
                max_attempts=6,
                should_retry=lambda x: True,
                retryable_exceptions=exceptions,
            )
            result.update(response)

        if result.get('regionalCertificateArn'):
            certificate_arn = result['regionalCertificateArn']
        else:
            certificate_arn = result['certificateArn']

        if result.get('regionalHostedZoneId'):
            hosted_zone_id = result['regionalHostedZoneId']
        else:
            hosted_zone_id = result['distributionHostedZoneId']

        if result.get('regionalDomainName') is not None:
            alias_domain_name = result['regionalDomainName']
        else:
            alias_domain_name = result['distributionDomainName']
        domain_name: DomainNameResponse = {
            'domain_name': result['domainName'],
            'security_policy': result['securityPolicy'],
            'certificate_arn': certificate_arn,
            'hosted_zone_id': hosted_zone_id,
            'alias_domain_name': alias_domain_name,
        }
        return domain_name

    def _update_domain_name_v2(
        self, api_args: Dict[str, Any]
    ) -> DomainNameResponse:
        client = self._client('apigatewayv2')
        exceptions = (client.exceptions.TooManyRequestsException,)

        result = self._call_client_method_with_retries(
            client.update_domain_name,
            api_args,
            max_attempts=6,
            should_retry=lambda x: True,
            retryable_exceptions=exceptions,
        )
        result_data = result['DomainNameConfigurations'][0]
        domain_name: DomainNameResponse = {
            'domain_name': result['DomainName'],
            'alias_domain_name': result_data['ApiGatewayDomainName'],
            'security_policy': result_data['SecurityPolicy'],
            'hosted_zone_id': result_data['HostedZoneId'],
            'certificate_arn': result_data['CertificateArn'],
        }
        return domain_name

    def delete_domain_name(self, domain_name: str) -> None:
        client = self._client('apigatewayv2')
        params = {'DomainName': domain_name}

        exceptions = (client.exceptions.TooManyRequestsException,)
        self._call_client_method_with_retries(
            client.delete_domain_name,
            params,
            max_attempts=6,
            should_retry=lambda x: True,
            retryable_exceptions=exceptions,
        )

    def delete_api_mapping(self, domain_name: str, path_key: str) -> None:
        client = self._client('apigateway')
        params = {'domainName': domain_name, 'basePath': path_key}
        client.delete_base_path_mapping(**params)

    def update_function(
        self,
        function_name: str,
        zip_contents: str,
        environment_variables: Optional[StrMap] = None,
        runtime: OptStr = None,
        tags: Optional[StrMap] = None,
        xray: Optional[bool] = None,
        timeout: OptInt = None,
        memory_size: OptInt = None,
        role_arn: OptStr = None,
        subnet_ids: OptStrList = None,
        security_group_ids: OptStrList = None,
        layers: OptStrList = None,
    ) -> Dict[str, Any]:
        """Update a Lambda function's code and configuration.

        This method only updates the values provided to it. If a parameter
        is not provided, no changes will be made for that that parameter on
        the targeted lambda function.
        """
        return_value = self._update_function_code(
            function_name=function_name, zip_contents=zip_contents
        )
        self._update_function_config(
            environment_variables=environment_variables,
            runtime=runtime,
            timeout=timeout,
            memory_size=memory_size,
            role_arn=role_arn,
            xray=xray,
            subnet_ids=subnet_ids,
            security_group_ids=security_group_ids,
            function_name=function_name,
            layers=layers,
        )
        if tags is not None:
            self._update_function_tags(return_value['FunctionArn'], tags)
        return return_value

    def _update_function_code(
        self, function_name: str, zip_contents: str
    ) -> Dict[str, Any]:
        lambda_client = self._client('lambda')
        try:
            result = lambda_client.update_function_code(
                FunctionName=function_name, ZipFile=zip_contents
            )
        except _REMOTE_CALL_ERRORS as e:
            context = LambdaErrorContext(
                function_name, 'update_function_code', len(zip_contents)
            )
            raise self._get_lambda_code_deployment_error(e, context)
        if result['LastUpdateStatus'] != 'Successful':
            self._wait_for_function_update(function_name)
        return result

    def _wait_for_function_update(self, function_name: str) -> None:
        client = self._client('lambda')
        waiter = client.get_waiter('function_updated')
        waiter.wait(FunctionName=function_name)

    def put_function_concurrency(
        self, function_name: str, reserved_concurrent_executions: int
    ) -> None:
        lambda_client = self._client('lambda')
        lambda_client.put_function_concurrency(
            FunctionName=function_name,
            ReservedConcurrentExecutions=reserved_concurrent_executions,
        )

    def delete_function_concurrency(self, function_name: str) -> None:
        lambda_client = self._client('lambda')
        lambda_client.delete_function_concurrency(FunctionName=function_name)

    def _update_function_config(
        self,
        environment_variables: StrMap,
        runtime: OptStr,
        timeout: OptInt,
        memory_size: OptInt,
        role_arn: OptStr,
        subnet_ids: OptStrList,
        security_group_ids: OptStrList,
        function_name: str,
        layers: OptStrList,
        xray: Optional[bool],
    ) -> None:
        kwargs: Dict[str, Any] = {}
        if environment_variables is not None:
            kwargs['Environment'] = {'Variables': environment_variables}
        if runtime is not None:
            kwargs['Runtime'] = runtime
        if timeout is not None:
            kwargs['Timeout'] = timeout
        if memory_size is not None:
            kwargs['MemorySize'] = memory_size
        if role_arn is not None:
            kwargs['Role'] =