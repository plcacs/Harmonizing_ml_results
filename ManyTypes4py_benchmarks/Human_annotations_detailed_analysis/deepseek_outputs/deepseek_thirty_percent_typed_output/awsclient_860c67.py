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
    LAMBDA_CREATE_ATTEMPTS = 30
    DELAY_TIME = 5

    def __init__(
        self,
        session: botocore.session.Session,
        sleep: Callable[[float], None] = time.sleep,
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
        zip_contents: bytes,
        runtime: str,
        handler: str,
        environment_variables: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        xray: Optional[bool] = None,
        timeout: Optional[int] = None,
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
        tags: Optional[Dict[str, str]] = None,
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
