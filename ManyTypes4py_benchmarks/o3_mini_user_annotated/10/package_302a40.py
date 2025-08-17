#!/usr/bin/env python3
# pylint: disable=too-many-lines

import copy
import json
import os
import re
import six
import yaml
from yaml.scanner import ScannerError
from yaml.nodes import Node, ScalarNode, SequenceNode, MappingNode
from typing import Any, Optional, Dict, List, Set, Union, cast

from chalice.deploy.swagger import CFNSwaggerGenerator, TerraformSwaggerGenerator
from chalice.utils import OSUtils, UI, serialize_to_json, to_cfn_resource_name
from chalice.awsclient import TypedAWSClient
from chalice.config import Config
from chalice.deploy import models
from chalice.deploy.appgraph import ApplicationGraphBuilder, DependencyBuilder
from chalice.deploy.deployer import BuildStage, create_build_stage


def create_app_packager(
    config: Config,
    options: "PackageOptions",
    package_format: str = 'cloudformation',
    template_format: str = 'json',
    merge_template: Optional[str] = None
) -> "AppPackager":
    osutils: OSUtils = OSUtils()
    ui: UI = UI()
    application_builder: ApplicationGraphBuilder = ApplicationGraphBuilder()
    deps_builder: DependencyBuilder = DependencyBuilder()
    post_processors: List[TemplatePostProcessor] = []
    generator: Optional[TemplateGenerator] = None

    template_serializer: TemplateSerializer = cast(TemplateSerializer, JSONTemplateSerializer())
    if package_format == 'cloudformation':
        build_stage: BuildStage = create_build_stage(osutils, ui, CFNSwaggerGenerator(), config)
        use_yaml_serializer: bool = template_format == 'yaml'
        if merge_template is not None and YAMLTemplateSerializer.is_yaml_template(merge_template):
            use_yaml_serializer = True
        if use_yaml_serializer:
            template_serializer = YAMLTemplateSerializer()
        post_processors.extend([
            SAMCodeLocationPostProcessor(osutils),
            TemplateMergePostProcessor(
                osutils=osutils,
                merger=TemplateDeepMerger(),
                template_serializer=template_serializer,
                merge_template=merge_template
            )
        ])
        generator = SAMTemplateGenerator(config, options)
    else:
        build_stage = create_build_stage(osutils, ui, TerraformSwaggerGenerator(), config)
        generator = TerraformGenerator(config, options)
        post_processors.append(TerraformCodeLocationPostProcessor(osutils))
    resource_builder: ResourceBuilder = ResourceBuilder(application_builder, deps_builder, build_stage)
    return AppPackager(generator, resource_builder, CompositePostProcessor(post_processors),
                       template_serializer, osutils)


class UnsupportedFeatureError(Exception):
    pass


class DuplicateResourceNameError(Exception):
    pass


class PackageOptions:
    def __init__(self, client: TypedAWSClient) -> None:
        self._client: TypedAWSClient = client

    def service_principal(self, service: str) -> str:
        dns_suffix: str = self._client.endpoint_dns_suffix(service, self._client.region_name)
        return self._client.service_principal(service, self._client.region_name, dns_suffix)


class ResourceBuilder:
    def __init__(
        self,
        application_builder: ApplicationGraphBuilder,
        deps_builder: DependencyBuilder,
        build_stage: BuildStage,
    ) -> None:
        self._application_builder: ApplicationGraphBuilder = application_builder
        self._deps_builder: DependencyBuilder = deps_builder
        self._build_stage: BuildStage = build_stage

    def construct_resources(self, config: Config, chalice_stage_name: str) -> List[models.Model]:
        application: Any = self._application_builder.build(config, chalice_stage_name)
        resources: List[models.Model] = self._deps_builder.build_dependencies(application)
        self._build_stage.execute(config, resources)
        resources = self._deps_builder.build_dependencies(application)
        return resources


class TemplateGenerator:
    template_file: Optional[str] = None

    def __init__(self, config: Config, options: PackageOptions) -> None:
        self._config: Config = config
        self._options: PackageOptions = options

    def dispatch(self, resource: models.Model, template: Dict[str, Any]) -> None:
        name: str = '_generate_%s' % resource.__class__.__name__.lower()
        handler = getattr(self, name, self._default)
        handler(resource, template)

    def generate(self, resources: List[models.Model]) -> Dict[str, Any]:
        raise NotImplementedError()

    def _generate_filebasediampolicy(self, resource: models.FileBasedIAMPolicy, template: Dict[str, Any]) -> None:
        pass

    def _generate_autogeniampolicy(self, resource: models.AutoGenIAMPolicy, template: Dict[str, Any]) -> None:
        pass

    def _generate_deploymentpackage(self, resource: models.DeploymentPackage, template: Dict[str, Any]) -> None:
        pass

    def _generate_precreatediamrole(self, resource: models.PreCreatedIAMRole, template: Dict[str, Any]) -> None:
        pass

    def _default(self, resource: models.Model, template: Dict[str, Any]) -> None:
        raise UnsupportedFeatureError(resource)


class SAMTemplateGenerator(TemplateGenerator):
    _BASE_TEMPLATE: Dict[str, Any] = {
        'AWSTemplateFormatVersion': '2010-09-09',
        'Transform': 'AWS::Serverless-2016-10-31',
        'Outputs': {},
        'Resources': {},
    }
    template_file = "sam"

    def __init__(self, config: Config, options: PackageOptions) -> None:
        super(SAMTemplateGenerator, self).__init__(config, options)
        self._seen_names: Set[str] = set()
        self._chalice_layer: str = ""

    def generate(self, resources: List[models.Model]) -> Dict[str, Any]:
        template: Dict[str, Any] = copy.deepcopy(self._BASE_TEMPLATE)
        self._seen_names.clear()
        for resource in resources:
            self.dispatch(resource, template)
        return template

    def _generate_lambdalayer(self, resource: models.LambdaLayer, template: Dict[str, Any]) -> None:
        layer: str = to_cfn_resource_name(resource.resource_name)
        template['Resources'][layer] = {
            "Type": "AWS::Serverless::LayerVersion",
            "Properties": {
                "CompatibleRuntimes": [resource.runtime],
                "ContentUri": resource.deployment_package.filename,
                "LayerName": resource.layer_name
            }
        }
        self._chalice_layer = layer

    def _generate_scheduledevent(self, resource: models.ScheduledEvent, template: Dict[str, Any]) -> None:
        function_cfn_name: str = to_cfn_resource_name(resource.lambda_function.resource_name)
        function_cfn: Dict[str, Any] = template['Resources'][function_cfn_name]
        event_cfn_name: str = self._register_cfn_resource_name(resource.resource_name)
        function_cfn['Properties']['Events'] = {
            event_cfn_name: {
                'Type': 'Schedule',
                'Properties': {
                    'Schedule': resource.schedule_expression,
                }
            }
        }

    def _generate_cloudwatchevent(self, resource: models.CloudWatchEvent, template: Dict[str, Any]) -> None:
        function_cfn_name: str = to_cfn_resource_name(resource.lambda_function.resource_name)
        function_cfn: Dict[str, Any] = template['Resources'][function_cfn_name]
        event_cfn_name: str = self._register_cfn_resource_name(resource.resource_name)
        function_cfn['Properties']['Events'] = {
            event_cfn_name: {
                'Type': 'CloudWatchEvent',
                'Properties': {
                    'Pattern': json.loads(resource.event_pattern)
                }
            }
        }

    def _generate_lambdafunction(self, resource: models.LambdaFunction, template: Dict[str, Any]) -> None:
        resources: Dict[str, Any] = template['Resources']
        cfn_name: str = self._register_cfn_resource_name(resource.resource_name)
        lambdafunction_definition: Dict[str, Any] = {
            'Type': 'AWS::Serverless::Function',
            'Properties': {
                'Runtime': resource.runtime,
                'Handler': resource.handler,
                'CodeUri': resource.deployment_package.filename,
                'Tags': resource.tags,
                'Tracing': 'Active' if resource.xray else 'PassThrough',
                'Timeout': resource.timeout,
                'MemorySize': resource.memory_size,
            }
        }
        if resource.environment_variables:
            environment_config: Dict[str, Dict[str, Dict[str, str]]] = {
                'Environment': {
                    'Variables': resource.environment_variables
                }
            }
            lambdafunction_definition['Properties'].update(environment_config)
        if resource.security_group_ids and resource.subnet_ids:
            vpc_config: Dict[str, Dict[str, List[str]]] = {
                'VpcConfig': {
                    'SecurityGroupIds': resource.security_group_ids,
                    'SubnetIds': resource.subnet_ids,
                }
            }
            lambdafunction_definition['Properties'].update(vpc_config)
        if resource.reserved_concurrency is not None:
            reserved_concurrency_config: Dict[str, Any] = {
                'ReservedConcurrentExecutions': resource.reserved_concurrency
            }
            lambdafunction_definition['Properties'].update(reserved_concurrency_config)
        layers: List[Any] = list(resource.layers) if resource.layers else []
        if self._chalice_layer:
            layers.insert(0, {'Ref': self._chalice_layer})
        if layers:
            layers_config: Dict[str, Any] = {
                'Layers': layers
            }
            lambdafunction_definition['Properties'].update(layers_config)
        if resource.log_group is not None:
            num_days: int = resource.log_group.retention_in_days
            log_name: str = self._register_cfn_resource_name(resource.log_group.resource_name)
            log_def: Dict[str, Any] = {
                'Type': 'AWS::Logs::LogGroup',
                'Properties': {
                    'LogGroupName': {'Fn::Sub': '/aws/lambda/${%s}' % cfn_name},
                    'RetentionInDays': num_days
                }
            }
            resources[log_name] = log_def
        resources[cfn_name] = lambdafunction_definition
        self._add_iam_role(resource, resources[cfn_name])

    def _add_iam_role(self, resource: models.LambdaFunction, cfn_resource: Dict[str, Any]) -> None:
        role = resource.role
        if isinstance(role, models.ManagedIAMRole):
            cfn_resource['Properties']['Role'] = {
                'Fn::GetAtt': [to_cfn_resource_name(role.resource_name), 'Arn']
            }
        else:
            role = cast(models.PreCreatedIAMRole, role)
            cfn_resource['Properties']['Role'] = role.role_arn

    def _generate_loggroup(self, resource: models.LogGroup, template: Dict[str, Any]) -> None:
        # Handled in LambdaFunction generation
        pass

    def _generate_restapi(self, resource: models.RestAPI, template: Dict[str, Any]) -> None:
        resources: Dict[str, Any] = template['Resources']
        resources['RestAPI'] = {
            'Type': 'AWS::Serverless::Api',
            'Properties': {
                'EndpointConfiguration': resource.endpoint_type,
                'StageName': resource.api_gateway_stage,
                'DefinitionBody': resource.swagger_doc,
            }
        }
        if resource.minimum_compression:
            properties = resources['RestAPI']['Properties']
            properties['MinimumCompressionSize'] = int(resource.minimum_compression)
        handler_cfn_name: str = to_cfn_resource_name(resource.lambda_function.resource_name)
        api_handler = template['Resources'].pop(handler_cfn_name)
        template['Resources']['APIHandler'] = api_handler
        resources['APIHandlerInvokePermission'] = {
            'Type': 'AWS::Lambda::Permission',
            'Properties': {
                'FunctionName': {'Ref': 'APIHandler'},
                'Action': 'lambda:InvokeFunction',
                'Principal': self._options.service_principal('apigateway'),
                'SourceArn': {
                    'Fn::Sub': [
                        ('arn:${AWS::Partition}:execute-api:${AWS::Region}:'
                         '${AWS::AccountId}:${RestAPIId}/*'),
                        {'RestAPIId': {'Ref': 'RestAPI'}}
                    ]
                },
            }
        }
        for auth in resource.authorizers:
            auth_cfn_name: str = to_cfn_resource_name(auth.resource_name)
            resources[auth_cfn_name + 'InvokePermission'] = {
                'Type': 'AWS::Lambda::Permission',
                'Properties': {
                    'FunctionName': {'Fn::GetAtt': [auth_cfn_name, 'Arn']},
                    'Action': 'lambda:InvokeFunction',
                    'Principal': self._options.service_principal('apigateway'),
                    'SourceArn': {
                        'Fn::Sub': [
                            ('arn:${AWS::Partition}:execute-api:${AWS::Region}:'
                             '${AWS::AccountId}:${RestAPIId}/*'),
                            {'RestAPIId': {'Ref': 'RestAPI'}}
                        ]
                    },
                }
            }
        self._add_domain_name(resource, template)
        self._inject_restapi_outputs(template)

    def _inject_restapi_outputs(self, template: Dict[str, Any]) -> None:
        stage_name = template['Resources']['RestAPI']['Properties']['StageName']
        outputs: Dict[str, Any] = template['Outputs']
        outputs['RestAPIId'] = {'Value': {'Ref': 'RestAPI'}}
        outputs['APIHandlerName'] = {'Value': {'Ref': 'APIHandler'}}
        outputs['APIHandlerArn'] = {'Value': {'Fn::GetAtt': ['APIHandler', 'Arn']}}
        outputs['EndpointURL'] = {
            'Value': {
                'Fn::Sub': (
                    'https://${RestAPI}.execute-api.${AWS::Region}'
                    '.${AWS::URLSuffix}/%s/'
                ) % stage_name
            }
        }

    def _add_websocket_lambda_integration(
        self, api_ref: Dict[str, str], websocket_handler: str, resources: Dict[str, Any]
    ) -> None:
        resources['%sAPIIntegration' % websocket_handler] = {
            'Type': 'AWS::ApiGatewayV2::Integration',
            'Properties': {
                'ApiId': api_ref,
                'ConnectionType': 'INTERNET',
                'ContentHandlingStrategy': 'CONVERT_TO_TEXT',
                'IntegrationType': 'AWS_PROXY',
                'IntegrationUri': {
                    'Fn::Sub': [
                        ('arn:${AWS::Partition}:apigateway:${AWS::Region}:'
                         'lambda:path/2015-03-31/functions/arn:${AWS::Partition}:'
                         'lambda:${AWS::Region}:${AWS::AccountId}:function:${WebsocketHandler}/invocations'),
                        {'WebsocketHandler': {'Ref': websocket_handler}}
                    ]
                }
            }
        }

    def _add_websocket_lambda_invoke_permission(
        self, api_ref: Dict[str, str], websocket_handler: str, resources: Dict[str, Any]
    ) -> None:
        resources['%sInvokePermission' % websocket_handler] = {
            'Type': 'AWS::Lambda::Permission',
            'Properties': {
                'FunctionName': {'Ref': websocket_handler},
                'Action': 'lambda:InvokeFunction',
                'Principal': self._options.service_principal('apigateway'),
                'SourceArn': {
                    'Fn::Sub': [
                        ('arn:${AWS::Partition}:execute-api:${AWS::Region}:'
                         '${AWS::AccountId}:${WebsocketAPIId}/*'),
                        {'WebsocketAPIId': api_ref},
                    ]
                },
            }
        }

    def _add_websocket_lambda_integrations(self, api_ref: Dict[str, str], resources: Dict[str, Any]) -> None:
        websocket_handlers: List[str] = ['WebsocketConnect', 'WebsocketMessage', 'WebsocketDisconnect']
        for handler in websocket_handlers:
            if handler in resources:
                self._add_websocket_lambda_integration(api_ref, handler, resources)
                self._add_websocket_lambda_invoke_permission(api_ref, handler, resources)

    def _create_route_for_key(self, route_key: str, api_ref: Dict[str, str]) -> Dict[str, Any]:
        integration_ref: str = {
            '$connect': 'WebsocketConnectAPIIntegration',
            '$disconnect': 'WebsocketDisconnectAPIIntegration',
        }.get(route_key, 'WebsocketMessageAPIIntegration')
        return {
            'Type': 'AWS::ApiGatewayV2::Route',
            'Properties': {
                'ApiId': api_ref,
                'RouteKey': route_key,
                'Target': {'Fn::Join': ['/', ['integrations', {'Ref': integration_ref}]]},
            },
        }

    def _generate_websocketapi(self, resource: models.WebsocketAPI, template: Dict[str, Any]) -> None:
        resources: Dict[str, Any] = template['Resources']
        api_ref: Dict[str, str] = {'Ref': 'WebsocketAPI'}
        resources['WebsocketAPI'] = {
            'Type': 'AWS::ApiGatewayV2::Api',
            'Properties': {
                'Name': resource.name,
                'RouteSelectionExpression': '$request.body.action',
                'ProtocolType': 'WEBSOCKET',
            }
        }
        self._add_websocket_lambda_integrations(api_ref, resources)
        route_key_names: List[str] = []
        for route in resource.routes:
            key_name: str = 'Websocket%sRoute' % route.replace('$', '').replace('default', 'message').capitalize()
            route_key_names.append(key_name)
            resources[key_name] = self._create_route_for_key(route, api_ref)
        resources['WebsocketAPIDeployment'] = {
            'Type': 'AWS::ApiGatewayV2::Deployment',
            'DependsOn': route_key_names,
            'Properties': {'ApiId': api_ref},
        }
        resources['WebsocketAPIStage'] = {
            'Type': 'AWS::ApiGatewayV2::Stage',
            'Properties': {
                'ApiId': api_ref,
                'DeploymentId': {'Ref': 'WebsocketAPIDeployment'},
                'StageName': resource.api_gateway_stage,
            }
        }
        self._add_websocket_domain_name(resource, template)
        self._inject_websocketapi_outputs(template)

    def _inject_websocketapi_outputs(self, template: Dict[str, Any]) -> None:
        stage_name: str = template['Resources']['WebsocketAPIStage']['Properties']['StageName']
        outputs: Dict[str, Any] = template['Outputs']
        resources: Dict[str, Any] = template['Resources']
        outputs['WebsocketAPIId'] = {'Value': {'Ref': 'WebsocketAPI'}}
        if 'WebsocketConnect' in resources:
            outputs['WebsocketConnectHandlerArn'] = {'Value': {'Fn::GetAtt': ['WebsocketConnect', 'Arn']}}
            outputs['WebsocketConnectHandlerName'] = {'Value': {'Ref': 'WebsocketConnect'}}
        if 'WebsocketMessage' in resources:
            outputs['WebsocketMessageHandlerArn'] = {'Value': {'Fn::GetAtt': ['WebsocketMessage', 'Arn']}}
            outputs['WebsocketMessageHandlerName'] = {'Value': {'Ref': 'WebsocketMessage'}}
        if 'WebsocketDisconnect' in resources:
            outputs['WebsocketDisconnectHandlerArn'] = {'Value': {'Fn::GetAtt': ['WebsocketDisconnect', 'Arn']}}
            outputs['WebsocketDisconnectHandlerName'] = {'Value': {'Ref': 'WebsocketDisconnect'}}
        outputs['WebsocketConnectEndpointURL'] = {
            'Value': {
                'Fn::Sub': (
                    'wss://${WebsocketAPI}.execute-api.${AWS::Region}'
                    '.${AWS::URLSuffix}/%s/'
                ) % stage_name
            }
        }

    def _generate_managediamrole(self, resource: models.ManagedIAMRole, template: Dict[str, Any]) -> None:
        role_cfn_name: str = self._register_cfn_resource_name(resource.resource_name)
        resource.trust_policy['Statement'][0]['Principal']['Service'] = self._options.service_principal('lambda')
        template['Resources'][role_cfn_name] = {
            'Type': 'AWS::IAM::Role',
            'Properties': {
                'AssumeRolePolicyDocument': resource.trust_policy,
                'Policies': [{'PolicyDocument': resource.policy.document, 'PolicyName': role_cfn_name + 'Policy'}],
            }
        }

    def _generate_s3bucketnotification(self, resource: models.S3BucketNotification, template: Dict[str, Any]) -> None:
        message: str = (
            "Unable to package chalice apps that @app.on_s3_event decorator. "
            "CloudFormation does not support modifying the event notifications of existing buckets. "
            "You can deploy this app using `chalice deploy`."
        )
        raise NotImplementedError(message)

    def _generate_snslambdasubscription(self, resource: models.SNSLambdaSubscription, template: Dict[str, Any]) -> None:
        function_cfn_name: str = to_cfn_resource_name(resource.lambda_function.resource_name)
        function_cfn: Dict[str, Any] = template['Resources'][function_cfn_name]
        sns_cfn_name: str = self._register_cfn_resource_name(resource.resource_name)
        if re.match(r"^arn:aws[a-z\-]*:sns:", resource.topic):
            topic_arn: Union[str, Dict[str, str]] = resource.topic
        else:
            topic_arn = {
                'Fn::Sub': ('arn:${AWS::Partition}:sns:${AWS::Region}:${AWS::AccountId}:%s' % resource.topic)
            }
        function_cfn['Properties']['Events'] = {
            sns_cfn_name: {
                'Type': 'SNS',
                'Properties': {
                    'Topic': topic_arn,
                }
            }
        }

    def _generate_sqseventsource(self, resource: models.SQSEventSource, template: Dict[str, Any]) -> None:
        function_cfn_name: str = to_cfn_resource_name(resource.lambda_function.resource_name)
        function_cfn: Dict[str, Any] = template['Resources'][function_cfn_name]
        sqs_cfn_name: str = self._register_cfn_resource_name(resource.resource_name)
        if isinstance(resource.queue, models.QueueARN):
            queue: Union[str, Dict[str, Any]] = resource.queue.arn
        else:
            queue = {'Fn::Sub': ('arn:${AWS::Partition}:sqs:${AWS::Region}:${AWS::AccountId}:%s' % resource.queue)}
        properties: Dict[str, Any] = {
            'Queue': queue,
            'BatchSize': resource.batch_size,
            'MaximumBatchingWindowInSeconds': resource.maximum_batching_window_in_seconds
        }
        if resource.maximum_concurrency:
            properties["ScalingConfig"] = {"MaximumConcurrency": resource.maximum_concurrency}
        function_cfn['Properties']['Events'] = {
            sqs_cfn_name: {
                'Type': 'SQS',
                'Properties': properties
            }
        }

    def _generate_kinesiseventsource(self, resource: models.KinesisEventSource, template: Dict[str, Any]) -> None:
        function_cfn_name: str = to_cfn_resource_name(resource.lambda_function.resource_name)
        function_cfn: Dict[str, Any] = template['Resources'][function_cfn_name]
        kinesis_cfn_name: str = self._register_cfn_resource_name(resource.resource_name)
        properties: Dict[str, Any] = {
            'Stream': {'Fn::Sub': ('arn:${AWS::Partition}:kinesis:${AWS::Region}:${AWS::AccountId}:stream/%s' % resource.stream)},
            'BatchSize': resource.batch_size,
            'StartingPosition': resource.starting_position,
            'MaximumBatchingWindowInSeconds': resource.maximum_batching_window_in_seconds,
        }
        function_cfn['Properties']['Events'] = {
            kinesis_cfn_name: {
                'Type': 'Kinesis',
                'Properties': properties
            }
        }

    def _generate_dynamodbeventsource(self, resource: models.DynamoDBEventSource, template: Dict[str, Any]) -> None:
        function_cfn_name: str = to_cfn_resource_name(resource.lambda_function.resource_name)
        function_cfn: Dict[str, Any] = template['Resources'][function_cfn_name]
        ddb_cfn_name: str = self._register_cfn_resource_name(resource.resource_name)
        properties: Dict[str, Any] = {
            'Stream': resource.stream_arn,
            'BatchSize': resource.batch_size,
            'StartingPosition': resource.starting_position,
            'MaximumBatchingWindowInSeconds': resource.maximum_batching_window_in_seconds,
        }
        function_cfn['Properties']['Events'] = {
            ddb_cfn_name: {
                'Type': 'DynamoDB',
                'Properties': properties
            }
        }

    def _generate_apimapping(self, resource: models.APIMapping, template: Dict[str, Any]) -> None:
        pass

    def _generate_domainname(self, resource: models.DomainName, template: Dict[str, Any]) -> None:
        pass

    def _add_domain_name(self, resource: models.RestAPI, template: Dict[str, Any]) -> None:
        if resource.domain_name is None:
            return
        domain_name = resource.domain_name
        endpoint_type: str = resource.endpoint_type
        cfn_name: str = to_cfn_resource_name(domain_name.resource_name)
        properties: Dict[str, Any] = {
            'DomainName': domain_name.domain_name,
            'EndpointConfiguration': {'Types': [endpoint_type]},
        }
        if endpoint_type == 'EDGE':
            properties['CertificateArn'] = domain_name.certificate_arn
        else:
            properties['RegionalCertificateArn'] = domain_name.certificate_arn
        if domain_name.tls_version is not None:
            properties['SecurityPolicy'] = domain_name.tls_version.value
        if domain_name.tags:
            properties['Tags'] = [{'Key': key, 'Value': value} for key, value in sorted(domain_name.tags.items())]
        template['Resources'][cfn_name] = {
            'Type': 'AWS::ApiGateway::DomainName',
            'Properties': properties
        }
        template['Resources'][cfn_name + 'Mapping'] = {
            'Type': 'AWS::ApiGateway::BasePathMapping',
            'Properties': {
                'DomainName': {'Ref': 'ApiGatewayCustomDomain'},
                'RestApiId': {'Ref': 'RestAPI'},
                'BasePath': domain_name.api_mapping.mount_path,
                'Stage': resource.api_gateway_stage,
            }
        }

    def _add_websocket_domain_name(self, resource: models.WebsocketAPI, template: Dict[str, Any]) -> None:
        if resource.domain_name is None:
            return
        domain_name = resource.domain_name
        cfn_name: str = to_cfn_resource_name(domain_name.resource_name)
        properties: Dict[str, Any] = {
            'DomainName': domain_name.domain_name,
            'DomainNameConfigurations': [{'CertificateArn': domain_name.certificate_arn, 'EndpointType': 'REGIONAL'}],
        }
        if domain_name.tags:
            properties['Tags'] = domain_name.tags
        template['Resources'][cfn_name] = {
            'Type': 'AWS::ApiGatewayV2::DomainName',
            'Properties': properties,
        }
        template['Resources'][cfn_name + 'Mapping'] = {
            'Type': 'AWS::ApiGatewayV2::ApiMapping',
            'Properties': {
                'DomainName': {'Ref': cfn_name},
                'ApiId': {'Ref': 'WebsocketAPI'},
                'ApiMappingKey': domain_name.api_mapping.mount_path,
                'Stage': {'Ref': 'WebsocketAPIStage'},
            }
        }

    def _register_cfn_resource_name(self, name: str) -> str:
        cfn_name: str = to_cfn_resource_name(name)
        if cfn_name in self._seen_names:
            raise DuplicateResourceNameError(
                'A duplicate resource name was generated for the SAM template: %s' % cfn_name
            )
        self._seen_names.add(cfn_name)
        return cfn_name


class TerraformGenerator(TemplateGenerator):
    template_file = "chalice.tf"

    def __init__(self, config: Config, options: PackageOptions) -> None:
        super(TerraformGenerator, self).__init__(config, options)
        self._chalice_layer: str = ""

    def generate(self, resources: List[models.Model]) -> Dict[str, Any]:
        template: Dict[str, Any] = {
            'resource': {},
            'locals': {},
            'terraform': {
                'required_version': '>= 0.12.26, < 1.4.0',
                'required_providers': {
                    'aws': {'version': '>= 2, < 5'},
                    'null': {'version': '>= 2, < 4'}
                }
            },
            'data': {
                'aws_caller_identity': {'chalice': {}},
                'aws_partition': {'chalice': {}},
                'aws_region': {'chalice': {}},
                'null_data_source': {
                    'chalice': {
                        'inputs': {
                            'app': self._config.app_name,
                            'stage': self._config.chalice_stage
                        }
                    }
                }
            }
        }
        for resource in resources:
            self.dispatch(resource, template)
        return template

    def _fref(self, lambda_function: models.ManagedModel, attr: str = 'arn') -> str:
        return '${aws_lambda_function.%s.%s}' % (lambda_function.resource_name, attr)

    def _arnref(self, arn_template: str, **kw: str) -> str:
        d: Dict[str, str] = {
            'partition': '${data.aws_partition.chalice.partition}',
            'region': '${data.aws_region.chalice.name}',
            'account_id': '${data.aws_caller_identity.chalice.account_id}'
        }
        d.update(kw)
        return arn_template % d

    def _generate_managediamrole(self, resource: models.ManagedIAMRole, template: Dict[str, Any]) -> None:
        resource.trust_policy['Statement'][0]['Principal']['Service'] = self._options.service_principal('lambda')
        template['resource'].setdefault('aws_iam_role', {})[resource.resource_name] = {
            'name': resource.role_name,
            'assume_role_policy': json.dumps(resource.trust_policy)
        }
        template['resource'].setdefault('aws_iam_role_policy', {})[resource.resource_name] = {
            'name': resource.resource_name + 'Policy',
            'policy': json.dumps(resource.policy.document),
            'role': '${aws_iam_role.%s.id}' % resource.resource_name,
        }

    def _add_websocket_lambda_integration(
        self, websocket_api_id: str, websocket_handler: str, template: Dict[str, Any]
    ) -> None:
        websocket_handler_function_name: str = "${aws_lambda_function.%s.function_name}" % websocket_handler
        resource_definition: Dict[str, Any] = {
            'api_id': websocket_api_id,
            'connection_type': 'INTERNET',
            'content_handling_strategy': 'CONVERT_TO_TEXT',
            'integration_type': 'AWS_PROXY',
            'integration_uri': self._arnref(
                "arn:%(partition)s:apigateway:%(region)s:lambda:path/2015-03-31/functions/arn:%(partition)s:lambda:%(region)s:%(account_id)s:function:%(websocket_handler_function_name)s/invocations",
                websocket_handler_function_name=websocket_handler_function_name
            )
        }
        template['resource'].setdefault('aws_apigatewayv2_integration', {})[
            '%s_api_integration' % websocket_handler
        ] = resource_definition

    def _add_websocket_lambda_invoke_permission(
        self, websocket_api_id: str, websocket_handler: str, template: Dict[str, Any]
    ) -> None:
        websocket_handler_function_name: str = "${aws_lambda_function.%s.function_name}" % websocket_handler
        resource_definition: Dict[str, Any] = {
            "function_name": websocket_handler_function_name,
            "action": "lambda:InvokeFunction",
            "principal": self._options.service_principal('apigateway'),
            "source_arn": self._arnref(
                "arn:%(partition)s:execute-api:%(region)s:%(account_id)s:%(websocket_api_id)s/*",
                websocket_api_id=websocket_api_id
            )
        }
        template['resource'].setdefault('aws_lambda_permission', {})[
            '%s_invoke_permission' % websocket_handler
        ] = resource_definition

    def _add_websockets_route(self, websocket_api_id: str, route_key: str, template: Dict[str, Any]) -> str:
        integration_target: str = {
            '$connect': 'integrations/${aws_apigatewayv2_integration.websocket_connect_api_integration.id}',
            '$disconnect': 'integrations/${aws_apigatewayv2_integration.websocket_disconnect_api_integration.id}',
        }.get(route_key,
              'integrations/${aws_apigatewayv2_integration.websocket_message_api_integration.id}')
        route_resource_name: str = {
            '$connect': 'websocket_connect_route',
            '$disconnect': 'websocket_disconnect_route',
            '$default': 'websocket_message_route',
        }.get(route_key, 'message')
        template['resource'].setdefault('aws_apigatewayv2_route', {})[route_resource_name] = {
            "api_id": websocket_api_id,
            "route_key": route_key,
            "target": integration_target
        }
        return route_resource_name

    def _add_websocket_domain_name(
        self, websocket_api_id: str, resource: models.WebsocketAPI, template: Dict[str, Any]
    ) -> None:
        if resource.domain_name is None:
            return
        domain_name = resource.domain_name
        ws_domain_name_definition: Dict[str, Any] = {
            "domain_name": domain_name.domain_name,
            "domain_name_configuration": {
                'certificate_arn': domain_name.certificate_arn,
                'endpoint_type': 'REGIONAL',
            },
        }
        if domain_name.tags:
            ws_domain_name_definition['tags'] = domain_name.tags
        template['resource'].setdefault('aws_apigatewayv2_domain_name', {})[domain_name.resource_name] = ws_domain_name_definition
        template['resource'].setdefault('aws_apigatewayv2_api_mapping', {})[domain_name.resource_name + '_mapping'] = {
            "api_id": websocket_api_id,
            "domain_name": "${aws_apigatewayv2_domain_name.%s.id}" % domain_name.resource_name,
            "stage": "${aws_apigatewayv2_stage.websocket_api_stage.id}",
        }

    def _inject_websocketapi_outputs(self, websocket_api_id: str, template: Dict[str, Any]) -> None:
        aws_lambda_functions: Dict[str, Any] = template['resource']['aws_lambda_function']
        stage_name: str = template['resource']['aws_apigatewayv2_stage']['websocket_api_stage']['name']
        output: Dict[str, Any] = template.setdefault('output', {})
        output['WebsocketAPIId'] = {"value": websocket_api_id}
        if 'websocket_connect' in aws_lambda_functions:
            output['WebsocketConnectHandlerArn'] = {"value": "${aws_lambda_function.websocket_connect.arn}"}
            output['WebsocketConnectHandlerName'] = {"value": "${aws_lambda_function.websocket_connect.function_name}"}
        if 'websocket_message' in aws_lambda_functions:
            output['WebsocketMessageHandlerArn'] = {"value": "${aws_lambda_function.websocket_message.arn}"}
            output['WebsocketMessageHandlerName'] = {"value": "${aws_lambda_function.websocket_message.function_name}"}
        if 'websocket_disconnect' in aws_lambda_functions:
            output['WebsocketDisconnectHandlerArn'] = {"value": "${aws_lambda_function.websocket_disconnect.arn}"}
            output['WebsocketDisconnectHandlerName'] = {"value": "${aws_lambda_function.websocket_disconnect.function_name}"}
        output['WebsocketConnectEndpointURL'] = {
            "value": (
                'wss://%(websocket_api_id)s.execute-api.${data.aws_region.chalice.name}.amazonaws.com/%(stage_name)s/'
            ) % {"stage_name": stage_name, "websocket_api_id": websocket_api_id}
        }

    def _generate_websocketapi(self, resource: models.WebsocketAPI, template: Dict[str, Any]) -> None:
        ws_definition: Dict[str, Any] = {
            'name': resource.name,
            'route_selection_expression': '$request.body.action',
            'protocol_type': 'WEBSOCKET',
        }
        template['resource'].setdefault('aws_apigatewayv2_api', {})[resource.resource_name] = ws_definition
        websocket_api_id: str = "${aws_apigatewayv2_api.%s.id}" % resource.resource_name
        websocket_handlers: List[str] = ['websocket_connect', 'websocket_message', 'websocket_disconnect']
        for handler in websocket_handlers:
            if handler in template['resource'].get('aws_lambda_function', {}):
                self._add_websocket_lambda_integration(websocket_api_id, handler, template)
                self._add_websocket_lambda_invoke_permission(websocket_api_id, handler, template)
        route_resource_names: List[str] = []
        for route_key in resource.routes:
            route_resource_name: str = self._add_websockets_route(websocket_api_id, route_key, template)
            route_resource_names.append(route_resource_name)
        template['resource'].setdefault('aws_apigatewayv2_deployment', {})['websocket_api_deployment'] = {
            "api_id": websocket_api_id,
            "depends_on": ["aws_apigatewayv2_route.%s" % name for name in route_resource_names]
        }
        template['resource'].setdefault('aws_apigatewayv2_stage', {})['websocket_api_stage'] = {
            "api_id": websocket_api_id,
            "deployment_id": "${aws_apigatewayv2_deployment.websocket_api_deployment.id}",
            "name": resource.api_gateway_stage
        }
        self._add_websocket_domain_name(websocket_api_id, resource, template)
        self._inject_websocketapi_outputs(websocket_api_id, template)

    def _generate_s3bucketnotification(self, resource: models.S3BucketNotification, template: Dict[str, Any]) -> None:
        bnotify: Dict[str, Any] = {
            'events': resource.events,
            'lambda_function_arn': self._fref(resource.lambda_function)
        }
        if resource.prefix:
            bnotify['filter_prefix'] = resource.prefix
        if resource.suffix:
            bnotify['filter_suffix'] = resource.suffix
        if '{aws_s3_bucket.' in resource.bucket:
            bucket_name: str = resource.bucket.split('.')[1]
        else:
            bucket_name = resource.bucket
        template['resource'].setdefault('aws_s3_bucket_notification', {}).setdefault(
            bucket_name + '_notify', {'bucket': resource.bucket}
        ).setdefault('lambda_function', []).append(bnotify)
        template['resource'].setdefault('aws_lambda_permission', {})[resource.resource_name] = {
            'statement_id': resource.resource_name,
            'action': 'lambda:InvokeFunction',
            'function_name': self._fref(resource.lambda_function),
            'principal': self._options.service_principal('s3'),
            'source_account': '${data.aws_caller_identity.chalice.account_id}',
            'source_arn': ('arn:${data.aws_partition.chalice.partition}:s3:::%s' % resource.bucket)
        }

    def _generate_sqseventsource(self, resource: models.SQSEventSource, template: Dict[str, Any]) -> None:
        if isinstance(resource.queue, models.QueueARN):
            event_source_arn: str = resource.queue.arn
        else:
            event_source_arn = self._arnref("arn:%(partition)s:sqs:%(region)s:%(account_id)s:%(queue)s", queue=resource.queue)
        aws_lambda_event_source_mapping: Dict[str, Any] = {
            'event_source_arn': event_source_arn,
            'batch_size': resource.batch_size,
            'maximum_batching_window_in_seconds': resource.maximum_batching_window_in_seconds,
            'function_name': self._fref(resource.lambda_function),
        }
        if resource.maximum_concurrency:
            aws_lambda_event_source_mapping["scaling_config"] = {"maximum_concurrency": resource.maximum_concurrency}
        template['resource'].setdefault('aws_lambda_event_source_mapping', {})[resource.resource_name] = aws_lambda_event_source_mapping

    def _generate_kinesiseventsource(self, resource: models.KinesisEventSource, template: Dict[str, Any]) -> None:
        template['resource'].setdefault('aws_lambda_event_source_mapping', {})[resource.resource_name] = {
            'event_source_arn': self._arnref("arn:%(partition)s:kinesis:%(region)s:%(account_id)s:stream/%(stream)s", stream=resource.stream),
            'batch_size': resource.batch_size,
            'starting_position': resource.starting_position,
            'maximum_batching_window_in_seconds': resource.maximum_batching_window_in_seconds,
            'function_name': self._fref(resource.lambda_function)
        }

    def _generate_dynamodbeventsource(self, resource: models.DynamoDBEventSource, template: Dict[str, Any]) -> None:
        template['resource'].setdefault('aws_lambda_event_source_mapping', {})[resource.resource_name] = {
            'event_source_arn': resource.stream_arn,
            'batch_size': resource.batch_size,
            'starting_position': resource.starting_position,
            'maximum_batching_window_in_seconds': resource.maximum_batching_window_in_seconds,
            'function_name': self._fref(resource.lambda_function),
        }

    def _generate_snslambdasubscription(self, resource: models.SNSLambdaSubscription, template: Dict[str, Any]) -> None:
        if resource.topic.startswith('arn:aws'):
            topic_arn: str = resource.topic
        else:
            topic_arn = self._arnref('arn:%(partition)s:sns:%(region)s:%(account_id)s:%(topic)s', topic=resource.topic)
        template['resource'].setdefault('aws_sns_topic_subscription', {})[resource.resource_name] = {
            'topic_arn': topic_arn,
            'protocol': 'lambda',
            'endpoint': self._fref(resource.lambda_function)
        }
        template['resource'].setdefault('aws_lambda_permission', {})[resource.resource_name] = {
            'function_name': self._fref(resource.lambda_function),
            'action': 'lambda:InvokeFunction',
            'principal': self._options.service_principal('sns'),
            'source_arn': topic_arn
        }

    def _generate_cloudwatchevent(self, resource: models.CloudWatchEvent, template: Dict[str, Any]) -> None:
        template['resource'].setdefault('aws_cloudwatch_event_rule', {})[resource.resource_name] = {
            'name': resource.resource_name,
            'event_pattern': resource.event_pattern
        }
        self._cwe_helper(resource, template)

    def _generate_scheduledevent(self, resource: models.ScheduledEvent, template: Dict[str, Any]) -> None:
        template['resource'].setdefault('aws_cloudwatch_event_rule', {})[resource.resource_name] = {
            'name': resource.resource_name,
            'schedule_expression': resource.schedule_expression,
            'description': resource.rule_description,
        }
        self._cwe_helper(resource, template)

    def _cwe_helper(self, resource: models.CloudWatchEventBase, template: Dict[str, Any]) -> None:
        template['resource'].setdefault('aws_cloudwatch_event_target', {})[resource.resource_name] = {
            'rule': '${aws_cloudwatch_event_rule.%s.name}' % resource.resource_name,
            'target_id': resource.resource_name,
            'arn': self._fref(resource.lambda_function)
        }
        template['resource'].setdefault('aws_lambda_permission', {})[resource.resource_name] = {
            'function_name': self._fref(resource.lambda_function),
            'action': 'lambda:InvokeFunction',
            'principal': self._options.service_principal('events'),
            'source_arn': "${aws_cloudwatch_event_rule.%s.arn}" % resource.resource_name
        }

    def _generate_lambdalayer(self, resource: models.LambdaLayer, template: Dict[str, Any]) -> None:
        template['resource'].setdefault("aws_lambda_layer_version", {})[resource.resource_name] = {
            'layer_name': resource.layer_name,
            'compatible_runtimes': [resource.runtime],
            'filename': resource.deployment_package.filename,
        }
        self._chalice_layer = resource.resource_name

    def _generate_lambdafunction(self, resource: models.LambdaFunction, template: Dict[str, Any]) -> None:
        func_definition: Dict[str, Any] = {
            'function_name': resource.function_name,
            'runtime': resource.runtime,
            'handler': resource.handler,
            'memory_size': resource.memory_size,
            'tags': resource.tags,
            'timeout': resource.timeout,
            'source_code_hash': '${filebase64sha256("%s")}' % resource.deployment_package.filename,
            'filename': resource.deployment_package.filename
        }
        if resource.security_group_ids and resource.subnet_ids:
            func_definition['vpc_config'] = {
                'subnet_ids': resource.subnet_ids,
                'security_group_ids': resource.security_group_ids
            }
        if resource.reserved_concurrency is not None:
            func_definition['reserved_concurrent_executions'] = resource.reserved_concurrency
        if resource.environment_variables:
            func_definition['environment'] = {'variables': resource.environment_variables}
        if resource.xray:
            func_definition['tracing_config'] = {'mode': 'Active'}
        if self._chalice_layer:
            func_definition['layers'] = ['${aws_lambda_layer_version.%s.arn}' % self._chalice_layer]
        if resource.layers:
            func_definition.setdefault('layers', []).extend(list(resource.layers))
        if isinstance(resource.role, models.ManagedIAMRole):
            func_definition['role'] = '${aws_iam_role.%s.arn}' % resource.role.resource_name
        else:
            role = cast(models.PreCreatedIAMRole, resource.role)
            func_definition['role'] = role.role_arn
        if resource.log_group is not None:
            log_group = resource.log_group
            num_days: int = log_group.retention_in_days
            template['resource'].setdefault('aws_cloudwatch_log_group', {})[log_group.resource_name] = {
                'name': log_group.resource_name,
                'retention_in_days': num_days,
            }
        template['resource'].setdefault('aws_lambda_function', {})[resource.resource_name] = func_definition

    def _generate_log_group(self, resource: models.LogGroup, remplate: Dict[str, Any]) -> None:
        pass

    def _generate_restapi(self, resource: models.RestAPI, template: Dict[str, Any]) -> None:
        swagger_doc: Dict[str, Any] = cast(Dict, resource.swagger_doc)
        template['locals']['chalice_api_swagger'] = json.dumps(swagger_doc)
        template['resource'].setdefault('aws_api_gateway_rest_api', {})[resource.resource_name] = {
            'body': '${local.chalice_api_swagger}',
            'name': swagger_doc['info']['title'],
            'binary_media_types': swagger_doc['x-amazon-apigateway-binary-media-types'],
            'endpoint_configuration': {'types': [resource.endpoint_type]}
        }
        if 'x-amazon-apigateway-policy' in swagger_doc:
            template['resource']['aws_api_gateway_rest_api'][resource.resource_name]['policy'] = json.dumps(
                swagger_doc['x-amazon-apigateway-policy'])
        if resource.minimum_compression.isdigit():
            template['resource']['aws_api_gateway_rest_api'][resource.resource_name]['minimum_compression_size'] = int(resource.minimum_compression)
        template['resource'].setdefault('aws_api_gateway_deployment', {})[resource.resource_name] = {
            'stage_name': resource.api_gateway_stage,
            'stage_description': "${md5(local.chalice_api_swagger)}",
            'rest_api_id': '${aws_api_gateway_rest_api.%s.id}' % resource.resource_name,
            'lifecycle': {'create_before_destroy': True}
        }
        template['resource'].setdefault('aws_lambda_permission', {})[resource.resource_name + '_invoke'] = {
            'function_name': self._fref(resource.lambda_function),
            'action': 'lambda:InvokeFunction',
            'principal': self._options.service_principal('apigateway'),
            'source_arn': "${aws_api_gateway_rest_api.%s.execution_arn}/*" % resource.resource_name
        }
        template.setdefault('output', {})['EndpointURL'] = {
            'value': '${aws_api_gateway_deployment.%s.invoke_url}' % resource.resource_name
        }
        template.setdefault('output', {})['RestAPIId'] = {
            'value': '${aws_api_gateway_rest_api.%s.id}' % resource.resource_name
        }
        for auth in resource.authorizers:
            template['resource']['aws_lambda_permission'][auth.resource_name + '_invoke'] = {
                'function_name': self._fref(auth),
                'action': 'lambda:InvokeFunction',
                'principal': self._options.service_principal('apigateway'),
                'source_arn': "${aws_api_gateway_rest_api.%s.execution_arn}/*" % resource.resource_name
            }
        self._add_domain_name(resource, template)

    def _add_domain_name(self, resource: models.RestAPI, template: Dict[str, Any]) -> None:
        if resource.domain_name is None:
            return
        domain_name = resource.domain_name
        endpoint_type: str = resource.endpoint_type
        properties: Dict[str, Any] = {
            'domain_name': domain_name.domain_name,
            'endpoint_configuration': {'types': [endpoint_type]},
        }
        if endpoint_type == 'EDGE':
            properties['certificate_arn'] = domain_name.certificate_arn
        else:
            properties['regional_certificate_arn'] = domain_name.certificate_arn
        if domain_name.tls_version is not None:
            properties['security_policy'] = domain_name.tls_version.value
        if domain_name.tags:
            properties['tags'] = domain_name.tags
        template['resource']['aws_api_gateway_domain_name'] = {
            domain_name.resource_name: properties
        }
        template['resource']['aws_api_gateway_base_path_mapping'] = {
            domain_name.resource_name + '_mapping': {
                'stage_name': resource.api_gateway_stage,
                'domain_name': domain_name.domain_name,
                'api_id': '${aws_api_gateway_rest_api.%s.id}' % resource.resource_name
            }
        }
        self._add_domain_name_outputs(domain_name.resource_name, endpoint_type, template)

    def _add_domain_name_outputs(self, domain_resource_name: str, endpoint_type: str, template: Dict[str, Any]) -> None:
        base: str = 'aws_api_gateway_domain_name.%s' % domain_resource_name
        if endpoint_type == 'EDGE':
            alias_domain_name: str = '${%s.cloudfront_domain_name}' % base
            hosted_zone_id: str = '${%s.cloudfront_zone_id}' % base
        else:
            alias_domain_name = '${%s.regional_domain_name}' % base
            hosted_zone_id = '${%s.regional_zone_id}' % base
        template.setdefault('output', {})['AliasDomainName'] = {'value': alias_domain_name}
        template.setdefault('output', {})['HostedZoneId'] = {'value': hosted_zone_id}

    def _generate_apimapping(self, resource: models.APIMapping, template: Dict[str, Any]) -> None:
        pass

    def _generate_domainname(self, resource: models.DomainName, template: Dict[str, Any]) -> None:
        pass


class AppPackager:
    def __init__(
        self,
        templater: TemplateGenerator,
        resource_builder: ResourceBuilder,
        post_processor: "TemplatePostProcessor",
        template_serializer: "TemplateSerializer",
        osutils: OSUtils,
    ) -> None:
        self._templater: TemplateGenerator = templater
        self._resource_builder: ResourceBuilder = resource_builder
        self._template_post_processor: TemplatePostProcessor = post_processor
        self._template_serializer: TemplateSerializer = template_serializer
        self._osutils: OSUtils = osutils

    def _to_json(self, doc: Any) -> str:
        return serialize_to_json(doc)

    def _to_yaml(self, doc: Any) -> str:
        return yaml.dump(doc, allow_unicode=True)

    def package_app(self, config: Config, outdir: str, chalice_stage_name: str) -> None:
        resources: List[models.Model] = self._resource_builder.construct_resources(config, chalice_stage_name)
        template: Dict[str, Any] = self._templater.generate(resources)
        if not self._osutils.directory_exists(outdir):
            self._osutils.makedirs(outdir)
        self._template_post_processor.process(template, config, outdir, chalice_stage_name)
        contents: str = self._template_serializer.serialize_template(template)
        extension: str = self._template_serializer.file_extension
        filename: str = os.path.join(outdir, self._templater.template_file) + '.' + extension
        self._osutils.set_file_contents(filename=filename, contents=contents, binary=False)


class TemplatePostProcessor:
    def __init__(self, osutils: OSUtils) -> None:
        self._osutils: OSUtils = osutils

    def process(self, template: Dict[str, Any], config: Config, outdir: str, chalice_stage_name: str) -> None:
        raise NotImplementedError()


class SAMCodeLocationPostProcessor(TemplatePostProcessor):
    def process(self, template: Dict[str, Any], config: Config, outdir: str, chalice_stage_name: str) -> None:
        self._fixup_deployment_package(template, outdir)

    def _fixup_deployment_package(self, template: Dict[str, Any], outdir: str) -> None:
        copied: bool = False
        for resource in template['Resources'].values():
            if resource['Type'] == 'AWS::Serverless::Function':
                original_location: str = resource['Properties']['CodeUri']
                new_location: str = os.path.join(outdir, 'deployment.zip')
                if not copied:
                    self._osutils.copy(original_location, new_location)
                    copied = True
                resource['Properties']['CodeUri'] = './deployment.zip'
            elif resource['Type'] == 'AWS::Serverless::LayerVersion':
                original_location = resource['Properties']['ContentUri']
                new_location = os.path.join(outdir, 'layer-deployment.zip')
                self._osutils.copy(original_location, new_location)
                resource['Properties']['ContentUri'] = './layer-deployment.zip'


class TerraformCodeLocationPostProcessor(TemplatePostProcessor):
    def process(self, template: Dict[str, Any], config: Config, outdir: str, chalice_stage_name: str) -> None:
        copied: bool = False
        resources: Dict[str, Any] = template['resource']
        for r in resources.get('aws_lambda_function', {}).values():
            if not copied:
                asset_path: str = os.path.join(outdir, 'deployment.zip')
                self._osutils.copy(r['filename'], asset_path)
                copied = True
            r['filename'] = "${path.module}/deployment.zip"
            r['source_code_hash'] = '${filebase64sha256("${path.module}/deployment.zip")}'
        copied = False
        for r in resources.get('aws_lambda_layer_version', {}).values():
            if not copied:
                asset_path = os.path.join(outdir, 'layer-deployment.zip')
                self._osutils.copy(r['filename'], asset_path)
                copied = True
            r['filename'] = "${path.module}/layer-deployment.zip"
            r['source_code_hash'] = '${filebase64sha256("${path.module}/layer-deployment.zip")}'


class TemplateMergePostProcessor(TemplatePostProcessor):
    def __init__(
        self,
        osutils: OSUtils,
        merger: "TemplateMerger",
        template_serializer: "TemplateSerializer",
        merge_template: Optional[str] = None,
    ) -> None:
        super(TemplateMergePostProcessor, self).__init__(osutils)
        self._merger: TemplateMerger = merger
        self._template_serializer: TemplateSerializer = template_serializer
        self._merge_template: Optional[str] = merge_template

    def process(self, template: Dict[str, Any], config: Config, outdir: str, chalice_stage_name: str) -> None:
        if self._merge_template is None:
            return
        loaded_template: Dict[str, Any] = self._load_template_to_merge()
        merged: Dict[str, Any] = self._merger.merge(loaded_template, template)
        template.clear()
        template.update(merged)

    def _load_template_to_merge(self) -> Dict[str, Any]:
        template_name: str = cast(str, self._merge_template)
        filepath: str = os.path.abspath(template_name)
        if not self._osutils.file_exists(filepath):
            raise RuntimeError('Cannot find template file: %s' % filepath)
        template_data: str = self._osutils.get_file_contents(filepath, binary=False)
        loaded_template: Dict[str, Any] = self._template_serializer.load_template(template_data, filepath)
        return loaded_template


class CompositePostProcessor(TemplatePostProcessor):
    def __init__(self, processors: List[TemplatePostProcessor]) -> None:
        self._processors: List[TemplatePostProcessor] = processors

    def process(self, template: Dict[str, Any], config: Config, outdir: str, chalice_stage_name: str) -> None:
        for processor in self._processors:
            processor.process(template, config, outdir, chalice_stage_name)


class TemplateMerger:
    def merge(self, file_template: Dict[str, Any], chalice_template: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError('merge')


class TemplateDeepMerger(TemplateMerger):
    def merge(self, file_template: Dict[str, Any], chalice_template: Dict[str, Any]) -> Dict[str, Any]:
        return self._merge(file_template, chalice_template)

    def _merge(self, file_template: Any, chalice_template: Any) -> Any:
        if isinstance(file_template, dict) and isinstance(chalice_template, dict):
            return self._merge_dict(file_template, chalice_template)
        return file_template

    def _merge_dict(self, file_template: Dict[str, Any], chalice_template: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = chalice_template.copy()
        for key, value in file_template.items():
            merged[key] = self._merge(value, chalice_template.get(key))
        return merged


class TemplateSerializer:
    file_extension: str = ''

    def load_template(self, file_contents: str, filename: str = '') -> Dict[str, Any]:
        raise NotImplementedError("load_template")

    def serialize_template(self, contents: Dict[str, Any]) -> str:
        raise NotImplementedError("serialize_template")


class JSONTemplateSerializer(TemplateSerializer):
    file_extension: str = 'json'

    def serialize_template(self, contents: Dict[str, Any]) -> str:
        return serialize_to_json(contents)

    def load_template(self, file_contents: str, filename: str = '') -> Dict[str, Any]:
        try:
            return json.loads(file_contents)
        except ValueError:
            raise RuntimeError('Expected %s to be valid JSON template.' % filename)


class YAMLTemplateSerializer(TemplateSerializer):
    file_extension: str = 'yaml'

    @classmethod
    def is_yaml_template(cls, template_name: str) -> bool:
        file_extension: str = os.path.splitext(template_name)[1].lower()
        return file_extension in [".yaml", ".yml"]

    def serialize_template(self, contents: Dict[str, Any]) -> str:
        return yaml.safe_dump(contents, allow_unicode=True)

    def load_template(self, file_contents: str, filename: str = '') -> Dict[str, Any]:
        yaml.SafeLoader.add_multi_constructor('!', self._custom_sam_instrinsics)
        try:
            return yaml.load(file_contents, Loader=yaml.SafeLoader)
        except ScannerError:
            raise RuntimeError('Expected %s to be valid YAML template.' % filename)

    def _custom_sam_instrinsics(self, loader: yaml.SafeLoader, tag_prefix: str, node: Node) -> Dict[str, Any]:
        tag: str = node.tag[1:]
        if tag not in ['Ref', 'Condition']:
            tag = 'Fn::%s' % tag
        value: Any = self._get_value(loader, node)
        return {tag: value}

    def _get_value(self, loader: yaml.SafeLoader, node: Node) -> Any:
        if node.tag[1:] == 'GetAtt' and isinstance(node.value, six.string_types):
            return node.value.split('.', 1)
        elif isinstance(node, ScalarNode):
            return loader.construct_scalar(node)
        elif isinstance(node, SequenceNode):
            return loader.construct_sequence(node)
        elif isinstance(node, MappingNode):
            return loader.construct_mapping(node)
        raise ValueError("Unknown YAML node: %s" % node)