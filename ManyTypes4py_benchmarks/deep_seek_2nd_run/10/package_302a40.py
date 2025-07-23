import copy
import json
import os
import re
import six
from typing import Any, Optional, Dict, List, Set, Union, cast, TypeVar, Type, Callable
from typing import Tuple
from chalice.deploy.swagger import CFNSwaggerGenerator, TerraformSwaggerGenerator
from chalice.utils import OSUtils, UI, serialize_to_json, to_cfn_resource_name
from chalice.awsclient import TypedAWSClient
from chalice.config import Config
from chalice.deploy import models
from chalice.deploy.appgraph import ApplicationGraphBuilder, DependencyBuilder
from chalice.deploy.deployer import BuildStage
from chalice.deploy.deployer import create_build_stage

T = TypeVar('T')

def create_app_packager(
    config: Config,
    options: PackageOptions,
    package_format: str = 'cloudformation',
    template_format: str = 'json',
    merge_template: Optional[str] = None
) -> 'AppPackager':
    osutils = OSUtils()
    ui = UI()
    application_builder = ApplicationGraphBuilder()
    deps_builder = DependencyBuilder()
    post_processors: List[TemplatePostProcessor] = []
    generator: Optional[TemplateGenerator] = None
    template_serializer: TemplateSerializer = cast(TemplateSerializer, JSONTemplateSerializer())
    if package_format == 'cloudformation':
        build_stage = create_build_stage(osutils, ui, CFNSwaggerGenerator(), config)
        use_yaml_serializer = template_format == 'yaml'
        if merge_template is not None and YAMLTemplateSerializer.is_yaml_template(merge_template):
            use_yaml_serializer = True
        if use_yaml_serializer:
            template_serializer = YAMLTemplateSerializer()
        post_processors.extend([
            SAMCodeLocationPostProcessor(osutils=osutils),
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
        post_processors.append(TerraformCodeLocationPostProcessor(osutils=osutils))
    resource_builder = ResourceBuilder(application_builder, deps_builder, build_stage)
    return AppPackager(
        generator, resource_builder, CompositePostProcessor(post_processors),
        template_serializer, osutils
    )

class UnsupportedFeatureError(Exception):
    pass

class DuplicateResourceNameError(Exception):
    pass

class PackageOptions:
    def __init__(self, client: TypedAWSClient) -> None:
        self._client = client

    def service_principal(self, service: str) -> str:
        dns_suffix = self._client.endpoint_dns_suffix(service, self._client.region_name)
        return self._client.service_principal(
            service, self._client.region_name, dns_suffix
        )

class ResourceBuilder:
    def __init__(
        self,
        application_builder: ApplicationGraphBuilder,
        deps_builder: DependencyBuilder,
        build_stage: BuildStage
    ) -> None:
        self._application_builder = application_builder
        self._deps_builder = deps_builder
        self._build_stage = build_stage

    def construct_resources(self, config: Config, chalice_stage_name: str) -> List[models.Model]:
        application = self._application_builder.build(config, chalice_stage_name)
        resources = self._deps_builder.build_dependencies(application)
        self._build_stage.execute(config, resources)
        resources = self._deps_builder.build_dependencies(application)
        return resources

class TemplateGenerator:
    template_file: Optional[str] = None

    def __init__(self, config: Config, options: PackageOptions) -> None:
        self._config = config
        self._options = options

    def dispatch(self, resource: models.Model, template: Dict[str, Any]) -> None:
        name = '_generate_%s' % resource.__class__.__name__.lower()
        handler = getattr(self, name, self._default)
        handler(resource, template)

    def generate(self, resources: List[models.Model]) -> Dict[str, Any]:
        raise NotImplementedError()

    def _generate_filebasediampolicy(self, resource: models.Model, template: Dict[str, Any]) -> None:
        pass

    def _generate_autogeniampolicy(self, resource: models.Model, template: Dict[str, Any]) -> None:
        pass

    def _generate_deploymentpackage(self, resource: models.Model, template: Dict[str, Any]) -> None:
        pass

    def _generate_precreatediamrole(self, resource: models.Model, template: Dict[str, Any]) -> None:
        pass

    def _default(self, resource: models.Model, template: Dict[str, Any]) -> None:
        raise UnsupportedFeatureError(resource)

class SAMTemplateGenerator(TemplateGenerator):
    _BASE_TEMPLATE: Dict[str, Any] = {
        'AWSTemplateFormatVersion': '2010-09-09',
        'Transform': 'AWS::Serverless-2016-10-31',
        'Outputs': {},
        'Resources': {}
    }
    template_file = 'sam'

    def __init__(self, config: Config, options: PackageOptions) -> None:
        super(SAMTemplateGenerator, self).__init__(config, options)
        self._seen_names: Set[str] = set()
        self._chalice_layer = ''

    def generate(self, resources: List[models.Model]) -> Dict[str, Any]:
        template = copy.deepcopy(self._BASE_TEMPLATE)
        self._seen_names.clear()
        for resource in resources:
            self.dispatch(resource, template)
        return template

    def _generate_lambdalayer(self, resource: models.LambdaLayer, template: Dict[str, Any]) -> None:
        layer = to_cfn_resource_name(resource.resource_name)
        template['Resources'][layer] = {
            'Type': 'AWS::Serverless::LayerVersion',
            'Properties': {
                'CompatibleRuntimes': [resource.runtime],
                'ContentUri': resource.deployment_package.filename,
                'LayerName': resource.layer_name
            }
        }
        self._chalice_layer = layer

    def _generate_scheduledevent(self, resource: models.ScheduledEvent, template: Dict[str, Any]) -> None:
        function_cfn_name = to_cfn_resource_name(resource.lambda_function.resource_name)
        function_cfn = template['Resources'][function_cfn_name]
        event_cfn_name = self._register_cfn_resource_name(resource.resource_name)
        function_cfn['Properties']['Events'] = {
            event_cfn_name: {
                'Type': 'Schedule',
                'Properties': {'Schedule': resource.schedule_expression}
            }
        }

    def _generate_cloudwatchevent(self, resource: models.CloudWatchEvent, template: Dict[str, Any]) -> None:
        function_cfn_name = to_cfn_resource_name(resource.lambda_function.resource_name)
        function_cfn = template['Resources'][function_cfn_name]
        event_cfn_name = self._register_cfn_resource_name(resource.resource_name)
        function_cfn['Properties']['Events'] = {
            event_cfn_name: {
                'Type': 'CloudWatchEvent',
                'Properties': {'Pattern': json.loads(resource.event_pattern)}
            }
        }

    def _generate_lambdafunction(self, resource: models.LambdaFunction, template: Dict[str, Any]) -> None:
        resources = template['Resources']
        cfn_name = self._register_cfn_resource_name(resource.resource_name)
        lambdafunction_definition = {
            'Type': 'AWS::Serverless::Function',
            'Properties': {
                'Runtime': resource.runtime,
                'Handler': resource.handler,
                'CodeUri': resource.deployment_package.filename,
                'Tags': resource.tags,
                'Tracing': resource.xray and 'Active' or 'PassThrough',
                'Timeout': resource.timeout,
                'MemorySize': resource.memory_size
            }
        }
        if resource.environment_variables:
            environment_config = {
                'Environment': {'Variables': resource.environment_variables}
            }
            lambdafunction_definition['Properties'].update(environment_config)
        if resource.security_group_ids and resource.subnet_ids:
            vpc_config = {
                'VpcConfig': {
                    'SecurityGroupIds': resource.security_group_ids,
                    'SubnetIds': resource.subnet_ids
                }
            }
            lambdafunction_definition['Properties'].update(vpc_config)
        if resource.reserved_concurrency is not None:
            reserved_concurrency_config = {
                'ReservedConcurrentExecutions': resource.reserved_concurrency
            }
            lambdafunction_definition['Properties'].update(reserved_concurrency_config)
        layers = list(resource.layers) or []
        if self._chalice_layer:
            layers.insert(0, {'Ref': self._chalice_layer})
        if layers:
            layers_config = {'Layers': layers}
            lambdafunction_definition['Properties'].update(layers_config)
        if resource.log_group is not None:
            num_days = resource.log_group.retention_in_days
            log_name = self._register_cfn_resource_name(resource.log_group.resource_name)
            log_def = {
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

    def _generate_loggroup(self, resource: models.Model, template: Dict[str, Any]) -> None:
        pass

    def _generate_restapi(self, resource: models.RestAPI, template: Dict[str, Any]) -> None:
        resources = template['Resources']
        resources['RestAPI'] = {
            'Type': 'AWS::Serverless::Api',
            'Properties': {
                'EndpointConfiguration': resource.endpoint_type,
                'StageName': resource.api_gateway_stage,
                'DefinitionBody': resource.swagger_doc
            }
        }
        if resource.minimum_compression:
            properties = resources['RestAPI']['Properties']
            properties['MinimumCompressionSize'] = int(resource.minimum_compression)
        handler_cfn_name = to_cfn_resource_name(resource.lambda_function.resource_name)
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
                        'arn:${AWS::Partition}:execute-api:${AWS::Region}:${AWS::AccountId}:${RestAPIId}/*',
                        {'RestAPIId': {'Ref': 'RestAPI'}}
                    ]
                }
            }
        }
        for auth in resource.authorizers:
            auth_cfn_name = to_cfn_resource_name(auth.resource_name)
            resources[auth_cfn_name + 'InvokePermission'] = {
                'Type': 'AWS::Lambda::Permission',
                'Properties': {
                    'FunctionName': {'Fn::GetAtt': [auth_cfn_name, 'Arn']},
                    'Action': 'lambda:InvokeFunction',
                    'Principal': self._options.service_principal('apigateway'),
                    'SourceArn': {
                        'Fn::Sub': [
                            'arn:${AWS::Partition}:execute-api:${AWS::Region}:${AWS::AccountId}:${RestAPIId}/*',
                            {'RestAPIId': {'Ref': 'RestAPI'}}
                        ]
                    }
                }
            }
        self._add_domain_name(resource, template)
        self._inject_restapi_outputs(template)

    def _inject_restapi_outputs(self, template: Dict[str, Any]) -> None:
        stage_name = template['Resources']['RestAPI']['Properties']['StageName']
        outputs = template['Outputs']
        outputs['RestAPIId'] = {'Value': {'Ref': 'RestAPI'}}
        outputs['APIHandlerName'] = {'Value': {'Ref': 'APIHandler'}}
        outputs['APIHandlerArn'] = {'Value': {'Fn::GetAtt': ['APIHandler', 'Arn']}}
        outputs['EndpointURL'] = {
            'Value': {
                'Fn::Sub': 'https://${RestAPI}.execute-api.${AWS::Region}.${AWS::URLSuffix}/%s/' % stage_name
            }
        }

    def _add_websocket_lambda_integration(
        self,
        api_ref: Dict[str, Any],
        websocket_handler: str,
        resources: Dict[str, Any]
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
                        'arn:${AWS::Partition}:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/arn:${AWS::Partition}:lambda:${AWS::Region}:${AWS::AccountId}:function:${WebsocketHandler}/invocations',
                        {'WebsocketHandler': {'Ref': websocket_handler}}
                    ]
                }
            }
        }

    def _add_websocket_lambda_invoke_permission(
        self,
        api_ref: Dict[str, Any],
        websocket_handler: str,
        resources: Dict[str, Any]
    ) -> None:
        resources['%sInvokePermission' % websocket_handler] = {
            'Type': 'AWS::Lambda::Permission',
            'Properties': {
                'FunctionName': {'Ref': websocket_handler},
                'Action': 'lambda:InvokeFunction',
                'Principal': self._options.service_principal('apigateway'),
                'SourceArn': {
                    'Fn::Sub': [
                        'arn:${AWS::Partition}:execute-api:${AWS::Region}:${AWS::AccountId}:${WebsocketAPIId}/*',
                        {'WebsocketAPIId': api_ref}
                    ]
                }
            }
        }

    def _add_websocket_lambda_integrations(
        self,
        api_ref: Dict[str, Any],
        resources: Dict[str, Any]
    ) -> None:
        websocket_handlers = ['WebsocketConnect', 'WebsocketMessage', 'WebsocketDisconnect']
        for handler in websocket_handlers:
            if handler in resources:
                self._add_websocket_lambda_integration(api_ref, handler, resources)
                self._add_websocket_lambda_invoke_permission(api_ref, handler, resources)

    def _create_route_for_key(
        self,
        route_key: str,
        api_ref: Dict[str, Any]
    ) -> Dict[str, Any]:
        integration_ref = {
            '$connect': 'WebsocketConnectAPIIntegration',
            '$disconnect': 'WebsocketDisconnectAPIIntegration'
        }.get(route_key, 'WebsocketMessageAPIIntegration')
        return {
            'Type': 'AWS::ApiGatewayV2::Route',
            'Properties': {
                'ApiId': api_ref,
                'RouteKey': route_key,
                'Target': {
                    'Fn::Join': ['/', ['integrations', {'Ref': integration_ref}]]
                }
            }
        }

    def _generate_websocketapi(self, resource: models.WebsocketAPI, template: Dict[str, Any]) -> None:
        resources = template['Resources']
        api_ref = {'Ref': 'WebsocketAPI'}
        resources['WebsocketAPI'] = {
            'Type': 'AWS::ApiGatewayV2::Api',
            'Properties': {
                'Name': resource.name,
                'RouteSelectionExpression': '$request.body.action',
                'ProtocolType': 'WEBSOCKET'
            }
        }
        self._add_websocket_lambda_integrations(api_ref, resources)
        route_key_names = []
        for route in resource.routes:
            key_name = 'Websocket%sRoute' % route.replace('$', '').replace('default', 'message').capitalize()
            route_key_names.append(key_name)
            resources[key_name] = self._create_route_for_key(route, api_ref)
        resources['WebsocketAPIDeployment'] = {
            'Type': 'AWS::ApiGatewayV2::Deployment',
            'DependsOn': route_key_names,
            'Properties': {'ApiId': api_ref}
        }
        resources['WebsocketAPIStage'] = {
            'Type': 'AWS::ApiGatewayV2::Stage',
            'Properties': {
                'ApiId': api_ref,
                'DeploymentId': {'Ref': 'WebsocketAPIDeployment'},
                'StageName': resource.api_gateway_stage
            }
        }
        self._add_websocket_domain_name(resource, template)
        self._inject_websocketapi_outputs(template)

    def _inject_websocketapi_outputs(self, template: Dict[str, Any]) -> None:
        stage_name = template['Resources']['WebsocketAPIStage']['Properties']['StageName']
        outputs = template['Outputs']
        resources = template['Resources']
        outputs['WebsocketAPIId