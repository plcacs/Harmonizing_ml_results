import json
import os
from dataclasses import asdict

from typing import cast
from typing import Dict, List, Tuple, Any, Set, Optional, Text, Union  # noqa

from chalice.config import Config  # noqa
from chalice import app
from chalice.constants import LAMBDA_TRUST_POLICY
from chalice.deploy import models
from chalice.utils import UI  # noqa

StrMapAny = Dict[str, Any]


class ChaliceBuildError(Exception):
    pass


class ApplicationGraphBuilder(object):
    def __init__(self) -> None:
        self._known_roles: Dict[str, models.IAMRole] = {}
        self._managed_layer: Optional[models.LambdaLayer] = None

    def build(self, config: Config, stage_name: str) -> models.Application:
        resources: List[models.Model] = []
        deployment = models.DeploymentPackage(models.Placeholder.BUILD_STAGE)
        for function in config.chalice_app.pure_lambda_functions:
            resource = self._create_lambda_model(
                config=config,
                deployment=deployment,
                name=function.name,
                handler_name=function.handler_string,
                stage_name=stage_name,
            )
            resources.append(resource)
        event_resources = self._create_lambda_event_resources(
            config, deployment, stage_name
        )
        resources.extend(event_resources)
        if config.chalice_app.routes:
            rest_api = self._create_rest_api_model(
                config, deployment, stage_name
            )
            resources.append(rest_api)
        if config.chalice_app.websocket_handlers:
            websocket_api = self._create_websocket_api_model(
                config, deployment, stage_name
            )
            resources.append(websocket_api)
        return models.Application(stage_name, resources)

    def _create_log_group(
        self, config: Config, resource_name: str, log_group_name: str
    ) -> models.LogGroup:
        return models.LogGroup(
            resource_name=resource_name,
            log_group_name=log_group_name,
            retention_in_days=config.log_retention_in_days,
        )

    def _create_custom_domain_name(
        self,
        api_type: models.APIType,
        domain_name_data: StrMapAny,
        endpoint_configuration: str,
        api_gateway_stage: str,
    ) -> models.DomainName:
        url_prefix = domain_name_data.get("url_prefix", '(none)')
        api_mapping_model = self._create_api_mapping_model(
            url_prefix, api_gateway_stage
        )
        domain_name = self._create_domain_name_model(
            api_type,
            domain_name_data,
            endpoint_configuration,
            api_mapping_model,
        )
        return domain_name

    def _create_api_mapping_model(
        self, key: str, stage: str
    ) -> models.APIMapping:
        if key == '/':
            key = '(none)'
        return models.APIMapping(
            resource_name='api_mapping',
            mount_path=key,
            api_gateway_stage=stage,
        )

    def _create_lambda_event_resources(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        stage_name: str,
    ) -> List[models.Model]:
        resources: List[models.Model] = []
        for event_source in config.chalice_app.event_sources:
            if isinstance(event_source, app.S3EventConfig):
                resources.append(
                    self._create_bucket_notification(
                        config, deployment, event_source, stage_name
                    )
                )
            elif isinstance(event_source, app.SNSEventConfig):
                resources.append(
                    self._create_sns_subscription(
                        config,
                        deployment,
                        event_source,
                        stage_name,
                    )
                )
            elif isinstance(event_source, app.CloudWatchEventConfig):
                resources.append(
                    self._create_cwe_subscription(
                        config, deployment, event_source, stage_name
                    )
                )
            elif isinstance(event_source, app.ScheduledEventConfig):
                resources.append(
                    self._create_scheduled_model(
                        config, deployment, event_source, stage_name
                    )
                )
            elif isinstance(event_source, app.SQSEventConfig):
                resources.append(
                    self._create_sqs_subscription(
                        config,
                        deployment,
                        event_source,
                        stage_name,
                    )
                )
            elif isinstance(event_source, app.KinesisEventConfig):
                resources.append(
                    self._create_kinesis_subscription(
                        config,
                        deployment,
                        event_source,
                        stage_name,
                    )
                )
            elif isinstance(event_source, app.DynamoDBEventConfig):
                resources.append(
                    self._create_ddb_subscription(
                        config,
                        deployment,
                        event_source,
                        stage_name,
                    )
                )
        return resources

    def _create_rest_api_model(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        stage_name: str,
    ) -> models.RestAPI:
        # Need to mess with the function name for back-compat.
        lambda_function = self._create_lambda_model(
            config=config,
            deployment=deployment,
            name='api_handler',
            handler_name='app.app',
            stage_name=stage_name,
        )
        # For backwards compatibility with the old deployer, the
        # lambda function for the API handler doesn't have the
        # resource_name appended to its complete function_name,
        # it's just <app>-<stage>.
        function_name = '%s-%s' % (config.app_name, config.chalice_stage)
        lambda_function.function_name = function_name
        if config.minimum_compression_size is None:
            minimum_compression = ''
        else:
            minimum_compression = str(config.minimum_compression_size)
        authorizers = []
        for auth in config.chalice_app.builtin_auth_handlers:
            auth_lambda = self._create_lambda_model(
                config=config,
                deployment=deployment,
                name=auth.name,
                handler_name=auth.handler_string,
                stage_name=stage_name,
            )
            authorizers.append(auth_lambda)

        policy = None
        policy_path = config.api_gateway_policy_file
        if config.api_gateway_endpoint_type == 'PRIVATE' and not policy_path:
            policy = models.IAMPolicy(
                document=self._get_default_private_api_policy(config)
            )
        elif policy_path:
            policy = models.FileBasedIAMPolicy(
                document=models.Placeholder.BUILD_STAGE,
                filename=os.path.join(
                    config.project_dir, '.chalice', policy_path
                ),
            )

        vpce_ids = None
        if config.api_gateway_endpoint_vpce:
            vpce = config.api_gateway_endpoint_vpce
            vpce_ids = [vpce] if isinstance(vpce, str) else vpce

        custom_domain_name = None
        if config.api_gateway_custom_domain:
            custom_domain_name = self._create_custom_domain_name(
                models.APIType.HTTP,
                config.api_gateway_custom_domain,
                config.api_gateway_endpoint_type,
                config.api_gateway_stage,
            )

        return models.RestAPI(
            resource_name='rest_api',
            swagger_doc=models.Placeholder.BUILD_STAGE,
            endpoint_type=config.api_gateway_endpoint_type,
            minimum_compression=minimum_compression,
            api_gateway_stage=config.api_gateway_stage,
            lambda_function=lambda_function,
            authorizers=authorizers,
            policy=policy,
            domain_name=custom_domain_name,
            xray=config.xray_enabled,
            vpce_ids=vpce_ids,
        )

    def _get_default_private_api_policy(self, config: Config) -> StrMapAny:
        statements = [
            {
                "Effect": "Allow",
                "Principal": "*",
                "Action": "execute-api:Invoke",
                "Resource": "arn:*:execute-api:*:*:*",
                "Condition": {
                    "StringEquals": {
                        "aws:SourceVpce": config.api_gateway_endpoint_vpce
                    }
                },
            }
        ]
        return {"Version": "2012-10-17", "Statement": statements}

    def _create_websocket_api_model(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        stage_name: str,
    ) -> models.WebsocketAPI:
        connect_handler: Optional[models.LambdaFunction] = None
        message_handler: Optional[models.LambdaFunction] = None
        disconnect_handler: Optional[models.LambdaFunction] = None

        routes = {
            h.route_key_handled: h.handler_string
            for h in config.chalice_app.websocket_handlers.values()
        }
        if '$connect' in routes:
            connect_handler = self._create_lambda_model(
                config=config,
                deployment=deployment,
                name='websocket_connect',
                handler_name=routes['$connect'],
                stage_name=stage_name,
            )
            routes.pop('$connect')
        if '$disconnect' in routes:
            disconnect_handler = self._create_lambda_model(
                config=config,
                deployment=deployment,
                name='websocket_disconnect',
                handler_name=routes['$disconnect'],
                stage_name=stage_name,
            )
            routes.pop('$disconnect')
        if routes:
            # If there are left over routes they are message handlers.
            handler_string = list(routes.values())[0]
            message_handler = self._create_lambda_model(
                config=config,
                deployment=deployment,
                name='websocket_message',
                handler_name=handler_string,
                stage_name=stage_name,
            )

        custom_domain_name = None
        if config.websocket_api_custom_domain:
            custom_domain_name = self._create_custom_domain_name(
                models.APIType.WEBSOCKET,
                config.websocket_api_custom_domain,
                config.api_gateway_endpoint_type,
                config.api_gateway_stage,
            )

        return models.WebsocketAPI(
            name='%s-%s-websocket-api' % (config.app_name, stage_name),
            resource_name='websocket_api',
            connect_function=connect_handler,
            message_function=message_handler,
            disconnect_function=disconnect_handler,
            routes=[
                h.route_key_handled
                for h in config.chalice_app.websocket_handlers.values()
            ],
            api_gateway_stage=config.api_gateway_stage,
            domain_name=custom_domain_name,
        )

    def _create_cwe_subscription(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        event_source: app.CloudWatchEventConfig,
        stage_name: str,
    ) -> models.CloudWatchEvent:
        lambda_function = self._create_lambda_model(
            config=config,
            deployment=deployment,
            name=event_source.name,
            handler_name=event_source.handler_string,
            stage_name=stage_name,
        )

        resource_name = event_source.name + '-event'
        rule_name = '%s-%s-%s' % (
            config.app_name,
            config.chalice_stage,
            resource_name,
        )
        cwe = models.CloudWatchEvent(
            resource_name=resource_name,
            rule_name=rule_name,
            event_pattern=json.dumps(event_source.event_pattern),
            lambda_function=lambda_function,
        )
        return cwe

    def _create_scheduled_model(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        event_source: app.ScheduledEventConfig,
        stage_name: str,
    ) -> models.ScheduledEvent:
        lambda_function = self._create_lambda_model(
            config=config,
            deployment=deployment,
            name=event_source.name,
            handler_name=event_source.handler_string,
            stage_name=stage_name,
        )
        # Resource names must be unique across a chalice app.
        # However, in the original deployer code, the cloudwatch
        # event + lambda function was considered a single resource.
        # Now that they're treated as two separate resources we need
        # a unique name for the event_source that's not the lambda
        # function resource name.  We handle this by just appending
        # '-event' to the name.  Ideally this is handled in app.py
        # but we won't be able to do that until the old deployer
        # is gone.
        resource_name = event_source.name + '-event'
        if isinstance(
            event_source.schedule_expression, app.ScheduleExpression
        ):
            expression = event_source.schedule_expression.to_string()
        else:
            expression = event_source.schedule_expression
        rule_name = '%s-%s-%s' % (
            config.app_name,
            config.chalice_stage,
            resource_name,
        )
        scheduled_event = models.ScheduledEvent(
            resource_name=resource_name,
            rule_name=rule_name,
            rule_description=event_source.description,
            schedule_expression=expression,
            lambda_function=lambda_function,
        )
        return scheduled_event

    def _create_domain_name_model(
        self,
        protocol: models.APIType,
        data: StrMapAny,
        endpoint_type: str,
        api_mapping: models.APIMapping,
    ) -> models.DomainName:
        default_name = 'api_gateway_custom_domain'
        resource_name_map: Dict[str, str] = {
            'HTTP': default_name,
            'WEBSOCKET': 'websocket_api_custom_domain',
        }

        domain_name = models.DomainName(
            protocol=protocol,
            resource_name=resource_name_map.get(protocol.value, default_name),
            domain_name=data['domain_name'],
            tls_version=models.TLSVersion.create(data.get('tls_version', '')),
            certificate_arn=data['certificate_arn'],
            tags=data.get('tags'),
            api_mapping=api_mapping,
        )
        return domain_name

    def _create_lambda_model(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        name: str,
        handler_name: str,
        stage_name: str,
    ) -> models.LambdaFunction:
        new_config = config.scope(
            chalice_stage=config.chalice_stage, function_name=name
        )
        role = self._get_role_reference(new_config, stage_name, name)
        resource = self._build_lambda_function(
            new_config, name, handler_name, deployment, role
        )
        if new_config.log_retention_in_days:
            log_resource_name = '%s-log-group' % name
            log_group_name = '/aws/lambda/%s-%s-%s' % (
                new_config.app_name,
                stage_name,
                name,
            )
            resource.log_group = self._create_log_group(
                new_config, log_resource_name, log_group_name
            )
        return resource

    def _get_managed_lambda_layer(
        self, config: Config
    ) -> Optional[models.LambdaLayer]:
        if not config.automatic_layer:
            return None
        if self._managed_layer is None:
            self._managed_layer = models.LambdaLayer(
                resource_name='managed-layer',
                layer_name='%s-%s-%s'
                % (config.app_name, config.chalice_stage, 'managed-layer'),
                runtime=config.lambda_python_version,
                deployment_package=models.DeploymentPackage(
                    models.Placeholder.BUILD_STAGE
                ),
            )
        return self._managed_layer

    def _get_role_reference(
        self, config: Config, stage_name: str, function_name: str
    ) -> models.IAMRole:
        role = self._create_role_reference(config, stage_name, function_name)
        role_identifier = self._get_role_identifier(role)
        if role_identifier in self._known_roles:
            # If we've already create a models.IAMRole with the same
            # identifier, we'll use the existing object instead of
            # creating a new one.
            return self._known_roles[role_identifier]
        self._known_roles[role_identifier] = role
        return role

    def _get_role_identifier(self, role: models.IAMRole) -> str:
        if isinstance(role, models.PreCreatedIAMRole):
            return role.role_arn
        # We know that if it's not a PreCreatedIAMRole, it's
        # a managed role, so we're using cast() to make mypy happy.
        role = cast(models.ManagedIAMRole, role)
        return role.resource_name

    def _create_role_reference(
        self, config: Config, stage_name: str, function_name: str
    ) -> models.IAMRole:
        # First option, the user doesn't want us to manage
        # the role at all.
        if not config.manage_iam_role:
            # We've already validated the iam_role_arn is provided
            # if manage_iam_role is set to False.
            return models.PreCreatedIAMRole(
                role_arn=config.iam_role_arn,
            )
        policy = models.IAMPolicy(document=models.Placeholder.BUILD_STAGE)
        if not config.autogen_policy:
            resource_name = '%s_role' % function_name
            role_name = '%s-%s-%s' % (
                config.app_name,
                stage_name,
                function_name,
            )
            if config.iam_policy_file is not None:
                filename = os.path.join(
                    config.project_dir, '.chalice', config.iam_policy_file
                )
            else:
                filename = os.path.join(
                    config.project_dir,
                    '.chalice',
                    'policy-%s.json' % stage_name,
                )
            policy = models.FileBasedIAMPolicy(
                filename=filename, document=models.Placeholder.BUILD_STAGE
            )
        else:
            resource_name = 'default-role'
            role_name = '%s-%s' % (config.app_name, stage_name)
            policy = models.AutoGenIAMPolicy(
                document=models.Placeholder.BUILD_STAGE,
                traits=set([]),
            )
        return models.ManagedIAMRole(
            resource_name=resource_name,
            role_name=role_name,
            trust_policy=LAMBDA_TRUST_POLICY,
            policy=policy,
        )

    def _get_vpc_params(
        self, function_name: str, config: Config
    ) -> Tuple[List[str], List[str]]:
        security_group_ids = config.security_group_ids
        subnet_ids = config.subnet_ids
       