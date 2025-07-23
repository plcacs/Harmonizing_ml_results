import json
import os
from dataclasses import asdict
from typing import cast, Dict, List, Tuple, Any, Set, Optional, Text, Union
from chalice.config import Config
from chalice import app
from chalice.constants import LAMBDA_TRUST_POLICY
from chalice.deploy import models
from chalice.utils import UI

StrMapAny = Dict[str, Any]


class ChaliceBuildError(Exception):
    pass


class ApplicationGraphBuilder(object):

    def __init__(self) -> None:
        self._known_roles: Dict[str, models.ManagedIAMRole] = {}
        self._managed_layer: Optional[models.LambdaLayer] = None

    def build(self, config: Config, stage_name: str) -> models.Application:
        resources: List[models.Resource] = []
        deployment = models.DeploymentPackage(models.Placeholder.BUILD_STAGE)
        for function in config.chalice_app.pure_lambda_functions:
            resource = self._create_lambda_model(
                config=config,
                deployment=deployment,
                name=function.name,
                handler_name=function.handler_string,
                stage_name=stage_name
            )
            resources.append(resource)
        event_resources = self._create_lambda_event_resources(config, deployment, stage_name)
        resources.extend(event_resources)
        if config.chalice_app.routes:
            rest_api = self._create_rest_api_model(config, deployment, stage_name)
            resources.append(rest_api)
        if config.chalice_app.websocket_handlers:
            websocket_api = self._create_websocket_api_model(config, deployment, stage_name)
            resources.append(websocket_api)
        return models.Application(stage_name, resources)

    def _create_log_group(self, config: Config, resource_name: str, log_group_name: str) -> models.LogGroup:
        return models.LogGroup(
            resource_name=resource_name,
            log_group_name=log_group_name,
            retention_in_days=config.log_retention_in_days
        )

    def _create_custom_domain_name(
        self,
        api_type: models.APIType,
        domain_name_data: Dict[str, Any],
        endpoint_configuration: str,
        api_gateway_stage: str
    ) -> models.DomainName:
        url_prefix: str = domain_name_data.get('url_prefix', '(none)')
        api_mapping_model: models.APIMapping = self._create_api_mapping_model(url_prefix, api_gateway_stage)
        domain_name: models.DomainName = self._create_domain_name_model(
            api_type,
            domain_name_data,
            endpoint_configuration,
            api_mapping_model
        )
        return domain_name

    def _create_api_mapping_model(self, key: str, stage: str) -> models.APIMapping:
        if key == '/':
            key = '(none)'
        return models.APIMapping(
            resource_name='api_mapping',
            mount_path=key,
            api_gateway_stage=stage
        )

    def _create_lambda_event_resources(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        stage_name: str
    ) -> List[models.Resource]:
        resources: List[models.Resource] = []
        for event_source in config.chalice_app.event_sources:
            if isinstance(event_source, app.S3EventConfig):
                resources.append(self._create_bucket_notification(config, deployment, event_source, stage_name))
            elif isinstance(event_source, app.SNSEventConfig):
                resources.append(self._create_sns_subscription(config, deployment, event_source, stage_name))
            elif isinstance(event_source, app.CloudWatchEventConfig):
                resources.append(self._create_cwe_subscription(config, deployment, event_source, stage_name))
            elif isinstance(event_source, app.ScheduledEventConfig):
                resources.append(self._create_scheduled_model(config, deployment, event_source, stage_name))
            elif isinstance(event_source, app.SQSEventConfig):
                resources.append(self._create_sqs_subscription(config, deployment, event_source, stage_name))
            elif isinstance(event_source, app.KinesisEventConfig):
                resources.append(self._create_kinesis_subscription(config, deployment, event_source, stage_name))
            elif isinstance(event_source, app.DynamoDBEventConfig):
                resources.append(self._create_ddb_subscription(config, deployment, event_source, stage_name))
        return resources

    def _create_rest_api_model(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        stage_name: str
    ) -> models.RestAPI:
        lambda_function: models.LambdaFunction = self._create_lambda_model(
            config=config,
            deployment=deployment,
            name='api_handler',
            handler_name='app.app',
            stage_name=stage_name
        )
        function_name: str = f'{config.app_name}-{config.chalice_stage}'
        lambda_function.function_name = function_name
        if config.minimum_compression_size is None:
            minimum_compression: str = ''
        else:
            minimum_compression = str(config.minimum_compression_size)
        authorizers: List[models.LambdaFunction] = []
        for auth in config.chalice_app.builtin_auth_handlers:
            auth_lambda: models.LambdaFunction = self._create_lambda_model(
                config=config,
                deployment=deployment,
                name=auth.name,
                handler_name=auth.handler_string,
                stage_name=stage_name
            )
            authorizers.append(auth_lambda)
        policy: Optional[models.IAMPolicy] = None
        policy_path: Optional[str] = config.api_gateway_policy_file
        if config.api_gateway_endpoint_type == 'PRIVATE' and not policy_path:
            policy = models.IAMPolicy(
                document=self._get_default_private_api_policy(config)
            )
        elif policy_path:
            policy = models.FileBasedIAMPolicy(
                document=models.Placeholder.BUILD_STAGE,
                filename=os.path.join(config.project_dir, '.chalice', policy_path)
            )
        vpce_ids: Optional[List[str]] = None
        if config.api_gateway_endpoint_vpce:
            vpce: Union[str, List[str]] = config.api_gateway_endpoint_vpce
            vpce_ids = [vpce] if isinstance(vpce, str) else vpce
        custom_domain_name: Optional[models.DomainName] = None
        if config.api_gateway_custom_domain:
            custom_domain_name = self._create_custom_domain_name(
                models.APIType.HTTP,
                config.api_gateway_custom_domain,
                config.api_gateway_endpoint_type,
                config.api_gateway_stage
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
            vpce_ids=vpce_ids
        )

    def _get_default_private_api_policy(self, config: Config) -> Dict[str, Any]:
        statements: List[Dict[str, Any]] = [{
            'Effect': 'Allow',
            'Principal': '*',
            'Action': 'execute-api:Invoke',
            'Resource': 'arn:*:execute-api:*:*:*',
            'Condition': {
                'StringEquals': {'aws:SourceVpce': config.api_gateway_endpoint_vpce}
            }
        }]
        return {'Version': '2012-10-17', 'Statement': statements}

    def _create_websocket_api_model(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        stage_name: str
    ) -> models.WebsocketAPI:
        connect_handler: Optional[models.LambdaFunction] = None
        message_handler: Optional[models.LambdaFunction] = None
        disconnect_handler: Optional[models.LambdaFunction] = None
        routes: Dict[str, str] = {
            h.route_key_handled: h.handler_string
            for h in config.chalice_app.websocket_handlers.values()
        }
        if '$connect' in routes:
            connect_handler = self._create_lambda_model(
                config=config,
                deployment=deployment,
                name='websocket_connect',
                handler_name=routes['$connect'],
                stage_name=stage_name
            )
            routes.pop('$connect')
        if '$disconnect' in routes:
            disconnect_handler = self._create_lambda_model(
                config=config,
                deployment=deployment,
                name='websocket_disconnect',
                handler_name=routes['$disconnect'],
                stage_name=stage_name
            )
            routes.pop('$disconnect')
        if routes:
            handler_string: str = list(routes.values())[0]
            message_handler = self._create_lambda_model(
                config=config,
                deployment=deployment,
                name='websocket_message',
                handler_name=handler_string,
                stage_name=stage_name
            )
        custom_domain_name: Optional[models.DomainName] = None
        if config.websocket_api_custom_domain:
            custom_domain_name = self._create_custom_domain_name(
                models.APIType.WEBSOCKET,
                config.websocket_api_custom_domain,
                config.api_gateway_endpoint_type,
                config.api_gateway_stage
            )
        return models.WebsocketAPI(
            name=f'{config.app_name}-{stage_name}-websocket-api',
            resource_name='websocket_api',
            connect_function=connect_handler,
            message_function=message_handler,
            disconnect_function=disconnect_handler,
            routes=[h.route_key_handled for h in config.chalice_app.websocket_handlers.values()],
            api_gateway_stage=config.api_gateway_stage,
            domain_name=custom_domain_name
        )

    def _create_cwe_subscription(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        event_source: app.CloudWatchEventConfig,
        stage_name: str
    ) -> models.CloudWatchEvent:
        lambda_function: models.LambdaFunction = self._create_lambda_model(
            config=config,
            deployment=deployment,
            name=event_source.name,
            handler_name=event_source.handler_string,
            stage_name=stage_name
        )
        resource_name: str = f'{event_source.name}-event'
        rule_name: str = f'{config.app_name}-{config.chalice_stage}-{resource_name}'
        cwe: models.CloudWatchEvent = models.CloudWatchEvent(
            resource_name=resource_name,
            rule_name=rule_name,
            event_pattern=json.dumps(event_source.event_pattern),
            lambda_function=lambda_function
        )
        return cwe

    def _create_scheduled_model(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        event_source: app.ScheduledEventConfig,
        stage_name: str
    ) -> models.ScheduledEvent:
        lambda_function: models.LambdaFunction = self._create_lambda_model(
            config=config,
            deployment=deployment,
            name=event_source.name,
            handler_name=event_source.handler_string,
            stage_name=stage_name
        )
        resource_name: str = f'{event_source.name}-event'
        if isinstance(event_source.schedule_expression, app.ScheduleExpression):
            expression: str = event_source.schedule_expression.to_string()
        else:
            expression = event_source.schedule_expression
        rule_name: str = f'{config.app_name}-{config.chalice_stage}-{resource_name}'
        scheduled_event: models.ScheduledEvent = models.ScheduledEvent(
            resource_name=resource_name,
            rule_name=rule_name,
            rule_description=event_source.description,
            schedule_expression=expression,
            lambda_function=lambda_function
        )
        return scheduled_event

    def _create_domain_name_model(
        self,
        protocol: models.APIType,
        data: Dict[str, Any],
        endpoint_type: str,
        api_mapping: models.APIMapping
    ) -> models.DomainName:
        default_name: str = 'api_gateway_custom_domain'
        resource_name_map: Dict[str, str] = {
            'HTTP': default_name,
            'WEBSOCKET': 'websocket_api_custom_domain'
        }
        domain_name: models.DomainName = models.DomainName(
            protocol=protocol,
            resource_name=resource_name_map.get(protocol.value, default_name),
            domain_name=data['domain_name'],
            tls_version=models.TLSVersion.create(data.get('tls_version', '')),
            certificate_arn=data['certificate_arn'],
            tags=data.get('tags'),
            api_mapping=api_mapping
        )
        return domain_name

    def _create_lambda_model(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        name: str,
        handler_name: str,
        stage_name: str
    ) -> models.LambdaFunction:
        new_config: Config = config.scope(
            chalice_stage=config.chalice_stage,
            function_name=name
        )
        role: models.IAMRole = self._get_role_reference(new_config, stage_name, name)
        resource: models.LambdaFunction = self._build_lambda_function(
            new_config,
            name,
            handler_name,
            deployment,
            role
        )
        if new_config.log_retention_in_days:
            log_resource_name: str = f'{name}-log-group'
            log_group_name: str = f'/aws/lambda/{new_config.app_name}-{stage_name}-{name}'
            resource.log_group = self._create_log_group(new_config, log_resource_name, log_group_name)
        return resource

    def _get_managed_lambda_layer(self, config: Config) -> Optional[models.LambdaLayer]:
        if not config.automatic_layer:
            return None
        if self._managed_layer is None:
            self._managed_layer = models.LambdaLayer(
                resource_name='managed-layer',
                layer_name=f'{config.app_name}-{config.chalice_stage}-managed-layer',
                runtime=config.lambda_python_version,
                deployment_package=models.DeploymentPackage(models.Placeholder.BUILD_STAGE)
            )
        return self._managed_layer

    def _get_role_reference(
        self,
        config: Config,
        stage_name: str,
        function_name: str
    ) -> models.IAMRole:
        role: models.IAMRole = self._create_role_reference(config, stage_name, function_name)
        role_identifier: str = self._get_role_identifier(role)
        if role_identifier in self._known_roles:
            return self._known_roles[role_identifier]
        self._known_roles[role_identifier] = role
        return role

    def _get_role_identifier(self, role: models.IAMRole) -> str:
        if isinstance(role, models.PreCreatedIAMRole):
            return role.role_arn
        role = cast(models.ManagedIAMRole, role)
        return role.resource_name

    def _create_role_reference(
        self,
        config: Config,
        stage_name: str,
        function_name: str
    ) -> models.IAMRole:
        if not config.manage_iam_role:
            return models.PreCreatedIAMRole(role_arn=config.iam_role_arn)
        policy: models.IAMPolicy
        resource_name: str
        role_name: str
        if not config.autogen_policy:
            resource_name = f'{function_name}_role'
            role_name = f'{config.app_name}-{stage_name}-{function_name}'
            if config.iam_policy_file is not None:
                filename: str = os.path.join(config.project_dir, '.chalice', config.iam_policy_file)
            else:
                filename = os.path.join(config.project_dir, '.chalice', f'policy-{stage_name}.json')
            policy = models.FileBasedIAMPolicy(
                document=models.Placeholder.BUILD_STAGE,
                filename=filename
            )
        else:
            resource_name = 'default-role'
            role_name = f'{config.app_name}-{stage_name}'
            policy = models.AutoGenIAMPolicy(
                document=models.Placeholder.BUILD_STAGE,
                traits=set()
            )
        return models.ManagedIAMRole(
            resource_name=resource_name,
            role_name=role_name,
            trust_policy=LAMBDA_TRUST_POLICY,
            policy=policy
        )

    def _get_vpc_params(
        self,
        function_name: str,
        config: Config
    ) -> Tuple[List[str], List[str]]:
        security_group_ids: Optional[List[str]] = config.security_group_ids
        subnet_ids: Optional[List[str]] = config.subnet_ids
        if security_group_ids and subnet_ids:
            return (security_group_ids, subnet_ids)
        elif not security_group_ids and not subnet_ids:
            return ([], [])
        else:
            raise ChaliceBuildError(
                f"Invalid VPC params for function '{function_name}', "
                f"in order to configure VPC for a Lambda function, you must provide the subnet_ids as well as the security_group_ids, "
                f"got subnet_ids: {subnet_ids}, security_group_ids: {security_group_ids}"
            )

    def _get_lambda_layers(self, config: Config) -> List[str]:
        layers: Optional[List[str]] = config.layers
        return layers if layers else []

    def _build_lambda_function(
        self,
        config: Config,
        name: str,
        handler_name: str,
        deployment: models.DeploymentPackage,
        role: models.IAMRole
    ) -> models.LambdaFunction:
        function_name: str = f'{config.app_name}-{config.chalice_stage}-{name}'
        security_group_ids, subnet_ids = self._get_vpc_params(name, config)
        lambda_layers: List[str] = self._get_lambda_layers(config)
        function: models.LambdaFunction = models.LambdaFunction(
            resource_name=name,
            function_name=function_name,
            environment_variables=config.environment_variables,
            runtime=config.lambda_python_version,
            handler=handler_name,
            tags=config.tags,
            timeout=config.lambda_timeout,
            memory_size=config.lambda_memory_size,
            deployment_package=deployment,
            role=role,
            security_group_ids=security_group_ids,
            subnet_ids=subnet_ids,
            reserved_concurrency=config.reserved_concurrency,
            layers=lambda_layers,
            managed_layer=self._get_managed_lambda_layer(config),
            xray=config.xray_enabled
        )
        self._inject_role_traits(function, role)
        return function

    def _inject_role_traits(
        self,
        function: models.LambdaFunction,
        role: models.IAMRole
    ) -> None:
        if not isinstance(role, models.ManagedIAMRole):
            return
        policy: models.IAMPolicy = role.policy
        if not isinstance(policy, models.AutoGenIAMPolicy):
            return
        if function.security_group_ids and function.subnet_ids:
            policy.traits.add(models.RoleTraits.VPC_NEEDED)

    def _create_bucket_notification(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        s3_event: app.S3EventConfig,
        stage_name: str
    ) -> models.S3BucketNotification:
        lambda_function: models.LambdaFunction = self._create_lambda_model(
            config=config,
            deployment=deployment,
            name=s3_event.name,
            handler_name=s3_event.handler_string,
            stage_name=stage_name
        )
        resource_name: str = f'{s3_event.name}-s3event'
        s3_bucket: models.S3BucketNotification = models.S3BucketNotification(
            resource_name=resource_name,
            bucket=s3_event.bucket,
            prefix=s3_event.prefix,
            suffix=s3_event.suffix,
            events=s3_event.events,
            lambda_function=lambda_function
        )
        return s3_bucket

    def _create_sns_subscription(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        sns_config: app.SNSEventConfig,
        stage_name: str
    ) -> models.SNSLambdaSubscription:
        lambda_function: models.LambdaFunction = self._create_lambda_model(
            config=config,
            deployment=deployment,
            name=sns_config.name,
            handler_name=sns_config.handler_string,
            stage_name=stage_name
        )
        resource_name: str = f'{sns_config.name}-sns-subscription'
        sns_subscription: models.SNSLambdaSubscription = models.SNSLambdaSubscription(
            resource_name=resource_name,
            topic=sns_config.topic,
            lambda_function=lambda_function
        )
        return sns_subscription

    def _create_sqs_subscription(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        sqs_config: app.SQSEventConfig,
        stage_name: str
    ) -> models.SQSEventSource:
        lambda_function: models.LambdaFunction = self._create_lambda_model(
            config=config,
            deployment=deployment,
            name=sqs_config.name,
            handler_name=sqs_config.handler_string,
            stage_name=stage_name
        )
        resource_name: str = f'{sqs_config.name}-sqs-event-source'
        queue: Union[models.QueueARN, str] = ''
        if sqs_config.queue_arn is not None:
            queue = models.QueueARN(arn=sqs_config.queue_arn)
        elif sqs_config.queue is not None:
            queue = sqs_config.queue
        batch_window: Optional[int] = sqs_config.maximum_batching_window_in_seconds
        sqs_event_source: models.SQSEventSource = models.SQSEventSource(
            resource_name=resource_name,
            queue=queue,
            batch_size=sqs_config.batch_size,
            lambda_function=lambda_function,
            maximum_batching_window_in_seconds=batch_window,
            maximum_concurrency=sqs_config.maximum_concurrency
        )
        return sqs_event_source

    def _create_kinesis_subscription(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        kinesis_config: app.KinesisEventConfig,
        stage_name: str
    ) -> models.KinesisEventSource:
        lambda_function: models.LambdaFunction = self._create_lambda_model(
            config=config,
            deployment=deployment,
            name=kinesis_config.name,
            handler_name=kinesis_config.handler_string,
            stage_name=stage_name
        )
        resource_name: str = f'{kinesis_config.name}-kinesis-event-source'
        batch_window: Optional[int] = kinesis_config.maximum_batching_window_in_seconds
        kinesis_event_source: models.KinesisEventSource = models.KinesisEventSource(
            resource_name=resource_name,
            stream=kinesis_config.stream,
            batch_size=kinesis_config.batch_size,
            maximum_batching_window_in_seconds=batch_window,
            starting_position=kinesis_config.starting_position,
            lambda_function=lambda_function
        )
        return kinesis_event_source

    def _create_ddb_subscription(
        self,
        config: Config,
        deployment: models.DeploymentPackage,
        ddb_config: app.DynamoDBEventConfig,
        stage_name: str
    ) -> models.DynamoDBEventSource:
        lambda_function: models.LambdaFunction = self._create_lambda_model(
            config=config,
            deployment=deployment,
            name=ddb_config.name,
            handler_name=ddb_config.handler_string,
            stage_name=stage_name
        )
        resource_name: str = f'{ddb_config.name}-dynamodb-event-source'
        batch_window: Optional[int] = ddb_config.maximum_batching_window_in_seconds
        ddb_event_source: models.DynamoDBEventSource = models.DynamoDBEventSource(
            resource_name=resource_name,
            stream_arn=ddb_config.stream_arn,
            batch_size=ddb_config.batch_size,
            maximum_batching_window_in_seconds=batch_window,
            starting_position=ddb_config.starting_position,
            lambda_function=lambda_function
        )
        return ddb_event_source


class DependencyBuilder(object):

    def __init__(self) -> None:
        pass

    def build_dependencies(self, graph: models.Application) -> List[models.Resource]:
        seen: Set[int] = set()
        ordered: List[models.Resource] = []
        for resource in graph.dependencies():
            self._traverse(resource, ordered, seen)
        return ordered

    def _traverse(
        self,
        resource: models.Resource,
        ordered: List[models.Resource],
        seen: Set[int]
    ) -> None:
        for dep in resource.dependencies():
            if id(dep) not in seen:
                seen.add(id(dep))
                self._traverse(dep, ordered, seen)
        if id(resource) not in {id(r) for r in ordered}:
            ordered.append(resource)


class GraphPrettyPrint(object):
    _NEW_SECTION: str = '├──'
    _LINE_VERTICAL: str = '│'

    def __init__(self, ui: UI) -> None:
        self._ui = ui

    def display_graph(self, graph: models.Application) -> None:
        self._ui.write('Application\n')
        for model in graph.dependencies():
            self._traverse(model, level=0)

    def _traverse(self, graph: models.Resource, level: int) -> None:
        prefix: str = f'{self._LINE_VERTICAL}   ' * level
        spaces: str = f'{prefix}{self._NEW_SECTION} '
        model_text: str = self._get_model_text(graph, spaces, level)
        current_line: str = f'{spaces}{model_text}\n'
        self._ui.write(current_line)
        for model in graph.dependencies():
            self._traverse(model, level + 1)

    def _get_model_text(
        self,
        model: models.Resource,
        spaces: str,
        level: int
    ) -> str:
        name: str = model.__class__.__name__
        filtered: Dict[str, Any] = self._get_filtered_params(model)
        if not filtered:
            return f'{name}()'
        total_len_prefix: int = len(spaces) + len(name) + 1
        prefix: str = f'{self._LINE_VERTICAL}   ' * (level + 2)
        full: str = f'{prefix}{" " * (total_len_prefix - len(prefix))}'
        param_items: List[Tuple[str, Any]] = list(filtered.items())
        first: Tuple[str, Any] = param_items[0]
        remaining: List[Tuple[str, Any]] = param_items[1:]
        lines: List[str] = [f'{name}({first[0]}={first[1]},']
        self._add_remaining_lines(lines, remaining, full)
        return '\n'.join(lines) + ')'

    def _add_remaining_lines(
        self,
        lines: List[str],
        remaining: List[Tuple[str, Any]],
        full: str
    ) -> None:
        for key, value in remaining:
            if isinstance(value, (list, dict)):
                value = key.upper()
            current: str = f'{full}{key}={value},'
            lines.append(current)

    def _get_filtered_params(self, model: models.Resource) -> Dict[str, Any]:
        dependencies = model.dependencies()
        filtered: Dict[str, Any] = {
            k: v for k, v in asdict(model).items() if v not in dependencies
        }
        return filtered
