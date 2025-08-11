import json
import os
from dataclasses import asdict
from typing import cast
from typing import Dict, List, Tuple, Any, Set, Optional, Text, Union
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
        self._known_roles = {}
        self._managed_layer = None

    def build(self, config: Union[pyramid.config.Configurator, str, None], stage_name: Union[str, config.Config]):
        resources = []
        deployment = models.DeploymentPackage(models.Placeholder.BUILD_STAGE)
        for function in config.chalice_app.pure_lambda_functions:
            resource = self._create_lambda_model(config=config, deployment=deployment, name=function.name, handler_name=function.handler_string, stage_name=stage_name)
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

    def _create_log_group(self, config: Union[str, isorsettings.Config, dict], resource_name: Union[str, isorsettings.Config, dict], log_group_name: Union[str, isorsettings.Config, dict]):
        return models.LogGroup(resource_name=resource_name, log_group_name=log_group_name, retention_in_days=config.log_retention_in_days)

    def _create_custom_domain_name(self, api_type: Union[str, None], domain_name_data: dict[str, typing.Any], endpoint_configuration: Union[str, None], api_gateway_stage: Union[str, int, None]) -> Union[str, set[str], dict]:
        url_prefix = domain_name_data.get('url_prefix', '(none)')
        api_mapping_model = self._create_api_mapping_model(url_prefix, api_gateway_stage)
        domain_name = self._create_domain_name_model(api_type, domain_name_data, endpoint_configuration, api_mapping_model)
        return domain_name

    def _create_api_mapping_model(self, key: Union[str, int, None], stage: Union[str, int, None]):
        if key == '/':
            key = '(none)'
        return models.APIMapping(resource_name='api_mapping', mount_path=key, api_gateway_stage=stage)

    def _create_lambda_event_resources(self, config: Union[cerise.config.Config, list[str], list[tuple[str]]], deployment: Union[list[str], dict[str, dict[str, typing.Any]]], stage_name: Union[list[str], dict[str, dict[str, typing.Any]]]) -> list:
        resources = []
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

    def _create_rest_api_model(self, config: Union[str, list[str], isorsettings.Config], deployment: Union[cmk.base.config.HostConfig, str, cmk.base.config.ConfigCache], stage_name: Union[cmk.base.config.HostConfig, str, cmk.base.config.ConfigCache]):
        lambda_function = self._create_lambda_model(config=config, deployment=deployment, name='api_handler', handler_name='app.app', stage_name=stage_name)
        function_name = '%s-%s' % (config.app_name, config.chalice_stage)
        lambda_function.function_name = function_name
        if config.minimum_compression_size is None:
            minimum_compression = ''
        else:
            minimum_compression = str(config.minimum_compression_size)
        authorizers = []
        for auth in config.chalice_app.builtin_auth_handlers:
            auth_lambda = self._create_lambda_model(config=config, deployment=deployment, name=auth.name, handler_name=auth.handler_string, stage_name=stage_name)
            authorizers.append(auth_lambda)
        policy = None
        policy_path = config.api_gateway_policy_file
        if config.api_gateway_endpoint_type == 'PRIVATE' and (not policy_path):
            policy = models.IAMPolicy(document=self._get_default_private_api_policy(config))
        elif policy_path:
            policy = models.FileBasedIAMPolicy(document=models.Placeholder.BUILD_STAGE, filename=os.path.join(config.project_dir, '.chalice', policy_path))
        vpce_ids = None
        if config.api_gateway_endpoint_vpce:
            vpce = config.api_gateway_endpoint_vpce
            vpce_ids = [vpce] if isinstance(vpce, str) else vpce
        custom_domain_name = None
        if config.api_gateway_custom_domain:
            custom_domain_name = self._create_custom_domain_name(models.APIType.HTTP, config.api_gateway_custom_domain, config.api_gateway_endpoint_type, config.api_gateway_stage)
        return models.RestAPI(resource_name='rest_api', swagger_doc=models.Placeholder.BUILD_STAGE, endpoint_type=config.api_gateway_endpoint_type, minimum_compression=minimum_compression, api_gateway_stage=config.api_gateway_stage, lambda_function=lambda_function, authorizers=authorizers, policy=policy, domain_name=custom_domain_name, xray=config.xray_enabled, vpce_ids=vpce_ids)

    def _get_default_private_api_policy(self, config: Union[dict[str, str], config.Config]) -> dict[typing.Text, typing.Union[typing.Text,list[dict[typing.Text, typing.Union[typing.Text,dict[typing.Text, dict[typing.Text, ]]]]]]]:
        statements = [{'Effect': 'Allow', 'Principal': '*', 'Action': 'execute-api:Invoke', 'Resource': 'arn:*:execute-api:*:*:*', 'Condition': {'StringEquals': {'aws:SourceVpce': config.api_gateway_endpoint_vpce}}}]
        return {'Version': '2012-10-17', 'Statement': statements}

    def _create_websocket_api_model(self, config: Union[Config, str, typing.NamedTuple], deployment: Union[str, cmk.base.config.ConfigCache, cerise.config.Config], stage_name: Union[str, cmk.base.config.ConfigCache, cerise.config.Config]):
        connect_handler = None
        message_handler = None
        disconnect_handler = None
        routes = {h.route_key_handled: h.handler_string for h in config.chalice_app.websocket_handlers.values()}
        if '$connect' in routes:
            connect_handler = self._create_lambda_model(config=config, deployment=deployment, name='websocket_connect', handler_name=routes['$connect'], stage_name=stage_name)
            routes.pop('$connect')
        if '$disconnect' in routes:
            disconnect_handler = self._create_lambda_model(config=config, deployment=deployment, name='websocket_disconnect', handler_name=routes['$disconnect'], stage_name=stage_name)
            routes.pop('$disconnect')
        if routes:
            handler_string = list(routes.values())[0]
            message_handler = self._create_lambda_model(config=config, deployment=deployment, name='websocket_message', handler_name=handler_string, stage_name=stage_name)
        custom_domain_name = None
        if config.websocket_api_custom_domain:
            custom_domain_name = self._create_custom_domain_name(models.APIType.WEBSOCKET, config.websocket_api_custom_domain, config.api_gateway_endpoint_type, config.api_gateway_stage)
        return models.WebsocketAPI(name='%s-%s-websocket-api' % (config.app_name, stage_name), resource_name='websocket_api', connect_function=connect_handler, message_function=message_handler, disconnect_function=disconnect_handler, routes=[h.route_key_handled for h in config.chalice_app.websocket_handlers.values()], api_gateway_stage=config.api_gateway_stage, domain_name=custom_domain_name)

    def _create_cwe_subscription(self, config: Union[str, types.Config, dict[str, str]], deployment: Union[str, types.Config, dict[str, str]], event_source: Union[str, types.Config, dict[str, str]], stage_name: Union[str, types.Config, dict[str, str]]):
        lambda_function = self._create_lambda_model(config=config, deployment=deployment, name=event_source.name, handler_name=event_source.handler_string, stage_name=stage_name)
        resource_name = event_source.name + '-event'
        rule_name = '%s-%s-%s' % (config.app_name, config.chalice_stage, resource_name)
        cwe = models.CloudWatchEvent(resource_name=resource_name, rule_name=rule_name, event_pattern=json.dumps(event_source.event_pattern), lambda_function=lambda_function)
        return cwe

    def _create_scheduled_model(self, config: Union[str, types.Config, dict[str, str]], deployment: Union[str, types.Config, dict[str, str]], event_source: Union[str, types.Config, dict[str, str]], stage_name: Union[str, types.Config, dict[str, str]]):
        lambda_function = self._create_lambda_model(config=config, deployment=deployment, name=event_source.name, handler_name=event_source.handler_string, stage_name=stage_name)
        resource_name = event_source.name + '-event'
        if isinstance(event_source.schedule_expression, app.ScheduleExpression):
            expression = event_source.schedule_expression.to_string()
        else:
            expression = event_source.schedule_expression
        rule_name = '%s-%s-%s' % (config.app_name, config.chalice_stage, resource_name)
        scheduled_event = models.ScheduledEvent(resource_name=resource_name, rule_name=rule_name, rule_description=event_source.description, schedule_expression=expression, lambda_function=lambda_function)
        return scheduled_event

    def _create_domain_name_model(self, protocol: Union[str, dict[typing.Any, str]], data: dict[typing.Any, str], endpoint_type: Union[dict, str, list[tuple[str]]], api_mapping: Union[str, dict[typing.Any, str]]) -> Union[str, set[str], tuple[str]]:
        default_name = 'api_gateway_custom_domain'
        resource_name_map = {'HTTP': default_name, 'WEBSOCKET': 'websocket_api_custom_domain'}
        domain_name = models.DomainName(protocol=protocol, resource_name=resource_name_map.get(protocol.value, default_name), domain_name=data['domain_name'], tls_version=models.TLSVersion.create(data.get('tls_version', '')), certificate_arn=data['certificate_arn'], tags=data.get('tags'), api_mapping=api_mapping)
        return domain_name

    def _create_lambda_model(self, config: Union[str, None, pyramid.config.Configurator], deployment: str, name: Union[str, None, pyramid.config.Configurator], handler_name: str, stage_name: str):
        new_config = config.scope(chalice_stage=config.chalice_stage, function_name=name)
        role = self._get_role_reference(new_config, stage_name, name)
        resource = self._build_lambda_function(new_config, name, handler_name, deployment, role)
        if new_config.log_retention_in_days:
            log_resource_name = '%s-log-group' % name
            log_group_name = '/aws/lambda/%s-%s-%s' % (new_config.app_name, stage_name, name)
            resource.log_group = self._create_log_group(new_config, log_resource_name, log_group_name)
        return resource

    def _get_managed_lambda_layer(self, config: Config) -> Union[None, int, dict[str, typing.Union[typing.Any,int]], typing.Iterator]:
        if not config.automatic_layer:
            return None
        if self._managed_layer is None:
            self._managed_layer = models.LambdaLayer(resource_name='managed-layer', layer_name='%s-%s-%s' % (config.app_name, config.chalice_stage, 'managed-layer'), runtime=config.lambda_python_version, deployment_package=models.DeploymentPackage(models.Placeholder.BUILD_STAGE))
        return self._managed_layer

    def _get_role_reference(self, config: Union[str, cmk.utils.type_defs.CheckPluginNameStr, dict], stage_name: Union[str, cmk.utils.type_defs.CheckPluginNameStr, dict], function_name: Union[str, cmk.utils.type_defs.CheckPluginNameStr, dict]) -> Union[dict[str, str], dict, str]:
        role = self._create_role_reference(config, stage_name, function_name)
        role_identifier = self._get_role_identifier(role)
        if role_identifier in self._known_roles:
            return self._known_roles[role_identifier]
        self._known_roles[role_identifier] = role
        return role

    def _get_role_identifier(self, role: Union[list[str], dict[str, str], accounts.models.KippoOrganization]):
        if isinstance(role, models.PreCreatedIAMRole):
            return role.role_arn
        role = cast(models.ManagedIAMRole, role)
        return role.resource_name

    def _create_role_reference(self, config: Union[str, list[tuple[str]], freqtrade.constants.ListPairsWithTimeframes], stage_name: Union[str, dict, None], function_name: Union[str, dict, cmk.utils.type_defs.CheckPluginNameStr]):
        if not config.manage_iam_role:
            return models.PreCreatedIAMRole(role_arn=config.iam_role_arn)
        policy = models.IAMPolicy(document=models.Placeholder.BUILD_STAGE)
        if not config.autogen_policy:
            resource_name = '%s_role' % function_name
            role_name = '%s-%s-%s' % (config.app_name, stage_name, function_name)
            if config.iam_policy_file is not None:
                filename = os.path.join(config.project_dir, '.chalice', config.iam_policy_file)
            else:
                filename = os.path.join(config.project_dir, '.chalice', 'policy-%s.json' % stage_name)
            policy = models.FileBasedIAMPolicy(filename=filename, document=models.Placeholder.BUILD_STAGE)
        else:
            resource_name = 'default-role'
            role_name = '%s-%s' % (config.app_name, stage_name)
            policy = models.AutoGenIAMPolicy(document=models.Placeholder.BUILD_STAGE, traits=set([]))
        return models.ManagedIAMRole(resource_name=resource_name, role_name=role_name, trust_policy=LAMBDA_TRUST_POLICY, policy=policy)

    def _get_vpc_params(self, function_name: str, config: Union[str, dict[str, typing.Any], None, dict]) -> Union[tuple, tuple[list]]:
        security_group_ids = config.security_group_ids
        subnet_ids = config.subnet_ids
        if security_group_ids and subnet_ids:
            return (security_group_ids, subnet_ids)
        elif not security_group_ids and (not subnet_ids):
            return ([], [])
        else:
            raise ChaliceBuildError("Invalid VPC params for function '%s', in order to configure VPC for a Lambda function, you must provide the subnet_ids as well as the security_group_ids, got subnet_ids: %s, security_group_ids: %s" % (function_name, subnet_ids, security_group_ids))

    def _get_lambda_layers(self, config: Union[dict[str, typing.Any], dict, routemaster.config.Config]) -> list:
        layers = config.layers
        return layers if layers else []

    def _build_lambda_function(self, config: Union[str, dict[str, str]], name: Union[str, cmk.utils.type_defs.CheckPluginName], handler_name: Union[str, None, dict], deployment: Union[str, None, dict], role: Union[str, None, dict]) -> Union[str, list[typing.Callable], list[str]]:
        function_name = '%s-%s-%s' % (config.app_name, config.chalice_stage, name)
        security_group_ids, subnet_ids = self._get_vpc_params(name, config)
        lambda_layers = self._get_lambda_layers(config)
        function = models.LambdaFunction(resource_name=name, function_name=function_name, environment_variables=config.environment_variables, runtime=config.lambda_python_version, handler=handler_name, tags=config.tags, timeout=config.lambda_timeout, memory_size=config.lambda_memory_size, deployment_package=deployment, role=role, security_group_ids=security_group_ids, subnet_ids=subnet_ids, reserved_concurrency=config.reserved_concurrency, layers=lambda_layers, managed_layer=self._get_managed_lambda_layer(config), xray=config.xray_enabled)
        self._inject_role_traits(function, role)
        return function

    def _inject_role_traits(self, function: Union[int, str, dict[str, typing.Any]], role: Union[app.models.User, django.db.models.QuerySet, typing.Callable]) -> None:
        if not isinstance(role, models.ManagedIAMRole):
            return
        policy = role.policy
        if not isinstance(policy, models.AutoGenIAMPolicy):
            return
        if function.security_group_ids and function.subnet_ids:
            policy.traits.add(models.RoleTraits.VPC_NEEDED)

    def _create_bucket_notification(self, config: Union[str, list[str], models.LTI1p3Provider], deployment: Union[str, list[str], models.LTI1p3Provider], s3_event: Union[str, list[str], cmk.utils.type_defs.HostName], stage_name: Union[str, list[str], models.LTI1p3Provider]):
        lambda_function = self._create_lambda_model(config=config, deployment=deployment, name=s3_event.name, handler_name=s3_event.handler_string, stage_name=stage_name)
        resource_name = s3_event.name + '-s3event'
        s3_bucket = models.S3BucketNotification(resource_name=resource_name, bucket=s3_event.bucket, prefix=s3_event.prefix, suffix=s3_event.suffix, events=s3_event.events, lambda_function=lambda_function)
        return s3_bucket

    def _create_sns_subscription(self, config: Union[str, None, types.Config], deployment: Union[str, None, types.Config], sns_config: Union[str, None, types.Config], stage_name: Union[str, None, types.Config]):
        lambda_function = self._create_lambda_model(config=config, deployment=deployment, name=sns_config.name, handler_name=sns_config.handler_string, stage_name=stage_name)
        resource_name = sns_config.name + '-sns-subscription'
        sns_subscription = models.SNSLambdaSubscription(resource_name=resource_name, topic=sns_config.topic, lambda_function=lambda_function)
        return sns_subscription

    def _create_sqs_subscription(self, config: Union[str, None, pyramid.config.Configurator], deployment: Union[str, None, pyramid.config.Configurator], sqs_config: Union[str, None, pyramid.config.Configurator], stage_name: Union[str, None, pyramid.config.Configurator]):
        lambda_function = self._create_lambda_model(config=config, deployment=deployment, name=sqs_config.name, handler_name=sqs_config.handler_string, stage_name=stage_name)
        resource_name = sqs_config.name + '-sqs-event-source'
        queue = ''
        if sqs_config.queue_arn is not None:
            queue = models.QueueARN(arn=sqs_config.queue_arn)
        elif sqs_config.queue is not None:
            queue = sqs_config.queue
        batch_window = sqs_config.maximum_batching_window_in_seconds
        sqs_event_source = models.SQSEventSource(resource_name=resource_name, queue=queue, batch_size=sqs_config.batch_size, lambda_function=lambda_function, maximum_batching_window_in_seconds=batch_window, maximum_concurrency=sqs_config.maximum_concurrency)
        return sqs_event_source

    def _create_kinesis_subscription(self, config: Union[str, None, cmk.base.config.ConfigCache, pyramid.config.Configurator], deployment: Union[str, None, cmk.base.config.ConfigCache, pyramid.config.Configurator], kinesis_config: Union[str, None, cmk.base.config.ConfigCache, pyramid.config.Configurator], stage_name: Union[str, None, cmk.base.config.ConfigCache, pyramid.config.Configurator]):
        lambda_function = self._create_lambda_model(config=config, deployment=deployment, name=kinesis_config.name, handler_name=kinesis_config.handler_string, stage_name=stage_name)
        resource_name = kinesis_config.name + '-kinesis-event-source'
        batch_window = kinesis_config.maximum_batching_window_in_seconds
        kinesis_event_source = models.KinesisEventSource(resource_name=resource_name, stream=kinesis_config.stream, batch_size=kinesis_config.batch_size, maximum_batching_window_in_seconds=batch_window, starting_position=kinesis_config.starting_position, lambda_function=lambda_function)
        return kinesis_event_source

    def _create_ddb_subscription(self, config: Union[str, None, cmk.base.config.ConfigCache], deployment: Union[str, None, cmk.base.config.ConfigCache], ddb_config: Union[str, None, cmk.base.config.ConfigCache], stage_name: Union[str, None, cmk.base.config.ConfigCache]):
        lambda_function = self._create_lambda_model(config=config, deployment=deployment, name=ddb_config.name, handler_name=ddb_config.handler_string, stage_name=stage_name)
        resource_name = ddb_config.name + '-dynamodb-event-source'
        batch_window = ddb_config.maximum_batching_window_in_seconds
        ddb_event_source = models.DynamoDBEventSource(resource_name=resource_name, stream_arn=ddb_config.stream_arn, batch_size=ddb_config.batch_size, maximum_batching_window_in_seconds=batch_window, starting_position=ddb_config.starting_position, lambda_function=lambda_function)
        return ddb_event_source

class DependencyBuilder(object):

    def __init__(self) -> None:
        pass

    def build_dependencies(self, graph: list[str]) -> list:
        seen = set()
        ordered = []
        for resource in graph.dependencies():
            self._traverse(resource, ordered, seen)
        return ordered

    def _traverse(self, resource, ordered, seen) -> None:
        for dep in resource.dependencies():
            if id(dep) not in seen:
                seen.add(id(dep))
                self._traverse(dep, ordered, seen)
        if id(resource) not in [id(r) for r in ordered]:
            ordered.append(resource)

class GraphPrettyPrint(object):
    _NEW_SECTION = '├──'
    _LINE_VERTICAL = '│'

    def __init__(self, ui: Union[Tracer, str]) -> None:
        self._ui = ui

    def display_graph(self, graph: Union[set[typing.Hashable], dict[str, set[str]]]) -> None:
        self._ui.write('Application\n')
        for model in graph.dependencies():
            self._traverse(model, level=0)

    def _traverse(self, graph: Union[str, int], level: int) -> None:
        prefix = '%s   ' % self._LINE_VERTICAL * level
        spaces = prefix + self._NEW_SECTION + ' '
        model_text = self._get_model_text(graph, spaces, level)
        current_line = cast(str, '%s%s\n' % (spaces, model_text))
        self._ui.write(current_line)
        for model in graph.dependencies():
            self._traverse(model, level + 1)

    def _get_model_text(self, model: Union[int, dict, db.models.Taxon, None], spaces: list[str], level: int) -> Union[typing.Text, str]:
        name = model.__class__.__name__
        filtered = self._get_filtered_params(model)
        if not filtered:
            return '%s()' % name
        total_len_prefix = len(spaces) + len(name) + 1
        prefix = '%s   ' % self._LINE_VERTICAL * (level + 2)
        full = '%s%s' % (prefix, ' ' * (total_len_prefix - len(prefix)))
        param_items = list(filtered.items())
        first = param_items[0]
        remaining = param_items[1:]
        lines = ['%s(%s=%s,' % (name, first[0], first[1])]
        self._add_remaining_lines(lines, remaining, full)
        return '\n'.join(lines) + ')'

    def _add_remaining_lines(self, lines: Any, remaining: Union[str, bytes], full: str) -> None:
        for key, value in remaining:
            if isinstance(value, (list, dict)):
                value = key.upper()
            current = cast(str, '%s%s=%s,' % (full, key, value))
            lines.append(current)

    def _get_filtered_params(self, model: Union[esm.models.service_instance.ServiceInstance, Contributor, Coverage, str]) -> dict:
        dependencies = model.dependencies()
        filtered = {k: v for k, v in asdict(model).items() if v not in dependencies}
        return filtered