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

    def __init__(self):
        self._known_roles = {}
        self._managed_layer = None

    def func_e954wy7n(self, config, stage_name):
        resources = []
        deployment = models.DeploymentPackage(models.Placeholder.BUILD_STAGE)
        for function in config.chalice_app.pure_lambda_functions:
            resource = self._create_lambda_model(config=config, deployment=
                deployment, name=function.name, handler_name=function.
                handler_string, stage_name=stage_name)
            resources.append(resource)
        event_resources = self._create_lambda_event_resources(config,
            deployment, stage_name)
        resources.extend(event_resources)
        if config.chalice_app.routes:
            rest_api = self._create_rest_api_model(config, deployment,
                stage_name)
            resources.append(rest_api)
        if config.chalice_app.websocket_handlers:
            websocket_api = self._create_websocket_api_model(config,
                deployment, stage_name)
            resources.append(websocket_api)
        return models.Application(stage_name, resources)

    def func_u45d0u5t(self, config, resource_name, log_group_name):
        return models.LogGroup(resource_name=resource_name, log_group_name=
            log_group_name, retention_in_days=config.log_retention_in_days)

    def func_jw34kgf7(self, api_type, domain_name_data,
        endpoint_configuration, api_gateway_stage):
        url_prefix = domain_name_data.get('url_prefix', '(none)')
        api_mapping_model = self._create_api_mapping_model(url_prefix,
            api_gateway_stage)
        domain_name = self._create_domain_name_model(api_type,
            domain_name_data, endpoint_configuration, api_mapping_model)
        return domain_name

    def func_i9ascg7i(self, key, stage):
        if key == '/':
            key = '(none)'
        return models.APIMapping(resource_name='api_mapping', mount_path=
            key, api_gateway_stage=stage)

    def func_1xiid40d(self, config, deployment, stage_name):
        resources = []
        for event_source in config.chalice_app.event_sources:
            if isinstance(event_source, app.S3EventConfig):
                resources.append(self._create_bucket_notification(config,
                    deployment, event_source, stage_name))
            elif isinstance(event_source, app.SNSEventConfig):
                resources.append(self._create_sns_subscription(config,
                    deployment, event_source, stage_name))
            elif isinstance(event_source, app.CloudWatchEventConfig):
                resources.append(self._create_cwe_subscription(config,
                    deployment, event_source, stage_name))
            elif isinstance(event_source, app.ScheduledEventConfig):
                resources.append(self._create_scheduled_model(config,
                    deployment, event_source, stage_name))
            elif isinstance(event_source, app.SQSEventConfig):
                resources.append(self._create_sqs_subscription(config,
                    deployment, event_source, stage_name))
            elif isinstance(event_source, app.KinesisEventConfig):
                resources.append(self._create_kinesis_subscription(config,
                    deployment, event_source, stage_name))
            elif isinstance(event_source, app.DynamoDBEventConfig):
                resources.append(self._create_ddb_subscription(config,
                    deployment, event_source, stage_name))
        return resources

    def func_m95yjz99(self, config, deployment, stage_name):
        lambda_function = self._create_lambda_model(config=config,
            deployment=deployment, name='api_handler', handler_name=
            'app.app', stage_name=stage_name)
        function_name = '%s-%s' % (config.app_name, config.chalice_stage)
        lambda_function.function_name = function_name
        if config.minimum_compression_size is None:
            minimum_compression = ''
        else:
            minimum_compression = str(config.minimum_compression_size)
        authorizers = []
        for auth in config.chalice_app.builtin_auth_handlers:
            auth_lambda = self._create_lambda_model(config=config,
                deployment=deployment, name=auth.name, handler_name=auth.
                handler_string, stage_name=stage_name)
            authorizers.append(auth_lambda)
        policy = None
        policy_path = config.api_gateway_policy_file
        if config.api_gateway_endpoint_type == 'PRIVATE' and not policy_path:
            policy = models.IAMPolicy(document=self.
                _get_default_private_api_policy(config))
        elif policy_path:
            policy = models.FileBasedIAMPolicy(document=models.Placeholder.
                BUILD_STAGE, filename=os.path.join(config.project_dir,
                '.chalice', policy_path))
        vpce_ids = None
        if config.api_gateway_endpoint_vpce:
            vpce = config.api_gateway_endpoint_vpce
            vpce_ids = [vpce] if isinstance(vpce, str) else vpce
        custom_domain_name = None
        if config.api_gateway_custom_domain:
            custom_domain_name = self._create_custom_domain_name(models.
                APIType.HTTP, config.api_gateway_custom_domain, config.
                api_gateway_endpoint_type, config.api_gateway_stage)
        return models.RestAPI(resource_name='rest_api', swagger_doc=models.
            Placeholder.BUILD_STAGE, endpoint_type=config.
            api_gateway_endpoint_type, minimum_compression=
            minimum_compression, api_gateway_stage=config.api_gateway_stage,
            lambda_function=lambda_function, authorizers=authorizers,
            policy=policy, domain_name=custom_domain_name, xray=config.
            xray_enabled, vpce_ids=vpce_ids)

    def func_n8uzj2ax(self, config):
        statements = [{'Effect': 'Allow', 'Principal': '*', 'Action':
            'execute-api:Invoke', 'Resource': 'arn:*:execute-api:*:*:*',
            'Condition': {'StringEquals': {'aws:SourceVpce': config.
            api_gateway_endpoint_vpce}}}]
        return {'Version': '2012-10-17', 'Statement': statements}

    def func_nclg9h0z(self, config, deployment, stage_name):
        connect_handler = None
        message_handler = None
        disconnect_handler = None
        routes = {h.route_key_handled: h.handler_string for h in config.
            chalice_app.websocket_handlers.values()}
        if '$connect' in routes:
            connect_handler = self._create_lambda_model(config=config,
                deployment=deployment, name='websocket_connect',
                handler_name=routes['$connect'], stage_name=stage_name)
            routes.pop('$connect')
        if '$disconnect' in routes:
            disconnect_handler = self._create_lambda_model(config=config,
                deployment=deployment, name='websocket_disconnect',
                handler_name=routes['$disconnect'], stage_name=stage_name)
            routes.pop('$disconnect')
        if routes:
            handler_string = list(routes.values())[0]
            message_handler = self._create_lambda_model(config=config,
                deployment=deployment, name='websocket_message',
                handler_name=handler_string, stage_name=stage_name)
        custom_domain_name = None
        if config.websocket_api_custom_domain:
            custom_domain_name = self._create_custom_domain_name(models.
                APIType.WEBSOCKET, config.websocket_api_custom_domain,
                config.api_gateway_endpoint_type, config.api_gateway_stage)
        return models.WebsocketAPI(name='%s-%s-websocket-api' % (config.
            app_name, stage_name), resource_name='websocket_api',
            connect_function=connect_handler, message_function=
            message_handler, disconnect_function=disconnect_handler, routes
            =[h.route_key_handled for h in config.chalice_app.
            websocket_handlers.values()], api_gateway_stage=config.
            api_gateway_stage, domain_name=custom_domain_name)

    def func_ddz0l6pn(self, config, deployment, event_source, stage_name):
        lambda_function = self._create_lambda_model(config=config,
            deployment=deployment, name=event_source.name, handler_name=
            event_source.handler_string, stage_name=stage_name)
        resource_name = event_source.name + '-event'
        rule_name = '%s-%s-%s' % (config.app_name, config.chalice_stage,
            resource_name)
        cwe = models.CloudWatchEvent(resource_name=resource_name, rule_name
            =rule_name, event_pattern=json.dumps(event_source.event_pattern
            ), lambda_function=lambda_function)
        return cwe

    def func_29pjt2ew(self, config, deployment, event_source, stage_name):
        lambda_function = self._create_lambda_model(config=config,
            deployment=deployment, name=event_source.name, handler_name=
            event_source.handler_string, stage_name=stage_name)
        resource_name = event_source.name + '-event'
        if isinstance(event_source.schedule_expression, app.ScheduleExpression
            ):
            expression = event_source.schedule_expression.to_string()
        else:
            expression = event_source.schedule_expression
        rule_name = '%s-%s-%s' % (config.app_name, config.chalice_stage,
            resource_name)
        scheduled_event = models.ScheduledEvent(resource_name=resource_name,
            rule_name=rule_name, rule_description=event_source.description,
            schedule_expression=expression, lambda_function=lambda_function)
        return scheduled_event

    def func_3bqsm8ny(self, protocol, data, endpoint_type, api_mapping):
        default_name = 'api_gateway_custom_domain'
        resource_name_map = {'HTTP': default_name, 'WEBSOCKET':
            'websocket_api_custom_domain'}
        domain_name = models.DomainName(protocol=protocol, resource_name=
            resource_name_map.get(protocol.value, default_name),
            domain_name=data['domain_name'], tls_version=models.TLSVersion.
            create(data.get('tls_version', '')), certificate_arn=data[
            'certificate_arn'], tags=data.get('tags'), api_mapping=api_mapping)
        return domain_name

    def func_jv5g9otu(self, config, deployment, name, handler_name, stage_name
        ):
        new_config = config.scope(chalice_stage=config.chalice_stage,
            function_name=name)
        role = self._get_role_reference(new_config, stage_name, name)
        resource = self._build_lambda_function(new_config, name,
            handler_name, deployment, role)
        if new_config.log_retention_in_days:
            log_resource_name = '%s-log-group' % name
            log_group_name = '/aws/lambda/%s-%s-%s' % (new_config.app_name,
                stage_name, name)
            resource.log_group = self._create_log_group(new_config,
                log_resource_name, log_group_name)
        return resource

    def func_di1ym8iz(self, config):
        if not config.automatic_layer:
            return None
        if self._managed_layer is None:
            self._managed_layer = models.LambdaLayer(resource_name=
                'managed-layer', layer_name='%s-%s-%s' % (config.app_name,
                config.chalice_stage, 'managed-layer'), runtime=config.
                lambda_python_version, deployment_package=models.
                DeploymentPackage(models.Placeholder.BUILD_STAGE))
        return self._managed_layer

    def func_tcy2pi8s(self, config, stage_name, function_name):
        role = self._create_role_reference(config, stage_name, function_name)
        role_identifier = self._get_role_identifier(role)
        if role_identifier in self._known_roles:
            return self._known_roles[role_identifier]
        self._known_roles[role_identifier] = role
        return role

    def func_hrzdf0h3(self, role):
        if isinstance(role, models.PreCreatedIAMRole):
            return role.role_arn
        role = cast(models.ManagedIAMRole, role)
        return role.resource_name

    def func_hjzr7qsg(self, config, stage_name, function_name):
        if not config.manage_iam_role:
            return models.PreCreatedIAMRole(role_arn=config.iam_role_arn)
        policy = models.IAMPolicy(document=models.Placeholder.BUILD_STAGE)
        if not config.autogen_policy:
            resource_name = '%s_role' % function_name
            role_name = '%s-%s-%s' % (config.app_name, stage_name,
                function_name)
            if config.iam_policy_file is not None:
                filename = os.path.join(config.project_dir, '.chalice',
                    config.iam_policy_file)
            else:
                filename = os.path.join(config.project_dir, '.chalice', 
                    'policy-%s.json' % stage_name)
            policy = models.FileBasedIAMPolicy(filename=filename, document=
                models.Placeholder.BUILD_STAGE)
        else:
            resource_name = 'default-role'
            role_name = '%s-%s' % (config.app_name, stage_name)
            policy = models.AutoGenIAMPolicy(document=models.Placeholder.
                BUILD_STAGE, traits=set([]))
        return models.ManagedIAMRole(resource_name=resource_name, role_name
            =role_name, trust_policy=LAMBDA_TRUST_POLICY, policy=policy)

    def func_9nbz0ck3(self, function_name, config):
        security_group_ids = config.security_group_ids
        subnet_ids = config.subnet_ids
        if security_group_ids and subnet_ids:
            return security_group_ids, subnet_ids
        elif not security_group_ids and not subnet_ids:
            return [], []
        else:
            raise ChaliceBuildError(
                "Invalid VPC params for function '%s', in order to configure VPC for a Lambda function, you must provide the subnet_ids as well as the security_group_ids, got subnet_ids: %s, security_group_ids: %s"
                 % (function_name, subnet_ids, security_group_ids))

    def func_z7rga5a7(self, config):
        layers = config.layers
        return layers if layers else []

    def func_l455w86l(self, config, name, handler_name, deployment, role):
        function_name = '%s-%s-%s' % (config.app_name, config.chalice_stage,
            name)
        security_group_ids, subnet_ids = self._get_vpc_params(name, config)
        lambda_layers = self._get_lambda_layers(config)
        function = models.LambdaFunction(resource_name=name, function_name=
            function_name, environment_variables=config.
            environment_variables, runtime=config.lambda_python_version,
            handler=handler_name, tags=config.tags, timeout=config.
            lambda_timeout, memory_size=config.lambda_memory_size,
            deployment_package=deployment, role=role, security_group_ids=
            security_group_ids, subnet_ids=subnet_ids, reserved_concurrency
            =config.reserved_concurrency, layers=lambda_layers,
            managed_layer=self._get_managed_lambda_layer(config), xray=
            config.xray_enabled)
        self._inject_role_traits(function, role)
        return function

    def func_m3wr0ix7(self, function, role):
        if not isinstance(role, models.ManagedIAMRole):
            return
        policy = role.policy
        if not isinstance(policy, models.AutoGenIAMPolicy):
            return
        if function.security_group_ids and function.subnet_ids:
            policy.traits.add(models.RoleTraits.VPC_NEEDED)

    def func_hqjb4tlm(self, config, deployment, s3_event, stage_name):
        lambda_function = self._create_lambda_model(config=config,
            deployment=deployment, name=s3_event.name, handler_name=
            s3_event.handler_string, stage_name=stage_name)
        resource_name = s3_event.name + '-s3event'
        s3_bucket = models.S3BucketNotification(resource_name=resource_name,
            bucket=s3_event.bucket, prefix=s3_event.prefix, suffix=s3_event
            .suffix, events=s3_event.events, lambda_function=lambda_function)
        return s3_bucket

    def func_hs362g5c(self, config, deployment, sns_config, stage_name):
        lambda_function = self._create_lambda_model(config=config,
            deployment=deployment, name=sns_config.name, handler_name=
            sns_config.handler_string, stage_name=stage_name)
        resource_name = sns_config.name + '-sns-subscription'
        sns_subscription = models.SNSLambdaSubscription(resource_name=
            resource_name, topic=sns_config.topic, lambda_function=
            lambda_function)
        return sns_subscription

    def func_vk3pdtxi(self, config, deployment, sqs_config, stage_name):
        lambda_function = self._create_lambda_model(config=config,
            deployment=deployment, name=sqs_config.name, handler_name=
            sqs_config.handler_string, stage_name=stage_name)
        resource_name = sqs_config.name + '-sqs-event-source'
        queue = ''
        if sqs_config.queue_arn is not None:
            queue = models.QueueARN(arn=sqs_config.queue_arn)
        elif sqs_config.queue is not None:
            queue = sqs_config.queue
        batch_window = sqs_config.maximum_batching_window_in_seconds
        sqs_event_source = models.SQSEventSource(resource_name=
            resource_name, queue=queue, batch_size=sqs_config.batch_size,
            lambda_function=lambda_function,
            maximum_batching_window_in_seconds=batch_window,
            maximum_concurrency=sqs_config.maximum_concurrency)
        return sqs_event_source

    def func_x9es7t86(self, config, deployment, kinesis_config, stage_name):
        lambda_function = self._create_lambda_model(config=config,
            deployment=deployment, name=kinesis_config.name, handler_name=
            kinesis_config.handler_string, stage_name=stage_name)
        resource_name = kinesis_config.name + '-kinesis-event-source'
        batch_window = kinesis_config.maximum_batching_window_in_seconds
        kinesis_event_source = models.KinesisEventSource(resource_name=
            resource_name, stream=kinesis_config.stream, batch_size=
            kinesis_config.batch_size, maximum_batching_window_in_seconds=
            batch_window, starting_position=kinesis_config.
            starting_position, lambda_function=lambda_function)
        return kinesis_event_source

    def func_7xc1mk7j(self, config, deployment, ddb_config, stage_name):
        lambda_function = self._create_lambda_model(config=config,
            deployment=deployment, name=ddb_config.name, handler_name=
            ddb_config.handler_string, stage_name=stage_name)
        resource_name = ddb_config.name + '-dynamodb-event-source'
        batch_window = ddb_config.maximum_batching_window_in_seconds
        ddb_event_source = models.DynamoDBEventSource(resource_name=
            resource_name, stream_arn=ddb_config.stream_arn, batch_size=
            ddb_config.batch_size, maximum_batching_window_in_seconds=
            batch_window, starting_position=ddb_config.starting_position,
            lambda_function=lambda_function)
        return ddb_event_source


class DependencyBuilder(object):

    def __init__(self):
        pass

    def func_xjthjcis(self, graph):
        seen = set()
        ordered = []
        for resource in graph.dependencies():
            self._traverse(resource, ordered, seen)
        return ordered

    def func_z8h1y3qg(self, resource, ordered, seen):
        for dep in resource.dependencies():
            if id(dep) not in seen:
                seen.add(id(dep))
                self._traverse(dep, ordered, seen)
        if id(resource) not in [id(r) for r in ordered]:
            ordered.append(resource)


class GraphPrettyPrint(object):
    _NEW_SECTION = '├──'
    _LINE_VERTICAL = '│'

    def __init__(self, ui):
        self._ui = ui

    def func_vi7aoqjs(self, graph):
        self._ui.write('Application\n')
        for model in graph.dependencies():
            self._traverse(model, level=0)

    def func_z8h1y3qg(self, graph, level):
        prefix = '%s   ' % self._LINE_VERTICAL * level
        spaces = prefix + self._NEW_SECTION + ' '
        model_text = self._get_model_text(graph, spaces, level)
        current_line = cast(str, '%s%s\n' % (spaces, model_text))
        self._ui.write(current_line)
        for model in graph.dependencies():
            self._traverse(model, level + 1)

    def func_o9k8qryz(self, model, spaces, level):
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

    def func_haeggxys(self, lines, remaining, full):
        for key, value in remaining:
            if isinstance(value, (list, dict)):
                value = key.upper()
            current = cast(str, '%s%s=%s,' % (full, key, value))
            lines.append(current)

    def func_8w52gpjw(self, model):
        dependencies = model.dependencies()
        filtered = {k: v for k, v in asdict(model).items() if v not in
            dependencies}
        return filtered
