from chalice.app import Chalice
from chalice.config import Config
from chalice.constants import LAMBDA_TRUST_POLICY
from chalice.deploy import models
from chalice.deploy.appgraph import ApplicationGraphBuilder, ChaliceBuildError
from chalice.deploy.deployer import BuildStage, PolicyGenerator
from chalice.utils import serialize_to_json, OSUtils
from typing import List, Dict, Any, Union

def create_config(self, app: Chalice, app_name: str = 'lambda-only', iam_role_arn: str = None, policy_file: str = None, api_gateway_stage: str = 'api', autogen_policy: bool = False, security_group_ids: List[str] = None, subnet_ids: List[str] = None, reserved_concurrency: int = None, layers: List[str] = None, automatic_layer: bool = False, api_gateway_endpoint_type: str = None, api_gateway_endpoint_vpce: str = None, api_gateway_policy_file: str = None, api_gateway_custom_domain: Dict[str, Any] = None, websocket_api_custom_domain: Dict[str, Any] = None, log_retention_in_days: int = None, project_dir: str = '.') -> Config:
    kwargs: Dict[str, Any] = {'chalice_app': app, 'app_name': app_name, 'project_dir': project_dir, 'automatic_layer': automatic_layer, 'api_gateway_stage': api_gateway_stage, 'api_gateway_policy_file': api_gateway_policy_file, 'api_gateway_endpoint_type': api_gateway_endpoint_type, 'api_gateway_endpoint_vpce': api_gateway_endpoint_vpce, 'api_gateway_custom_domain': api_gateway_custom_domain, 'websocket_api_custom_domain': websocket_api_custom_domain}
    if iam_role_arn is not None:
        kwargs['manage_iam_role'] = False
        kwargs['iam_role_arn'] = 'role:arn'
    elif policy_file is not None:
        kwargs['autogen_policy'] = False
        kwargs['iam_policy_file'] = policy_file
    elif autogen_policy:
        kwargs['autogen_policy'] = True
    if security_group_ids is not None and subnet_ids is not None:
        kwargs['security_group_ids'] = security_group_ids
        kwargs['subnet_ids'] = subnet_ids
    if reserved_concurrency is not None:
        kwargs['reserved_concurrency'] = reserved_concurrency
    if log_retention_in_days is not None:
        kwargs['log_retention_in_days'] = log_retention_in_days
    kwargs['layers'] = layers
    config: Config = Config.create(**kwargs)
    return config

def test_can_build_single_lambda_function_app(self, sample_app_lambda_only: Chalice) -> None:
    ...

# Add type annotations to the remaining test methods as well
