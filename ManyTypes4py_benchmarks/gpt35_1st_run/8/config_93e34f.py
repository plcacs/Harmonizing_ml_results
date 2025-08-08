from __future__ import annotations
import os
import sys
import json
from typing import Dict, Any, Optional, List, Union
from chalice import __version__ as current_chalice_version
from chalice.app import Chalice
from chalice.constants import DEFAULT_STAGE_NAME
from chalice.constants import DEFAULT_HANDLER_NAME
StrMap = Dict[str, Any]

class Config(object):
    def __init__(self, chalice_stage: str = DEFAULT_STAGE_NAME, function_name: str = DEFAULT_HANDLER_NAME, user_provided_params: Optional[Dict[str, Any]] = None, config_from_disk: Optional[Dict[str, Any]] = None, default_params: Optional[Dict[str, Any]] = None, layers: Optional[List[str]] = None) -> None:
    
    @classmethod
    def create(cls, chalice_stage: str = DEFAULT_STAGE_NAME, function_name: str = DEFAULT_HANDLER_NAME, **kwargs: Any) -> Config:
    
    @property
    def profile(self) -> Any:
    
    @property
    def app_name(self) -> Any:
    
    @property
    def project_dir(self) -> Any:
    
    @property
    def chalice_app(self) -> Chalice:
    
    @property
    def config_from_disk(self) -> Dict[str, Any]:
    
    @property
    def lambda_python_version(self) -> str:
    
    @property
    def log_retention_in_days(self) -> Any:
    
    @property
    def layers(self) -> Any:
    
    @property
    def api_gateway_custom_domain(self) -> Any:
    
    @property
    def websocket_api_custom_domain(self) -> Any:
    
    def _chain_lookup(self, name: str, varies_per_chalice_stage: bool = False, varies_per_function: bool = False) -> Any:
    
    def _chain_merge(self, name: str) -> Dict[str, Any]:
    
    @property
    def config_file_version(self) -> str:
    
    @property
    def api_gateway_stage(self) -> Any:
    
    @property
    def api_gateway_endpoint_type(self) -> Any:
    
    @property
    def api_gateway_endpoint_vpce(self) -> Any:
    
    @property
    def api_gateway_policy_file(self) -> Any:
    
    @property
    def minimum_compression_size(self) -> Any:
    
    @property
    def iam_policy_file(self) -> Any:
    
    @property
    def lambda_memory_size(self) -> Any:
    
    @property
    def lambda_timeout(self) -> Any:
    
    @property
    def automatic_layer(self) -> bool:
    
    @property
    def iam_role_arn(self) -> Any:
    
    @property
    def manage_iam_role(self) -> bool:
    
    @property
    def autogen_policy(self) -> Any:
    
    @property
    def xray_enabled(self) -> Any:
    
    @property
    def environment_variables(self) -> Dict[str, Any]:
    
    @property
    def tags(self) -> Dict[str, Any]:
    
    @property
    def security_group_ids(self) -> Any:
    
    @property
    def subnet_ids(self) -> Any:
    
    @property
    def reserved_concurrency(self) -> Any:
    
    def scope(self, chalice_stage: str, function_name: str) -> Config:
    
    def deployed_resources(self, chalice_stage_name: str) -> DeployedResources:
    
    def _try_old_deployer_values(self, chalice_stage_name: str) -> DeployedResources:
    
    def _load_json_file(self, deployed_file: str) -> Optional[Dict[str, Any]]:
    
    def _upgrade_deployed_values(self, chalice_stage_name: str, data: Dict[str, Any]) -> DeployedResources:
    
    def _upgrade_lambda_functions(self, resources: List[Dict[str, Any]], deployed: Dict[str, Any], prefix: str) -> None:
    
    def _upgrade_rest_api(self, resources: List[Dict[str, Any]], deployed: Dict[str, Any]) -> None:

class DeployedResources(object):
    def __init__(self, deployed_values: Dict[str, Any]) -> None:
    
    @classmethod
    def empty(cls) -> DeployedResources:
    
    def resource_values(self, name: str) -> Dict[str, Any]:
    
    def resource_names(self) -> List[str]:
