```python
from __future__ import annotations
import sys
import os
import logging
import functools
from typing import Any, Optional, Dict, MutableMapping, cast
from botocore.config import Config as BotocoreConfig
from botocore.session import Session
from chalice import __version__ as chalice_version
from chalice.awsclient import TypedAWSClient
from chalice.app import Chalice
from chalice.config import Config
from chalice.config import DeployedResources
from chalice.package import AppPackager
from chalice.package import PackageOptions
from chalice.constants import DEFAULT_STAGE_NAME
from chalice.constants import DEFAULT_APIGATEWAY_STAGE_NAME
from chalice.constants import DEFAULT_ENDPOINT_TYPE
from chalice.logs import LogRetriever, LogEventGenerator
from chalice.logs import FollowLogEventGenerator
from chalice.logs import BaseLogEventGenerator
from chalice import local
from chalice.utils import UI
from chalice.utils import PipeReader
from chalice.deploy import deployer
from chalice.invoke import LambdaInvokeHandler
from chalice.invoke import LambdaInvoker
from chalice.invoke import LambdaResponseFormatter

OptStr: type = Optional[str]
OptInt: type = Optional[int]

def create_botocore_session(
    profile: Any = ...,
    debug: bool = ...,
    connection_timeout: Any = ...,
    read_timeout: Any = ...,
    max_retries: Any = ...
) -> Session: ...

def _add_chalice_user_agent(session: Any) -> None: ...

def _inject_large_request_body_filter() -> None: ...

class NoSuchFunctionError(Exception):
    name: Any
    def __init__(self, name: Any) -> None: ...

class UnknownConfigFileVersion(Exception):
    def __init__(self, version: Any) -> None: ...

class LargeRequestBodyFilter(logging.Filter):
    def filter(self, record: Any) -> bool: ...

class CLIFactory:
    project_dir: Any
    debug: bool
    profile: Any
    _environ: MutableMapping[str, str]
    
    def __init__(
        self,
        project_dir: Any,
        debug: bool = ...,
        profile: Any = ...,
        environ: Optional[MutableMapping[str, str]] = ...
    ) -> None: ...
    
    def create_botocore_session(
        self,
        connection_timeout: Any = ...,
        read_timeout: Any = ...,
        max_retries: Any = ...
    ) -> Session: ...
    
    def create_default_deployer(
        self,
        session: Any,
        config: Any,
        ui: Any
    ) -> Any: ...
    
    def create_plan_only_deployer(
        self,
        session: Any,
        config: Any,
        ui: Any
    ) -> Any: ...
    
    def create_deletion_deployer(
        self,
        session: Any,
        ui: Any
    ) -> Any: ...
    
    def create_deployment_reporter(self, ui: Any) -> Any: ...
    
    def create_config_obj(
        self,
        chalice_stage_name: str = ...,
        autogen_policy: Any = ...,
        api_gateway_stage: Any = ...,
        user_provided_params: Optional[Dict[str, Any]] = ...
    ) -> Config: ...
    
    def _validate_config_from_disk(self, config: Any) -> None: ...
    
    def create_app_packager(
        self,
        config: Any,
        options: Any,
        package_format: Any,
        template_format: Any,
        merge_template: Any = ...
    ) -> AppPackager: ...
    
    def create_log_retriever(
        self,
        session: Any,
        lambda_arn: Any,
        follow_logs: bool
    ) -> LogRetriever: ...
    
    def create_stdin_reader(self) -> PipeReader: ...
    
    def create_lambda_invoke_handler(
        self,
        name: Any,
        stage: Any
    ) -> LambdaInvokeHandler: ...
    
    def load_chalice_app(
        self,
        environment_variables: Optional[Dict[str, str]] = ...,
        validate_feature_flags: bool = ...
    ) -> Chalice: ...
    
    def load_project_config(self) -> Dict[str, Any]: ...
    
    def create_local_server(
        self,
        app_obj: Any,
        config: Any,
        host: Any,
        port: Any
    ) -> Any: ...
    
    def create_package_options(self) -> PackageOptions: ...
```