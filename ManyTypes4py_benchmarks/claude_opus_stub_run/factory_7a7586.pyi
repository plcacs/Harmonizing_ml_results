from __future__ import annotations
import logging
from botocore.session import Session
from typing import Any, Optional, Dict, MutableMapping
from chalice.app import Chalice
from chalice.config import Config
from chalice.package import AppPackager, PackageOptions
from chalice.logs import LogRetriever
from chalice.utils import UI, PipeReader
from chalice.deploy import deployer
from chalice.invoke import LambdaInvokeHandler

OptStr = Optional[str]
OptInt = Optional[int]

def create_botocore_session(
    profile: OptStr = ...,
    debug: bool = ...,
    connection_timeout: OptInt = ...,
    read_timeout: OptInt = ...,
    max_retries: OptInt = ...,
) -> Session: ...

def _add_chalice_user_agent(session: Session) -> None: ...

def _inject_large_request_body_filter() -> None: ...

class NoSuchFunctionError(Exception):
    name: str
    def __init__(self, name: str) -> None: ...

class UnknownConfigFileVersion(Exception):
    def __init__(self, version: str) -> None: ...

class LargeRequestBodyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool: ...

class CLIFactory:
    project_dir: str
    debug: bool
    profile: OptStr
    _environ: Dict[str, str]

    def __init__(
        self,
        project_dir: str,
        debug: bool = ...,
        profile: OptStr = ...,
        environ: Optional[Dict[str, str]] = ...,
    ) -> None: ...

    def create_botocore_session(
        self,
        connection_timeout: OptInt = ...,
        read_timeout: OptInt = ...,
        max_retries: OptInt = ...,
    ) -> Session: ...

    def create_default_deployer(
        self, session: Session, config: Config, ui: UI
    ) -> deployer.Deployer: ...

    def create_plan_only_deployer(
        self, session: Session, config: Config, ui: UI
    ) -> deployer.Deployer: ...

    def create_deletion_deployer(
        self, session: Session, ui: UI
    ) -> deployer.Deployer: ...

    def create_deployment_reporter(
        self, ui: UI
    ) -> deployer.DeploymentReporter: ...

    def create_config_obj(
        self,
        chalice_stage_name: str = ...,
        autogen_policy: Optional[bool] = ...,
        api_gateway_stage: OptStr = ...,
        user_provided_params: Optional[Dict[str, Any]] = ...,
    ) -> Config: ...

    def _validate_config_from_disk(self, config: Dict[str, Any]) -> None: ...

    def create_app_packager(
        self,
        config: Config,
        options: PackageOptions,
        package_format: str,
        template_format: str,
        merge_template: Optional[str] = ...,
    ) -> AppPackager: ...

    def create_log_retriever(
        self, session: Session, lambda_arn: str, follow_logs: bool
    ) -> LogRetriever: ...

    def create_stdin_reader(self) -> PipeReader: ...

    def create_lambda_invoke_handler(
        self, name: str, stage: str
    ) -> LambdaInvokeHandler: ...

    def load_chalice_app(
        self,
        environment_variables: Optional[MutableMapping[str, str]] = ...,
        validate_feature_flags: bool = ...,
    ) -> Chalice: ...

    def load_project_config(self) -> Dict[str, Any]: ...

    def create_local_server(
        self, app_obj: Chalice, config: Config, host: str, port: int
    ) -> local.LocalDevServer: ...

    def create_package_options(self) -> PackageOptions: ...