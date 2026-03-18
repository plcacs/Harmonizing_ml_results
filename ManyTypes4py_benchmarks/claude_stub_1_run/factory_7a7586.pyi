```pyi
from __future__ import annotations

import logging
from typing import Any, Optional, Dict, MutableMapping, cast

from botocore.config import Config as BotocoreConfig
from botocore.session import Session
from chalice.app import Chalice
from chalice.config import Config, DeployedResources
from chalice.package import AppPackager, PackageOptions
from chalice.logs import LogRetriever, BaseLogEventGenerator
from chalice.invoke import LambdaInvokeHandler
from chalice.utils import UI, PipeReader
from chalice.deploy import deployer

OptStr = Optional[str]
OptInt = Optional[int]

def create_botocore_session(
    profile: Optional[str] = None,
    debug: bool = False,
    connection_timeout: Optional[int] = None,
    read_timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
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
    profile: Optional[str]
    _environ: Dict[str, str]

    def __init__(
        self,
        project_dir: str,
        debug: bool = False,
        profile: Optional[str] = None,
        environ: Optional[Dict[str, str]] = None,
    ) -> None: ...

    def create_botocore_session(
        self,
        connection_timeout: Optional[int] = None,
        read_timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> Session: ...

    def create_default_deployer(self, session: Session, config: Config, ui: UI) -> Any: ...

    def create_plan_only_deployer(self, session: Session, config: Config, ui: UI) -> Any: ...

    def create_deletion_deployer(self, session: Session, ui: UI) -> Any: ...

    def create_deployment_reporter(self, ui: UI) -> deployer.DeploymentReporter: ...

    def create_config_obj(
        self,
        chalice_stage_name: str = ...,
        autogen_policy: Optional[bool] = None,
        api_gateway_stage: Optional[str] = None,
        user_provided_params: Optional[Dict[str, Any]] = None,
    ) -> Config: ...

    def _validate_config_from_disk(self, config: Dict[str, Any]) -> None: ...

    def create_app_packager(
        self,
        config: Config,
        options: PackageOptions,
        package_format: str,
        template_format: str,
        merge_template: Optional[str] = None,
    ) -> AppPackager: ...

    def create_log_retriever(
        self,
        session: Session,
        lambda_arn: str,
        follow_logs: bool,
    ) -> LogRetriever: ...

    def create_stdin_reader(self) -> PipeReader: ...

    def create_lambda_invoke_handler(self, name: str, stage: str) -> LambdaInvokeHandler: ...

    def load_chalice_app(
        self,
        environment_variables: Optional[Dict[str, str]] = None,
        validate_feature_flags: bool = True,
    ) -> Chalice: ...

    def load_project_config(self) -> Dict[str, Any]: ...

    def create_local_server(
        self,
        app_obj: Chalice,
        config: Config,
        host: str,
        port: int,
    ) -> Any: ...

    def create_package_options(self) -> PackageOptions: ...
```