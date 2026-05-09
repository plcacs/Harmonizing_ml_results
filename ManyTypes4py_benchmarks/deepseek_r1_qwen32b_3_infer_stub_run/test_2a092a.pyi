from __future__ import annotations
from typing import Any, Dict, Generator, List, Optional, Union
from chalice import Chalice
from chalice.config import Config
from chalice.local import LocalGateway, LambdaContext
from types import TracebackType

class FunctionNotFoundError(Exception):
    ...

class Client:
    def __init__(self, app: Chalice, stage_name: str = ..., project_dir: str = ...):
        ...
    
    @property
    def http(self) -> TestHTTPClient:
        ...
    
    @property
    def lambda_(self) -> TestLambdaClient:
        ...
    
    @property
    def events(self) -> TestEventsClient:
        ...
    
    def __enter__(self) -> Client:
        ...
    
    def __exit__(self, exception_type: Optional[Type[BaseException]], exception_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        ...

class BaseClient:
    def _patched_env_vars(self, environment_variables: Dict[str, str]) -> Generator[None, None, None]:
        ...

class TestHTTPClient(BaseClient):
    def __init__(self, app: Chalice, config: Config):
        ...
    
    def request(self, method: str, path: str, headers: Optional[Dict[str, str]] = ..., body: bytes = ...) -> HTTPResponse:
        ...
    
    def get(self, path: str, **kwargs: Any) -> HTTPResponse:
        ...
    
    def post(self, path: str, **kwargs: Any) -> HTTPResponse:
        ...
    
    def put(self, path: str, **kwargs: Any) -> HTTPResponse:
        ...
    
    def patch(self, path: str, **kwargs: Any) -> HTTPResponse:
        ...
    
    def options(self, path: str, **kwargs: Any) -> HTTPResponse:
        ...
    
    def delete(self, path: str, **kwargs: Any) -> HTTPResponse:
        ...
    
    def head(self, path: str, **kwargs: Any) -> HTTPResponse:
        ...

class HTTPResponse:
    def __init__(self, body: bytes, headers: Dict[str, str], status_code: int):
        ...
    
    @property
    def json_body(self) -> Optional[Dict[Any, Any]]:
        ...
    
    @classmethod
    def create_from_dict(cls, response_dict: Dict[str, Any]) -> HTTPResponse:
        ...

class TestEventsClient(BaseClient):
    def __init__(self, app: Chalice):
        ...
    
    def generate_sns_event(self, message: str, subject: str = ..., message_attributes: Optional[Dict[str, Dict[str, Union[str, Dict[str, str]]]]] = ...) -> Dict[str, Any]:
        ...
    
    def generate_s3_event(self, bucket: str, key: str, event_name: str = ...) -> Dict[str, Any]:
        ...
    
    def generate_sqs_event(self, message_bodies: List[str], queue_name: str = ...) -> Dict[str, Any]:
        ...
    
    def generate_cw_event(self, source: str, detail_type: str, detail: Dict[str, Any], resources: List[str], region: str = ...) -> Dict[str, Any]:
        ...
    
    def generate_kinesis_event(self, message_bodies: List[str], stream_name: str = ...) -> Dict[str, Any]:
        ...

class TestLambdaClient(BaseClient):
    def __init__(self, app: Chalice, config: Config):
        ...
    
    def invoke(self, function_name: str, payload: Optional[Dict[str, Any]] = ...) -> InvokeResponse:
        ...

class InvokeResponse:
    def __init__(self, payload: Any):
        ...
    
    @property
    def payload(self) -> Any:
        ...