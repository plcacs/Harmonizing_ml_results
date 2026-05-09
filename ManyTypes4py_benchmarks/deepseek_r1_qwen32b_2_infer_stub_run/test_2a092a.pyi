from __future__ import annotations
from typing import Dict, Optional, Generator, Any, List, Type, Union
from chalice import Chalice
from chalice.config import Config
from chalice.local import LocalGateway, LambdaContext

class FunctionNotFoundError(Exception):
    ...

class Client:
    def __init__(self, app: Chalice, stage_name: str = 'dev', project_dir: str = '.') -> None:
        ...
    
    @property
    def _chalice_config(self) -> Config:
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
    
    def __enter__(self) -> 'Client':
        ...
    
    def __exit__(self, exception_type: Optional[Type[BaseException]], exception_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        ...

class BaseClient:
    @contextlib.contextmanager
    def _patched_env_vars(self, environment_variables: Dict[str, str]) -> Generator[None, None, None]:
        ...

class TestHTTPClient(BaseClient):
    def __init__(self, app: Chalice, config: Config) -> None:
        ...
    
    def request(self, method: str, path: str, headers: Optional[Dict[str, str]] = None, body: bytes = b'') -> HTTPResponse:
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
    def __init__(self, body: bytes, headers: Dict[str, str], status_code: int) -> None:
        ...
    
    @property
    def json_body(self) -> Optional[Dict[str, Any]]:
        ...
    
    @classmethod
    def create_from_dict(cls, response_dict: Dict[str, Any]) -> 'HTTPResponse':
        ...

class TestEventsClient(BaseClient):
    def __init__(self, app: Chalice) -> None:
        ...
    
    def generate_sns_event(self, message: str, subject: str = '', message_attributes: Optional[Dict[str, Dict[str, Union[str, bool]]]] = None) -> Dict[str, Any]:
        ...
    
    def generate_s3_event(self, bucket: str, key: str, event_name: str = 'ObjectCreated:Put') -> Dict[str, Any]:
        ...
    
    def generate_sqs_event(self, message_bodies: List[str], queue_name: str = 'queue-name') -> Dict[str, Any]:
        ...
    
    def generate_cw_event(self, source: str, detail_type: str, detail: Dict[str, Any], resources: List[str], region: str = 'us-west-2') -> Dict[str, Any]:
        ...
    
    def generate_kinesis_event(self, message_bodies: List[str], stream_name: str = 'stream-name') -> Dict[str, Any]:
        ...

class TestLambdaClient(BaseClient):
    def __init__(self, app: Chalice, config: Config) -> None:
        ...
    
    def invoke(self, function_name: str, payload: Optional[Dict[str, Any]] = None) -> InvokeResponse:
        ...

class InvokeResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        ...
    
    @property
    def payload(self) -> Dict[str, Any]:
        ...