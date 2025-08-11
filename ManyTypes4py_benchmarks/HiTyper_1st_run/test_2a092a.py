from __future__ import annotations
import os
import json
import base64
import contextlib
from types import TracebackType
from typing import Optional, Type, Generator, Dict, Any, List
from chalice import Chalice
from chalice.config import Config
from chalice.local import LocalGateway, LambdaContext, LocalGatewayException
from chalice.cli.factory import CLIFactory

class FunctionNotFoundError(Exception):
    pass

class Client(object):

    def __init__(self, app: Union[Config, typing.Callable], stage_name='dev', project_dir='.') -> None:
        self._app = app
        self._project_dir = project_dir
        self._stage_name = stage_name
        self._http_client = None
        self._events_client = None
        self._lambda_client = None
        self._chalice_config_obj = None
        self._cli_factory = CLIFactory(project_dir)

    @property
    def _chalice_config(self) -> Union[dict[str, typing.Union[str,None]], telepresence.cli.PortMapping, dict[str, dict[str, typing.Any]]]:
        if self._chalice_config_obj is None:
            try:
                self._chalice_config_obj = self._cli_factory.create_config_obj(chalice_stage_name=self._stage_name)
            except RuntimeError:
                self._chalice_config_obj = Config.create()
        return self._chalice_config_obj

    @property
    def http(self) -> TestHTTPClient:
        if self._http_client is None:
            self._http_client = TestHTTPClient(self._app, self._chalice_config)
        return self._http_client

    @property
    def lambda_(self) -> Union[str, None, list[str], TestLambdaClient]:
        if self._lambda_client is None:
            self._lambda_client = TestLambdaClient(self._app, self._chalice_config)
        return self._lambda_client

    @property
    def events(self) -> TestEventsClient:
        if self._events_client is None:
            self._events_client = TestEventsClient(self._app)
        return self._events_client

    def __enter__(self) -> Client:
        return self

    def __exit__(self, exception_type: Union[types.TracebackType, None, typing.Any, BaseException], exception_value: Union[types.TracebackType, None, typing.Any, BaseException], traceback: Union[types.TracebackType, None, typing.Any, BaseException]) -> None:
        pass

class BaseClient(object):

    @contextlib.contextmanager
    def _patched_env_vars(self, environment_variables: Union[dict[str, list[typing.Callable]], dict[str, str]]) -> typing.Generator:
        original = os.environ
        patched = os.environ.copy()
        patched.update(environment_variables)
        os.environ = patched
        try:
            yield
        finally:
            os.environ = original

class TestHTTPClient(BaseClient):

    def __init__(self, app: Union[Config, typing.Callable], config: Union[list[str], None, dict, pyramid.config.Configurator]) -> None:
        self._app = app
        self._config = config
        self._local_gateway = LocalGateway(app, self._config)

    def request(self, method: Union[str, bytes], path: Union[str, None, bytes], headers: Union[None, str, dict]=None, body: bytes=b''):
        if headers is None:
            headers = {}
        scoped = self._config.scope(self._config.chalice_stage, 'api_handler')
        with self._patched_env_vars(scoped.environment_variables):
            try:
                response = self._local_gateway.handle_request(method=method.upper(), path=path, headers=headers, body=body)
            except LocalGatewayException as e:
                return self._error_response(e)
        return HTTPResponse.create_from_dict(response)

    def _error_response(self, e: Union[list, dict[str, typing.Any], tuple[typing.Union[str,int]]]) -> HTTPResponse:
        return HTTPResponse(headers=e.headers, body=e.body if e.body else b'', status_code=e.CODE)

    def get(self, path: str, **kwargs):
        return self.request('GET', path, **kwargs)

    def post(self, path: str, **kwargs):
        return self.request('POST', path, **kwargs)

    def put(self, path: str, **kwargs):
        return self.request('PUT', path, **kwargs)

    def patch(self, path: str, **kwargs):
        return self.request('PATCH', path, **kwargs)

    def options(self, path: Union[str, typing.Sequence[str]], **kwargs):
        return self.request('OPTIONS', path, **kwargs)

    def delete(self, path: str, **kwargs):
        return self.request('DELETE', path, **kwargs)

    def head(self, path: str, **kwargs):
        return self.request('HEAD', path, **kwargs)

class HTTPResponse(object):

    def __init__(self, body, headers, status_code) -> None:
        self.body = body
        self.headers = headers
        self.status_code = status_code

    @property
    def json_body(self) -> None:
        try:
            return json.loads(self.body)
        except ValueError:
            return None

    @classmethod
    def create_from_dict(cls: Union[dict, dict[str, typing.Any], dict[str, str]], response_dict: Union[dict, dict[str, typing.Any]]) -> Union[tuple[typing.Union[dict,bool]], mypy.types.Any, dict, None]:
        if response_dict.get('isBase64Encoded', False):
            body = base64.b64decode(response_dict['body'])
        else:
            body = response_dict['body'].encode('utf-8')
        combined_headers = response_dict['headers']
        combined_headers.update(response_dict['multiValueHeaders'])
        return cls(body=body, status_code=response_dict['statusCode'], headers=combined_headers)

class TestEventsClient(BaseClient):

    def __init__(self, app: Union[Config, typing.Callable]) -> None:
        self._app = app

    def generate_sns_event(self, message: Union[str, bool, tuple[typing.Union[str,int]]], subject: typing.Text='', message_attributes: Union[None, str, dict]=None) -> dict[typing.Text, list[dict[typing.Text, typing.Union[typing.Text,dict[typing.Text, typing.Union[str,bool,tuple[typing.Union[str,int]],set[str],typing.Type,dict[typing.Text, dict[typing.Text, typing.Text]]]]]]]]:
        if message_attributes is None:
            message_attributes = {'AttributeKey': {'Type': 'String', 'Value': 'AttributeValue'}}
        sns_event = {'Records': [{'EventSource': 'aws:sns', 'EventSubscriptionArn': 'arn:subscription-arn', 'EventVersion': '1.0', 'Sns': {'Message': message, 'MessageAttributes': message_attributes, 'MessageId': 'abcdefgh-51e4-5ae2-9964-b296c8d65d1a', 'Signature': 'signature', 'SignatureVersion': '1', 'SigningCertUrl': 'https://sns.us-west-2.amazonaws.com/cert.pem', 'Subject': subject, 'Timestamp': '2018-06-26T19:41:38.695Z', 'TopicArn': 'arn:aws:sns:us-west-2:12345:TopicName', 'Type': 'Notification', 'UnsubscribeUrl': 'https://unsubscribe-url/'}}]}
        return sns_event

    def generate_s3_event(self, bucket: str, key: str, event_name: typing.Text='ObjectCreated:Put') -> dict[typing.Text, list[dict[typing.Text, typing.Union[typing.Text,dict[typing.Text, typing.Union[dict[typing.Text, typing.Union[typing.Text,dict[typing.Text, typing.Text]]],typing.Text,dict[typing.Text, typing.Union[typing.Text,int]]]]]]]]:
        s3_event = {'Records': [{'awsRegion': 'us-west-2', 'eventName': event_name, 'eventSource': 'aws:s3', 'eventTime': '2018-05-22T04:41:23.823Z', 'eventVersion': '2.0', 'requestParameters': {'sourceIPAddress': '1.1.1.1'}, 'responseElements': {'x-amz-id-2': 'request-id-2', 'x-amz-request-id': 'request-id-1'}, 's3': {'bucket': {'arn': 'arn:aws:s3:::%s' % bucket, 'name': bucket, 'ownerIdentity': {'principalId': 'ABCD'}}, 'configurationId': 'config-id', 'object': {'eTag': 'd41d8cd98f00b204e9800998ecf8427e', 'key': key, 'sequencer': '005B039F73C627CE8B', 'size': 0}, 's3SchemaVersion': '1.0'}, 'userIdentity': {'principalId': 'AWS:XYZ'}}]}
        return s3_event

    def generate_sqs_event(self, message_bodies: Union[str, bytes], queue_name: typing.Text='queue-name') -> dict[typing.Text, list[dict[typing.Text, typing.Union[dict[typing.Text, typing.Text],typing.Text]]]]:
        records = [{'attributes': {'ApproximateFirstReceiveTimestamp': '1530576251596', 'ApproximateReceiveCount': '1', 'SenderId': 'sender-id', 'SentTimestamp': '1530576251595'}, 'awsRegion': 'us-west-2', 'body': body, 'eventSource': 'aws:sqs', 'eventSourceARN': 'arn:aws:sqs:us-west-2:12345:%s' % queue_name, 'md5OfBody': '754ac2f7a12df38320e0c5eafd060145', 'messageAttributes': {}, 'messageId': 'message-id', 'receiptHandle': 'receipt-handle'} for body in message_bodies]
        sqs_event = {'Records': records}
        return sqs_event

    def generate_cw_event(self, source: Union[str, dict[str, bool]], detail_type: Union[str, dict[str, bool]], detail: Union[str, dict[str, bool]], resources: Union[str, dict[str, bool]], region: typing.Text='us-west-2') -> dict[typing.Text, typing.Union[int,typing.Text,dict[str, bool]]]:
        event = {'version': 0, 'id': '7bf73129-1428-4cd3-a780-95db273d1602', 'detail-type': detail_type, 'source': source, 'account': '123456789012', 'time': '2015-11-11T21:29:54Z', 'region': region, 'resources': resources, 'detail': detail}
        return event

    def generate_kinesis_event(self, message_bodies: Union[str, bytes], stream_name: typing.Text='stream-name') -> dict[typing.Text, list[dict[typing.Text, typing.Union[dict[typing.Text, typing.Union[typing.Text,float]],typing.Text]]]]:
        records = [{'kinesis': {'kinesisSchemaVersion': '1.0', 'partitionKey': '1', 'sequenceNumber': '12345', 'data': base64.b64encode(body).decode('ascii'), 'approximateArrivalTimestamp': 1545084650.987}, 'eventSource': 'aws:kinesis', 'eventVersion': '1.0', 'eventID': 'shardId-000000000006:12345', 'eventName': 'aws:kinesis:record', 'invokeIdentityArn': 'arn:aws:iam::123:role/lambda-role', 'awsRegion': 'us-west-2', 'eventSourceARN': 'arn:aws:kinesis:us-east-2:123:stream/%s' % stream_name} for body in message_bodies]
        return {'Records': records}

class TestLambdaClient(BaseClient):

    def __init__(self, app: Union[Config, typing.Callable], config: Union[list[str], None, dict, pyramid.config.Configurator]) -> None:
        self._app = app
        self._config = config

    def invoke(self, function_name: Union[lightbus.config.Config, dict[str, typing.Any], str], payload: Union[None, dict[str, typing.Any], str]=None) -> InvokeResponse:
        if payload is None:
            payload = {}
        scoped = self._config.scope(self._config.chalice_stage, function_name)
        lambda_context = LambdaContext(function_name, memory_size=scoped.lambda_memory_size)
        if function_name not in self._app.handler_map:
            raise FunctionNotFoundError(function_name)
        with self._patched_env_vars(scoped.environment_variables):
            response = self._app.handler_map[function_name](payload, lambda_context)
        return InvokeResponse(payload=response)

class InvokeResponse(object):

    def __init__(self, payload) -> None:
        self.payload = payload