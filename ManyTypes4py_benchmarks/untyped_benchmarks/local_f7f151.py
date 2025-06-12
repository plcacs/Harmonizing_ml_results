"""Dev server used for running a chalice app locally.

This is intended only for local development purposes.

"""
from __future__ import print_function
from __future__ import annotations
import re
import threading
import time
import uuid
import base64
import functools
import warnings
from collections import namedtuple
import json
from six.moves.BaseHTTPServer import HTTPServer
from six.moves.BaseHTTPServer import BaseHTTPRequestHandler
from six.moves.socketserver import ThreadingMixIn
from typing import List, Any, Dict, Tuple, Callable, Optional, Union
from chalice.app import Chalice
from chalice.app import CORSConfig
from chalice.app import ChaliceAuthorizer
from chalice.app import CognitoUserPoolAuthorizer
from chalice.app import RouteEntry
from chalice.app import Request
from chalice.app import AuthResponse
from chalice.app import BuiltinAuthConfig
from chalice.config import Config
from chalice.compat import urlparse, parse_qs
MatchResult = namedtuple('MatchResult', ['route', 'captured', 'query_params'])
EventType = Dict[str, Any]
ContextType = Dict[str, Any]
HeaderType = Dict[str, Any]
ResponseType = Dict[str, Any]
HandlerCls = Callable[..., 'ChaliceRequestHandler']
ServerCls = Callable[..., 'HTTPServer']

class Clock(object):

    def time(self):
        return time.time()

def create_local_server(app_obj, config, host, port):
    CustomLocalChalice.__bases__ = (LocalChalice, app_obj.__class__)
    app_obj.__class__ = CustomLocalChalice
    return LocalDevServer(app_obj, config, host, port)

class LocalARNBuilder(object):
    ARN_FORMAT = 'arn:aws:execute-api:{region}:{account_id}:{api_id}/{stage}/{method}/{resource_path}'
    LOCAL_REGION = 'mars-west-1'
    LOCAL_ACCOUNT_ID = '123456789012'
    LOCAL_API_ID = 'ymy8tbxw7b'
    LOCAL_STAGE = 'api'

    def build_arn(self, method, path):
        if path != '/':
            path = path[1:]
        path = path.split('?')[0]
        return self.ARN_FORMAT.format(region=self.LOCAL_REGION, account_id=self.LOCAL_ACCOUNT_ID, api_id=self.LOCAL_API_ID, stage=self.LOCAL_STAGE, method=method, resource_path=path)

class ARNMatcher(object):

    def __init__(self, target_arn):
        self._arn = target_arn

    def _resource_match(self, resource):
        escaped_resource = re.escape(resource)
        resource_regex = escaped_resource.replace('\\?', '.').replace('\\*', '.*?')
        resource_regex = '^%s$' % resource_regex
        return re.match(resource_regex, self._arn) is not None

    def does_any_resource_match(self, resources):
        for resource in resources:
            if self._resource_match(resource):
                return True
        return False

class RouteMatcher(object):

    def __init__(self, route_urls):
        self.route_urls = sorted(route_urls)

    def match_route(self, url):
        """Match the url against known routes.

        This method takes a concrete route "/foo/bar", and
        matches it against a set of routes.  These routes can
        use param substitution corresponding to API gateway patterns.
        For example::

            match_route('/foo/bar') -> '/foo/{name}'

        """
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query, keep_blank_values=True)
        path = parsed_url.path
        if path != '/' and path.endswith('/'):
            path = path[:-1]
        parts = path.split('/')
        captured = {}
        for route_url in self.route_urls:
            url_parts = route_url.split('/')
            if len(parts) == len(url_parts):
                for i, j in zip(parts, url_parts):
                    if j.startswith('{') and j.endswith('}'):
                        captured[j[1:-1]] = i
                        continue
                    if i != j:
                        break
                else:
                    return MatchResult(route_url, captured, query_params)
        raise ValueError('No matching route found for: %s' % url)

class LambdaEventConverter(object):
    LOCAL_SOURCE_IP = '127.0.0.1'
    'Convert an HTTP request to an event dict used by lambda.'

    def __init__(self, route_matcher, binary_types=None):
        self._route_matcher = route_matcher
        if binary_types is None:
            binary_types = []
        self._binary_types = binary_types

    def _is_binary(self, headers):
        return headers.get('content-type', '') in self._binary_types

    def create_lambda_event(self, method, path, headers, body=None):
        view_route = self._route_matcher.match_route(path)
        event = {'requestContext': {'httpMethod': method, 'resourcePath': view_route.route, 'identity': {'sourceIp': self.LOCAL_SOURCE_IP}, 'path': path.split('?')[0]}, 'headers': {k.lower(): v for k, v in headers.items()}, 'pathParameters': view_route.captured, 'stageVariables': {}}
        if view_route.query_params:
            event['multiValueQueryStringParameters'] = view_route.query_params
        else:
            event['multiValueQueryStringParameters'] = None
        if self._is_binary(headers) and body is not None:
            event['body'] = base64.b64encode(body).decode('ascii')
            event['isBase64Encoded'] = True
        else:
            event['body'] = body
        return event

class LocalGatewayException(Exception):
    CODE = 0

    def __init__(self, headers, body=None):
        self.headers = headers
        self.body = body

class InvalidAuthorizerError(LocalGatewayException):
    CODE = 500

class ForbiddenError(LocalGatewayException):
    CODE = 403

class NotAuthorizedError(LocalGatewayException):
    CODE = 401

class LambdaContext(object):

    def __init__(self, function_name, memory_size, max_runtime_ms=3000, time_source=None):
        if time_source is None:
            time_source = Clock()
        self._time_source = time_source
        self._start_time = self._current_time_millis()
        self._max_runtime = max_runtime_ms
        self.function_name = function_name
        self.function_version = '$LATEST'
        self.invoked_function_arn = ''
        self.memory_limit_in_mb = memory_size
        self.aws_request_id = str(uuid.uuid4())
        self.log_group_name = ''
        self.log_stream_name = ''
        self.identity = None
        self.client_context = None

    def _current_time_millis(self):
        return self._time_source.time() * 1000

    def get_remaining_time_in_millis(self):
        runtime = self._current_time_millis() - self._start_time
        return self._max_runtime - runtime
LocalAuthPair = Tuple[EventType, LambdaContext]

class LocalGatewayAuthorizer(object):
    """A class for running user defined authorizers in local mode."""

    def __init__(self, app_object):
        self._app_object = app_object
        self._arn_builder = LocalARNBuilder()

    def authorize(self, raw_path, lambda_event, lambda_context):
        method = lambda_event['requestContext']['httpMethod']
        route_entry = self._route_for_event(lambda_event)
        if not route_entry:
            return (lambda_event, lambda_context)
        authorizer = route_entry.authorizer
        if not authorizer:
            return (lambda_event, lambda_context)
        if isinstance(authorizer, CognitoUserPoolAuthorizer):
            if 'headers' in lambda_event and 'authorization' in lambda_event['headers']:
                token = lambda_event['headers']['authorization']
                claims = self._decode_jwt_payload(token)
                try:
                    cognito_username = claims['cognito:username']
                except KeyError:
                    warnings.warn('%s for machine-to-machine communicaiton is not supported in local mode. All requests made against a route will be authorized to allow local testing.' % authorizer.__class__.__name__)
                    return (lambda_event, lambda_context)
                auth_result = {'context': {'claims': claims}, 'principalId': cognito_username}
                lambda_event = self._update_lambda_event(lambda_event, auth_result)
        if not isinstance(authorizer, ChaliceAuthorizer):
            warnings.warn('%s is not a supported in local mode. All requests made against a route will be authorized to allow local testing.' % authorizer.__class__.__name__)
            return (lambda_event, lambda_context)
        arn = self._arn_builder.build_arn(method, raw_path)
        auth_event = self._prepare_authorizer_event(arn, lambda_event, lambda_context)
        auth_result = authorizer(auth_event, lambda_context)
        if auth_result is None:
            raise InvalidAuthorizerError({'x-amzn-RequestId': lambda_context.aws_request_id, 'x-amzn-ErrorType': 'AuthorizerConfigurationException'}, b'{"message":null}')
        authed = self._check_can_invoke_view_function(arn, auth_result)
        if authed:
            lambda_event = self._update_lambda_event(lambda_event, auth_result)
        else:
            raise ForbiddenError({'x-amzn-RequestId': lambda_context.aws_request_id, 'x-amzn-ErrorType': 'AccessDeniedException'}, b'{"Message": "User is not authorized to access this resource"}')
        return (lambda_event, lambda_context)

    def _check_can_invoke_view_function(self, arn, auth_result):
        policy = auth_result.get('policyDocument', {})
        statements = policy.get('Statement', [])
        allow_resource_statements = []
        for statement in statements:
            if statement.get('Effect') == 'Allow' and (statement.get('Action') == 'execute-api:Invoke' or 'execute-api:Invoke' in statement.get('Action')):
                for resource in statement.get('Resource'):
                    allow_resource_statements.append(resource)
        arn_matcher = ARNMatcher(arn)
        return arn_matcher.does_any_resource_match(allow_resource_statements)

    def _route_for_event(self, lambda_event):
        resource_path = lambda_event.get('requestContext', {}).get('resourcePath')
        http_method = lambda_event['requestContext']['httpMethod']
        try:
            route_entry = self._app_object.routes[resource_path][http_method]
        except KeyError:
            return None
        return route_entry

    def _update_lambda_event(self, lambda_event, auth_result):
        auth_context = auth_result['context']
        auth_context.update({'principalId': auth_result['principalId']})
        lambda_event['requestContext']['authorizer'] = auth_context
        return lambda_event

    def _prepare_authorizer_event(self, arn, lambda_event, lambda_context):
        """Translate event for an authorizer input."""
        authorizer_event = lambda_event.copy()
        authorizer_event['type'] = 'TOKEN'
        try:
            authorizer_event['authorizationToken'] = authorizer_event.get('headers', {})['authorization']
        except KeyError:
            raise NotAuthorizedError({'x-amzn-RequestId': lambda_context.aws_request_id, 'x-amzn-ErrorType': 'UnauthorizedException'}, b'{"message":"Unauthorized"}')
        authorizer_event['methodArn'] = arn
        return authorizer_event

    def _decode_jwt_payload(self, jwt):
        payload_segment = jwt.split('.', 2)[1]
        payload = base64.urlsafe_b64decode(self._base64_pad(payload_segment))
        return json.loads(payload)

    def _base64_pad(self, value):
        rem = len(value) % 4
        if rem > 0:
            value += '=' * (4 - rem)
        return value

class LocalGateway(object):
    """A class for faking the behavior of API Gateway."""
    MAX_LAMBDA_EXECUTION_TIME = 900

    def __init__(self, app_object, config):
        self._app_object = app_object
        self._config = config
        self.event_converter = LambdaEventConverter(RouteMatcher(list(app_object.routes)), self._app_object.api.binary_types)
        self._authorizer = LocalGatewayAuthorizer(app_object)

    def _generate_lambda_context(self):
        if self._config.lambda_timeout is None:
            timeout = self.MAX_LAMBDA_EXECUTION_TIME * 1000
        else:
            timeout = self._config.lambda_timeout * 1000
        return LambdaContext(function_name=self._config.function_name, memory_size=self._config.lambda_memory_size, max_runtime_ms=timeout)

    def _generate_lambda_event(self, method, path, headers, body):
        lambda_event = self.event_converter.create_lambda_event(method=method, path=path, headers=headers, body=body)
        return lambda_event

    def _has_user_defined_options_method(self, lambda_event):
        route_key = lambda_event['requestContext']['resourcePath']
        return 'OPTIONS' in self._app_object.routes[route_key]

    def handle_request(self, method, path, headers, body):
        lambda_context = self._generate_lambda_context()
        try:
            lambda_event = self._generate_lambda_event(method, path, headers, body)
        except ValueError:
            error_headers = {'x-amzn-RequestId': lambda_context.aws_request_id, 'x-amzn-ErrorType': 'UnauthorizedException'}
            auth_header = headers.get('authorization')
            if auth_header is None:
                auth_header = headers.get('Authorization')
            if auth_header is not None:
                raise ForbiddenError(error_headers, b'{"message": "Authorization header requires \'Credential\' parameter. Authorization header requires \'Signature\' parameter. Authorization header requires \'SignedHeaders\' parameter. Authorization header requires existence of either a \'X-Amz-Date\' or a \'Date\' header. Authorization=%s"}' % auth_header.encode('ascii'))
            raise ForbiddenError(error_headers, b'{"message": "Missing Authentication Token"}')
        if method == 'OPTIONS' and (not self._has_user_defined_options_method(lambda_event)):
            options_headers = self._autogen_options_headers(lambda_event)
            return {'statusCode': 200, 'headers': options_headers, 'multiValueHeaders': {}, 'body': None}
        lambda_event, lambda_context = self._authorizer.authorize(path, lambda_event, lambda_context)
        response = self._app_object(lambda_event, lambda_context)
        return response

    def _autogen_options_headers(self, lambda_event):
        route_key = lambda_event['requestContext']['resourcePath']
        route_dict = self._app_object.routes[route_key]
        route_methods = [method for method in route_dict.keys() if route_dict[method].cors is not None]
        if not route_methods:
            return {'Access-Control-Allow-Methods': 'OPTIONS'}
        cors_config = route_dict[route_methods[0]].cors
        cors_headers = cors_config.get_access_control_headers()
        route_methods.append('OPTIONS')
        cors_headers.update({'Access-Control-Allow-Methods': '%s' % ','.join(route_methods)})
        return cors_headers

class ChaliceRequestHandler(BaseHTTPRequestHandler):
    """A class for mapping raw HTTP events to and from LocalGateway."""
    protocol_version = 'HTTP/1.1'

    def __init__(self, request, client_address, server, app_object, config):
        self.local_gateway = LocalGateway(app_object, config)
        BaseHTTPRequestHandler.__init__(self, request, client_address, server)

    def _parse_payload(self):
        body = None
        content_length = int(self.headers.get('content-length', '0'))
        if content_length > 0:
            body = self.rfile.read(content_length)
        converted_headers = dict(self.headers)
        return (converted_headers, body)

    def _generic_handle(self):
        headers, body = self._parse_payload()
        try:
            response = self.local_gateway.handle_request(method=self.command, path=self.path, headers=headers, body=body)
            status_code = response['statusCode']
            headers = response['headers'].copy()
            headers.update(response['multiValueHeaders'])
            response = self._handle_binary(response)
            body = response['body']
            self._send_http_response(status_code, headers, body)
        except LocalGatewayException as e:
            self._send_error_response(e)

    def _handle_binary(self, response):
        if response.get('isBase64Encoded'):
            body = base64.b64decode(response['body'])
            response['body'] = body
        return response

    def _send_error_response(self, error):
        code = error.CODE
        headers = error.headers
        body = error.body
        self._send_http_response(code, headers, body)

    def _send_http_response(self, code, headers, body):
        if body is None:
            self._send_http_response_no_body(code, headers)
        else:
            self._send_http_response_with_body(code, headers, body)

    def _send_http_response_with_body(self, code, headers, body):
        self.send_response(code)
        if not isinstance(body, bytes):
            body = body.encode('utf-8')
        self.send_header('Content-Length', str(len(body)))
        content_type = headers.pop('Content-Type', 'application/json')
        self.send_header('Content-Type', content_type)
        self._send_headers(headers)
        self.wfile.write(body)
    do_GET = do_PUT = do_POST = do_HEAD = do_DELETE = do_PATCH = do_OPTIONS = _generic_handle

    def _send_http_response_no_body(self, code, headers):
        headers['Content-Length'] = '0'
        self.send_response(code)
        self._send_headers(headers)

    def _send_headers(self, headers):
        for header_name, header_value in headers.items():
            if isinstance(header_value, list):
                for value in header_value:
                    self.send_header(header_name, value)
            else:
                self.send_header(header_name, header_value)
        self.end_headers()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Threading mixin to better support browsers.

    When a browser sends a GET request to Chalice it keeps the connection open
    for reuse. In the single threaded model this causes Chalice local to become
    unresponsive to all clients other than that browser socket. Even sending a
    header requesting that the client close the connection is not good enough,
    the browswer will simply open another one and sit on it.
    """
    daemon_threads = True

class LocalDevServer(object):

    def __init__(self, app_object, config, host, port, handler_cls=ChaliceRequestHandler, server_cls=ThreadedHTTPServer):
        self.app_object = app_object
        self.host = host
        self.port = port
        self._wrapped_handler = functools.partial(handler_cls, app_object=app_object, config=config)
        self.server = server_cls((host, port), self._wrapped_handler)

    def handle_single_request(self):
        self.server.handle_request()

    def serve_forever(self):
        print('Serving on http://%s:%s' % (self.host, self.port))
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

class HTTPServerThread(threading.Thread):
    """Thread that manages starting/stopping local HTTP server.

    This is a small wrapper around a normal threading.Thread except
    that it adds shutdown capability of the HTTP server, which is
    not part of the normal threading.Thread interface.

    """

    def __init__(self, server_factory):
        threading.Thread.__init__(self)
        self._server_factory = server_factory
        self._server = None
        self.daemon = True

    def run(self):
        self._server = self._server_factory()
        self._server.serve_forever()

    def shutdown(self):
        if self._server is not None:
            self._server.shutdown()

class LocalChalice(Chalice):
    _THREAD_LOCAL = threading.local()

    @property
    def current_request(self):
        return self._THREAD_LOCAL.current_request

    @current_request.setter
    def current_request(self, value):
        self._THREAD_LOCAL.current_request = value

class CustomLocalChalice(LocalChalice):
    pass