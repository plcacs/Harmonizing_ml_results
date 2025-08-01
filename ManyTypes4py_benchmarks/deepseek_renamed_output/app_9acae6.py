"""Chalice app and routing code."""
import re
import sys
import os
import logging
import json
import traceback
import decimal
import base64
import copy
import functools
import datetime
from collections import defaultdict
from urllib.parse import unquote_plus
from collections.abc import Mapping, MutableMapping
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, 
    Type, TypeVar, Union, Generic, cast, overload
)
from typing_extensions import Protocol

__version__ = '1.31.4'

if TYPE_CHECKING:
    from chalice.local import LambdaContext

T = TypeVar('T')
R = TypeVar('R')
HandlerType = TypeVar('HandlerType', bound=Callable[..., Any])
EventType = TypeVar('EventType', bound='BaseLambdaEvent')
MiddlewareType = Callable[[Any, Callable[[Any], Any]], Any]

_PARAMS = re.compile('{\\w+}')
MiddlewareFuncType = Callable[[Any, Callable[[Any], Any]], Any]
UserHandlerFuncType = Callable[..., Any]
HeadersType = Dict[str, Union[str, List[str]]]
_ANY_STRING = (str, bytes)


def func_b51am1qp(obj: Any) -> Any:
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, MultiDict):
        return dict(obj)
    raise TypeError('Object of type %s is not JSON serializable' % obj.
        __class__.__name__)


def func_x2u6x5cg(message: str, error_code: str, http_status_code: int,
                 headers: Optional[Dict[str, str]] = None) -> 'Response':
    body = {'Code': error_code, 'Message': message}
    response = Response(body=body, status_code=http_status_code, headers=
        headers)
    return response


def func_i6qpvb6g(content_type: str, valid_content_types: List[str]) -> bool:
    content_type = content_type.lower()
    valid_content_types = [x.lower() for x in valid_content_types]
    return ('*/*' in content_type or '*/*' in valid_content_types or
        _content_type_header_contains(content_type, valid_content_types))


def func_a8514pra(content_type_header: str, valid_content_types: List[str]) -> bool:
    content_type_header_parts = [p.strip() for p in re.split('[,;]',
        content_type_header)]
    valid_parts = set(valid_content_types).intersection(
        content_type_header_parts)
    return len(valid_parts) > 0


class ChaliceError(Exception):
    pass


class WebsocketDisconnectedError(ChaliceError):
    def __init__(self, connection_id: str) -> None:
        self.connection_id = connection_id


class ChaliceViewError(ChaliceError):
    STATUS_CODE: int = 500


class ChaliceUnhandledError(ChaliceError):
    """This error is not caught from a Chalice view function.

    This exception is allowed to propagate from a view function so
    that middleware handlers can process the exception.
    """


class BadRequestError(ChaliceViewError):
    STATUS_CODE = 400


class UnauthorizedError(ChaliceViewError):
    STATUS_CODE = 401


class ForbiddenError(ChaliceViewError):
    STATUS_CODE = 403


class NotFoundError(ChaliceViewError):
    STATUS_CODE = 404


class MethodNotAllowedError(ChaliceViewError):
    STATUS_CODE = 405


class RequestTimeoutError(ChaliceViewError):
    STATUS_CODE = 408


class ConflictError(ChaliceViewError):
    STATUS_CODE = 409


class UnprocessableEntityError(ChaliceViewError):
    STATUS_CODE = 422


class TooManyRequestsError(ChaliceViewError):
    STATUS_CODE = 429


ALL_ERRORS: List[Type[ChaliceViewError]] = [
    ChaliceViewError, BadRequestError, NotFoundError,
    UnauthorizedError, ForbiddenError, MethodNotAllowedError,
    RequestTimeoutError, ConflictError, UnprocessableEntityError,
    TooManyRequestsError
]


class MultiDict(MutableMapping[str, Any]):
    """A mapping of key to list of values.

    Accessing it in the usual way will return the last value in the list.
    Calling getlist will return a list of all the values associated with
    the same key.
    """

    def __init__(self, mapping: Optional[Dict[str, List[Any]]] = None) -> None:
        if mapping is None:
            mapping = {}
        self._dict = mapping

    def __getitem__(self, k: str) -> Any:
        try:
            return self._dict[k][-1]
        except IndexError:
            raise KeyError(k)

    def __setitem__(self, k: str, v: Any) -> None:
        self._dict[k] = [v]

    def __delitem__(self, k: str) -> None:
        del self._dict[k]

    def func_8luclvrw(self, k: str) -> List[Any]:
        return list(self._dict[k])

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __repr__(self) -> str:
        return 'MultiDict(%s)' % self._dict

    def __str__(self) -> str:
        return repr(self)


class CaseInsensitiveMapping(Mapping[str, Any]):
    """Case insensitive and read-only mapping."""

    def __init__(self, mapping: Optional[Dict[str, Any]]] = None) -> None:
        mapping = mapping or {}
        self._dict = {k.lower(): v for k, v in mapping.items()}

    def __getitem__(self, key: str) -> Any:
        return self._dict[key.lower()]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return 'CaseInsensitiveMapping(%s)' % repr(self._dict)


class Authorizer:
    name: str = ''
    scopes: List[str] = []

    def func_ed8viqid(self) -> Dict[str, Any]:
        raise NotImplementedError('to_swagger')

    def func_scoker0h(self, scopes: List[str]) -> 'Authorizer':
        raise NotImplementedError('with_scopes')


class IAMAuthorizer(Authorizer):
    _AUTH_TYPE = 'aws_iam'

    def __init__(self) -> None:
        self.name = 'sigv4'
        self.scopes = []

    def func_ed8viqid(self) -> Dict[str, Any]:
        return {'in': 'header', 'type': 'apiKey', 'name': 'Authorization',
            'x-amazon-apigateway-authtype': 'awsSigv4'}

    def func_scoker0h(self, scopes: List[str]) -> 'Authorizer':
        raise NotImplementedError('with_scopes')


class CognitoUserPoolAuthorizer(Authorizer):
    _AUTH_TYPE = 'cognito_user_pools'

    def __init__(self, name: str, provider_arns: List[str], header: str = 'Authorization',
                 scopes: Optional[List[str]] = None) -> None:
        self.name = name
        self._header = header
        if not isinstance(provider_arns, list):
            raise TypeError(
                'provider_arns should be a list of ARNs, received: %s' %
                provider_arns)
        self._provider_arns = provider_arns
        self.scopes = scopes or []

    def func_ed8viqid(self) -> Dict[str, Any]:
        return {'in': 'header', 'type': 'apiKey', 'name': self._header,
            'x-amazon-apigateway-authtype': self._AUTH_TYPE,
            'x-amazon-apigateway-authorizer': {'type': self._AUTH_TYPE,
            'providerARNs': self._provider_arns}}

    def func_scoker0h(self, scopes: List[str]) -> 'Authorizer':
        authorizer_with_scopes = copy.deepcopy(self)
        authorizer_with_scopes.scopes = scopes
        return authorizer_with_scopes


class CustomAuthorizer(Authorizer):
    _AUTH_TYPE = 'custom'

    def __init__(self, name: str, authorizer_uri: str, ttl_seconds: int = 300,
                 header: str = 'Authorization', invoke_role_arn: Optional[str] = None,
                 scopes: Optional[List[str]] = None) -> None:
        self.name = name
        self._header = header
        self._authorizer_uri = authorizer_uri
        self._ttl_seconds = ttl_seconds
        self._invoke_role_arn = invoke_role_arn
        self.scopes = scopes or []

    def func_ed8viqid(self) -> Dict[str, Any]:
        swagger = {'in': 'header', 'type': 'apiKey', 'name': self._header,
            'x-amazon-apigateway-authtype': self._AUTH_TYPE,
            'x-amazon-apigateway-authorizer': {'type': 'token',
            'authorizerUri': self._authorizer_uri,
            'authorizerResultTtlInSeconds': self._ttl_seconds}}
        if self._invoke_role_arn is not None:
            swagger['x-amazon-apigateway-authorizer']['authorizerCredentials'
                ] = self._invoke_role_arn
        return swagger

    def func_scoker0h(self, scopes: List[str]) -> 'Authorizer':
        authorizer_with_scopes = copy.deepcopy(self)
        authorizer_with_scopes.scopes = scopes
        return authorizer_with_scopes


class CORSConfig:
    """A cors configuration to attach to a route."""
    _REQUIRED_HEADERS = ['Content-Type', 'X-Amz-Date', 'Authorization',
        'X-Api-Key', 'X-Amz-Security-Token']

    def __init__(self, allow_origin: str = '*', allow_headers: Optional[List[str]] = None,
                 expose_headers: Optional[List[str]] = None, max_age: Optional[int] = None,
                 allow_credentials: Optional[bool] = None) -> None:
        self.allow_origin = allow_origin
        if allow_headers is None:
            self._allow_headers = set(self._REQUIRED_HEADERS)
        else:
            self._allow_headers = set(list(allow_headers) + self.
                _REQUIRED_HEADERS)
        if expose_headers is None:
            expose_headers = []
        self._expose_headers = expose_headers
        self._max_age = max_age
        self._allow_credentials = allow_credentials

    @property
    def func_mqk3qj1t(self) -> str:
        return ','.join(sorted(self._allow_headers))

    def func_buo5f3kr(self) -> Dict[str, str]:
        headers = {'Access-Control-Allow-Origin': self.allow_origin,
            'Access-Control-Allow-Headers': self.allow_headers}
        if self._expose_headers:
            headers.update({'Access-Control-Expose-Headers': ','.join(self.
                _expose_headers)})
        if self._max_age is not None:
            headers.update({'Access-Control-Max-Age': str(self._max_age)})
        if self._allow_credentials is True:
            headers.update({'Access-Control-Allow-Credentials': 'true'})
        return headers

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.get_access_control_headers(
                ) == other.get_access_control_headers()
        return False


class Request:
    """The current request from API gateway."""
    _NON_SERIALIZED_ATTRS = ['lambda_context']

    def __init__(self, event_dict: Dict[str, Any], lambda_context: Optional['LambdaContext'] = None) -> None:
        query_params = event_dict['multiValueQueryStringParameters']
        self.query_params = None if query_params is None else MultiDict(
            query_params)
        self.headers = CaseInsensitiveMapping(event_dict['headers'])
        self.uri_params = event_dict['pathParameters']
        self.method = event_dict['requestContext']['httpMethod']
        self._is_base64_encoded = event_dict.get('isBase64Encoded', False)
        self._body = event_dict['body']
        self._json_body: Optional[Any] = None
        self._raw_body = b''
        self.context = event_dict['requestContext']
        self.stage_vars = event_dict['stageVariables']
        self.path = event_dict['requestContext']['resourcePath']
        self.lambda_context = lambda_context
        self._event_dict = event_dict

    def func_jbpe448t(self, encoded: Union[str, bytes]) -> bytes:
        if not isinstance(encoded, bytes):
            encoded = encoded.encode('ascii')
        output = base64.b64decode(encoded)
        return output

    @property
    def func_vaeky95c(self) -> bytes:
        if not self._raw_body and self._body is not None:
            if self._is_base64_encoded:
                self._raw_body = self._base64decode(self._body)
            elif not isinstance(self._body, bytes):
                self._raw_body = self._body.encode('utf-8')
            else:
                self._raw_body = self._body
        return self._raw_body

    @property
    def func_da23b6ip(self) -> Any:
        if self.headers.get('content-type', '').startswith('application/json'):
            if self._json_body is None:
                try:
                    self._json_body = json.loads(self.raw_body)
                except ValueError:
                    raise BadRequestError('Error Parsing JSON')
            return self._json_body

    def func_n1mm27l7(self) -> Dict[str, Any]:
        copied = {k: v for k, v in self.__dict__.items() if not k.
            startswith('_') and k not in self._NON_SERIALIZED_ATTRS}
        copied['headers'] = dict(copied['headers'])
        if copied['query_params'] is not None:
            copied['query_params'] = dict(copied['query_params'])
        return copied

    def func_r2m92irr(self) -> Dict[str, Any]:
        return self._event_dict


class Response:
    def __init__(self, body: Any, headers: Optional[Dict[str, str]] = None, status_code: int = 200) -> None:
        self.body = body
        if headers is None:
            headers = {}
        self.headers = headers
        self.status_code = status_code

    def func_n1mm27l7(self, binary_types: Optional[List[str]] = None) -> Dict[str, Any]:
        body = self.body
        if not isinstance(body, _ANY_STRING):
            body = json.dumps(body, separators=(',', ':'), default=
                handle_extra_types)
        single_headers, multi_headers = self._sort_headers(self.headers)
        response = {'headers': single_headers, 'multiValueHeaders':
            multi_headers, 'statusCode': self.status_code, 'body': body}
        if binary_types is not None:
            self._b64encode_body_if_needed(response, binary_types)
        return response

    def func_8sjdkhe5(self, all_headers: Dict[str, Union[str, List[str]]]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        multi_headers = {}
        single_headers = {}
        for name, value in all_headers.items():
            if isinstance(value, list):
                multi_headers[name] = value
            else:
                single_headers[name] = value
        return single_headers, multi_headers

    def func_mlayjrmg(self, response_dict: Dict[str, Any], binary_types: List[str]) -> None:
        response_headers = CaseInsensitiveMapping(response_dict['headers'])
        content_type = response_headers.get('content-type', '')
        body = response_dict['body']
        if func_i6qpvb6g(content_type, binary_types):
            if func_i6qpvb6g(content_type, ['application/json']
                ) or not content_type:
                body = body if isinstance(body, bytes) else body.encode('utf-8'
                    )
            body = self._base64encode(body)
            response_dict['isBase64Encoded'] = True
        response_dict['body'] = body

    def func_gb83cah5(self, data: bytes) -> str:
        if not isinstance(data, bytes):
            raise ValueError(
                'Expected bytes type for body with binary Content-Type. Got %s type body instead.'
                 % type(data))
        data = base64.b64encode(data)
        return data.decode('ascii')


class RouteEntry:
    def __init__(self, view_function: Callable[..., Any], view_name: str, path: str, method: str,
                 api_key_required: Optional[bool] = None, content_types: Optional[List[str]] = None,
                 cors: Union[bool, CORSConfig, None] = False, authorizer: Optional[Authorizer] = None) -> None:
        self.view_function = view_function
        self.view_name = view_name
        self.uri_pattern = path
        self.method = method
        self.api_key_required = api_key_required
        self.view_args = self._parse_view_args()
        self.content_types = content_types or []
        if cors is True:
            cors = CORSConfig()
        elif cors is False:
            cors = None
        self.cors = cors
        self.authorizer = authorizer

    def func_21ritgdi(self) -> List[str]:
        if '{' not in self.uri_pattern:
            return []
        results = [r[1:-1] for r in _PARAMS.findall(self.uri_pattern)]
        return results

    def __eq__(self, other: Any) -> bool:
        return self.__dict__ == other.__dict__


class APIGateway:
    _DEFAULT_BINARY_TYPES = ['application/octet-stream',
        'application/x-t