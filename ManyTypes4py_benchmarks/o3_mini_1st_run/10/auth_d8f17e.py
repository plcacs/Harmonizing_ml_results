#!/usr/bin/env python3
"""This module contains implementations of various third-party
authentication schemes.

All the classes in this file are class mixins designed to be used with
the `tornado.web.RequestHandler` class.  They are used in two ways:

* On a login handler, use methods such as ``authenticate_redirect()``,
  ``authorize_redirect()``, and ``get_authenticated_user()`` to
  establish the user's identity and store authentication tokens to your
  database and/or cookies.
* In non-login handlers, use methods such as ``facebook_request()`` or
  ``twitter_request()`` to use the authentication tokens to make
  requests to the respective services.

They all take slightly different arguments due to the fact all these
services implement authentication and authorization slightly differently.
See the individual service classes below for complete documentation.

Example usage for Google OAuth:

.. testsetup::

    import urllib

.. testcode::

    class GoogleOAuth2LoginHandler(tornado.web.RequestHandler,
                                    tornado.auth.GoogleOAuth2Mixin):
        async def get(self):
            # Google requires an exact match for redirect_uri, so it's
            # best to get it from your app configuration instead of from
            # self.request.full_uri().
            redirect_uri = urllib.parse.urljoin(self.application.settings['redirect_base_uri'],
                self.reverse_url('google_oauth'))
            if self.get_argument('code', False):
                access = await self.get_authenticated_user(
                    redirect_uri=redirect_uri,
                    code=self.get_argument('code'))
                user = await self.oauth2_request(
                    "https://www.googleapis.com/oauth2/v1/userinfo",
                    access_token=access["access_token"])
                # Save the user and access token. For example:
                user_cookie = dict(id=user["id"], access_token=access["access_token"])
                self.set_signed_cookie("user", json.dumps(user_cookie))
                self.redirect("/")
            else:
                self.authorize_redirect(
                    redirect_uri=redirect_uri,
                    client_id=self.get_google_oauth_settings()['key'],
                    scope=['profile', 'email'],
                    response_type='code',
                    extra_params={'approval_prompt': 'auto'})
"""

import base64
import binascii
import hashlib
import hmac
import time
import urllib.parse
import uuid
import warnings
from typing import Any, Dict, List, Optional, Union

from tornado import escape, httpclient
from tornado.httputil import url_concat
from tornado.util import unicode_type
from tornado.web import RequestHandler
from typing import cast

class AuthError(Exception):
    pass

class OpenIdMixin:
    _OPENID_ENDPOINT: str

    def authenticate_redirect(self, callback_uri: Optional[str] = None,
                              ax_attrs: List[str] = ['name', 'email', 'language', 'username']) -> None:
        handler = cast(RequestHandler, self)
        callback_uri = callback_uri or handler.request.uri  # type: ignore
        assert callback_uri is not None
        args: Dict[str, Any] = self._openid_args(callback_uri, ax_attrs=ax_attrs)
        endpoint: str = self._OPENID_ENDPOINT
        handler.redirect(endpoint + '?' + urllib.parse.urlencode(args))

    async def get_authenticated_user(self, http_client: Optional[httpclient.AsyncHTTPClient] = None) -> Dict[str, Any]:
        handler = cast(RequestHandler, self)
        args = {k: v[-1] for k, v in handler.request.arguments.items()}
        args['openid.mode'] = 'check_authentication'
        url: str = self._OPENID_ENDPOINT
        if http_client is None:
            http_client = self.get_auth_http_client()
        resp: httpclient.HTTPResponse = await http_client.fetch(url, method='POST', body=urllib.parse.urlencode(args))
        return self._on_authentication_verified(resp)

    def _openid_args(self, callback_uri: str, ax_attrs: List[str] = [],
                     oauth_scope: Optional[Any] = None) -> Dict[str, Any]:
        handler = cast(RequestHandler, self)
        url: str = urllib.parse.urljoin(handler.request.full_url(), callback_uri)
        args: Dict[str, Any] = {
            'openid.ns': 'http://specs.openid.net/auth/2.0',
            'openid.claimed_id': 'http://specs.openid.net/auth/2.0/identifier_select',
            'openid.identity': 'http://specs.openid.net/auth/2.0/identifier_select',
            'openid.return_to': url,
            'openid.realm': urllib.parse.urljoin(url, '/'),
            'openid.mode': 'checkid_setup'
        }
        if ax_attrs:
            args.update({
                'openid.ns.ax': 'http://openid.net/srv/ax/1.0',
                'openid.ax.mode': 'fetch_request'
            })
            ax_attrs = list(set(ax_attrs))
            required: List[str] = []
            if 'name' in ax_attrs:
                for attr in ['name', 'firstname', 'fullname', 'lastname']:
                    if attr in ax_attrs:
                        ax_attrs.remove(attr)
                required += ['firstname', 'fullname', 'lastname']
                args.update({
                    'openid.ax.type.firstname': 'http://axschema.org/namePerson/first',
                    'openid.ax.type.fullname': 'http://axschema.org/namePerson',
                    'openid.ax.type.lastname': 'http://axschema.org/namePerson/last'
                })
            known_attrs: Dict[str, str] = {
                'email': 'http://axschema.org/contact/email',
                'language': 'http://axschema.org/pref/language',
                'username': 'http://axschema.org/namePerson/friendly'
            }
            for name in ax_attrs:
                args['openid.ax.type.' + name] = known_attrs[name]
                required.append(name)
            args['openid.ax.required'] = ','.join(required)
        if oauth_scope:
            args.update({
                'openid.ns.oauth': 'http://specs.openid.net/extensions/oauth/1.0',
                'openid.oauth.consumer': handler.request.host.split(':')[0],
                'openid.oauth.scope': oauth_scope
            })
        return args

    def _on_authentication_verified(self, response: httpclient.HTTPResponse) -> Dict[str, Any]:
        handler = cast(RequestHandler, self)
        if b'is_valid:true' not in response.body:
            raise AuthError('Invalid OpenID response: %r' % response.body)
        ax_ns: Optional[str] = None
        for key in handler.request.arguments:
            if key.startswith('openid.ns.') and handler.get_argument(key) == 'http://openid.net/srv/ax/1.0':
                ax_ns = key[10:]
                break

        def get_ax_arg(uri: str) -> str:
            if not ax_ns:
                return ''
            prefix: str = 'openid.' + ax_ns + '.type.'
            ax_name: Optional[str] = None
            for name in handler.request.arguments.keys():
                if handler.get_argument(name) == uri and name.startswith(prefix):
                    part = name[len(prefix):]
                    ax_name = 'openid.' + ax_ns + '.value.' + part
                    break
            if not ax_name:
                return ''
            return handler.get_argument(ax_name, '')
        email: str = get_ax_arg('http://axschema.org/contact/email')
        name: str = get_ax_arg('http://axschema.org/namePerson')
        first_name: str = get_ax_arg('http://axschema.org/namePerson/first')
        last_name: str = get_ax_arg('http://axschema.org/namePerson/last')
        username: str = get_ax_arg('http://axschema.org/namePerson/friendly')
        locale: str = get_ax_arg('http://axschema.org/pref/language').lower()
        user: Dict[str, Any] = {}
        name_parts: List[str] = []
        if first_name:
            user['first_name'] = first_name
            name_parts.append(first_name)
        if last_name:
            user['last_name'] = last_name
            name_parts.append(last_name)
        if name:
            user['name'] = name
        elif name_parts:
            user['name'] = ' '.join(name_parts)
        elif email:
            user['name'] = email.split('@')[0]
        if email:
            user['email'] = email
        if locale:
            user['locale'] = locale
        if username:
            user['username'] = username
        claimed_id: str = handler.get_argument('openid.claimed_id', None)
        if claimed_id:
            user['claimed_id'] = claimed_id
        return user

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        return httpclient.AsyncHTTPClient()

class OAuthMixin:
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_VERSION: str = '1.0a'
    _OAUTH_NO_CALLBACKS: bool = False
    _OAUTH_REQUEST_TOKEN_URL: str

    async def authorize_redirect(self, callback_uri: Optional[str] = None,
                                 extra_params: Optional[Dict[str, Any]] = None,
                                 http_client: Optional[httpclient.AsyncHTTPClient] = None) -> None:
        if callback_uri and getattr(self, '_OAUTH_NO_CALLBACKS', False):
            raise Exception('This service does not support oauth_callback')
        if http_client is None:
            http_client = self.get_auth_http_client()
        assert http_client is not None
        if getattr(self, '_OAUTH_VERSION', '1.0a') == '1.0a':
            response: httpclient.HTTPResponse = await http_client.fetch(
                self._oauth_request_token_url(callback_uri=callback_uri, extra_params=extra_params))
        else:
            response = await http_client.fetch(self._oauth_request_token_url())
        url: str = self._OAUTH_AUTHORIZE_URL
        self._on_request_token(url, callback_uri, response)

    async def get_authenticated_user(self, http_client: Optional[httpclient.AsyncHTTPClient] = None) -> Dict[str, Any]:
        handler = cast(RequestHandler, self)
        request_key: bytes = escape.utf8(handler.get_argument('oauth_token'))
        oauth_verifier: Optional[str] = handler.get_argument('oauth_verifier', None)
        request_cookie: Optional[str] = handler.get_cookie('_oauth_request_token')
        if not request_cookie:
            raise AuthError('Missing OAuth request token cookie')
        handler.clear_cookie('_oauth_request_token')
        cookie_parts = request_cookie.split('|')
        cookie_key: bytes = base64.b64decode(escape.utf8(cookie_parts[0]))
        cookie_secret: bytes = base64.b64decode(escape.utf8(cookie_parts[1]))
        if cookie_key != request_key:
            raise AuthError('Request token does not match cookie')
        token: Dict[str, Any] = dict(key=cookie_key, secret=cookie_secret)
        if oauth_verifier:
            token['verifier'] = oauth_verifier
        if http_client is None:
            http_client = self.get_auth_http_client()
        assert http_client is not None
        response: httpclient.HTTPResponse = await http_client.fetch(self._oauth_access_token_url(token))
        access_token: Dict[str, Any] = _oauth_parse_response(response.body)
        user: Dict[str, Any] = await self._oauth_get_user_future(access_token)
        if not user:
            raise AuthError('Error getting user')
        user['access_token'] = access_token
        return user

    def _oauth_request_token_url(self, callback_uri: Optional[str] = None,
                                 extra_params: Optional[Dict[str, Any]] = None) -> str:
        handler = cast(RequestHandler, self)
        consumer_token: Dict[str, str] = self._oauth_consumer_token()
        url: str = self._OAUTH_REQUEST_TOKEN_URL
        args: Dict[str, Any] = {
            'oauth_consumer_key': escape.to_basestring(consumer_token['key']),
            'oauth_signature_method': 'HMAC-SHA1',
            'oauth_timestamp': str(int(time.time())),
            'oauth_nonce': escape.to_basestring(binascii.b2a_hex(uuid.uuid4().bytes)),
            'oauth_version': '1.0'
        }
        if getattr(self, '_OAUTH_VERSION', '1.0a') == '1.0a':
            if callback_uri == 'oob':
                args['oauth_callback'] = 'oob'
            elif callback_uri:
                args['oauth_callback'] = urllib.parse.urljoin(handler.request.full_url(), callback_uri)
            if extra_params:
                args.update(extra_params)
            signature: bytes = _oauth10a_signature(consumer_token, 'GET', url, args)
        else:
            signature = _oauth_signature(consumer_token, 'GET', url, args)
        args['oauth_signature'] = signature
        return url + '?' + urllib.parse.urlencode(args)

    def _on_request_token(self, authorize_url: str, callback_uri: Optional[str],
                           response: httpclient.HTTPResponse) -> None:
        handler = cast(RequestHandler, self)
        request_token: Dict[str, Any] = _oauth_parse_response(response.body)
        data: bytes = base64.b64encode(escape.utf8(request_token['key'])) + b'|' + base64.b64encode(escape.utf8(request_token['secret']))
        handler.set_cookie('_oauth_request_token', data.decode('utf-8'))
        args: Dict[str, Any] = dict(oauth_token=request_token['key'])
        if callback_uri == 'oob':
            handler.finish(authorize_url + '?' + urllib.parse.urlencode(args))
            return
        elif callback_uri:
            args['oauth_callback'] = urllib.parse.urljoin(handler.request.full_url(), callback_uri)
        handler.redirect(authorize_url + '?' + urllib.parse.urlencode(args))

    def _oauth_access_token_url(self, request_token: Dict[str, Any]) -> str:
        consumer_token: Dict[str, str] = self._oauth_consumer_token()
        url: str = self._OAUTH_ACCESS_TOKEN_URL
        args: Dict[str, Any] = {
            'oauth_consumer_key': escape.to_basestring(consumer_token['key']),
            'oauth_token': escape.to_basestring(request_token['key']),
            'oauth_signature_method': 'HMAC-SHA1',
            'oauth_timestamp': str(int(time.time())),
            'oauth_nonce': escape.to_basestring(binascii.b2a_hex(uuid.uuid4().bytes)),
            'oauth_version': '1.0'
        }
        if 'verifier' in request_token:
            args['oauth_verifier'] = request_token['verifier']
        if getattr(self, '_OAUTH_VERSION', '1.0a') == '1.0a':
            signature: bytes = _oauth10a_signature(consumer_token, 'GET', url, args, request_token)
        else:
            signature = _oauth_signature(consumer_token, 'GET', url, args, request_token)
        args['oauth_signature'] = signature
        return url + '?' + urllib.parse.urlencode(args)

    def _oauth_consumer_token(self) -> Dict[str, str]:
        raise NotImplementedError()

    async def _oauth_get_user_future(self, access_token: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()

    def _oauth_request_parameters(self, url: str, access_token: Dict[str, Any],
                                  parameters: Dict[str, Any] = {}, method: str = 'GET') -> Dict[str, Any]:
        consumer_token: Dict[str, str] = self._oauth_consumer_token()
        base_args: Dict[str, Any] = {
            'oauth_consumer_key': escape.to_basestring(consumer_token['key']),
            'oauth_token': escape.to_basestring(access_token['key']),
            'oauth_signature_method': 'HMAC-SHA1',
            'oauth_timestamp': str(int(time.time())),
            'oauth_nonce': escape.to_basestring(binascii.b2a_hex(uuid.uuid4().bytes)),
            'oauth_version': '1.0'
        }
        args: Dict[str, Any] = {}
        args.update(base_args)
        args.update(parameters)
        if getattr(self, '_OAUTH_VERSION', '1.0a') == '1.0a':
            signature: bytes = _oauth10a_signature(consumer_token, method, url, args, access_token)
        else:
            signature = _oauth_signature(consumer_token, method, url, args, access_token)
        base_args['oauth_signature'] = escape.to_basestring(signature)
        return base_args

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        return httpclient.AsyncHTTPClient()

class OAuth2Mixin:
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str

    def authorize_redirect(self, redirect_uri: Optional[str] = None, client_id: Optional[str] = None,
                           client_secret: Optional[str] = None,
                           extra_params: Optional[Dict[str, Any]] = None,
                           scope: Optional[List[str]] = None, response_type: str = 'code') -> None:
        if client_secret is not None:
            warnings.warn('client_secret argument is deprecated', DeprecationWarning)
        handler = cast(RequestHandler, self)
        args: Dict[str, Any] = {'response_type': response_type}
        if redirect_uri is not None:
            args['redirect_uri'] = redirect_uri
        if client_id is not None:
            args['client_id'] = client_id
        if extra_params:
            args.update(extra_params)
        if scope:
            args['scope'] = ' '.join(scope)
        url: str = self._OAUTH_AUTHORIZE_URL
        handler.redirect(url_concat(url, args))

    def _oauth_request_token_url(self, redirect_uri: Optional[str] = None, client_id: Optional[str] = None,
                                 client_secret: Optional[str] = None, code: Optional[str] = None,
                                 extra_params: Optional[Dict[str, Any]] = None) -> str:
        url: str = self._OAUTH_ACCESS_TOKEN_URL
        args: Dict[str, Any] = {}
        if redirect_uri is not None:
            args['redirect_uri'] = redirect_uri
        if code is not None:
            args['code'] = code
        if client_id is not None:
            args['client_id'] = client_id
        if client_secret is not None:
            args['client_secret'] = client_secret
        if extra_params:
            args.update(extra_params)
        return url_concat(url, args)

    async def oauth2_request(self, url: str, access_token: Optional[str] = None,
                             post_args: Optional[Dict[str, Any]] = None, **args: Any) -> Any:
        all_args: Dict[str, Any] = {}
        if access_token:
            all_args['access_token'] = access_token
            all_args.update(args)
        if all_args:
            url += '?' + urllib.parse.urlencode(all_args)
        http = self.get_auth_http_client()
        if post_args is not None:
            response: httpclient.HTTPResponse = await http.fetch(url, method='POST', body=urllib.parse.urlencode(post_args))
        else:
            response = await http.fetch(url)
        return escape.json_decode(response.body)

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        return httpclient.AsyncHTTPClient()

class TwitterMixin(OAuthMixin):
    _OAUTH_REQUEST_TOKEN_URL: str = 'https://api.twitter.com/oauth/request_token'
    _OAUTH_ACCESS_TOKEN_URL: str = 'https://api.twitter.com/oauth/access_token'
    _OAUTH_AUTHORIZE_URL: str = 'https://api.twitter.com/oauth/authorize'
    _OAUTH_AUTHENTICATE_URL: str = 'https://api.twitter.com/oauth/authenticate'
    _OAUTH_NO_CALLBACKS: bool = False
    _TWITTER_BASE_URL: str = 'https://api.twitter.com/1.1'

    async def authenticate_redirect(self, callback_uri: Optional[str] = None) -> None:
        http = self.get_auth_http_client()
        response: httpclient.HTTPResponse = await http.fetch(
            self._oauth_request_token_url(callback_uri=callback_uri))
        self._on_request_token(self._OAUTH_AUTHENTICATE_URL, None, response)

    async def twitter_request(self, path: str, access_token: Dict[str, Any],
                              post_args: Optional[Dict[str, Any]] = None, **args: Any) -> Any:
        if path.startswith('http:') or path.startswith('https:'):
            url: str = path
        else:
            url = self._TWITTER_BASE_URL + path + '.json'
        if access_token:
            all_args: Dict[str, Any] = {}
            all_args.update(args)
            all_args.update(post_args or {})
            method: str = 'POST' if post_args is not None else 'GET'
            oauth: Dict[str, Any] = self._oauth_request_parameters(url, access_token, all_args, method=method)
            args.update(oauth)
        if args:
            url += '?' + urllib.parse.urlencode(args)
        http = self.get_auth_http_client()
        if post_args is not None:
            response: httpclient.HTTPResponse = await http.fetch(url, method='POST', body=urllib.parse.urlencode(post_args))
        else:
            response = await http.fetch(url)
        return escape.json_decode(response.body)

    def _oauth_consumer_token(self) -> Dict[str, str]:
        handler = cast(RequestHandler, self)
        handler.require_setting('twitter_consumer_key', 'Twitter OAuth')
        handler.require_setting('twitter_consumer_secret', 'Twitter OAuth')
        return dict(key=handler.settings['twitter_consumer_key'], secret=handler.settings['twitter_consumer_secret'])

    async def _oauth_get_user_future(self, access_token: Dict[str, Any]) -> Dict[str, Any]:
        user: Dict[str, Any] = await self.twitter_request('/account/verify_credentials', access_token=access_token)
        if user:
            user['username'] = user['screen_name']
        return user

class GoogleOAuth2Mixin(OAuth2Mixin):
    _OAUTH_AUTHORIZE_URL: str = 'https://accounts.google.com/o/oauth2/v2/auth'
    _OAUTH_ACCESS_TOKEN_URL: str = 'https://www.googleapis.com/oauth2/v4/token'
    _OAUTH_USERINFO_URL: str = 'https://www.googleapis.com/oauth2/v1/userinfo'
    _OAUTH_NO_CALLBACKS: bool = False
    _OAUTH_SETTINGS_KEY: str = 'google_oauth'

    def get_google_oauth_settings(self) -> Dict[str, Any]:
        handler = cast(RequestHandler, self)
        return handler.settings[self._OAUTH_SETTINGS_KEY]

    async def get_authenticated_user(self, redirect_uri: str, code: str,
                                     client_id: Optional[str] = None,
                                     client_secret: Optional[str] = None) -> Dict[str, Any]:
        if client_id is None or client_secret is None:
            settings: Dict[str, Any] = self.get_google_oauth_settings()
            if client_id is None:
                client_id = settings['key']
            if client_secret is None:
                client_secret = settings['secret']
        http = self.get_auth_http_client()
        body: str = urllib.parse.urlencode({
            'redirect_uri': redirect_uri,
            'code': code,
            'client_id': client_id,
            'client_secret': client_secret,
            'grant_type': 'authorization_code'
        })
        response: httpclient.HTTPResponse = await http.fetch(
            self._OAUTH_ACCESS_TOKEN_URL,
            method='POST',
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            body=body)
        return escape.json_decode(response.body)

class FacebookGraphMixin(OAuth2Mixin):
    _OAUTH_ACCESS_TOKEN_URL: str = 'https://graph.facebook.com/oauth/access_token?'
    _OAUTH_AUTHORIZE_URL: str = 'https://www.facebook.com/dialog/oauth?'
    _OAUTH_NO_CALLBACKS: bool = False
    _FACEBOOK_BASE_URL: str = 'https://graph.facebook.com'

    async def get_authenticated_user(self, redirect_uri: str, client_id: str, client_secret: str,
                                     code: str, extra_fields: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        http = self.get_auth_http_client()
        args: Dict[str, Any] = {
            'redirect_uri': redirect_uri,
            'code': code,
            'client_id': client_id,
            'client_secret': client_secret
        }
        fields: set = {'id', 'name', 'first_name', 'last_name', 'locale', 'picture', 'link'}
        if extra_fields:
            fields.update(extra_fields)
        response: httpclient.HTTPResponse = await http.fetch(self._oauth_request_token_url(**args))
        args_response: Dict[str, Any] = escape.json_decode(response.body)
        session: Dict[str, Any] = {
            'access_token': args_response.get('access_token'),
            'expires_in': args_response.get('expires_in')
        }
        assert session['access_token'] is not None
        appsecret_proof: str = hmac.new(key=client_secret.encode('utf8'),
                                          msg=session['access_token'].encode('utf8'),
                                          digestmod=hashlib.sha256).hexdigest()
        user: Any = await self.facebook_request(path='/me', access_token=session['access_token'],
                                                appsecret_proof=appsecret_proof,
                                                fields=','.join(fields))
        if user is None:
            return None
        fieldmap: Dict[str, Any] = {}
        for field in fields:
            fieldmap[field] = user.get(field)
        fieldmap.update({
            'access_token': session['access_token'],
            'session_expires': str(session.get('expires_in'))
        })
        return fieldmap

    async def facebook_request(self, path: str, access_token: Optional[str] = None,
                               post_args: Optional[Dict[str, Any]] = None, **args: Any) -> Any:
        url: str = self._FACEBOOK_BASE_URL + path
        return await self.oauth2_request(url, access_token=access_token, post_args=post_args, **args)

def _oauth_signature(consumer_token: Dict[str, str], method: str, url: str,
                     parameters: Dict[str, Any] = {}, token: Optional[Dict[str, str]] = None) -> bytes:
    parts = urllib.parse.urlparse(url)
    scheme, netloc, path = parts[:3]
    normalized_url: str = scheme.lower() + '://' + netloc.lower() + path
    base_elems: List[str] = []
    base_elems.append(method.upper())
    base_elems.append(normalized_url)
    base_elems.append('&'.join((f'{k}={_oauth_escape(str(v))}' for k, v in sorted(parameters.items()))))
    base_string: str = '&'.join((_oauth_escape(e) for e in base_elems))
    key_elems: List[bytes] = [escape.utf8(consumer_token['secret'])]
    key_elems.append(escape.utf8(token['secret'] if token else ''))
    key: bytes = b'&'.join(key_elems)
    hash_obj = hmac.new(key, escape.utf8(base_string), hashlib.sha1)
    return binascii.b2a_base64(hash_obj.digest())[:-1]

def _oauth10a_signature(consumer_token: Dict[str, str], method: str, url: str,
                        parameters: Dict[str, Any] = {}, token: Optional[Dict[str, str]] = None) -> bytes:
    parts = urllib.parse.urlparse(url)
    scheme, netloc, path = parts[:3]
    normalized_url: str = scheme.lower() + '://' + netloc.lower() + path
    base_elems: List[str] = []
    base_elems.append(method.upper())
    base_elems.append(normalized_url)
    base_elems.append('&'.join((f'{k}={_oauth_escape(str(v))}' for k, v in sorted(parameters.items()))))
    base_string: str = '&'.join((_oauth_escape(e) for e in base_elems))
    key_elems: List[bytes] = [escape.utf8(urllib.parse.quote(consumer_token['secret'], safe='~'))]
    key_elems.append(escape.utf8(urllib.parse.quote(token['secret'], safe='~') if token else ''))
    key: bytes = b'&'.join(key_elems)
    hash_obj = hmac.new(key, escape.utf8(base_string), hashlib.sha1)
    return binascii.b2a_base64(hash_obj.digest())[:-1]

def _oauth_escape(val: Union[str, bytes]) -> str:
    if isinstance(val, unicode_type):
        val = val.encode('utf-8')
    return urllib.parse.quote(val, safe='~')

def _oauth_parse_response(body: bytes) -> Dict[str, str]:
    body_str: str = escape.native_str(body)
    p: Dict[str, List[str]] = urllib.parse.parse_qs(body_str, keep_blank_values=False)
    token: Dict[str, str] = dict(key=p['oauth_token'][0], secret=p['oauth_token_secret'][0])
    special = ('oauth_token', 'oauth_token_secret')
    token.update(((k, p[k][0]) for k in p if k not in special))
    return token
