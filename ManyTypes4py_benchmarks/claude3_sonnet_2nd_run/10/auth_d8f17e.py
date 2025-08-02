import base64
import binascii
import hashlib
import hmac
import time
import urllib.parse
import uuid
import warnings
from tornado import httpclient
from tornado import escape
from tornado.httputil import url_concat
from tornado.util import unicode_type
from tornado.web import RequestHandler
from typing import List, Any, Dict, cast, Iterable, Union, Optional, Awaitable, Callable, Set, Tuple, TypeVar, overload

class AuthError(Exception):
    pass

class OpenIdMixin:
    """Abstract implementation of OpenID and Attribute Exchange.

    Class attributes:

    * ``_OPENID_ENDPOINT``: the identity provider's URI.
    """
    _OPENID_ENDPOINT: str

    def authenticate_redirect(self, callback_uri: Optional[str] = None, ax_attrs: List[str] = ['name', 'email', 'language', 'username']) -> None:
        """Redirects to the authentication URL for this service."""
        handler = cast(RequestHandler, self)
        callback_uri = callback_uri or handler.request.uri
        assert callback_uri is not None
        args = self._openid_args(callback_uri, ax_attrs=ax_attrs)
        endpoint = self._OPENID_ENDPOINT
        handler.redirect(endpoint + '?' + urllib.parse.urlencode(args))

    async def get_authenticated_user(self, http_client: Optional[httpclient.AsyncHTTPClient] = None) -> Dict[str, Any]:
        """Fetches the authenticated user data upon redirect."""
        handler = cast(RequestHandler, self)
        args: Dict[str, str] = {k: v[-1] for k, v in handler.request.arguments.items()}
        args['openid.mode'] = 'check_authentication'
        url = self._OPENID_ENDPOINT
        if http_client is None:
            http_client = self.get_auth_http_client()
        resp = await http_client.fetch(url, method='POST', body=urllib.parse.urlencode(args))
        return self._on_authentication_verified(resp)

    def _openid_args(self, callback_uri: str, ax_attrs: List[str] = [], oauth_scope: Optional[str] = None) -> Dict[str, str]:
        handler = cast(RequestHandler, self)
        url = urllib.parse.urljoin(handler.request.full_url(), callback_uri)
        args: Dict[str, str] = {
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
            ax_attrs = set(ax_attrs)
            required: List[str] = []
            if 'name' in ax_attrs:
                ax_attrs -= {'name', 'firstname', 'fullname', 'lastname'}
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

    def _on_authentication_verified(self, response: httpclient.HTTPResponse) -> Dict[str, str]:
        handler = cast(RequestHandler, self)
        if b'is_valid:true' not in response.body:
            raise AuthError('Invalid OpenID response: %r' % response.body)
        ax_ns = None
        for key in handler.request.arguments:
            if key.startswith('openid.ns.') and handler.get_argument(key) == 'http://openid.net/srv/ax/1.0':
                ax_ns = key[10:]
                break

        def get_ax_arg(uri: str) -> str:
            if not ax_ns:
                return ''
            prefix = 'openid.' + ax_ns + '.type.'
            ax_name = None
            for name in handler.request.arguments.keys():
                if handler.get_argument(name) == uri and name.startswith(prefix):
                    part = name[len(prefix):]
                    ax_name = 'openid.' + ax_ns + '.value.' + part
                    break
            if not ax_name:
                return ''
            return handler.get_argument(ax_name, '')

        email = get_ax_arg('http://axschema.org/contact/email')
        name = get_ax_arg('http://axschema.org/namePerson')
        first_name = get_ax_arg('http://axschema.org/namePerson/first')
        last_name = get_ax_arg('http://axschema.org/namePerson/last')
        username = get_ax_arg('http://axschema.org/namePerson/friendly')
        locale = get_ax_arg('http://axschema.org/pref/language').lower()
        user: Dict[str, str] = {}
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
        claimed_id = handler.get_argument('openid.claimed_id', None)
        if claimed_id:
            user['claimed_id'] = claimed_id
        return user

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        """Returns the `.AsyncHTTPClient` instance to be used for auth requests."""
        return httpclient.AsyncHTTPClient()

class OAuthMixin:
    """Abstract implementation of OAuth 1.0 and 1.0a."""
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_REQUEST_TOKEN_URL: str
    _OAUTH_VERSION: str = "1.0a"
    _OAUTH_NO_CALLBACKS: bool = False

    async def authorize_redirect(self, callback_uri: Optional[str] = None, 
                                extra_params: Optional[Dict[str, str]] = None, 
                                http_client: Optional[httpclient.AsyncHTTPClient] = None) -> None:
        """Redirects the user to obtain OAuth authorization for this service."""
        if callback_uri and getattr(self, '_OAUTH_NO_CALLBACKS', False):
            raise Exception('This service does not support oauth_callback')
        if http_client is None:
            http_client = self.get_auth_http_client()
        assert http_client is not None
        if getattr(self, '_OAUTH_VERSION', '1.0a') == '1.0a':
            response = await http_client.fetch(self._oauth_request_token_url(callback_uri=callback_uri, extra_params=extra_params))
        else:
            response = await http_client.fetch(self._oauth_request_token_url())
        url = self._OAUTH_AUTHORIZE_URL
        self._on_request_token(url, callback_uri, response)

    async def get_authenticated_user(self, http_client: Optional[httpclient.AsyncHTTPClient] = None) -> Dict[str, Any]:
        """Gets the OAuth authorized user and access token."""
        handler = cast(RequestHandler, self)
        request_key = escape.utf8(handler.get_argument('oauth_token'))
        oauth_verifier = handler.get_argument('oauth_verifier', None)
        request_cookie = handler.get_cookie('_oauth_request_token')
        if not request_cookie:
            raise AuthError('Missing OAuth request token cookie')
        handler.clear_cookie('_oauth_request_token')
        cookie_key, cookie_secret = (base64.b64decode(escape.utf8(i)) for i in request_cookie.split('|'))
        if cookie_key != request_key:
            raise AuthError('Request token does not match cookie')
        token: Dict[str, Any] = dict(key=cookie_key, secret=cookie_secret)
        if oauth_verifier:
            token['verifier'] = oauth_verifier
        if http_client is None:
            http_client = self.get_auth_http_client()
        assert http_client is not None
        response = await http_client.fetch(self._oauth_access_token_url(token))
        access_token = _oauth_parse_response(response.body)
        user = await self._oauth_get_user_future(access_token)
        if not user:
            raise AuthError('Error getting user')
        user['access_token'] = access_token
        return user

    def _oauth_request_token_url(self, callback_uri: Optional[str] = None, 
                                extra_params: Optional[Dict[str, str]] = None) -> str:
        handler = cast(RequestHandler, self)
        consumer_token = self._oauth_consumer_token()
        url = self._OAUTH_REQUEST_TOKEN_URL
        args: Dict[str, str] = {
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
            signature = _oauth10a_signature(consumer_token, 'GET', url, args)
        else:
            signature = _oauth_signature(consumer_token, 'GET', url, args)
        args['oauth_signature'] = signature
        return url + '?' + urllib.parse.urlencode(args)

    def _on_request_token(self, authorize_url: str, callback_uri: Optional[str], 
                         response: httpclient.HTTPResponse) -> None:
        handler = cast(RequestHandler, self)
        request_token = _oauth_parse_response(response.body)
        data = base64.b64encode(escape.utf8(request_token['key'])) + b'|' + base64.b64encode(escape.utf8(request_token['secret']))
        handler.set_cookie('_oauth_request_token', data)
        args = dict(oauth_token=request_token['key'])
        if callback_uri == 'oob':
            handler.finish(authorize_url + '?' + urllib.parse.urlencode(args))
            return
        elif callback_uri:
            args['oauth_callback'] = urllib.parse.urljoin(handler.request.full_url(), callback_uri)
        handler.redirect(authorize_url + '?' + urllib.parse.urlencode(args))

    def _oauth_access_token_url(self, request_token: Dict[str, Any]) -> str:
        consumer_token = self._oauth_consumer_token()
        url = self._OAUTH_ACCESS_TOKEN_URL
        args: Dict[str, str] = {
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
            signature = _oauth10a_signature(consumer_token, 'GET', url, args, request_token)
        else:
            signature = _oauth_signature(consumer_token, 'GET', url, args, request_token)
        args['oauth_signature'] = signature
        return url + '?' + urllib.parse.urlencode(args)

    def _oauth_consumer_token(self) -> Dict[str, str]:
        """Subclasses must override this to return their OAuth consumer keys."""
        raise NotImplementedError()

    async def _oauth_get_user_future(self, access_token: Dict[str, Any]) -> Dict[str, Any]:
        """Subclasses must override this to get basic information about the user."""
        raise NotImplementedError()

    def _oauth_request_parameters(self, url: str, access_token: Dict[str, Any], 
                                 parameters: Dict[str, str] = {}, 
                                 method: str = 'GET') -> Dict[str, str]:
        """Returns the OAuth parameters as a dict for the given request."""
        consumer_token = self._oauth_consumer_token()
        base_args: Dict[str, str] = {
            'oauth_consumer_key': escape.to_basestring(consumer_token['key']),
            'oauth_token': escape.to_basestring(access_token['key']),
            'oauth_signature_method': 'HMAC-SHA1',
            'oauth_timestamp': str(int(time.time())),
            'oauth_nonce': escape.to_basestring(binascii.b2a_hex(uuid.uuid4().bytes)),
            'oauth_version': '1.0'
        }
        args: Dict[str, str] = {}
        args.update(base_args)
        args.update(parameters)
        if getattr(self, '_OAUTH_VERSION', '1.0a') == '1.0a':
            signature = _oauth10a_signature(consumer_token, method, url, args, access_token)
        else:
            signature = _oauth_signature(consumer_token, method, url, args, access_token)
        base_args['oauth_signature'] = escape.to_basestring(signature)
        return base_args

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        """Returns the `.AsyncHTTPClient` instance to be used for auth requests."""
        return httpclient.AsyncHTTPClient()

class OAuth2Mixin:
    """Abstract implementation of OAuth 2.0."""
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str

    def authorize_redirect(self, redirect_uri: Optional[str] = None, 
                          client_id: Optional[str] = None, 
                          client_secret: Optional[str] = None, 
                          extra_params: Optional[Dict[str, str]] = None, 
                          scope: Optional[List[str]] = None, 
                          response_type: str = 'code') -> None:
        """Redirects the user to obtain OAuth authorization for this service."""
        if client_secret is not None:
            warnings.warn('client_secret argument is deprecated', DeprecationWarning)
        handler = cast(RequestHandler, self)
        args: Dict[str, str] = {'response_type': response_type}
        if redirect_uri is not None:
            args['redirect_uri'] = redirect_uri
        if client_id is not None:
            args['client_id'] = client_id
        if extra_params:
            args.update(extra_params)
        if scope:
            args['scope'] = ' '.join(scope)
        url = self._OAUTH_AUTHORIZE_URL
        handler.redirect(url_concat(url, args))

    def _oauth_request_token_url(self, redirect_uri: Optional[str] = None, 
                                client_id: Optional[str] = None, 
                                client_secret: Optional[str] = None, 
                                code: Optional[str] = None, 
                                extra_params: Optional[Dict[str, str]] = None) -> str:
        url = self._OAUTH_ACCESS_TOKEN_URL
        args: Dict[str, str] = {}
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

    async def oauth2_request(self, url: str, 
                            access_token: Optional[str] = None, 
                            post_args: Optional[Dict[str, str]] = None, 
                            **args: str) -> Any:
        """Fetches the given URL auth an OAuth2 access token."""
        all_args: Dict[str, str] = {}
        if access_token:
            all_args['access_token'] = access_token
            all_args.update(args)
        if all_args:
            url += '?' + urllib.parse.urlencode(all_args)
        http = self.get_auth_http_client()
        if post_args is not None:
            response = await http.fetch(url, method='POST', body=urllib.parse.urlencode(post_args))
        else:
            response = await http.fetch(url)
        return escape.json_decode(response.body)

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        """Returns the `.AsyncHTTPClient` instance to be used for auth requests."""
        return httpclient.AsyncHTTPClient()

class TwitterMixin(OAuthMixin):
    """Twitter OAuth authentication."""
    _OAUTH_REQUEST_TOKEN_URL: str = 'https://api.twitter.com/oauth/request_token'
    _OAUTH_ACCESS_TOKEN_URL: str = 'https://api.twitter.com/oauth/access_token'
    _OAUTH_AUTHORIZE_URL: str = 'https://api.twitter.com/oauth/authorize'
    _OAUTH_AUTHENTICATE_URL: str = 'https://api.twitter.com/oauth/authenticate'
    _OAUTH_NO_CALLBACKS: bool = False
    _TWITTER_BASE_URL: str = 'https://api.twitter.com/1.1'

    async def authenticate_redirect(self, callback_uri: Optional[str] = None) -> None:
        """Just like `~OAuthMixin.authorize_redirect`, but auto-redirects if authorized."""
        http = self.get_auth_http_client()
        response = await http.fetch(self._oauth_request_token_url(callback_uri=callback_uri))
        self._on_request_token(self._OAUTH_AUTHENTICATE_URL, None, response)

    async def twitter_request(self, path: str, 
                             access_token: Optional[Dict[str, Any]], 
                             post_args: Optional[Dict[str, str]] = None, 
                             **args: str) -> Any:
        """Fetches the given API path, e.g., ``statuses/user_timeline/btaylor``"""
        if path.startswith('http:') or path.startswith('https:'):
            url = path
        else:
            url = self._TWITTER_BASE_URL + path + '.json'
        if access_token:
            all_args: Dict[str, str] = {}
            all_args.update(args)
            all_args.update(post_args or {})
            method = 'POST' if post_args is not None else 'GET'
            oauth = self._oauth_request_parameters(url, access_token, all_args, method=method)
            args.update(oauth)
        if args:
            url += '?' + urllib.parse.urlencode(args)
        http = self.get_auth_http_client()
        if post_args is not None:
            response = await http.fetch(url, method='POST', body=urllib.parse.urlencode(post_args))
        else:
            response = await http.fetch(url)
        return escape.json_decode(response.body)

    def _oauth_consumer_token(self) -> Dict[str, str]:
        handler = cast(RequestHandler, self)
        handler.require_setting('twitter_consumer_key', 'Twitter OAuth')
        handler.require_setting('twitter_consumer_secret', 'Twitter OAuth')
        return dict(key=handler.settings['twitter_consumer_key'], 
                   secret=handler.settings['twitter_consumer_secret'])

    async def _oauth_get_user_future(self, access_token: Dict[str, Any]) -> Dict[str, Any]:
        user = await self.twitter_request('/account/verify_credentials', access_token=access_token)
        if user:
            user['username'] = user['screen_name']
        return user

class GoogleOAuth2Mixin(OAuth2Mixin):
    """Google authentication using OAuth2."""
    _OAUTH_AUTHORIZE_URL: str = 'https://accounts.google.com/o/oauth2/v2/auth'
    _OAUTH_ACCESS_TOKEN_URL: str = 'https://www.googleapis.com/oauth2/v4/token'
    _OAUTH_USERINFO_URL: str = 'https://www.googleapis.com/oauth2/v1/userinfo'
    _OAUTH_NO_CALLBACKS: bool = False
    _OAUTH_SETTINGS_KEY: str = 'google_oauth'

    def get_google_oauth_settings(self) -> Dict[str, str]:
        """Return the Google OAuth 2.0 credentials."""
        handler = cast(RequestHandler, self)
        return handler.settings[self._OAUTH_SETTINGS_KEY]

    async def get_authenticated_user(self, redirect_uri: str, code: str, 
                                    client_id: Optional[str] = None, 
                                    client_secret: Optional[str] = None) -> Dict[str, Any]:
        """Handles the login for the Google user, returning an access token."""
        if client_id is None or client_secret is None:
            settings = self.get_google_oauth_settings()
            if client_id is None:
                client_id = settings['key']
            if client_secret is None:
                client_secret = settings['secret']
        http = self.get_auth_http_client()
        body = urllib.parse.urlencode({
            'redirect_uri': redirect_uri,
            'code': code,
            'client_id': client_id,
            'client_secret': client_secret,
            'grant_type': 'authorization_code'
        })
        response = await http.fetch(
            self._OAUTH_ACCESS_TOKEN_URL,
            method='POST',
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            body=body)
        return escape.json_decode(response.body)

class FacebookGraphMixin(OAuth2Mixin):
    """Facebook authentication using the new Graph API and OAuth2."""
    _OAUTH_ACCESS_TOKEN_URL: str = 'https://graph.facebook.com/oauth/access_token?'
    _OAUTH_AUTHORIZE_URL: str = 'https://www.facebook.com/dialog/oauth?'
    _OAUTH_NO_CALLBACKS: bool = False
    _FACEBOOK_BASE_URL: str = 'https://graph.facebook.com'

    async def get_authenticated_user(self, redirect_uri: str, client_id: str, 
                                    client_secret: str, code: str, 
                                    extra_fields: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
        """Handles the login for the Facebook user, returning a user object."""
        http = self.get_auth_http_client()
        args = {
            'redirect_uri': redirect_uri,
            'code': code,
            'client_id': client_id,
            'client_secret': client_secret
        }
        fields: Set[str] = {'id', 'name', 'first_name', 'last_name', 'locale', 'picture', 'link'}
        if extra_fields:
            fields.update(extra_fields)
        response = await http.fetch(self._oauth_request_token_url(**args))
        args = escape.json_decode(response.body)
        session = {
            'access_token': args.get('access_token'),
            'expires_in': args.get('expires_in')
        }
        assert session['access_token'] is not None
        user = await self.facebook_request(
            path='/me',
            access_token=session['access_token'],
            appsecret_proof=hmac.new(
                key=client_secret.encode('utf8'),
                msg=session['access_token'].encode('utf8'),
                digestmod=hashlib.sha256
            ).hexdigest(),
            fields=','.join(fields)
        )
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

    async def facebook_request(self, path: str, 
                              access_token: Optional[str] = None, 
                              post_args: Optional[Dict[str, str]] = None, 
                              **args: Any) -> Any:
        """Fetches the given relative API path, e.g., "/btaylor/picture" """
        url = self._FACEBOOK_BASE_URL + path
        return await self.oauth2_request(url, access_token=access_token, post_args=post_args, **args)

def _oauth_signature(consumer_token: Dict[str, str], method: str, url: str, 
                    parameters: Dict[str, str] = {}, 
                    token: Optional[Dict[str, Any]] = None) -> bytes:
    """Calculates the HMAC-SHA1 OAuth signature for the given request."""
    parts = urllib.parse.urlparse(url)
    scheme, netloc, path = parts[:3]
    normalized_url = scheme.lower() + '://' + netloc.lower() + path
    base_elems: List[str] = []
    base_elems.append(method.upper())
    base_elems.append(normalized_url)
    base_elems.append('&'.join((f'{k}={_oauth_escape(str(v))}' for k, v in sorted(parameters.items()))))
    base_string = '&'.join((_oauth_escape(e) for e in base_elems))
    key_elems = [escape.utf8(consumer_token['secret'])]
    key_elems.append(escape.utf8(token['secret'] if token else ''))
    key = b'&'.join(key_elems)
    hash = hmac.new(key, escape.utf8(base_string), hashlib.sha1)
    return binascii.b2a_base64(hash.digest())[:-1]

def _oauth10a_signature(consumer_token: Dict[str, str], method: str, url: str, 
                       parameters: Dict[str, str] = {}, 
                       token: Optional[Dict[str, Any]] = None) -> bytes:
    """Calculates the HMAC-SHA1 OAuth 1.0a signature for the given request."""
    parts = urllib.parse.urlparse(url)
    scheme, netloc, path = parts[:3]
    normalized_url = scheme.lower() + '://' + netloc.lower() + path
    base_elems: List[str] = []
    base_elems.append(method.upper())
    base_elems.append(normalized_url)
    base_elems.append('&'.join((f'{k}={_oauth_escape(str(v))}' for k, v in sorted(parameters.items()))))
    base_string = '&'.join((_oauth_escape(e) for e in base_elems))
    key_elems = [escape.utf8(urllib.parse.quote(consumer_token['secret'], safe='~'))]
    key_elems.append(escape.utf8(urllib.parse.quote(token['secret'], safe='~') if token else ''))
    key = b'&'.join(key_elems)
    hash = hmac.new(key, escape.utf8(base_string), hashlib.sha1)
    return binascii.b2a_base64(hash.digest())[:-1]

def _oauth_escape(val: Union[str, bytes]) -> str:
    if isinstance(val, unicode_type):
        val = val.encode('utf-8')
    return urllib.parse.quote(val, safe='~')

def _oauth_parse_response(body: bytes) -> Dict[str, str]:
    body_str = escape.native_str(body)
    p = urllib.parse.parse_qs(body_str, keep_blank_values=False)
    token = dict(key=p['oauth_token'][0], secret=p['oauth_token_secret'][0])
    special = ('oauth_token', 'oauth_token_secret')
    token.update(((k, p[k][0]) for k in p if k not in special))
    return token
