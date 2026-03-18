```python
import tornado.httpclient
import tornado.httputil
import tornado.web
from typing import Any, Dict, List, Optional, Union

class AuthError(Exception):
    ...

class OpenIdMixin:
    _OPENID_ENDPOINT: str

    def authenticate_redirect(
        self,
        callback_uri: Optional[str] = None,
        ax_attrs: List[str] = ...
    ) -> None: ...

    async def get_authenticated_user(
        self,
        http_client: Optional[tornado.httpclient.AsyncHTTPClient] = None
    ) -> Dict[str, Any]: ...

    def _openid_args(
        self,
        callback_uri: str,
        ax_attrs: List[str] = ...,
        oauth_scope: Optional[str] = None
    ) -> Dict[str, str]: ...

    def _on_authentication_verified(
        self,
        response: tornado.httpclient.HTTPResponse
    ) -> Dict[str, Any]: ...

    def get_auth_http_client(self) -> tornado.httpclient.AsyncHTTPClient: ...

class OAuthMixin:
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_VERSION: str
    _OAUTH_NO_CALLBACKS: bool

    async def authorize_redirect(
        self,
        callback_uri: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        http_client: Optional[tornado.httpclient.AsyncHTTPClient] = None
    ) -> None: ...

    async def get_authenticated_user(
        self,
        http_client: Optional[tornado.httpclient.AsyncHTTPClient] = None
    ) -> Dict[str, Any]: ...

    def _oauth_request_token_url(
        self,
        callback_uri: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> str: ...

    def _on_request_token(
        self,
        authorize_url: str,
        callback_uri: Optional[str],
        response: tornado.httpclient.HTTPResponse
    ) -> None: ...

    def _oauth_access_token_url(
        self,
        request_token: Dict[str, Any]
    ) -> str: ...

    def _oauth_consumer_token(self) -> Dict[str, str]: ...

    async def _oauth_get_user_future(
        self,
        access_token: Dict[str, Any]
    ) -> Dict[str, Any]: ...

    def _oauth_request_parameters(
        self,
        url: str,
        access_token: Dict[str, Any],
        parameters: Dict[str, Any] = ...,
        method: str = ...
    ) -> Dict[str, Any]: ...

    def get_auth_http_client(self) -> tornado.httpclient.AsyncHTTPClient: ...

class OAuth2Mixin:
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str

    def authorize_redirect(
        self,
        redirect_uri: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        scope: Optional[List[str]] = None,
        response_type: str = ...
    ) -> None: ...

    def _oauth_request_token_url(
        self,
        redirect_uri: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        code: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> str: ...

    async def oauth2_request(
        self,
        url: str,
        access_token: Optional[str] = None,
        post_args: Optional[Dict[str, Any]] = None,
        **args: Any
    ) -> Any: ...

    def get_auth_http_client(self) -> tornado.httpclient.AsyncHTTPClient: ...

class TwitterMixin(OAuthMixin):
    _OAUTH_REQUEST_TOKEN_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_AUTHENTICATE_URL: str
    _OAUTH_NO_CALLBACKS: bool
    _TWITTER_BASE_URL: str

    async def authenticate_redirect(
        self,
        callback_uri: Optional[str] = None
    ) -> None: ...

    async def twitter_request(
        self,
        path: str,
        access_token: Dict[str, Any],
        post_args: Optional[Dict[str, Any]] = None,
        **args: Any
    ) -> Any: ...

    def _oauth_consumer_token(self) -> Dict[str, str]: ...

    async def _oauth_get_user_future(
        self,
        access_token: Dict[str, Any]
    ) -> Dict[str, Any]: ...

class GoogleOAuth2Mixin(OAuth2Mixin):
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_USERINFO_URL: str
    _OAUTH_NO_CALLBACKS: bool
    _OAUTH_SETTINGS_KEY: str

    def get_google_oauth_settings(self) -> Dict[str, str]: ...

    async def get_authenticated_user(
        self,
        redirect_uri: str,
        code: str,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None
    ) -> Dict[str, Any]: ...

class FacebookGraphMixin(OAuth2Mixin):
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_NO_CALLBACKS: bool
    _FACEBOOK_BASE_URL: str

    async def get_authenticated_user(
        self,
        redirect_uri: str,
        client_id: str,
        client_secret: str,
        code: str,
        extra_fields: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]: ...

    async def facebook_request(
        self,
        path: str,
        access_token: Optional[str] = None,
        post_args: Optional[Dict[str, Any]] = None,
        **args: Any
    ) -> Any: ...

def _oauth_signature(
    consumer_token: Dict[str, str],
    method: str,
    url: str,
    parameters: Dict[str, Any] = ...,
    token: Optional[Dict[str, Any]] = None
) -> bytes: ...

def _oauth10a_signature(
    consumer_token: Dict[str, str],
    method: str,
    url: str,
    parameters: Dict[str, Any] = ...,
    token: Optional[Dict[str, Any]] = None
) -> bytes: ...

def _oauth_escape(val: Any) -> str: ...

def _oauth_parse_response(body: bytes) -> Dict[str, Any]: ...
```