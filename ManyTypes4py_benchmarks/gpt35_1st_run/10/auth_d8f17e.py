from typing import List, Any, Dict, cast, Iterable, Union, Optional

class AuthError(Exception):
    pass

class OpenIdMixin:
    _OPENID_ENDPOINT: str

    def authenticate_redirect(self, callback_uri=None, ax_attrs: List[str] = ['name', 'email', 'language', 'username']) -> None:
        ...

    async def get_authenticated_user(self, http_client=None) -> Dict[str, Any]:
        ...

    def _openid_args(self, callback_uri, ax_attrs=[], oauth_scope=None) -> Dict[str, str]:
        ...

    def _on_authentication_verified(self, response) -> Dict[str, Any]:
        ...

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        ...

class OAuthMixin:
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_VERSION: str
    _OAUTH_NO_CALLBACKS: bool

    async def authorize_redirect(self, callback_uri=None, extra_params=None, http_client=None) -> None:
        ...

    async def get_authenticated_user(self, http_client=None) -> Dict[str, Any]:
        ...

    def _oauth_request_token_url(self, callback_uri=None, extra_params=None) -> str:
        ...

    def _on_request_token(self, authorize_url, callback_uri, response) -> None:
        ...

    def _oauth_access_token_url(self, request_token) -> str:
        ...

    def _oauth_consumer_token(self) -> Dict[str, str]:
        ...

    async def _oauth_get_user_future(self, access_token) -> Dict[str, Any]:
        ...

    def _oauth_request_parameters(self, url, access_token, parameters={}, method='GET') -> Dict[str, str]:
        ...

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        ...

class OAuth2Mixin:
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str

    def authorize_redirect(self, redirect_uri=None, client_id=None, client_secret=None, extra_params=None, scope=None, response_type='code') -> None:
        ...

    def _oauth_request_token_url(self, redirect_uri=None, client_id=None, client_secret=None, code=None, extra_params=None) -> str:
        ...

    async def oauth2_request(self, url, access_token=None, post_args=None, **args) -> Dict[str, Any]:
        ...

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        ...

class TwitterMixin(OAuthMixin):
    _OAUTH_REQUEST_TOKEN_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_AUTHENTICATE_URL: str
    _OAUTH_NO_CALLBACKS: bool
    _TWITTER_BASE_URL: str

    async def authenticate_redirect(self, callback_uri=None) -> None:
        ...

    async def twitter_request(self, path, access_token, post_args=None, **args) -> Dict[str, Any]:
        ...

    def _oauth_consumer_token(self) -> Dict[str, str]:
        ...

    async def _oauth_get_user_future(self, access_token) -> Dict[str, Any]:
        ...

class GoogleOAuth2Mixin(OAuth2Mixin):
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_USERINFO_URL: str
    _OAUTH_NO_CALLBACKS: bool
    _OAUTH_SETTINGS_KEY: str

    def get_google_oauth_settings(self) -> Dict[str, str]:
        ...

    async def get_authenticated_user(self, redirect_uri, code, client_id=None, client_secret=None) -> Dict[str, Any]:
        ...

class FacebookGraphMixin(OAuth2Mixin):
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_NO_CALLBACKS: bool
    _FACEBOOK_BASE_URL: str

    async def get_authenticated_user(self, redirect_uri, client_id, client_secret, code, extra_fields=None) -> Dict[str, Any]:
        ...

    async def facebook_request(self, path, access_token=None, post_args=None, **args) -> Dict[str, Any]:
        ...
