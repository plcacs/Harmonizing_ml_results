"""Stub file for tornado.auth module."""

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
from typing import List, Any, Dict, cast, Iterable, Union, Optional

class AuthError(Exception):
    ...

class OpenIdMixin:
    """Abstract implementation of OpenID and Attribute Exchange."""
    _OPENID_ENDPOINT: str

    def authenticate_redirect(self, callback_uri: Optional[str] = None, ax_attrs: List[str] = ['name', 'email', 'language', 'username']) -> None:
        ...

    async def get_authenticated_user(self, http_client: Optional[httpclient.AsyncHTTPClient] = None) -> dict:
        ...

    def _openid_args(self, callback_uri: str, ax_attrs: Optional[List[str]] = None, oauth_scope: Optional[str] = None) -> dict:
        ...

    def _on_authentication_verified(self, response: httpclient.HTTPResponse) -> dict:
        ...

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        ...

class OAuthMixin:
    """Abstract implementation of OAuth 1.0 and 1.0a."""
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_VERSION: str
    _OAUTH_NO_CALLBACKS: bool

    async def authorize_redirect(self, callback_uri: Optional[str] = None, extra_params: Optional[dict] = None, http_client: Optional[httpclient.AsyncHTTPClient] = None) -> None:
        ...

    async def get_authenticated_user(self, http_client: Optional[httpclient.AsyncHTTPClient] = None) -> dict:
        ...

    def _oauth_request_token_url(self, callback_uri: Optional[str] = None, extra_params: Optional[dict] = None) -> str:
        ...

    def _on_request_token(self, authorize_url: str, callback_uri: Optional[str], response: httpclient.HTTPResponse) -> None:
        ...

    def _oauth_access_token_url(self, request_token: dict) -> str:
        ...

    def _oauth_consumer_token(self) -> dict:
        ...

    async def _oauth_get_user_future(self, access_token: dict) -> dict:
        ...

    def _oauth_request_parameters(self, url: str, access_token: dict, parameters: dict = ..., method: str = 'GET') -> dict:
        ...

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        ...

class OAuth2Mixin:
    """Abstract implementation of OAuth 2.0."""
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str

    def authorize_redirect(self, redirect_uri: Optional[str] = None, client_id: Optional[str] = None, client_secret: Optional[str] = None, extra_params: Optional[dict] = None, scope: Optional[List[str]] = None, response_type: str = 'code') -> None:
        ...

    def _oauth_request_token_url(self, redirect_uri: Optional[str] = None, client_id: Optional[str] = None, client_secret: Optional[str] = None, code: Optional[str] = None, extra_params: Optional[dict] = None) -> str:
        ...

    async def oauth2_request(self, url: str, access_token: Optional[str] = None, post_args: Optional[dict] = None, **args: Any) -> dict:
        ...

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        ...

class TwitterMixin(OAuthMixin):
    """Twitter OAuth authentication."""
    _OAUTH_REQUEST_TOKEN_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_AUTHENTICATE_URL: str
    _OAUTH_NO_CALLBACKS: bool
    _TWITTER_BASE_URL: str

    async def authenticate_redirect(self, callback_uri: Optional[str] = None) -> None:
        ...

    async def twitter_request(self, path: str, access_token: str, post_args: Optional[dict] = None, **args: Any) -> dict:
        ...

    def _oauth_consumer_token(self) -> dict:
        ...

    async def _oauth_get_user_future(self, access_token: dict) -> dict:
        ...

class GoogleOAuth2Mixin(OAuth2Mixin):
    """Google authentication using OAuth2."""
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_USERINFO_URL: str
    _OAUTH_NO_CALLBACKS: bool
    _OAUTH_SETTINGS_KEY: str

    def get_google_oauth_settings(self) -> dict:
        ...

    async def get_authenticated_user(self, redirect_uri: str, code: str, client_id: Optional[str] = None, client_secret: Optional[str] = None) -> dict:
        ...

class FacebookGraphMixin(OAuth2Mixin):
    """Facebook authentication using the new Graph API and OAuth2."""
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_NO_CALLBACKS: bool
    _FACEBOOK_BASE_URL: str

    async def get_authenticated_user(self, redirect_uri: str, client_id: str, client_secret: str, code: str, extra_fields: Optional[List[str]] = None) -> dict:
        ...

    async def facebook_request(self, path: str, access_token: Optional[str] = None, post_args: Optional[dict] = None, **args: Any) -> dict:
        ...

def _oauth_signature(consumer_token: dict, method: str, url: str, parameters: dict = ..., token: Optional[dict] = None) -> str:
    ...

def _oauth10a_signature(consumer_token: dict, method: str, url: str, parameters: dict = ..., token: Optional[dict] = None) -> str:
    ...

def _oauth_escape(val: Any) -> str:
    ...

def _oauth_parse_response(body: bytes) -> dict:
    ...