#
# Copyright 2009 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""This module contains implementations of various third-party
authentication schemes.
"""

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

from typing import List, Any, Dict, cast, Iterable, Union, Optional, Tuple


class AuthError(Exception):
    pass


class OpenIdMixin:
    """Abstract implementation of OpenID and Attribute Exchange."""

    _OPENID_ENDPOINT: str

    def authenticate_redirect(
        self,
        callback_uri: Optional[str] = None,
        ax_attrs: List[str] = ["name", "email", "language", "username"],
    ) -> None:
        handler = cast(RequestHandler, self)
        callback_uri = callback_uri or handler.request.uri
        assert callback_uri is not None
        args = self._openid_args(callback_uri, ax_attrs=ax_attrs)
        endpoint = self._OPENID_ENDPOINT
        handler.redirect(endpoint + "?" + urllib.parse.urlencode(args))

    async def get_authenticated_user(
        self, http_client: Optional[httpclient.AsyncHTTPClient] = None
    ) -> Dict[str, Any]:
        handler = cast(RequestHandler, self)
        args = {
            k: v[-1] for k, v in handler.request.arguments.items()
        }
        args["openid.mode"] = "check_authentication"
        url = self._OPENID_ENDPOINT
        if http_client is None:
            http_client = self.get_auth_http_client()
        resp = await http_client.fetch(
            url, method="POST", body=urllib.parse.urlencode(args)
        )
        return self._on_authentication_verified(resp)

    def _openid_args(
        self,
        callback_uri: str,
        ax_attrs: Iterable[str] = [],
        oauth_scope: Optional[str] = None,
    ) -> Dict[str, str]:
        handler = cast(RequestHandler, self)
        url = urllib.parse.urljoin(handler.request.full_url(), callback_uri)
        args = {
            "openid.ns": "http://specs.openid.net/auth/2.0",
            "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select",
            "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
            "openid.return_to": url,
            "openid.realm": urllib.parse.urljoin(url, "/"),
            "openid.mode": "checkid_setup",
        }
        if ax_attrs:
            args.update(
                {
                    "openid.ns.ax": "http://openid.net/srv/ax/1.0",
                    "openid.ax.mode": "fetch_request",
                }
            )
            ax_attrs = set(ax_attrs)
            required = []
            if "name" in ax_attrs:
                ax_attrs -= {"name", "firstname", "fullname", "lastname"}
                required += ["firstname", "fullname", "lastname"]
                args.update(
                    {
                        "openid.ax.type.firstname": "http://axschema.org/namePerson/first",
                        "openid.ax.type.fullname": "http://axschema.org/namePerson",
                        "openid.ax.type.lastname": "http://axschema.org/namePerson/last",
                    }
                )
            known_attrs = {
                "email": "http://axschema.org/contact/email",
                "language": "http://axschema.org/pref/language",
                "username": "http://axschema.org/namePerson/friendly",
            }
            for name in ax_attrs:
                args["openid.ax.type." + name] = known_attrs[name]
                required.append(name)
            args["openid.ax.required"] = ",".join(required)
        if oauth_scope:
            args.update(
                {
                    "openid.ns.oauth": "http://specs.openid.net/extensions/oauth/1.0",
                    "openid.oauth.consumer": handler.request.host.split(":")[0],
                    "openid.oauth.scope": oauth_scope,
                }
            )
        return args

    def _on_authentication_verified(
        self, response: httpclient.HTTPResponse
    ) -> Dict[str, Any]:
        handler = cast(RequestHandler, self)
        if b"is_valid:true" not in response.body:
            raise AuthError("Invalid OpenID response: %r" % response.body)

        ax_ns = None
        for key in handler.request.arguments:
            if (
                key.startswith("openid.ns.")
                and handler.get_argument(key) == "http://openid.net/srv/ax/1.0"
            ):
                ax_ns = key[10:]
                break

        def get_ax_arg(uri: str) -> str:
            if not ax_ns:
                return ""
            prefix = "openid." + ax_ns + ".type."
            ax_name = None
            for name in handler.request.arguments.keys():
                if handler.get_argument(name) == uri and name.startswith(prefix):
                    part = name[len(prefix):]
                    ax_name = "openid." + ax_ns + ".value." + part
                    break
            if not ax_name:
                return ""
            return handler.get_argument(ax_name, "")

        email = get_ax_arg("http://axschema.org/contact/email")
        name = get_ax_arg("http://axschema.org/namePerson")
        first_name = get_ax_arg("http://axschema.org/namePerson/first")
        last_name = get_ax_arg("http://axschema.org/namePerson/last")
        username = get_ax_arg("http://axschema.org/namePerson/friendly")
        locale = get_ax_arg("http://axschema.org/pref/language").lower()
        user = dict()
        name_parts = []
        if first_name:
            user["first_name"] = first_name
            name_parts.append(first_name)
        if last_name:
            user["last_name"] = last_name
            name_parts.append(last_name)
        if name:
            user["name"] = name
        elif name_parts:
            user["name"] = " ".join(name_parts)
        elif email:
            user["name"] = email.split("@")[0]
        if email:
            user["email"] = email
        if locale:
            user["locale"] = locale
        if username:
            user["username"] = username
        claimed_id = handler.get_argument("openid.claimed_id", None)
        if claimed_id:
            user["claimed_id"] = claimed_id
        return user

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        return httpclient.AsyncHTTPClient()


class OAuthMixin:
    """Abstract implementation of OAuth 1.0 and 1.0a."""

    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_REQUEST_TOKEN_URL: str
    _OAUTH_VERSION: str = "1.0a"
    _OAUTH_NO_CALLBACKS: bool = False

    async def authorize_redirect(
        self,
        callback_uri: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        http_client: Optional[httpclient.AsyncHTTPClient] = None,
    ) -> None:
        if callback_uri and getattr(self, "_OAUTH_NO_CALLBACKS", False):
            raise Exception("This service does not support oauth_callback")
        if http_client is None:
            http_client = self.get_auth_http_client()
        assert http_client is not None
        if getattr(self, "_OAUTH_VERSION", "1.0a") == "1.0a":
            response = await http_client.fetch(
                self._oauth_request_token_url(
                    callback_uri=callback_uri, extra_params=extra_params
                )
            )
        else:
            response = await http_client.fetch(self._oauth_request_token_url())
        url = self._OAUTH_AUTHORIZE_URL
        self._on_request_token(url, callback_uri, response)

    async def get_authenticated_user(
        self, http_client: Optional[httpclient.AsyncHTTPClient] = None
    ) -> Dict[str, Any]:
        handler = cast(RequestHandler, self)
        request_key = escape.utf8(handler.get_argument("oauth_token"))
        oauth_verifier = handler.get_argument("oauth_verifier", None)
        request_cookie = handler.get_cookie("_oauth_request_token")
        if not request_cookie:
            raise AuthError("Missing OAuth request token cookie")
        handler.clear_cookie("_oauth_request_token")
        cookie_key, cookie_secret = (
            base64.b64decode(escape.utf8(i)) for i in request_cookie.split("|")
        )
        if cookie_key != request_key:
            raise AuthError("Request token does not match cookie")
        token = dict(key=cookie_key, secret=cookie_secret)
        if oauth_verifier:
            token["verifier"] = oauth_verifier
        if http_client is None:
            http_client = self.get_auth_http_client()
        assert http_client is not None
        response = await http_client.fetch(self._oauth_access_token_url(token))
        access_token = _oauth_parse_response(response.body)
        user = await self._oauth_get_user_future(access_token)
        if not user:
            raise AuthError("Error getting user")
        user["access_token"] = access_token
        return user

    def _oauth_request_token_url(
        self,
        callback_uri: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        handler = cast(RequestHandler, self)
        consumer_token = self._oauth_consumer_token()
        url = self._OAUTH_REQUEST_TOKEN_URL
        args = dict(
            oauth_consumer_key=escape.to_basestring(consumer_token["key"]),
            oauth_signature_method="HMAC-SHA1",
            oauth_timestamp=str(int(time.time())),
            oauth_nonce=escape.to_basestring(binascii.b2a_hex(uuid.uuid4().bytes)),
            oauth_version="1.0",
        )
        if getattr(self, "_OAUTH_VERSION", "1.0a") == "1.0a":
            if callback_uri == "oob":
                args["oauth_callback"] = "oob"
            elif callback_uri:
                args["oauth_callback"] = urllib.parse.urljoin(
                    handler.request.full_url(), callback_uri
                )
            if extra_params:
                args.update(extra_params)
            signature = _oauth10a_signature(consumer_token, "GET", url, args)
        else:
            signature = _oauth_signature(consumer_token, "GET", url, args)

        args["oauth_signature"] = signature
        return url + "?" + urllib.parse.urlencode(args)

    def _on_request_token(
        self,
        authorize_url: str,
        callback_uri: Optional[str],
        response: httpclient.HTTPResponse,
    ) -> None:
        handler = cast(RequestHandler, self)
        request_token = _oauth_parse_response(response.body)
        data = (
            base64.b64encode(escape.utf8(request_token["key"]))
            + b"|"
            + base64.b64encode(escape.utf8(request_token["secret"]))
        handler.set_cookie("_oauth_request_token", data)
        args = dict(oauth_token=request_token["key"])
        if callback_uri == "oob":
            handler.finish(authorize_url + "?" + urllib.parse.urlencode(args))
            return
        elif callback_uri:
            args["oauth_callback"] = urllib.parse.urljoin(
                handler.request.full_url(), callback_uri
            )
        handler.redirect(authorize_url + "?" + urllib.parse.urlencode(args))

    def _oauth_access_token_url(self, request_token: Dict[str, Any]) -> str:
        consumer_token = self._oauth_consumer_token()
        url = self._OAUTH_ACCESS_TOKEN_URL
        args = dict(
            oauth_consumer_key=escape.to_basestring(consumer_token["key"]),
            oauth_token=escape.to_basestring(request_token["key"]),
            oauth_signature_method="HMAC-SHA1",
            oauth_timestamp=str(int(time.time())),
            oauth_nonce=escape.to_basestring(binascii.b2a_hex(uuid.uuid4().bytes)),
            oauth_version="1.0",
        )
        if "verifier" in request_token:
            args["oauth_verifier"] = request_token["verifier"]

        if getattr(self, "_OAUTH_VERSION", "1.0a") == "1.0a":
            signature = _oauth10a_signature(
                consumer_token, "GET", url, args, request_token
            )
        else:
            signature = _oauth_signature(
                consumer_token, "GET", url, args, request_token
            )

        args["oauth_signature"] = signature
        return url + "?" + urllib.parse.urlencode(args)

    def _oauth_consumer_token(self) -> Dict[str, Any]:
        raise NotImplementedError()

    async def _oauth_get_user_future(
        self, access_token: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    def _oauth_request_parameters(
        self,
        url: str,
        access_token: Dict[str, Any],
        parameters: Dict[str, Any] = {},
        method: str = "GET",
    ) -> Dict[str, Any]:
        consumer_token = self._oauth_consumer_token()
        base_args = dict(
            oauth_consumer_key=escape.to_basestring(consumer_token["key"]),
            oauth_token=escape.to_basestring(access_token["key"]),
            oauth_signature_method="HMAC-SHA1",
            oauth_timestamp=str(int(time.time())),
            oauth_nonce=escape.to_basestring(binascii.b2a_hex(uuid.uuid4().bytes)),
            oauth_version="1.0",
        )
        args = {}
        args.update(base_args)
        args.update(parameters)
        if getattr(self, "_OAUTH_VERSION", "1.0a") == "1.0a":
            signature = _oauth10a_signature(
                consumer_token, method, url, args, access_token
            )
        else:
            signature = _oauth_signature(
                consumer_token, method, url, args, access_token
            )
        base_args["oauth_signature"] = escape.to_basestring(signature)
        return base_args

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        return httpclient.AsyncHTTPClient()


class OAuth2Mixin:
    """Abstract implementation of OAuth 2.0."""

    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str

    def authorize_redirect(
        self,
        redirect_uri: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        scope: Optional[List[str]] = None,
        response_type: str = "code",
    ) -> None:
        if client_secret is not None:
            warnings.warn("client_secret argument is deprecated", DeprecationWarning)
        handler = cast(RequestHandler, self)
        args = {"response_type": response_type}
        if redirect_uri is not None:
            args["redirect_uri"] = redirect_uri
        if client_id is not None:
            args["client_id"] = client_id
        if extra_params:
            args.update(extra_params)
        if scope:
            args["scope"] = " ".join(scope)
        url = self._OAUTH_AUTHORIZE_URL
        handler.redirect(url_concat(url, args))

    def _oauth_request_token_url(
        self,
        redirect_uri: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        code: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = self._OAUTH_ACCESS_TOKEN_URL
        args = {}
        if redirect_uri is not None:
            args["redirect_uri"] = redirect_uri
        if code is not None:
            args["code"] = code
        if client_id is not None:
            args["client_id"] = client_id
        if client_secret is not None:
            args["client_secret"] = client_secret
        if extra_params:
            args.update(extra_params)
        return url_concat(url, args)

    async def oauth2_request(
        self,
        url: str,
        access_token: Optional[str] = None,
        post_args: Optional[Dict[str, Any]] = None,
        **args: Any,
    ) -> Any:
        all_args = {}
        if access_token:
            all_args["access_token"] = access_token
            all_args.update(args)

        if all_args:
            url += "?" + urllib.parse.urlencode(all_args)
        http = self.get_auth_http_client()
        if post_args is not None:
            response = await http.fetch(
                url, method="POST", body=urllib.parse.urlencode(post_args)
            )
        else:
            response = await http.fetch(url)
        return escape.json_decode(response.body)

    def get_auth_http_client(self)