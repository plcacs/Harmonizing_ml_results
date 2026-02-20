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

All the classes in this file are class mixins designed to be used with
the `tornado.web.RequestHandler` class.  They are used in two ways:

* On a login handler, use methods such as ``authenticate_redirect()``,
  ``authorize_redirect()``, and ``get_authenticated_user()`` to
  establish the user's identity and store authentication tokens to your
  database and/or cookies.
* In non-login handlers, use methods such as ``facebook_request()``
  or ``twitter_request()`` to use the authentication tokens to make
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
            async def get(self):
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

from tornado import httpclient
from tornado import escape
from tornado.httputil import url_concat
from tornado.util import unicode_type
from tornado.web import RequestHandler

from typing import List, Any, Dict, cast, Iterable, Union, Optional


class AuthError(Exception):
    pass


class OpenIdMixin:
    """Abstract implementation of OpenID and Attribute Exchange.

    Class attributes:

    * ``_OPENID_ENDPOINT``: the identity provider's URI.
    """

    def authenticate_redirect(
        self,
        callback_uri: Optional[str] = None,
        ax_attrs: Iterable[str] = ["name", "email", "language", "username"],
    ) -> None:
        """Redirects to the authentication URL for this service.

        After authentication, the service will redirect back to the given
        callback URI with additional parameters including ``openid.mode``.

        We request the given attributes for the authenticated user by
        default (name, email, language, and username). If you don't need
        all those attributes for your app, you can request fewer with
        the ax_attrs keyword argument.

        .. versionchanged:: 6.0

            The ``callback`` argument was removed and this method no
            longer returns an awaitable object. It is now an ordinary
            synchronous function.
        """
        handler = cast(RequestHandler, self)
        callback_uri = callback_uri or handler.request.uri
        assert callback_uri is not None
        args = self._openid_args(callback_uri, ax_attrs=ax_attrs)
        endpoint = self._OPENID_ENDPOINT  # type: ignore
        handler.redirect(endpoint + "?" + urllib.parse.urlencode(args))

    async def get_authenticated_user(
        self, http_client: Optional[httpclient.AsyncHTTPClient] = None
    ) -> Dict[str, Any]:
        """Fetches the authenticated user data upon redirect.

        This method should be called by the handler that receives the
        redirect from the `authenticate_redirect()` method (which is
        often the same as the one that calls it; in that case you would
        call `get_authenticated_user` if the ``openid.mode`` parameter
        is present and `authenticate_redirect` if it is not).

        The result of this method will generally be used to set a cookie.

        .. versionchanged:: 6.0

            The ``callback`` argument was removed. Use the returned
            awaitable object instead.
        """
        handler = cast(RequestHandler, self)
        # Verify the OpenID response via direct request to the OP
        args = {
            k: v[-1] for k, v in handler.request.arguments.items()
        }  # type: Dict[str, Union[str, bytes]]
        args["openid.mode"] = "check_authentication"
        url = self._OPENID_ENDPOINT  # type: ignore
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
            required = []  # type: List[str]
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

        # Make sure we got back at least an email from attribute exchange
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
                    part = name[len(prefix) :]
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
        """Returns the `.AsyncHTTPClient` instance to be used for auth requests.

        May be overridden by subclasses to use an HTTP client other than
        the default.
        """
        return httpclient.AsyncHTTPClient()


class OAuthMixin:
    """Abstract implementation of OAuth 1.0 and 1.0a.

    See `TwitterMixin` below for an example implementation.

    Class attributes:

    * ``_OAUTH_AUTHORIZE_URL``: The service's OAuth authorization url.
    * ``_OAUTH_ACCESS_TOKEN_URL``: The service's OAuth access token url.
    * ``_OAUTH_VERSION``: May be either "1.0" or "1.0a".
    * ``_OAUTH_NO_CALLBACKS``: Set this to True if the service requires
      advance registration of callbacks.

    Subclasses must also override the `_oauth_get_user_future` and
    `_oauth_consumer_token` methods.
    """

    async def authorize_redirect(
        self,
        callback_uri: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        http_client: Optional[httpclient.AsyncHTTPClient] = None,
    ) -> None:
        """Redirects the user to obtain OAuth authorization for this service.

        The ``callback_uri`` may be omitted if you have previously
        registered a callback URI with the third-party service. For
        some services, you must use a previously-registered callback
        URI and cannot specify a callback via this method.

        This method sets a cookie called ``_oauth_request_token`` which is
        subsequently used (and cleared) in `get_authenticated_user` for
        security purposes.

        This method is asynchronous and must be called with ``await``
        or ``yield`` (This is different from other ``auth*_redirect``
        methods defined in this module). It calls
        `.RequestHandler.finish` for you so you should not write any
        other response after it returns.

        .. versionchanged:: 3.1
           Now returns a `.Future` and takes an optional callback, for
           compatibility with `.gen.coroutine`.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           awaitable object instead.

        """
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
        url = self._OAUTH_AUTHORIZE_URL  # type: ignore
        self._on_request_token(url, callback_uri, response)

    async def get_authenticated_user(
        self, http_client: Optional[httpclient.AsyncHTTPClient] = None
    ) -> Dict[str, Any]:
        """Gets the OAuth authorized user and access token.

        This method should be called from the handler for your
        OAuth callback URL to complete the registration process. We run the
        callback with the authenticated user dictionary.  This dictionary
        will contain an ``access_key`` which can be used to make authorized
        requests to this service on behalf of the user.  The dictionary will
        also contain other fields such as ``name``, depending on the service
        used.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           awaitable object instead.
        """
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
        token = dict(
            key=cookie_key, secret=cookie_secret
        )  # type: Dict[str, Union[str, bytes]]
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
        url = self._OAUTH_REQUEST_TOKEN_URL  # type: ignore
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
        )
        handler.set_cookie("_oauth_request_token", data)
        args = dict(oauth_token=request_token["key"])
        if callback_uri == "oob":
            handler