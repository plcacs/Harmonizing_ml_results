import json
import os
import pickle
import random
import shutil
import sys
import textwrap
import time
import urllib.parse
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import requests
import requests.utils
from .exceptions import *

def copy_session(session: requests.Session, request_timeout: Optional[float] = None) -> requests.Session:
    """Duplicates a requests.Session."""
    new = requests.Session()
    new.cookies = requests.utils.cookiejar_from_dict(requests.utils.dict_from_cookiejar(session.cookies))
    new.headers = session.headers.copy()
    new.request = partial(new.request, timeout=request_timeout)
    return new

def default_user_agent() -> str:
    return 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'

def default_iphone_headers() -> Dict[str, str]:
    return {'User-Agent': 'Instagram 273.0.0.16.70 (iPad13,8; iOS 16_3; en_US; en-US; scale=2.00; 2048x2732; 452417278) AppleWebKit/420+', 'x-ads-opt-out': '1', 'x-bloks-is-panorama-enabled': 'true', 'x-bloks-version-id': '01507c21540f73e2216b6f62a11a5b5e51aa85491b72475c080da35b1228ddd6', 'x-fb-client-ip': 'True', 'x-fb-connection-type': 'wifi', 'x-fb-http-engine': 'Liger', 'x-fb-server-cluster': 'True', 'x-fb': '1', 'x-ig-abr-connection-speed-kbps': '2', 'x-ig-app-id': '124024574287414', 'x-ig-app-locale': 'en-US', 'x-ig-app-startup-country': 'US', 'x-ig-bandwidth-speed-kbps': '0.000', 'x-ig-capabilities': '36r/F/8=', 'x-ig-connection-speed': '{}kbps'.format(random.randint(1000, 20000)), 'x-ig-connection-type': 'WiFi', 'x-ig-device-locale': 'en-US', 'x-ig-mapped-locale': 'en-US', 'x-ig-timezone-offset': str((datetime.now().astimezone().utcoffset() or timedelta(seconds=0)).seconds), 'x-ig-www-claim': '0', 'x-pigeon-session-id': str(uuid.uuid4()), 'x-tigon-is-retry': 'False', 'x-whatsapp': '0'}

class InstaloaderContext:
    """Class providing methods for (error) logging and low-level communication with Instagram.

    It is not thought to be instantiated directly, rather :class:`Instaloader` instances maintain a context
    object.

    For logging, it provides :meth:`log`, :meth:`error`, :meth:`error_catcher`.

    It provides low-level communication routines :meth:`get_json`, :meth:`graphql_query`, :meth:`graphql_node_list`,
    :meth:`get_and_write_raw` and implements mechanisms for rate controlling and error handling.

    Further, it provides methods for logging in and general session handles, which are used by that routines in
    class :class:`Instaloader`.
    """

    def __init__(self, sleep: bool = True, quiet: bool = False, user_agent: Optional[str] = None, max_connection_attempts: int = 3, request_timeout: float = 300.0, rate_controller: Optional[Callable[['InstaloaderContext'], RateController]] = None, fatal_status_codes: Optional[List[int]] = None, iphone_support: bool = True):
        self.user_agent = user_agent if user_agent is not None else default_user_agent()
        self.request_timeout = request_timeout
        self._session = self.get_anonymous_session()
        self.username = None
        self.user_id = None
        self.sleep = sleep
        self.quiet = quiet
        self.max_connection_attempts = max_connection_attempts
        self._graphql_page_length = 50
        self.two_factor_auth_pending = None
        self.iphone_support = iphone_support
        self.iphone_headers = default_iphone_headers()
        self.error_log = []
        self._rate_controller = rate_controller(self) if rate_controller is not None else RateController(self)
        self.raise_all_errors = False
        self.fatal_status_codes = fatal_status_codes or []
        self.profile_id_cache = dict()

    @contextmanager
    def anonymous_copy(self) -> Iterator['InstaloaderContext']:
        session = self._session
        username = self.username
        user_id = self.user_id
        iphone_headers = self.iphone_headers
        self._session = self.get_anonymous_session()
        self.username = None
        self.user_id = None
        self.iphone_headers = default_iphone_headers()
        try:
            yield self
        finally:
            self._session.close()
            self.username = username
            self._session = session
            self.user_id = user_id
            self.iphone_headers = iphone_headers

    @property
    def is_logged_in(self) -> bool:
        """True, if this Instaloader instance is logged in."""
        return bool(self.username)

    def log(self, *msg: Any, sep: str = '', end: str = '\n', flush: bool = False) -> None:
        """Log a message to stdout that can be suppressed with --quiet."""
        if not self.quiet:
            print(*msg, sep=sep, end=end, flush=flush)

    def error(self, msg: str, repeat_at_end: bool = True) -> None:
        """Log a non-fatal error message to stderr, which is repeated at program termination.

        :param msg: Message to be printed.
        :param repeat_at_end: Set to false if the message should be printed, but not repeated at program termination."""
        print(msg, file=sys.stderr)
        if repeat_at_end:
            self.error_log.append(msg)

    @property
    def has_stored_errors(self) -> bool:
        """Returns whether any error has been reported and stored to be repeated at program termination.

        .. versionadded: 4.12"""
        return bool(self.error_log)

    def close(self) -> None:
        """Print error log and close session"""
        if self.error_log and (not self.quiet):
            print('\nErrors or warnings occurred:', file=sys.stderr)
            for err in self.error_log:
                print(err, file=sys.stderr)
        self._session.close()

    @contextmanager
    def error_catcher(self, extra_info: Optional[str] = None) -> None:
        """
        Context manager to catch, print and record InstaloaderExceptions.

        :param extra_info: String to prefix error message with."""
        try:
            yield
        except InstaloaderException as err:
            if extra_info:
                self.error('{}: {}'.format(extra_info, err))
            else:
                self.error('{}'.format(err))
            if self.raise_all_errors:
                raise

    def _default_http_header(self, empty_session_only: bool = False) -> Dict[str, str]:
        """Returns default HTTP header we use for requests."""
        header = {'Accept-Encoding': 'gzip, deflate', 'Accept-Language': 'en-US,en;q=0.8', 'Connection': 'keep-alive', 'Content-Length': '0', 'Host': 'www.instagram.com', 'Origin': 'https://www.instagram.com', 'Referer': 'https://www.instagram.com/', 'User-Agent': self.user_agent, 'X-Instagram-AJAX': '1', 'X-Requested-With': 'XMLHttpRequest'}
        if empty_session_only:
            del header['Host']
            del header['Origin']
            del header['X-Instagram-AJAX']
            del header['X-Requested-With']
        return header

    def get_anonymous_session(self) -> requests.Session:
        """Returns our default anonymous requests.Session object."""
        session = requests.Session()
        session.cookies.update({'sessionid': '', 'mid': '', 'ig_pr': '1', 'ig_vw': '1920', 'csrftoken': '', 's_network': '', 'ds_user_id': ''})
        session.headers.update(self._default_http_header(empty_session_only=True))
        session.request = partial(session.request, timeout=self.request_timeout)
        return session

    def save_session(self) -> Dict[str, str]:
        """Not meant to be used directly, use :meth:`Instaloader.save_session`."""
        return requests.utils.dict_from_cookiejar(self._session.cookies)

    def update_cookies(self, cookie: Dict[str, str]) -> None:
        """.. versionadded:: 4.11"""
        self._session.cookies.update(cookie)

    def load_session(self, username: str, sessiondata: Dict[str, str]) -> None:
        """Not meant to be used directly, use :meth:`Instaloader.load_session`."""
        session = requests.Session()
        session.cookies = requests.utils.cookiejar_from_dict(sessiondata)
        session.headers.update(self._default_http_header())
        session.headers.update({'X-CSRFToken': session.cookies.get_dict()['csrftoken']})
        session.request = partial(session.request, timeout=self.request_timeout)
        self._session = session
        self.username = username

    def save_session_to_file(self, sessionfile: Any) -> None:
        """Not meant to be used directly, use :meth:`Instaloader.save_session_to_file`."""
        pickle.dump(self.save_session(), sessionfile)

    def load_session_from_file(self, username: str, sessionfile: Any) -> None:
        """Not meant to be used directly, use :meth:`Instaloader.load_session_from_file`."""
        self.load_session(username, pickle.load(sessionfile))

    def test_login(self) -> Optional[str]:
        """Not meant to be used directly, use :meth:`Instaloader.test_login`."""
        try:
            data = self.graphql_query('d6f4427fbe92d846298cf93df0b937d3', {})
            return data['data']['user']['username'] if data['data']['user'] is not None else None
        except (AbortDownloadException, ConnectionException) as err:
            self.error(f'Error when checking if logged in: {err}')
            return None

    def login(self, user: str, passwd: str) -> None:
        """Not meant to be used directly, use :meth:`Instaloader.login`.

        :raises BadCredentialsException: If the provided password is wrong.
        :raises TwoFactorAuthRequiredException: First step of 2FA login done, now call
           :meth:`Instaloader.two_factor_login`.
        :raises LoginException: An error happened during login (for example, and invalid response).
           Or if the provided username does not exist.

        .. versionchanged:: 4.12
           Raises LoginException instead of ConnectionException when an error happens.
           Raises LoginException instead of InvalidArgumentException when the username does not exist.
        """
        import http.client
        http.client._MAXHEADERS = 200
        session = requests.Session()
        session.cookies.update({'sessionid': '', 'mid': '', 'ig_pr': '1', 'ig_vw': '1920', 'ig_cb': '1', 'csrftoken': '', 's_network': '', 'ds_user_id': ''})
        session.headers.update(self._default_http_header())
        session.get('https://www.instagram.com/')
        csrf_token = session.cookies.get_dict()['csrftoken']
        session.headers.update({'X-CSRFToken': csrf_token})
        self.do_sleep()
        enc_password = '#PWD_INSTAGRAM_BROWSER:0:{}:{}'.format(int(datetime.now().timestamp()), passwd)
        login = session.post('https://www.instagram.com/api/v1/web/accounts/login/ajax/', data={'enc_password': enc_password, 'username': user}, allow_redirects=True)
        try:
            resp_json = login.json()
        except json.decoder.JSONDecodeError as err:
            raise LoginException('Login error: JSON decode fail, {} - {}.'.format(login.status_code, login.reason)) from err
        if resp_json.get('two_factor_required'):
            two_factor_session = copy_session(session, self.request_timeout)
            two_factor_session.headers.update({'X-CSRFToken': csrf_token})
            two_factor_session.cookies.update({'csrftoken': csrf_token})
            self.two_factor_auth_pending = (two_factor_session, user, resp_json['two_factor_info']['two_factor_identifier'])
            raise TwoFactorAuthRequiredException('Login error: two-factor authentication required.')
        if resp_json.get('checkpoint_url'):
            raise LoginException(f'Login: Checkpoint required. Point your browser to {resp_json.get('checkpoint_url')} - follow the instructions, then retry.')
        if resp_json['status'] != 'ok':
            if 'message' in resp_json:
                raise LoginException('Login error: "{}" status, message "{}".'.format(resp_json['status'], resp_json['message']))
            else:
                raise LoginException('Login error: "{}" status.'.format(resp_json['status']))
        if 'authenticated' not in resp_json:
            if 'message' in resp_json:
                raise LoginException('Login error: Unexpected response, "{}".'.format(resp_json['message']))
            else:
                raise LoginException('Login error: Unexpected response, this might indicate a blocked IP.')
        if not resp_json['authenticated']:
            if resp_json['user']:
                raise BadCredentialsException('Login error: Wrong password.')
            else:
                raise LoginException('Login error: User {} does not exist.'.format(user))
        session.headers.update({'X-CSRFToken': login.cookies['csrftoken']})
        self._session = session
        self.username = user
        self.user_id = resp_json['userId']

    def two_factor_login(self, two_factor_code: str) -> None:
        """Second step of login if 2FA is enabled.
        Not meant to be used directly, use :meth:`Instaloader.two_factor_login`.

        :raises InvalidArgumentException: No two-factor authentication pending.
        :raises BadCredentialsException: 2FA verification code invalid.

        .. versionadded:: 4.2"""
        if not self.two_factor_auth_pending:
            raise InvalidArgumentException('No two-factor authentication pending.')
        session, user, two_factor_id = self.two_factor_auth_pending
        login = session.post('https://www.instagram.com/accounts/login/ajax/two_factor/', data={'username': user, 'verificationCode': two_factor_code, 'identifier': two_factor_id}, allow_redirects=True)
        resp_json = login.json()
        if resp_json['status'] != 'ok':
            if 'message' in resp_json:
                raise BadCredentialsException('2FA error: {}'.format(resp_json['message']))
            else:
                raise BadCredentialsException('2FA error: "{}" status.'.format(resp_json['status']))
        session.headers.update({'X-CSRFToken': login.cookies['csrftoken']})
        self._session = session
        self.username = user
        self.two_factor_auth_pending = None

    def do_sleep(self) -> None:
        """Sleep a short time if self.sleep is set. Called before each request to instagram.com."""
        if self.sleep:
            time.sleep(min(random.expovariate(0.6), 15.0))

    @staticmethod
    def _response_error(resp: requests.Response) -> str:
        extra_from_json = None
        with suppress(json.decoder.JSONDecodeError):
            resp_json = resp.json()
            if 'status' in resp_json:
                extra_from_json = f'"{resp_json['status']}" status, message "{resp_json['message']}"' if 'message' in resp_json else f'"{resp_json['status']}" status'
        return f'{resp.status_code} {resp.reason}{(f' - {extra_from_json}' if extra_from_json is not None else '')} when accessing {resp.url}'

    def get_json(self, path: str, params: Dict[str, Any], host: str = 'www.instagram.com', session: Optional[requests.Session] = None, _attempt: int = 1, response_headers: Optional[Dict[str, str]] = None, use_post: bool = False) -> Dict[str, Any]:
        """JSON request to Instagram.

        :param path: URL, relative to the given domain which defaults to www.instagram.com/
        :param params: request parameters
        :param host: Domain part of the URL from where to download the requested JSON; defaults to www.instagram.com
        :param session: Session to use, or None to use self.session
        :param use_post: Use POST instead of GET to make the request
        :return: Decoded response dictionary
        :raises QueryReturnedBadRequestException: When the server responds with a 400.
        :raises QueryReturnedNotFoundException: When the server responds with a 404.
        :raises ConnectionException: When query repeatedly failed.

        .. versionchanged:: 4.13
           Added `use_post` parameter.
        """
        is_graphql_query = 'query_hash' in params and 'graphql/query' in path
        is_doc_id_query = 'doc_id' in params and 'graphql/query' in path
        is_iphone_query = host == 'i.instagram.com'
        is_other_query = not is_graphql_query and (not is_doc_id_query) and (host == 'www.instagram.com')
        sess = session if session else self._session
        try:
            self.do_sleep()
            if is_graphql_query:
                self._rate_controller.wait_before_query(params['query_hash'])
            if is_doc_id_query:
                self._rate_controller.wait_before_query(params['doc_id'])
            if is_iphone_query:
                self._rate_controller.wait_before_query('iphone')
            if is_other_query:
                self._rate_controller.wait_before_query('other')
            if use_post:
                resp = sess.post(f'https://{host}/{path}', data=params, allow_redirects=False)
            else:
                resp = sess.get(f'https://{host}/{path}', params=params, allow_redirects=False)
            if resp.status_code in self.fatal_status_codes:
                redirect = f' redirect to {resp.headers['location']}' if 'location' in resp.headers else ''
                body = ''
                if resp.headers['Content-Type'].startswith('application/json'):
                    body = f': {resp.text[:500]}' + ('…' if len(resp.text) > 501 else '')
                raise AbortDownloadException(f'Query to https://{host}/{path} responded with "{resp.status_code} {resp.reason}"{redirect}{body}')
            while resp.is_redirect:
                redirect_url = resp.headers['location']
                self.log(f'\nHTTP redirect from https://{host}/{path} to {redirect_url}')
                if redirect_url.startswith('https://www.instagram.com/accounts/login') or redirect_url.startswith('https://i.instagram.com/accounts/login'):
                    if not self.is_logged_in:
                        raise LoginRequiredException('Redirected to login page. Use --login or --load-cookies.')
                   