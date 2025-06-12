import re
from functools import partial
from tornado import httputil
from tornado.httpserver import _CallableAdapter
from tornado.escape import url_escape, url_unescape, utf8
from tornado.log import app_log
from tornado.util import basestring_type, import_object, re_unescape, unicode_type
from typing import Any, Union, Optional, Awaitable, List, Dict, Pattern, Tuple, overload

class Router(httputil.HTTPServerConnectionDelegate):
    """Abstract router interface."""

    def find_handler(self, request: httputil.HTTPServerRequest, **kwargs: Any) -> Optional[httputil.HTTPMessageDelegate]:
        raise NotImplementedError()

    def start_request(self, server_conn: httputil.HTTPServerConnection, request_conn: httputil.HTTPConnection) -> httputil.HTTPMessageDelegate:
        return _RoutingDelegate(self, server_conn, request_conn)

class ReversibleRouter(Router):
    """Abstract router interface for routers that can handle named routes
    and support reversing them to original urls.
    """

    def reverse_url(self, name: str, *args: Any) -> Optional[str]:
        raise NotImplementedError()

class _RoutingDelegate(httputil.HTTPMessageDelegate):

    def __init__(self, router: Router, server_conn: httputil.HTTPServerConnection, request_conn: httputil.HTTPConnection) -> None:
        self.server_conn = server_conn
        self.request_conn = request_conn
        self.delegate: Optional[httputil.HTTPMessageDelegate] = None
        self.router = router

    def headers_received(self, start_line: httputil.RequestStartLine, headers: httputil.HTTPHeaders) -> Optional[Awaitable[None]]:
        assert isinstance(start_line, httputil.RequestStartLine)
        request = httputil.HTTPServerRequest(connection=self.request_conn, server_connection=self.server_conn, start_line=start_line, headers=headers)
        self.delegate = self.router.find_handler(request)
        if self.delegate is None:
            app_log.debug('Delegate for %s %s request not found', start_line.method, start_line.path)
            self.delegate = _DefaultMessageDelegate(self.request_conn)
        return self.delegate.headers_received(start_line, headers)

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        assert self.delegate is not None
        return self.delegate.data_received(chunk)

    def finish(self) -> None:
        assert self.delegate is not None
        self.delegate.finish()

    def on_connection_close(self) -> None:
        assert self.delegate is not None
        self.delegate.on_connection_close()

class _DefaultMessageDelegate(httputil.HTTPMessageDelegate):

    def __init__(self, connection: httputil.HTTPConnection) -> None:
        self.connection = connection

    def finish(self) -> None:
        self.connection.write_headers(httputil.ResponseStartLine('HTTP/1.1', 404, 'Not Found'), httputil.HTTPHeaders())
        self.connection.finish()

_RuleList = List[Union['Rule', List[Any], Tuple[Union[str, 'Matcher'], Any], Tuple[Union[str, 'Matcher'], Any, Dict[str, Any]], Tuple[Union[str, 'Matcher'], Any, Dict[str, Any], str]]]

class RuleRouter(Router):
    """Rule-based router implementation."""

    def __init__(self, rules: Optional[_RuleList] = None) -> None:
        self.rules: List[Rule] = []
        if rules:
            self.add_rules(rules)

    def add_rules(self, rules: _RuleList) -> None:
        for rule in rules:
            if isinstance(rule, (tuple, list)):
                assert len(rule) in (2, 3, 4)
                if isinstance(rule[0], basestring_type):
                    rule = Rule(PathMatches(rule[0]), *rule[1:])
                else:
                    rule = Rule(*rule)
            self.rules.append(self.process_rule(rule))

    def process_rule(self, rule: 'Rule') -> 'Rule':
        return rule

    def find_handler(self, request: httputil.HTTPServerRequest, **kwargs: Any) -> Optional[httputil.HTTPMessageDelegate]:
        for rule in self.rules:
            target_params = rule.matcher.match(request)
            if target_params is not None:
                if rule.target_kwargs:
                    target_params['target_kwargs'] = rule.target_kwargs
                delegate = self.get_target_delegate(rule.target, request, **target_params)
                if delegate is not None:
                    return delegate
        return None

    def get_target_delegate(self, target: Any, request: httputil.HTTPServerRequest, **target_params: Any) -> Optional[httputil.HTTPMessageDelegate]:
        if isinstance(target, Router):
            return target.find_handler(request, **target_params)
        elif isinstance(target, httputil.HTTPServerConnectionDelegate):
            assert request.connection is not None
            return target.start_request(request.server_connection, request.connection)
        elif callable(target):
            assert request.connection is not None
            return _CallableAdapter(partial(target, **target_params), request.connection)
        return None

class ReversibleRuleRouter(ReversibleRouter, RuleRouter):
    """A rule-based router that implements ``reverse_url`` method."""

    def __init__(self, rules: Optional[_RuleList] = None) -> None:
        self.named_rules: Dict[str, Rule] = {}
        super().__init__(rules)

    def process_rule(self, rule: 'Rule') -> 'Rule':
        rule = super().process_rule(rule)
        if rule.name:
            if rule.name in self.named_rules:
                app_log.warning('Multiple handlers named %s; replacing previous value', rule.name)
            self.named_rules[rule.name] = rule
        return rule

    def reverse_url(self, name: str, *args: Any) -> Optional[str]:
        if name in self.named_rules:
            return self.named_rules[name].matcher.reverse(*args)
        for rule in self.rules:
            if isinstance(rule.target, ReversibleRouter):
                reversed_url = rule.target.reverse_url(name, *args)
                if reversed_url is not None:
                    return reversed_url
        return None

class Rule:
    """A routing rule."""

    def __init__(self, matcher: 'Matcher', target: Any, target_kwargs: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> None:
        if isinstance(target, str):
            target = import_object(target)
        self.matcher = matcher
        self.target = target
        self.target_kwargs = target_kwargs if target_kwargs else {}
        self.name = name

    def reverse(self, *args: Any) -> Optional[str]:
        return self.matcher.reverse(*args)

    def __repr__(self) -> str:
        return '{}({!r}, {}, kwargs={!r}, name={!r})'.format(self.__class__.__name__, self.matcher, self.target, self.target_kwargs, self.name)

class Matcher:
    """Represents a matcher for request features."""

    def match(self, request: httputil.HTTPServerRequest) -> Optional[Dict[str, Any]]:
        raise NotImplementedError()

    def reverse(self, *args: Any) -> Optional[str]:
        return None

class AnyMatches(Matcher):
    """Matches any request."""

    def match(self, request: httputil.HTTPServerRequest) -> Dict[str, Any]:
        return {}

class HostMatches(Matcher):
    """Matches requests from hosts specified by ``host_pattern`` regex."""

    def __init__(self, host_pattern: Union[str, Pattern[str]]) -> None:
        if isinstance(host_pattern, basestring_type):
            if not host_pattern.endswith('$'):
                host_pattern += '$'
            self.host_pattern = re.compile(host_pattern)
        else:
            self.host_pattern = host_pattern

    def match(self, request: httputil.HTTPServerRequest) -> Optional[Dict[str, Any]]:
        if self.host_pattern.match(request.host_name):
            return {}
        return None

class DefaultHostMatches(Matcher):
    """Matches requests from host that is equal to application's default_host.
    Always returns no match if ``X-Real-Ip`` header is present.
    """

    def __init__(self, application: Any, host_pattern: Pattern[str]) -> None:
        self.application = application
        self.host_pattern = host_pattern

    def match(self, request: httputil.HTTPServerRequest) -> Optional[Dict[str, Any]]:
        if 'X-Real-Ip' not in request.headers:
            if self.host_pattern.match(self.application.default_host):
                return {}
        return None

class PathMatches(Matcher):
    """Matches requests with paths specified by ``path_pattern`` regex."""

    def __init__(self, path_pattern: Union[str, Pattern[str]]) -> None:
        if isinstance(path_pattern, basestring_type):
            if not path_pattern.endswith('$'):
                path_pattern += '$'
            self.regex = re.compile(path_pattern)
        else:
            self.regex = path_pattern
        assert len(self.regex.groupindex) in (0, self.regex.groups), 'groups in url regexes must either be all named or all positional: %r' % self.regex.pattern
        self._path, self._group_count = self._find_groups()

    def match(self, request: httputil.HTTPServerRequest) -> Optional[Dict[str, Any]]:
        match = self.regex.match(request.path)
        if match is None:
            return None
        if not self.regex.groups:
            return {}
        path_args = []
        path_kwargs = {}
        if self.regex.groupindex:
            path_kwargs = {str(k): _unquote_or_none(v) for k, v in match.groupdict().items()}
        else:
            path_args = [_unquote_or_none(s) for s in match.groups()]
        return dict(path_args=path_args, path_kwargs=path_kwargs)

    def reverse(self, *args: Any) -> str:
        if self._path is None:
            raise ValueError('Cannot reverse url regex ' + self.regex.pattern)
        assert len(args) == self._group_count, 'required number of arguments not found'
        if not len(args):
            return self._path
        converted_args = []
        for a in args:
            if not isinstance(a, (unicode_type, bytes)):
                a = str(a)
            converted_args.append(url_escape(utf8(a), plus=False))
        return self._path % tuple(converted_args)

    def _find_groups(self) -> Tuple[Optional[str], Optional[int]]:
        pattern = self.regex.pattern
        if pattern.startswith('^'):
            pattern = pattern[1:]
        if pattern.endswith('$'):
            pattern = pattern[:-1]
        if self.regex.groups != pattern.count('('):
            return (None, None)
        pieces = []
        for fragment in pattern.split('('):
            if ')' in fragment:
                paren_loc = fragment.index(')')
                if paren_loc >= 0:
                    try:
                        unescaped_fragment = re_unescape(fragment[paren_loc + 1:])
                    except ValueError:
                        return (None, None)
                    pieces.append('%s' + unescaped_fragment)
            else:
                try:
                    unescaped_fragment = re_unescape(fragment)
                except ValueError:
                    return (None, None)
                pieces.append(unescaped_fragment)
        return (''.join(pieces), self.regex.groups)

class URLSpec(Rule):
    """Specifies mappings between URLs and handlers."""

    def __init__(self, pattern: str, handler: Any, kwargs: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> None:
        matcher = PathMatches(pattern)
        super().__init__(matcher, handler, kwargs, name)
        self.regex = matcher.regex
        self.handler_class = self.target
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return '{}({!r}, {}, kwargs={!r}, name={!r})'.format(self.__class__.__name__, self.regex.pattern, self.handler_class, self.kwargs, self.name)

@overload
def _unquote_or_none(s: None) -> None:
    pass

@overload
def _unquote_or_none(s: bytes) -> Optional[bytes]:
    pass

def _unquote_or_none(s: Optional[bytes]) -> Optional[bytes]:
    if s is None:
        return s
    return url_unescape(s, encoding=None, plus=False)
