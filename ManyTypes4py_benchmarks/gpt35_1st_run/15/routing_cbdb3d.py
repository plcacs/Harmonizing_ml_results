from typing import Any, Union, Optional, Awaitable, List, Dict, Pattern, Tuple, overload

class Router(httputil.HTTPServerConnectionDelegate):
    def find_handler(self, request, **kwargs) -> Any:
        raise NotImplementedError()

    def start_request(self, server_conn, request_conn) -> Any:
        return _RoutingDelegate(self, server_conn, request_conn)

class ReversibleRouter(Router):
    def reverse_url(self, name, *args) -> Optional[str]:
        raise NotImplementedError()

class _RoutingDelegate(httputil.HTTPMessageDelegate):
    def __init__(self, router, server_conn, request_conn):
        self.server_conn = server_conn
        self.request_conn = request_conn
        self.delegate = None
        self.router = router

class _DefaultMessageDelegate(httputil.HTTPMessageDelegate):
    def __init__(self, connection):
        self.connection = connection

class RuleRouter(Router):
    def __init__(self, rules=None):
        self.rules = []

    def add_rules(self, rules):
        pass

    def process_rule(self, rule) -> Rule:
        return rule

    def find_handler(self, request, **kwargs) -> Optional[httputil.HTTPMessageDelegate]:
        pass

    def get_target_delegate(self, target, request, **target_params) -> Optional[httputil.HTTPMessageDelegate]:
        pass

class ReversibleRuleRouter(ReversibleRouter, RuleRouter):
    def __init__(self, rules=None):
        self.named_rules = {}

    def process_rule(self, rule) -> Rule:
        return rule

    def reverse_url(self, name, *args) -> Optional[str]:
        pass

class Rule:
    def __init__(self, matcher, target, target_kwargs=None, name=None):
        pass

    def reverse(self, *args) -> Optional[str]:
        pass

class Matcher:
    def match(self, request) -> Optional[Dict[str, Any]]:
        raise NotImplementedError()

    def reverse(self, *args) -> Optional[str]:
        return None

class AnyMatches(Matcher):
    def match(self, request) -> Dict[str, Any]:
        return {}

class HostMatches(Matcher):
    def __init__(self, host_pattern):
        pass

    def match(self, request) -> Optional[Dict[str, Any]]:
        pass

class DefaultHostMatches(Matcher):
    def __init__(self, application, host_pattern):
        pass

    def match(self, request) -> Optional[Dict[str, Any]]:
        pass

class PathMatches(Matcher):
    def __init__(self, path_pattern):
        pass

    def match(self, request) -> Optional[Dict[str, Any]]:
        pass

    def reverse(self, *args) -> Optional[str]:
        pass

class URLSpec(Rule):
    def __init__(self, pattern, handler, kwargs=None, name=None):
        pass
