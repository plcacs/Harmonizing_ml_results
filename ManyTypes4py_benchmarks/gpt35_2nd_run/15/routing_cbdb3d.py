from typing import Any, Union, Optional, Awaitable, List, Dict, Pattern, Tuple, overload

class Router(httputil.HTTPServerConnectionDelegate):
    def find_handler(self, request, **kwargs) -> Any:
        ...

class ReversibleRouter(Router):
    def reverse_url(self, name: str, *args: Any) -> Optional[str]:
        ...

class _RoutingDelegate(httputil.HTTPMessageDelegate):
    def __init__(self, router: Router, server_conn: Any, request_conn: Any) -> None:
        ...

class _DefaultMessageDelegate(httputil.HTTPMessageDelegate):
    def __init__(self, connection: Any) -> None:
        ...

_RuleList = List[Union['Rule', List[Any], Tuple[Union[str, 'Matcher'], Any], Tuple[Union[str, 'Matcher'], Any, Dict[str, Any]], Tuple[Union[str, 'Matcher'], Any, Dict[str, Any], str]]]

class RuleRouter(Router):
    def __init__(self, rules: Optional[_RuleList] = None) -> None:
        ...

    def add_rules(self, rules: _RuleList) -> None:
        ...

    def process_rule(self, rule: 'Rule') -> 'Rule':
        ...

    def find_handler(self, request: Any, **kwargs) -> Any:
        ...

    def get_target_delegate(self, target: Any, request: Any, **target_params: Any) -> Any:
        ...

class ReversibleRuleRouter(ReversibleRouter, RuleRouter):
    def __init__(self, rules: Optional[_RuleList] = None) -> None:
        ...

    def process_rule(self, rule: 'Rule') -> 'Rule':
        ...

    def reverse_url(self, name: str, *args: Any) -> Optional[str]:
        ...

class Rule:
    def __init__(self, matcher: 'Matcher', target: Any, target_kwargs: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> None:
        ...

    def reverse(self, *args: Any) -> Optional[str]:
        ...

class Matcher:
    def match(self, request: Any) -> Optional[Dict[str, Any]]:
        ...

    def reverse(self, *args: Any) -> Optional[str]:
        ...

class AnyMatches(Matcher):
    def match(self, request: Any) -> Dict[str, Any]:
        ...

class HostMatches(Matcher):
    def __init__(self, host_pattern: Union[str, Pattern]) -> None:
        ...

    def match(self, request: Any) -> Dict[str, Any]:
        ...

class DefaultHostMatches(Matcher):
    def __init__(self, application: Any, host_pattern: Pattern) -> None:
        ...

    def match(self, request: Any) -> Dict[str, Any]:
        ...

class PathMatches(Matcher):
    def __init__(self, path_pattern: Union[str, Pattern]) -> None:
        ...

    def match(self, request: Any) -> Dict[str, Any]:
        ...

    def reverse(self, *args: Any) -> Optional[str]:
        ...

class URLSpec(Rule):
    def __init__(self, pattern: str, handler: Any, kwargs: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> None:
        ...
