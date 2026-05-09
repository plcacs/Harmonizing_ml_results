class Router(httputil.HTTPServerConnectionDelegate):
    """Abstract router interface."""

    def find_handler(self, request: httputil.HTTPServerRequest, **kwargs: Any) -> httputil.HTTPMessageDelegate:
        """Must be implemented to return an appropriate instance of `~.httputil.HTTPMessageDelegate`
        that can serve the request.
        Routing implementations may pass additional kwargs to extend the routing logic.

        :arg httputil.HTTPServerRequest request: current HTTP request.
        :arg kwargs: additional keyword arguments passed by routing implementation.
        :returns: an instance of `~.httputil.HTTPMessageDelegate` that will be used to
            process the request.
        """
        raise NotImplementedError()

    def start_request(self, server_conn: Any, request_conn: Any) -> httputil.HTTPMessageDelegate:
        return _RoutingDelegate(self, server_conn, request_conn)

class ReversibleRouter(Router):
    """Abstract router interface for routers that can handle named routes
    and support reversing them to original urls.
    """

    def reverse_url(self, name: str, *args: Any) -> str:
        """Returns url string for a given route name and arguments
        or ``None`` if no match is found.

        :arg str name: route name.
        :arg args: url parameters.
        :returns: parametrized url string for a given route name (or ``None``).
        """
        raise NotImplementedError()

class RuleRouter(Router):
    """Rule-based router implementation."""

    def __init__(self, rules: _RuleList = None):
        """Constructs a router from an ordered list of rules::

            RuleRouter([
                Rule(PathMatches("/handler"), Target),
                # ... more rules
            ])

        You can also omit explicit `Rule` constructor and use tuples of arguments::

            RuleRouter([
                (PathMatches("/handler"), Target),
            ])

        `PathMatches` is a default matcher, so the example above can be simplified::

            RuleRouter([
                ("/handler", Target),
            ])

        In the examples above, ``Target`` can be a nested `Router` instance, an instance of
        `~.httputil.HTTPServerConnectionDelegate` or an old-style callable,
        accepting a request argument.

        :arg rules: a list of `Rule` instances or tuples of `Rule`
            constructor arguments.
        """
        self.rules = []
        if rules:
            self.add_rules(rules)

    def add_rules(self, rules: _RuleList):
        """Appends new rules to the router.

        :arg rules: a list of Rule instances (or tuples of arguments, which are
            passed to Rule constructor).
        """
        for rule in rules:
            if isinstance(rule, (tuple, list)):
                assert len(rule) in (2, 3, 4)
                if isinstance(rule[0], basestring_type):
                    rule = Rule(PathMatches(rule[0]), *rule[1:])
                else:
                    rule = Rule(*rule)
            self.rules.append(self.process_rule(rule))

    def process_rule(self, rule: Rule) -> Rule:
        """Override this method for additional preprocessing of each rule.

        :arg Rule rule: a rule to be processed.
        :returns: the same or modified Rule instance.
        """
        return rule

    def find_handler(self, request: httputil.HTTPServerRequest, **kwargs: Any) -> httputil.HTTPMessageDelegate:
        for rule in self.rules:
            target_params = rule.matcher.match(request)
            if target_params is not None:
                if rule.target_kwargs:
                    target_params['target_kwargs'] = rule.target_kwargs
                delegate = self.get_target_delegate(rule.target, request, **target_params)
                if delegate is not None:
                    return delegate
        return None

    def get_target_delegate(self, target: Any, request: httputil.HTTPServerRequest, **target_params: Any) -> httputil.HTTPMessageDelegate:
        """Returns an instance of `~.httputil.HTTPMessageDelegate` for a
        Rule's target. This method is called by `~.find_handler` and can be
        extended to provide additional target types.

        :arg target: a Rule's target.
        :arg httputil.HTTPServerRequest request: current request.
        :arg target_params: additional parameters that can be useful
            for `~.httputil.HTTPMessageDelegate` creation.
        """
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
    """A rule-based router that implements ``reverse_url`` method.

    Each rule added to this router may have a ``name`` attribute that can be
    used to reconstruct an original uri. The actual reconstruction takes place
    in a rule's matcher (see `Matcher.reverse`).
    """

    def __init__(self, rules: _RuleList = None):
        self.named_rules = {}
        super().__init__(rules)

    def process_rule(self, rule: Rule) -> Rule:
        rule = super().process_rule(rule)
        if rule.name:
            if rule.name in self.named_rules:
                app_log.warning('Multiple handlers named %s; replacing previous value', rule.name)
            self.named_rules[rule.name] = rule
        return rule

    def reverse_url(self, name: str, *args: Any) -> str:
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

    def __init__(self, matcher: Matcher, target: Any, target_kwargs: dict = None, name: str = None):
        """Constructs a Rule instance.

        :arg Matcher matcher: a `Matcher` instance used for determining
            whether the rule should be considered a match for a specific
            request.
        :arg target: a Rule's target (typically a ``RequestHandler`` or
            `~.httputil.HTTPServerConnectionDelegate` subclass or even a nested `Router`,
            depending on routing implementation).
        :arg dict target_kwargs: a dict of parameters that can be useful
            at the moment of target instantiation (for example, ``status_code``
            for a ``RequestHandler`` subclass). They end up in
            ``target_params['target_kwargs']`` of `RuleRouter.get_target_delegate`
            method.
        :arg str name: the name of the rule that can be used to find it
            in `ReversibleRouter.reverse_url` implementation.
        """
        if isinstance(target, str):
            target = import_object(target)
        self.matcher = matcher
        self.target = target
        self.target_kwargs = target_kwargs if target_kwargs else {}
        self.name = name

    def reverse(self, *args: Any) -> str:
        return self.matcher.reverse(*args)

    def __repr__(self) -> str:
        return '{}({!r}, {}, kwargs={!r}, name={!r})'.format(self.__class__.__name__, self.matcher, self.target, self.target_kwargs, self.name)

class Matcher:
    """Represents a matcher for request features."""

    def match(self, request: httputil.HTTPServerRequest) -> dict:
        """Matches current instance against the request.

        :arg httputil.HTTPServerRequest request: current HTTP request
        :returns: a dict of parameters to be passed to the target handler
            (for example, ``handler_kwargs``, ``path_args``, ``path_kwargs``
            can be passed for proper `~.web.RequestHandler` instantiation).
            An empty dict is a valid (and common) return value to indicate a match
            when the argument-passing features are not used.
            ``None`` must be returned to indicate that there is no match."""
        raise NotImplementedError()

    def reverse(self, *args: Any) -> str:
        """Reconstructs full url from matcher instance and additional arguments."""
        return None

class AnyMatches(Matcher):
    """Matches any request."""

    def match(self, request: httputil.HTTPServerRequest) -> dict:
        return {}

class HostMatches(Matcher):
    """Matches requests from hosts specified by ``host_pattern`` regex."""

    def __init__(self, host_pattern: Pattern):
        if isinstance(host_pattern, basestring_type):
            if not host_pattern.endswith('$'):
                host_pattern += '$'
            self.host_pattern = re.compile(host_pattern)
        else:
            self.host_pattern = host_pattern

    def match(self, request: httputil.HTTPServerRequest) -> dict:
        if self.host_pattern.match(request.host_name):
            return {}
        return None

class DefaultHostMatches(Matcher):
    """Matches requests from host that is equal to application's default_host.
    Always returns no match if ``X-Real-Ip`` header is present.
    """

    def __init__(self, application: Any, host_pattern: Pattern):
        self.application = application
        self.host_pattern = host_pattern

    def match(self, request: httputil.HTTPServerRequest) -> dict:
        if 'X-Real-Ip' not in request.headers:
            if self.host_pattern.match(self.application.default_host):
                return {}
        return None

class URLSpec(Rule):
    """Specifies mappings between URLs and handlers.

    .. versionchanged: 4.5
       `URLSpec` is now a subclass of a `Rule` with `PathMatches` matcher and is preserved for
       backwards compatibility.
    """

    def __init__(self, pattern: Pattern, handler: Any, kwargs: dict = None, name: str = None):
        """Parameters:

        * ``pattern``: Regular expression to be matched. Any capturing
          groups in the regex will be passed in to the handler's
          get/post/etc methods as arguments (by keyword if named, by
          position if unnamed. Named and unnamed capturing groups
          may not be mixed in the same rule).

        * ``handler``: `~.web.RequestHandler` subclass to be invoked.

        * ``kwargs`` (optional): A dictionary of additional arguments
          to be passed to the handler's constructor.

        * ``name`` (optional): A name for this handler.  Used by
          `~.web.Application.reverse_url`.

        """
        matcher = PathMatches(pattern)
        super().__init__(matcher, handler, kwargs, name)
        self.regex = matcher.regex
        self.handler_class = self.target
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return '{}({!r}, {}, kwargs={!r}, name={!r})'.format(self.__class__.__name__, self.regex.pattern, self.handler_class, self.kwargs, self.name)

@overload
def _unquote_or_none(s: Any) -> Any:
    pass

@overload
def _unquote_or_none(s: Any) -> None:
    pass

def _unquote_or_none(s: Any) -> Any:
    """None-safe wrapper around url_unescape to handle unmatched optional
    groups correctly.

    Note that args are passed as bytes so the handler can decide what
    encoding to use.
    """
    if s is None:
        return s
    return url_unescape(s, encoding=None, plus=False)
