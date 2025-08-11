from contextlib import contextmanager
from typing import Dict, List

class _NormalizerMeta(type):

    def __new__(cls: Union[dict[str, typing.Any], str, tuple[typing.Type]], name: Union[dict[str, typing.Any], str, tuple[typing.Type]], bases: Union[dict[str, typing.Any], str, tuple[typing.Type]], dct: Union[dict[str, typing.Any], str, tuple[typing.Type]]):
        new_cls = type.__new__(cls, name, bases, dct)
        new_cls.rule_value_classes = {}
        new_cls.rule_type_classes = {}
        return new_cls

class Normalizer(metaclass=_NormalizerMeta):
    _rule_type_instances = {}
    _rule_value_instances = {}

    def __init__(self, grammar, config) -> None:
        self.grammar = grammar
        self._config = config
        self.issues = []
        self._rule_type_instances = self._instantiate_rules('rule_type_classes')
        self._rule_value_instances = self._instantiate_rules('rule_value_classes')

    def _instantiate_rules(self, attr: Union[str, None]) -> dict:
        dct = {}
        for base in type(self).mro():
            rules_map = getattr(base, attr, {})
            for type_, rule_classes in rules_map.items():
                new = [rule_cls(self) for rule_cls in rule_classes]
                dct.setdefault(type_, []).extend(new)
        return dct

    def walk(self, node: int) -> Union[int, str, list[None]]:
        self.initialize(node)
        value = self.visit(node)
        self.finalize()
        return value

    def visit(self, node: Any) -> str:
        try:
            children = node.children
        except AttributeError:
            return self.visit_leaf(node)
        else:
            with self.visit_node(node):
                return ''.join((self.visit(child) for child in children))

    @contextmanager
    def visit_node(self, node: Any) -> typing.Generator:
        self._check_type_rules(node)
        yield

    def _check_type_rules(self, node: Any) -> None:
        for rule in self._rule_type_instances.get(node.type, []):
            rule.feed_node(node)

    def visit_leaf(self, leaf: Any) -> str:
        self._check_type_rules(leaf)
        for rule in self._rule_value_instances.get(leaf.value, []):
            rule.feed_node(leaf)
        return leaf.prefix + leaf.value

    def initialize(self, node: Union[int, list[int]]) -> None:
        pass

    def finalize(self) -> None:
        pass

    def add_issue(self, node: Union[str, int], code: Union[None, str], message: Union[None, str, int]) -> None:
        issue = Issue(node, code, message)
        if issue not in self.issues:
            self.issues.append(issue)
        return True

    @classmethod
    def register_rule(cls: Union[str, None, dict[str, typing.Any]], *, value: Union[None, str, dict[str, typing.Any]]=None, values: tuple=(), type=None, types=()):
        """
        Use it as a class decorator::

            normalizer = Normalizer('grammar', 'config')
            @normalizer.register_rule(value='foo')
            class MyRule(Rule):
                error_code = 42
        """
        values = list(values)
        types = list(types)
        if value is not None:
            values.append(value)
        if type is not None:
            types.append(type)
        if not values and (not types):
            raise ValueError('You must register at least something.')

        def decorator(rule_cls: Any):
            for v in values:
                cls.rule_value_classes.setdefault(v, []).append(rule_cls)
            for t in types:
                cls.rule_type_classes.setdefault(t, []).append(rule_cls)
            return rule_cls
        return decorator

class NormalizerConfig:
    normalizer_class = Normalizer

    def create_normalizer(self, grammar: dict) -> Union[None, str, typing.Type]:
        if self.normalizer_class is None:
            return None
        return self.normalizer_class(grammar, self)

class Issue:

    def __init__(self, node, code, message) -> None:
        self.code = code
        '\n        An integer code that stands for the type of error.\n        '
        self.message = message
        '\n        A message (string) for the issue.\n        '
        self.start_pos = node.start_pos
        '\n        The start position position of the error as a tuple (line, column). As\n        always in |parso| the first line is 1 and the first column 0.\n        '
        self.end_pos = node.end_pos

    def __eq__(self, other: float) -> bool:
        return self.start_pos == other.start_pos and self.code == other.code

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.code, self.start_pos))

    def __repr__(self) -> typing.Text:
        return '<%s: %s>' % (self.__class__.__name__, self.code)

class Rule:

    def __init__(self, normalizer: Union[typing.Callable, list[str]]) -> None:
        self._normalizer = normalizer

    def is_issue(self, node: bool) -> None:
        raise NotImplementedError()

    def get_node(self, node: bool) -> bool:
        return node

    def _get_message(self, message: Union[bytes, None, dict, str], node: Union[dict[str, str], str, None, bytes]) -> Union[str, None, object]:
        if message is None:
            message = self.message
            if message is None:
                raise ValueError('The message on the class is not set.')
        return message

    def add_issue(self, node: Union[str, int], code: Union[None, str]=None, message: Union[None, str, int]=None) -> None:
        if code is None:
            code = self.code
            if code is None:
                raise ValueError('The error code on the class is not set.')
        message = self._get_message(message, node)
        self._normalizer.add_issue(node, code, message)

    def feed_node(self, node: str) -> None:
        if self.is_issue(node):
            issue_node = self.get_node(node)
            self.add_issue(issue_node)

class RefactoringNormalizer(Normalizer):

    def __init__(self, node_to_str_map) -> None:
        self._node_to_str_map = node_to_str_map

    def visit(self, node: Any) -> str:
        try:
            return self._node_to_str_map[node]
        except KeyError:
            return super().visit(node)

    def visit_leaf(self, leaf: Any) -> str:
        try:
            return self._node_to_str_map[leaf]
        except KeyError:
            return super().visit_leaf(leaf)