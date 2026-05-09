from contextlib import contextmanager
from typing import Dict, List, Type, TypeVar, Generic, Protocol, Any

T = TypeVar('T')

class _NormalizerMeta(type):

    def __new__(cls: type, name: str, bases: tuple[type, ...], dct: dict[str, Any]) -> type:
        new_cls = type.__new__(cls, name, bases, dct)
        new_cls.rule_value_classes: Dict[str, List[type]] = {}
        new_cls.rule_type_classes: Dict[str, List[type]] = {}
        return new_cls

class Normalizer(metaclass=_NormalizerMeta):
    _rule_type_instances: Dict[str, List['Rule']]
    _rule_value_instances: Dict[str, List['Rule']]
    _config: 'NormalizerConfig'

    def __init__(self, grammar: str, config: 'NormalizerConfig') -> None:
        self.grammar = grammar
        self._config = config
        self.issues: List['Issue'] = []
        self._rule_type_instances = self._instantiate_rules('rule_type_classes')
        self._rule_value_instances = self._instantiate_rules('rule_value_classes')

    def _instantiate_rules(self, attr: str) -> Dict[str, List['Rule']]:
        dct: Dict[str, List['Rule']] = {}
        for base in type(self).mro():
            rules_map = getattr(base, attr, {})
            for type_, rule_classes in rules_map.items():
                new = [rule_cls(self) for rule_cls in rule_classes]
                dct.setdefault(type_, []).extend(new)
        return dct

    def walk(self, node: T) -> T:
        self.initialize(node)
        value = self.visit(node)
        self.finalize()
        return value

    def visit(self, node: T) -> T:
        try:
            children = node.children
        except AttributeError:
            return self.visit_leaf(node)
        else:
            with self.visit_node(node):
                return ''.join((self.visit(child) for child in children))

    @contextmanager
    def visit_node(self, node: T) -> Iterator[T]:
        self._check_type_rules(node)
        yield

    def _check_type_rules(self, node: T) -> None:
        for rule in self._rule_type_instances.get(node.type, []):
            rule.feed_node(node)

    def visit_leaf(self, leaf: T) -> T:
        self._check_type_rules(leaf)
        for rule in self._rule_value_instances.get(leaf.value, []):
            rule.feed_node(leaf)
        return leaf.prefix + leaf.value

    def initialize(self, node: T) -> None:
        pass

    def finalize(self) -> None:
        pass

    def add_issue(self, node: T, code: int, message: str) -> bool:
        issue = Issue(node, code, message)
        if issue not in self.issues:
            self.issues.append(issue)
        return True

    @classmethod
    def register_rule(cls: type, *, value: str = None, values: tuple[str, ...] = (), type: str = None, types: tuple[str, ...] = ()) -> Callable[[type], type]:
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

        def decorator(rule_cls: type) -> type:
            for v in values:
                cls.rule_value_classes.setdefault(v, []).append(rule_cls)
            for t in types:
                cls.rule_type_classes.setdefault(t, []).append(rule_cls)
            return rule_cls
        return decorator

class NormalizerConfig:
    normalizer_class: type = Normalizer

    def create_normalizer(self, grammar: str) -> 'Normalizer':
        if self.normalizer_class is None:
            return None
        return self.normalizer_class(grammar, self)

class Issue(Protocol):
    code: int
    message: str
    start_pos: tuple[int, int]
    end_pos: tuple[int, int]

    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError

    def __ne__(self, other: Any) -> bool:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

class Rule(Generic[T]):
    def __init__(self, normalizer: 'Normalizer') -> None:
        self._normalizer = normalizer

    def is_issue(self, node: T) -> bool:
        raise NotImplementedError()

    def get_node(self, node: T) -> T:
        return node

    def _get_message(self, message: str, node: T) -> str:
        if message is None:
            message = self.message
            if message is None:
                raise ValueError('The message on the class is not set.')
        return message

    def add_issue(self, node: T, code: int = None, message: str = None) -> bool:
        if code is None:
            code = self.code
            if code is None:
                raise ValueError('The error code on the class is not set.')
        message = self._get_message(message, node)
        self._normalizer.add_issue(node, code, message)

    def feed_node(self, node: T) -> None:
        if self.is_issue(node):
            issue_node = self.get_node(node)
            self.add_issue(issue_node)

class RefactoringNormalizer(Normalizer):
    def __init__(self, node_to_str_map: Dict[Any, str]) -> None:
        self._node_to_str_map = node_to_str_map

    def visit(self, node: T) -> T:
        try:
            return self._node_to_str_map[node]
        except KeyError:
            return super().visit(node)

    def visit_leaf(self, leaf: T) -> T:
        try:
            return self._node_to_str_map[leaf]
        except KeyError:
            return super().visit_leaf(leaf)
