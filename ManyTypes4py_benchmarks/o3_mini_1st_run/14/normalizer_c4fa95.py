from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, TypeVar

T = TypeVar("T", bound="Rule")


class _NormalizerMeta(type):
    def __new__(cls: Type['_NormalizerMeta'], name: str, bases: Tuple[type, ...], dct: Dict[str, Any]) -> Type[Any]:
        new_cls = super().__new__(cls, name, bases, dct)
        new_cls.rule_value_classes: Dict[Any, List[Type['Rule']]] = {}
        new_cls.rule_type_classes: Dict[Any, List[Type['Rule']]] = {}
        return new_cls


class Normalizer(metaclass=_NormalizerMeta):
    _rule_type_instances: Dict[Any, List['Rule']]
    _rule_value_instances: Dict[Any, List['Rule']]

    def __init__(self, grammar: Any, config: Any) -> None:
        self.grammar = grammar
        self._config = config
        self.issues: List[Issue] = []
        self._rule_type_instances = self._instantiate_rules('rule_type_classes')
        self._rule_value_instances = self._instantiate_rules('rule_value_classes')

    def _instantiate_rules(self, attr: str) -> Dict[Any, List['Rule']]:
        rules_dict: Dict[Any, List['Rule']] = {}
        for base in type(self).mro():
            rules_map: Dict[Any, List[Type['Rule']]] = getattr(base, attr, {})
            for key, rule_classes in rules_map.items():
                instances = [rule_cls(self) for rule_cls in rule_classes]
                rules_dict.setdefault(key, []).extend(instances)
        return rules_dict

    def walk(self, node: Any) -> str:
        self.initialize(node)
        value: str = self.visit(node)
        self.finalize()
        return value

    def visit(self, node: Any) -> str:
        try:
            children = node.children
        except AttributeError:
            return self.visit_leaf(node)
        else:
            with self.visit_node(node):
                return ''.join(self.visit(child) for child in children)

    @contextmanager
    def visit_node(self, node: Any) -> Generator[None, None, None]:
        self._check_type_rules(node)
        yield

    def _check_type_rules(self, node: Any) -> None:
        for rule in self._rule_type_instances.get(getattr(node, "type", None), []):
            rule.feed_node(node)

    def visit_leaf(self, leaf: Any) -> str:
        self._check_type_rules(leaf)
        for rule in self._rule_value_instances.get(getattr(leaf, "value", None), []):
            rule.feed_node(leaf)
        # Assuming leaf has attributes 'prefix' and 'value' of type str.
        return leaf.prefix + leaf.value

    def initialize(self, node: Any) -> None:
        pass

    def finalize(self) -> None:
        pass

    def add_issue(self, node: Any, code: Any, message: str) -> bool:
        issue = Issue(node, code, message)
        if issue not in self.issues:
            self.issues.append(issue)
        return True

    @classmethod
    def register_rule(cls, *, value: Optional[Any] = None, values: Tuple[Any, ...] = (), type: Optional[Any] = None, types: Tuple[Any, ...] = ()) -> Callable[[Type[T]], Type[T]]:
        """
        Use it as a class decorator::

            normalizer = Normalizer('grammar', 'config')
            @normalizer.register_rule(value='foo')
            class MyRule(Rule):
                error_code = 42
        """
        values_list: List[Any] = list(values)
        types_list: List[Any] = list(types)
        if value is not None:
            values_list.append(value)
        if type is not None:
            types_list.append(type)
        if not values_list and not types_list:
            raise ValueError('You must register at least something.')

        def decorator(rule_cls: Type[T]) -> Type[T]:
            for v in values_list:
                cls.rule_value_classes.setdefault(v, []).append(rule_cls)
            for t in types_list:
                cls.rule_type_classes.setdefault(t, []).append(rule_cls)
            return rule_cls

        return decorator


class NormalizerConfig:
    normalizer_class: Optional[Type[Normalizer]] = Normalizer

    def create_normalizer(self, grammar: Any) -> Optional[Normalizer]:
        if self.normalizer_class is None:
            return None
        return self.normalizer_class(grammar, self)


class Issue:
    def __init__(self, node: Any, code: Any, message: str) -> None:
        self.code = code
        # An integer code that stands for the type of error.
        self.message = message
        # A message (string) for the issue.
        self.start_pos = node.start_pos  # Assumes node.start_pos is a tuple (line, column).
        self.end_pos = node.end_pos

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Issue):
            return NotImplemented
        return self.start_pos == other.start_pos and self.code == other.code

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.code, self.start_pos))

    def __repr__(self) -> str:
        return '<%s: %s>' % (self.__class__.__name__, self.code)


class Rule:
    code: Optional[Any] = None
    message: Optional[str] = None

    def __init__(self, normalizer: Normalizer) -> None:
        self._normalizer = normalizer

    def is_issue(self, node: Any) -> bool:
        raise NotImplementedError()

    def get_node(self, node: Any) -> Any:
        return node

    def _get_message(self, message: Optional[str], node: Any) -> str:
        if message is None:
            message = self.message
            if message is None:
                raise ValueError('The message on the class is not set.')
        return message

    def add_issue(self, node: Any, code: Optional[Any] = None, message: Optional[str] = None) -> None:
        if code is None:
            code = self.code
            if code is None:
                raise ValueError('The error code on the class is not set.')
        message_to_use: str = self._get_message(message, node)
        self._normalizer.add_issue(node, code, message_to_use)

    def feed_node(self, node: Any) -> None:
        if self.is_issue(node):
            issue_node = self.get_node(node)
            self.add_issue(issue_node)


class RefactoringNormalizer(Normalizer):
    def __init__(self, node_to_str_map: Dict[Any, str]) -> None:
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