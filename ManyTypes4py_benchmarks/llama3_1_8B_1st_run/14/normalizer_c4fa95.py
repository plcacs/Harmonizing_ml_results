from contextlib import contextmanager
from typing import Dict, List, TypeVar, Generic, Protocol, Any, Type, Optional, Callable, ClassVar

T = TypeVar('T')

class _NormalizerMeta(type):

    def __new__(cls: Type, name: str, bases: tuple, dct: dict) -> type:
        new_cls = type.__new__(cls, name, bases, dct)
        new_cls.rule_value_classes: Dict[str, List[type]] = {}
        new_cls.rule_type_classes: Dict[str, List[type]] = {}
        return new_cls

class Normalizer(metaclass=_NormalizerMeta):
    _rule_type_instances: Dict[str, List['Normalizer']]
    _rule_value_instances: Dict[str, List['Normalizer']]
    _config: object

    def __init__(self, grammar: str, config: object):
        self.grammar = grammar
        self._config = config
        self.issues: List['Issue'] = []
        self._rule_type_instances = self._instantiate_rules('rule_type_classes')
        self._rule_value_instances = self._instantiate_rules('rule_value_classes')

    def _instantiate_rules(self, attr: str) -> Dict[str, List['Normalizer']]:
        dct: Dict[str, List['Normalizer']] = {}
        for base in type(self).mro():
            rules_map: Dict[str, List[type]] = getattr(base, attr, {})
            for type_, rule_classes in rules_map.items():
                new = [rule_cls(self) for rule_cls in rule_classes]
                dct.setdefault(type_, []).extend(new)
        return dct

    def walk(self, node: object) -> T:
        self.initialize(node)
        value = self.visit(node)
        self.finalize()
        return value

    def visit(self, node: object) -> str:
        try:
            children = node.children
        except AttributeError:
            return self.visit_leaf(node)
        else:
            with self.visit_node(node):
                return ''.join((self.visit(child) for child in children))

    @contextmanager
    def visit_node(self, node: object):
        self._check_type_rules(node)
        yield

    def _check_type_rules(self, node: object) -> None:
        for rule in self._rule_type_instances.get(node.type, []):
            rule.feed_node(node)

    def visit_leaf(self, leaf: object) -> str:
        self._check_type_rules(leaf)
        for rule in self._rule_value_instances.get(leaf.value, []):
            rule.feed_node(leaf)
        return leaf.prefix + leaf.value

    def initialize(self, node: object) -> None:
        pass

    def finalize(self) -> None:
        pass

    def add_issue(self, node: object, code: int, message: str) -> bool:
        issue = Issue(node, code, message)
        if issue not in self.issues:
            self.issues.append(issue)
        return True

    @classmethod
    def register_rule(cls, *, value: Optional[str] = None, values: tuple = (), type: Optional[str] = None, types: tuple = ()) -> Callable[[type], type]:
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
    normalizer_class: ClassVar[Type['Normalizer']] = Normalizer

    def create_normalizer(self, grammar: str) -> Optional['Normalizer']:
        if self.normalizer_class is None:
            return None
        return self.normalizer_class(grammar, self)

class Issue(Generic[T]):
    code: int
    message: str
    start_pos: tuple[int, int]
    end_pos: tuple[int, int]

    def __init__(self, node: object, code: int, message: str):
        self.code = code
        self.message = message
        self.start_pos = node.start_pos
        self.end_pos = node.end_pos

    def __eq__(self, other: object) -> bool:
        return self.start_pos == other.start_pos and self.code == other.code

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.code, self.start_pos))

    def __repr__(self) -> str:
        return '<%s: %s>' % (self.__class__.__name__, self.code)

class Rule(Protocol):
    _normalizer: object
    code: Optional[int]
    message: Optional[str]

    def is_issue(self, node: object) -> bool:
        ...

    def get_node(self, node: object) -> object:
        ...

    def _get_message(self, message: str, node: object) -> str:
        ...

    def add_issue(self, node: object, code: Optional[int] = None, message: Optional[str] = None) -> None:
        ...

    def feed_node(self, node: object) -> None:
        ...

class RuleClass(type):
    def __new__(cls, name: str, bases: tuple, dct: dict) -> type:
        new_cls = type.__new__(cls, name, bases, dct)
        new_cls.__origin__ = Rule
        return new_cls

class RefactoringNormalizer(Normalizer):
    _node_to_str_map: Dict[Any, str]

    def __init__(self, node_to_str_map: Dict[Any, str]):
        self._node_to_str_map = node_to_str_map

    def visit(self, node: object) -> str:
        try:
            return self._node_to_str_map[node]
        except KeyError:
            return super().visit(node)

    def visit_leaf(self, leaf: object) -> str:
        try:
            return self._node_to_str_map[leaf]
        except KeyError:
            return super().visit_leaf(leaf)
