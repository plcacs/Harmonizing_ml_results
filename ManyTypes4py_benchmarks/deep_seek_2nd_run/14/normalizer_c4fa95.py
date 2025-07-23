from contextlib import contextmanager
from typing import Dict, List, Type, TypeVar, Any, Optional, Union, Iterator, Tuple, Set, Callable, Generic, cast

T = TypeVar('T')
NodeType = TypeVar('NodeType')
LeafType = TypeVar('LeafType')

class _NormalizerMeta(type):
    def __new__(cls, name: str, bases: Tuple[type, ...], dct: Dict[str, Any]) -> '_NormalizerMeta':
        new_cls = cast('_NormalizerMeta', type.__new__(cls, name, bases, dct))
        new_cls.rule_value_classes: Dict[str, List[Type['Rule']] = {}
        new_cls.rule_type_classes: Dict[str, List[Type['Rule']] = {}
        return new_cls

class Normalizer(metaclass=_NormalizerMeta):
    _rule_type_instances: Dict[str, List['Rule']]
    _rule_value_instances: Dict[str, List['Rule']]
    issues: List['Issue']

    def __init__(self, grammar: Any, config: 'NormalizerConfig') -> None:
        self.grammar = grammar
        self._config = config
        self.issues = []
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

    def walk(self, node: Union[NodeType, LeafType]) -> str:
        self.initialize(node)
        value = self.visit(node)
        self.finalize()
        return value

    def visit(self, node: Union[NodeType, LeafType]) -> str:
        try:
            children = node.children
        except AttributeError:
            return self.visit_leaf(node)
        else:
            with self.visit_node(node):
                return ''.join((self.visit(child) for child in children))

    @contextmanager
    def visit_node(self, node: NodeType) -> Iterator[None]:
        self._check_type_rules(node)
        yield

    def _check_type_rules(self, node: Union[NodeType, LeafType]) -> None:
        for rule in self._rule_type_instances.get(node.type, []):
            rule.feed_node(node)

    def visit_leaf(self, leaf: LeafType) -> str:
        self._check_type_rules(leaf)
        for rule in self._rule_value_instances.get(leaf.value, []):
            rule.feed_node(leaf)
        return leaf.prefix + leaf.value

    def initialize(self, node: Union[NodeType, LeafType]) -> None:
        pass

    def finalize(self) -> None:
        pass

    def add_issue(self, node: Union[NodeType, LeafType], code: int, message: str) -> bool:
        issue = Issue(node, code, message)
        if issue not in self.issues:
            self.issues.append(issue)
        return True

    @classmethod
    def register_rule(
        cls,
        *,
        value: Optional[str] = None,
        values: Tuple[str, ...] = (),
        type: Optional[str] = None,
        types: Tuple[str, ...] = ()
    ) -> Callable[[Type[T]], Type[T]]:
        values_list = list(values)
        types_list = list(types)
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
    normalizer_class: Type[Normalizer] = Normalizer

    def create_normalizer(self, grammar: Any) -> Optional[Normalizer]:
        if self.normalizer_class is None:
            return None
        return self.normalizer_class(grammar, self)

class Issue:
    def __init__(self, node: Union[NodeType, LeafType], code: int, message: str) -> None:
        self.code = code
        self.message = message
        self.start_pos = node.start_pos
        self.end_pos = node.end_pos

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Issue):
            return NotImplemented
        return self.start_pos == other.start_pos and self.code == other.code

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.code, self.start_pos))

    def __repr__(self) -> str:
        return '<%s: %s>' % (self.__class__.__name__, self.code)

class Rule:
    code: Optional[int] = None
    message: Optional[str] = None

    def __init__(self, normalizer: Normalizer) -> None:
        self._normalizer = normalizer

    def is_issue(self, node: Union[NodeType, LeafType]) -> bool:
        raise NotImplementedError()

    def get_node(self, node: Union[NodeType, LeafType]) -> Union[NodeType, LeafType]:
        return node

    def _get_message(self, message: Optional[str], node: Union[NodeType, LeafType]) -> str:
        if message is None:
            message = self.message
            if message is None:
                raise ValueError('The message on the class is not set.')
        return message

    def add_issue(
        self,
        node: Union[NodeType, LeafType],
        code: Optional[int] = None,
        message: Optional[str] = None
    ) -> None:
        if code is None:
            code = self.code
            if code is None:
                raise ValueError('The error code on the class is not set.')
        message_str = self._get_message(message, node)
        self._normalizer.add_issue(node, code, message_str)

    def feed_node(self, node: Union[NodeType, LeafType]) -> None:
        if self.is_issue(node):
            issue_node = self.get_node(node)
            self.add_issue(issue_node)

class RefactoringNormalizer(Normalizer):
    def __init__(self, node_to_str_map: Dict[Union[NodeType, LeafType], str]) -> None:
        self._node_to_str_map = node_to_str_map

    def visit(self, node: Union[NodeType, LeafType]) -> str:
        try:
            return self._node_to_str_map[node]
        except KeyError:
            return super().visit(node)

    def visit_leaf(self, leaf: LeafType) -> str:
        try:
            return self._node_to_str_map[leaf]
        except KeyError:
            return super().visit_leaf(leaf)
