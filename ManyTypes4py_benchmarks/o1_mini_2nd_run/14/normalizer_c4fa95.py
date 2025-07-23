from contextlib import contextmanager
from typing import (
    Dict,
    List,
    Type,
    Optional,
    Callable,
    Generator,
    Any,
    Tuple,
)


class _NormalizerMeta(type):
    def __new__(
        cls, name: str, bases: tuple, dct: dict
    ) -> "Normalizer":
        new_cls = type.__new__(cls, name, bases, dct)
        new_cls.rule_value_classes: Dict[str, List[Type["Rule"]]] = {}
        new_cls.rule_type_classes: Dict[str, List[Type["Rule"]]] = {}
        return new_cls


class Normalizer(metaclass=_NormalizerMeta):
    _rule_type_instances: Dict[str, List["Rule"]] = {}
    _rule_value_instances: Dict[str, List["Rule"]] = {}

    def __init__(self, grammar: Any, config: "NormalizerConfig") -> None:
        self.grammar: Any = grammar
        self._config: "NormalizerConfig" = config
        self.issues: List["Issue"] = []
        self._rule_type_instances: Dict[str, List["Rule"]] = self._instantiate_rules(
            "rule_type_classes"
        )
        self._rule_value_instances: Dict[str, List["Rule"]] = self._instantiate_rules(
            "rule_value_classes"
        )

    def _instantiate_rules(self, attr: str) -> Dict[str, List["Rule"]]:
        dct: Dict[str, List["Rule"]] = {}
        for base in type(self).mro():
            rules_map = getattr(base, attr, {})
            for type_, rule_classes in rules_map.items():
                new = [rule_cls(self) for rule_cls in rule_classes]
                dct.setdefault(type_, []).extend(new)
        return dct

    def walk(self, node: Any) -> str:
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
                return "".join((self.visit(child) for child in children))

    @contextmanager
    def visit_node(self, node: Any) -> Generator[None, None, None]:
        self._check_type_rules(node)
        yield

    def _check_type_rules(self, node: Any) -> None:
        rules = self._rule_type_instances.get(node.type, [])
        for rule in rules:
            rule.feed_node(node)

    def visit_leaf(self, leaf: Any) -> str:
        self._check_type_rules(leaf)
        rules = self._rule_value_instances.get(leaf.value, [])
        for rule in rules:
            rule.feed_node(leaf)
        return leaf.prefix + leaf.value

    def initialize(self, node: Any) -> None:
        pass

    def finalize(self) -> None:
        pass

    def add_issue(self, node: Any, code: int, message: str) -> bool:
        issue = Issue(node, code, message)
        if issue not in self.issues:
            self.issues.append(issue)
        return True

    @classmethod
    def register_rule(
        cls,
        *,
        value: Optional[str] = None,
        values: Optional[List[str]] = None,
        type: Optional[str] = None,
        types: Optional[List[str]] = None,
    ) -> Callable[[Type["Rule"]], Type["Rule"]]:
        """
        Use it as a class decorator::

            normalizer = Normalizer('grammar', 'config')
            @normalizer.register_rule(value='foo')
            class MyRule(Rule):
                error_code = 42
        """
        if values is None:
            values = []
        if types is None:
            types = []
        if value is not None:
            values.append(value)
        if type is not None:
            types.append(type)
        if not values and not types:
            raise ValueError("You must register at least something.")

        def decorator(rule_cls: Type["Rule"]) -> Type["Rule"]:
            for v in values:
                cls.rule_value_classes.setdefault(v, []).append(rule_cls)
            for t in types:
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
    def __init__(self, node: Any, code: int, message: str) -> None:
        self.code: int = code
        """
        An integer code that stands for the type of error.
        """
        self.message: str = message
        """
        A message (string) for the issue.
        """
        self.start_pos: Tuple[int, int] = node.start_pos
        """
        The start position position of the error as a tuple (line, column). As
        always in |parso| the first line is 1 and the first column 0.
        """
        self.end_pos: Tuple[int, int] = node.end_pos

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Issue):
            return False
        return self.start_pos == other.start_pos and self.code == other.code

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.code, self.start_pos))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.code}>"


class Rule:
    code: Optional[int] = None
    message: Optional[str] = None

    def __init__(self, normalizer: Normalizer) -> None:
        self._normalizer: Normalizer = normalizer

    def is_issue(self, node: Any) -> bool:
        raise NotImplementedError()

    def get_node(self, node: Any) -> Any:
        return node

    def _get_message(self, message: Optional[str], node: Any) -> str:
        if message is None:
            if self.message is None:
                raise ValueError("The message on the class is not set.")
            message = self.message
        return message

    def add_issue(
        self, node: Any, code: Optional[int] = None, message: Optional[str] = None
    ) -> bool:
        if code is None:
            if self.code is None:
                raise ValueError("The error code on the class is not set.")
            code = self.code
        message = self._get_message(message, node)
        self._normalizer.add_issue(node, code, message)
        return True

    def feed_node(self, node: Any) -> None:
        if self.is_issue(node):
            issue_node = self.get_node(node)
            self.add_issue(issue_node)


class RefactoringNormalizer(Normalizer):
    def __init__(self, node_to_str_map: Dict[Any, str]) -> None:
        super().__init__(grammar=None, config=None)  # Adjust as needed
        self._node_to_str_map: Dict[Any, str] = node_to_str_map

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
