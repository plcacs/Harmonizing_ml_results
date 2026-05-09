from typing import List, Dict
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.common import monkeypatch

class AbstractLazyValue:
    """Base class for lazy values."""

    def __init__(self, data: object, min: int = 1, max: int = 1) -> None:
        self.data: object = data
        self.min: int = min
        self.max: int = max

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {self.data}>'

    def infer(self) -> ValueSet:
        raise NotImplementedError

class LazyKnownValue(AbstractLazyValue):
    """Lazy value representing a known value."""

    def __init__(self, data: 'Value') -> None:
        super().__init__(data, min=1, max=1)

    def infer(self) -> ValueSet:
        return ValueSet([self.data])

class LazyKnownValues(AbstractLazyValue):
    """Lazy value representing a set of known values."""

    def __init__(self, data: ValueSet) -> None:
        super().__init__(data, min=1, max=1)

    def infer(self) -> ValueSet:
        return self.data

class LazyUnknownValue(AbstractLazyValue):
    """Lazy value representing an unknown value."""

    def __init__(self, min: int = 1, max: int = 1) -> None:
        super().__init__(None, min, max)

    def infer(self) -> ValueSet:
        return NO_VALUES

class LazyTreeValue(AbstractLazyValue):
    """Lazy value representing a tree value."""

    def __init__(self, context: object, node: object, min: int = 1, max: int = 1) -> None:
        super().__init__(node, min, max)
        self.context: object = context
        self._predefined_names: Dict[str, object] = dict(context.predefined_names)

    def infer(self) -> ValueSet:
        with monkeypatch(self.context, 'predefined_names', self._predefined_names):
            return self.context.infer_node(self.data)

def get_merged_lazy_value(lazy_values: List['AbstractLazyValue']) -> 'AbstractLazyValue':
    if len(lazy_values) > 1:
        return MergedLazyValues(lazy_values)
    else:
        return lazy_values[0]

class MergedLazyValues(AbstractLazyValue):
    """Lazy value representing a merged set of values."""

    def __init__(self, data: List['AbstractLazyValue']) -> None:
        super().__init__(data, min=1, max=1)

    def infer(self) -> ValueSet:
        return ValueSet.from_sets((l.infer() for l in self.data))
