```python
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.common import monkeypatch
from typing import Any, Dict, List, Union


class AbstractLazyValue:
    def __init__(self, data: Any, min: int = 1, max: int = 1) -> None:
        self.data = data
        self.min = min
        self.max = max

    def __repr__(self) -> str:
        return '<%s: %s>' % (self.__class__.__name__, self.data)

    def infer(self) -> ValueSet:
        raise NotImplementedError


class LazyKnownValue(AbstractLazyValue):
    """data is a Value."""
    def infer(self) -> ValueSet:
        return ValueSet([self.data])


class LazyKnownValues(AbstractLazyValue):
    """data is a ValueSet."""
    def infer(self) -> ValueSet:
        return self.data


class LazyUnknownValue(AbstractLazyValue):
    def __init__(self, min: int = 1, max: int = 1) -> None:
        super().__init__(None, min, max)

    def infer(self) -> ValueSet:
        return NO_VALUES


class LazyTreeValue(AbstractLazyValue):
    def __init__(self, context: Any, node: Any, min: int = 1, max: int = 1) -> None:
        super().__init__(node, min, max)
        self.context = context
        # We need to save the predefined names. It's an unfortunate side effect
        # that needs to be tracked otherwise results will be wrong.
        self._predefined_names: Dict[str, Any] = dict(context.predefined_names)

    def infer(self) -> ValueSet:
        with monkeypatch(self.context, 'predefined_names', self._predefined_names):
            return self.context.infer_node(self.data)


def get_merged_lazy_value(lazy_values: List[AbstractLazyValue]) -> AbstractLazyValue:
    if len(lazy_values) > 1:
        return MergedLazyValues(lazy_values)
    else:
        return lazy_values[0]


class MergedLazyValues(AbstractLazyValue):
    """data is a list of lazy values."""
    def infer(self) -> ValueSet:
        return ValueSet.from_sets(l.infer() for l in self.data)
```