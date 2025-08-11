from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.common import monkeypatch

class AbstractLazyValue:

    def __init__(self, data, min: int=1, max: int=1) -> None:
        self.data = data
        self.min = min
        self.max = max

    def __repr__(self) -> typing.Text:
        return '<%s: %s>' % (self.__class__.__name__, self.data)

    def infer(self):
        raise NotImplementedError

class LazyKnownValue(AbstractLazyValue):
    """data is a Value."""

    def infer(self):
        return ValueSet([self.data])

class LazyKnownValues(AbstractLazyValue):
    """data is a ValueSet."""

    def infer(self):
        return self.data

class LazyUnknownValue(AbstractLazyValue):

    def __init__(self, min: int=1, max: int=1) -> None:
        super().__init__(None, min, max)

    def infer(self):
        return NO_VALUES

class LazyTreeValue(AbstractLazyValue):

    def __init__(self, context, node, min: int=1, max: int=1) -> None:
        super().__init__(node, min, max)
        self.context = context
        self._predefined_names = dict(context.predefined_names)

    def infer(self):
        with monkeypatch(self.context, 'predefined_names', self._predefined_names):
            return self.context.infer_node(self.data)

def get_merged_lazy_value(lazy_values: Union[list[list[typing.Any]], list, list[dict[str, typing.Any]]]) -> Union[MergedLazyValues, list[typing.Any], dict[str, typing.Any]]:
    if len(lazy_values) > 1:
        return MergedLazyValues(lazy_values)
    else:
        return lazy_values[0]

class MergedLazyValues(AbstractLazyValue):
    """data is a list of lazy values."""

    def infer(self):
        return ValueSet.from_sets((l.infer() for l in self.data))