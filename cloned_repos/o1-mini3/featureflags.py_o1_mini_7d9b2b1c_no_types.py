from typing import Optional, Iterable, Set, Dict, Any, Tuple
from hypothesis.internal.conjecture import utils as cu
from hypothesis.strategies._internal.strategies import SearchStrategy
FEATURE_LABEL = cu.calc_label_from_name('feature flag')

class FeatureFlags:
    """Object that can be used to control a number of feature flags for a
    given test run.

    This enables an approach to data generation called swarm testing (
    see Groce, Alex, et al. "Swarm testing." Proceedings of the 2012
    International Symposium on Software Testing and Analysis. ACM, 2012), in
    which generation is biased by selectively turning some features off for
    each test case generated. When there are many interacting features this can
    find bugs that a pure generation strategy would otherwise have missed.

    FeatureFlags are designed to "shrink open", so that during shrinking they
    become less restrictive. This allows us to potentially shrink to smaller
    test cases that were forbidden during the generation phase because they
    required disabled features.
    """

    def __init__(self, data=None, enabled=(), disabled=(), at_least_one_of=()):
        self.__data = data
        self.__is_disabled: Dict[str, bool] = {}
        for f in enabled:
            self.__is_disabled[f] = False
        for f in disabled:
            self.__is_disabled[f] = True
        if self.__data is not None:
            self.__p_disabled: float = data.draw_integer(0, 254) / 255
        else:
            self.__p_disabled = 0.0
        self.__at_least_one_of: Set[str] = set(at_least_one_of)

    def is_enabled(self, name):
        """Tests whether the feature named ``name`` should be enabled on this
        test run."""
        if self.__data is None or getattr(self.__data, 'frozen', False):
            return not self.__is_disabled.get(name, False)
        data = self.__data
        data.start_example(label=FEATURE_LABEL)
        oneof = self.__at_least_one_of
        is_disabled: bool = data.draw_boolean(self.__p_disabled, forced=False if len(oneof) == 1 and name in oneof else self.__is_disabled.get(name))
        self.__is_disabled[name] = is_disabled
        if name in oneof and (not is_disabled):
            oneof.clear()
        oneof.discard(name)
        data.stop_example()
        return not is_disabled

    def __repr__(self):
        enabled: List[str] = []
        disabled: List[str] = []
        for name, is_disabled in self.__is_disabled.items():
            if is_disabled:
                disabled.append(name)
            else:
                enabled.append(name)
        return f'FeatureFlags(enabled={enabled}, disabled={disabled})'

class FeatureStrategy(SearchStrategy):

    def __init__(self, at_least_one_of=()):
        super().__init__()
        self._at_least_one_of: frozenset[str] = frozenset(at_least_one_of)

    def do_draw(self, data):
        return FeatureFlags(data, at_least_one_of=self._at_least_one_of)