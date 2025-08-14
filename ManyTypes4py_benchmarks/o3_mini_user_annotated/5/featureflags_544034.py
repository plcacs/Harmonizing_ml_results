from typing import Optional, Iterable, Set, FrozenSet, Any
from hypothesis.internal.conjecture import utils as cu
from hypothesis.strategies._internal.strategies import SearchStrategy

FEATURE_LABEL = cu.calc_label_from_name("feature flag")


class FeatureFlags:
    def __init__(
        self,
        data: Optional[Any] = None,
        enabled: Iterable[str] = (),
        disabled: Iterable[str] = (),
        at_least_one_of: Iterable[str] = (),
    ) -> None:
        self.__data: Optional[Any] = data
        self.__is_disabled: dict[str, bool] = {}

        for f in enabled:
            self.__is_disabled[f] = False

        for f in disabled:
            self.__is_disabled[f] = True

        if self.__data is not None:
            self.__p_disabled: float = data.draw_integer(0, 254) / 255
        else:
            self.__p_disabled = 0.0

        self.__at_least_one_of: Set[str] = set(at_least_one_of)

    def is_enabled(self, name: str) -> bool:
        if self.__data is None or self.__data.frozen:
            return not self.__is_disabled.get(name, False)

        data = self.__data
        data.start_example(label=FEATURE_LABEL)

        oneof: Set[str] = self.__at_least_one_of
        is_disabled: bool = self.__data.draw_boolean(
            self.__p_disabled,
            forced=(
                False
                if len(oneof) == 1 and name in oneof
                else self.__is_disabled.get(name)
            ),
        )
        self.__is_disabled[name] = is_disabled
        if name in oneof and not is_disabled:
            oneof.clear()
        oneof.discard(name)
        data.stop_example()
        return not is_disabled

    def __repr__(self) -> str:
        enabled, disabled = [], []
        for name, is_disabled in self.__is_disabled.items():
            if is_disabled:
                disabled.append(name)
            else:
                enabled.append(name)
        return f"FeatureFlags({enabled=}, {disabled=})"


class FeatureStrategy(SearchStrategy):
    def __init__(self, at_least_one_of: Iterable[str] = ()) -> None:
        super().__init__()
        self._at_least_one_of: FrozenSet[str] = frozenset(at_least_one_of)

    def do_draw(self, data: Any) -> FeatureFlags:
        return FeatureFlags(data, at_least_one_of=self._at_least_one_of)