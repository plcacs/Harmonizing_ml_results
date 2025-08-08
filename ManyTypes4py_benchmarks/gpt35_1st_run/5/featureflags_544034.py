from typing import List, Tuple, Set

class FeatureFlags:
    def __init__(self, data=None, enabled: List[str] = (), disabled: List[str] = (), at_least_one_of: List[str] = ()):
        ...

    def is_enabled(self, name: str) -> bool:
        ...

    def __repr__(self) -> str:
        ...

class FeatureStrategy(SearchStrategy):
    def __init__(self, at_least_one_of: Set[str] = ()):
        ...

    def do_draw(self, data) -> FeatureFlags:
        ...
