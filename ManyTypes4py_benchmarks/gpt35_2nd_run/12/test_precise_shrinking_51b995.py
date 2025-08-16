from typing import Callable, TypeVar

T = TypeVar('T')

def safe_draw(data: ConjectureData, strategy: st.SearchStrategy) -> T:
def precisely_shrink(strategy: st.SearchStrategy, is_interesting: Callable[[T], bool] = lambda x: True, initial_condition: Callable[[T], bool] = lambda x: True, end_marker: st.SearchStrategy = st.integers(), seed: int = 0) -> Tuple[ConjectureData, T]:
def minimal_for_strategy(s: st.SearchStrategy) -> ConjectureData:
def minimal_nodes_for_strategy(s: st.SearchStrategy) -> int:
def find_random(s: st.SearchStrategy, condition: Callable[[T], bool], seed: Optional[int] = None) -> Tuple[ConjectureData, T]:
def shrinks(strategy: st.SearchStrategy, nodes: List[ConjectureDataNode], allow_sloppy: bool = True, seed: int = 0) -> List[T]:
