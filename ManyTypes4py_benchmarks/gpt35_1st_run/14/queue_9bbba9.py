    def __init__(self, graph: nx.DiGraph, manifest: Manifest, selected: Set[str], preserve_edges: bool = True) -> None:
    def get_selected_nodes(self) -> Set[str]:
    def _include_in_cost(self, node_id: str) -> bool:
    @staticmethod
    def _grouped_topological_sort(graph: nx.DiGraph) -> Generator[List[str], None, None]:
    def _get_scores(self, graph: nx.DiGraph) -> Dict[str, int]:
    def get(self, block: bool = True, timeout: Optional[float] = None) -> GraphMemberNode:
    def __len__(self) -> int:
    def empty(self) -> bool:
    def _already_known(self, node: str) -> bool:
    def _find_new_additions(self, candidates: List[str]) -> None:
    def mark_done(self, node_id: str) -> None:
    def _mark_in_progress(self, node_id: str) -> None:
    def wait_until_something_was_done(self) -> int:
