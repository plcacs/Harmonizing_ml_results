def get_stripped_prefix(source: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    ...

def build_catalog_table(data: Dict[str, Any]) -> CatalogTable:
    ...

class Catalog(Dict[CatalogKey, CatalogTable]):

    def __init__(self, columns: List[Dict[str, Any]]) -> None:
        ...

    def get_table(self, data: Dict[str, Any]) -> CatalogTable:
        ...

    def add_column(self, data: Dict[str, Any]) -> None:
        ...

    def make_unique_id_map(self, manifest: Manifest, selected_node_ids: Optional[Set[UniqueId]] = None) -> Tuple[Dict[UniqueId, CatalogTable], Dict[UniqueId, CatalogTable]]:
        ...

def format_stats(stats: Dict[str, Any]) -> StatsDict:
    ...

def mapping_key(node: ResultNode) -> CatalogKey:
    ...

def get_unique_id_mapping(manifest: Manifest) -> Tuple[Dict[CatalogKey, UniqueId], Dict[CatalogKey, Set[UniqueId]]]:
    ...

class GenerateTask(CompileTask):

    def run(self) -> CatalogArtifact:
        ...

    def get_node_selector(self) -> ResourceTypeSelector:
        ...

    def get_catalog_results(self, nodes: Dict[UniqueId, CatalogTable], sources: Dict[UniqueId, CatalogTable], generated_at: datetime, compile_results: Optional[List[ResultNode]], errors: Optional[List[str]) -> CatalogArtifact:
        ...

    @classmethod
    def interpret_results(cls, results: Optional[CatalogArtifact]) -> bool:
        ...

    @staticmethod
    def _get_nodes_from_ids(manifest: Manifest, node_ids: Set[UniqueId]) -> List[ResultNode]:
        ...

    def _get_selected_source_ids(self) -> Set[UniqueId]:
        ...
