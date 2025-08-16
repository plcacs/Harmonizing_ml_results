    def __init__(self, project: Any, manifest: Any, root_project: Any) -> None:
    def parse_file(self, block: Any, dct: Optional[Dict[str, Any]] = None) -> None:
    def parse_from_dict(self, dct: Dict[str, Any], validate: bool = True) -> GenericTestNode:
    def parse_column_tests(self, block: Any, column: UnparsedColumn, version: Optional[str]) -> None:
    def create_test_node(self, target: Any, path: str, config: Dict[str, Any], tags: List[str], fqn: str, name: str, raw_code: str, test_metadata: Dict[str, Any], file_key_name: str, column_name: str, description: str) -> GenericTestNode:
    def parse_generic_test(self, target: Any, data_test: Union[str, Dict[str, Any]], tags: List[str], column_name: str, schema_file_id: str, version: str) -> GenericTestNode:
    def _lookup_attached_node(self, target: Any, version: str) -> Optional[ManifestNode]:
    def store_env_vars(self, target: Any, schema_file_id: str, env_vars: Dict[str, Any]) -> None:
    def render_test_update(self, node: GenericTestNode, config: ContextConfig, builder: TestBuilder, schema_file_id: str) -> None:
    def parse_node(self, block: FileBlock) -> GenericTestNode:
    def add_test_node(self, block: FileBlock, node: GenericTestNode) -> None:
    def render_with_context(self, node: GenericTestNode, config: ContextConfig) -> None:
    def parse_test(self, target_block: TestBlock, data_test: Union[str, Dict[str, Any]], column: Optional[UnparsedColumn], version: str) -> None:
    def parse_tests(self, block: Testable) -> None:
    def parse_versioned_tests(self, block: VersionedTestBlock) -> None:
    def generate_unique_id(self, resource_name: str, hash: Optional[str] = None) -> str:
