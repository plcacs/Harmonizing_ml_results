def make_model(pkg: str, name: str, code: str, language: str = 'sql', refs: List[Any] = None, sources: List[Any] = None, tags: List[str] = None, path: str = None, alias: str = None, config_kwargs: Dict[str, Any] = None, fqn_extras: List[str] = None, depends_on_macros: List[str] = None, version: Any = None, latest_version: Any = None, access: Any = None, patch_path: str = None) -> ModelNode:

def make_seed(pkg: str, name: str, path: str = None, loader: Any = None, alias: str = None, tags: List[str] = None, fqn_extras: List[str] = None, checksum: Any = None) -> SeedNode:

def make_source(pkg: str, source_name: str, table_name: str, path: str = None, loader: Any = None, identifier: str = None, fqn_extras: List[str] = None) -> SourceDefinition:

def make_macro(pkg: str, name: str, macro_sql: str, path: str = None, depends_on_macros: List[str] = None) -> Macro:

def make_unique_test(pkg: str, test_model: Any, column_name: str, path: str = None, refs: List[Any] = None, sources: List[Any] = None, tags: List[str] = None) -> GenericTestNode:

def make_not_null_test(pkg: str, test_model: Any, column_name: str, path: str = None, refs: List[Any] = None, sources: List[Any] = None, tags: List[str] = None) -> GenericTestNode:

def make_generic_test(pkg: str, test_name: str, test_model: Any, test_kwargs: Dict[str, Any], path: str = None, refs: List[Any] = None, sources: List[Any] = None, tags: List[str] = None, column_name: str = None) -> GenericTestNode:

def make_unit_test(pkg: str, test_name: str, test_model: Any) -> UnitTestDefinition:

def make_singular_test(pkg: str, name: str, sql: str, refs: List[Any] = None, sources: List[Any] = None, tags: List[str] = None, path: str = None, config_kwargs: Dict[str, Any] = None) -> SingularTestNode:

def make_exposure(pkg: str, name: str, path: str = None, fqn_extras: List[str] = None, owner: Any = None) -> Exposure:

def make_metric(pkg: str, name: str, path: str = None) -> Metric:

def make_group(pkg: str, name: str, path: str = None) -> Group:

def make_semantic_model(pkg: str, name: str, model: Any, path: str = None) -> SemanticModel:

def make_saved_query(pkg: str, name: str, metric: Any, path: str = None) -> SavedQuery:

def make_manifest(disabled: Dict[str, Any] = {}, docs: List[Any] = [], exposures: List[Any] = [], files: Dict[str, Any] = {}, groups: List[Any] = [], macros: List[Any] = [], metrics: List[Any] = [], nodes: List[Any] = [], saved_queries: List[Any] = [], selectors: Dict[str, Any] = {}, semantic_models: List[Any] = [], sources: List[Any] = [], unit_tests: List[Any] = []) -> Manifest:
