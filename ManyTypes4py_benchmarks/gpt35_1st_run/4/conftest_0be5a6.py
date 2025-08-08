def cli_tests_contracts_version() -> str:
def raiden_testchain(blockchain_type: str, port_generator: Iterable[int], cli_tests_contracts_version: str) -> ContextManager[Dict[str, Any]]:
def removed_args() -> Optional[Dict[str, Any]]:
def changed_args() -> Optional[Dict[str, Any]]:
def cli_args(logs_storage: str, raiden_testchain: Dict[str, Any], local_matrix_servers: List[ParsedURL], removed_args: Optional[Dict[str, Any]], changed_args: Optional[Dict[str, Any]], environment_type: Environment) -> List[str]:
def raiden_spawner(tmp_path: str, request: Any) -> Callable[[List[str]], pexpect.spawn]:
