def cli_tests_contracts_version() -> str:
    return RAIDEN_CONTRACT_VERSION

def raiden_testchain(blockchain_type: str, port_generator: Iterable[int], cli_tests_contracts_version: str) -> ContextManager[Dict[str, Any]]:
    ...

def removed_args() -> Optional[None]:
    return None

def changed_args() -> Optional[None]:
    return None

def cli_args(logs_storage: str, raiden_testchain: Dict[str, Any], local_matrix_servers: List[str], removed_args: Optional[None], changed_args: Optional[None], environment_type: Environment) -> List[str]:
    ...

def raiden_spawner(tmp_path: str, request: Any) -> Callable[[List[str]], pexpect.spawn]:
    ...
