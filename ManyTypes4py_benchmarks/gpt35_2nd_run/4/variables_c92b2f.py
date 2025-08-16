def escape_for_format(string: str) -> str:
    """Escape `string` so that it can be used with `.format()`.

    >>> escaped = escape_for_format('{}')
    >>> escaped + '{}'.format(0)
    '{}0'
    """
    return string.translate(DUPLICATED_BRACKETS)

def settle_timeout(reveal_timeout: int) -> int:
    """
    NettingChannel default settle timeout for tests.
    If using geth we set it considerably lower since waiting for
    too many blocks to be mined is very costly time-wise.
    """
    return reveal_timeout * 3

def chain_id() -> int:
    return UNIT_CHAIN_ID

def settle_timeout_min() -> int:
    return TEST_SETTLE_TIMEOUT_MIN

def settle_timeout_max() -> int:
    return TEST_SETTLE_TIMEOUT_MAX

def reveal_timeout(number_of_nodes: int) -> int:
    """NettingChannel default reveal timeout for tests.

    If using geth we set it considerably lower since waiting for
    too many blocks to be mined is very costly time-wise.
    """
    return number_of_nodes * 4 + DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS

def retry_timeout() -> int:
    return DEFAULT_RETRY_TIMEOUT

def random_marker() -> str:
    """A random marker used to identify a pytest run.

    Some tests will spawn a private chain, the private chain will be one or
    more ethereum nodes on a new subprocesss. These nodes may fail to start on
    concurrent test runs, mostly because of port number conflicts, but even
    though the test fails to start its private chain it may run interacting
    with the geth process from a different test run! This leads to
    unreasonable test errors.

    This fixture creates a random marker used to distinguish pytest runs and
    avoid test failures. Note this could fail for other reasons and fail to
    detect unwanted interations if the user sets the PYTHONHASHSEED to the same
    value.
    """
    return remove_0x_prefix(HexStr(hex(random.getrandbits(100)))

def logs_storage(request, tmpdir) -> str:
    """Returns the path where debugging data should be saved.

    Use this to preserve the databases and logs necessary to debug test
    failures on the CI system.
    """
    short_path = shortened_artifacts_storage(request.node) or str(tmpdir)
    return unique_path(short_path)

def deposit() -> TokenAmount:
    """Raiden chain default deposit."""
    return TokenAmount(200)

def number_of_tokens() -> int:
    """Number of tokens pre-registered in the test Registry."""
    return 1

def register_tokens() -> bool:
    """Should fixture generated tokens be registered with raiden."""
    return True

def number_of_nodes() -> int:
    """Number of raiden nodes in the test network."""
    return 3

def channels_per_node() -> int:
    """Number of pre-created channels per test raiden node."""
    return 1

def retry_interval_initial(transport_protocol: TransportProtocol) -> int:
    return 2

def retry_interval_max(transport_protocol: TransportProtocol) -> int:
    return 2

def retries_before_backoff() -> int:
    return 2

def privatekey_seed(request) -> str:
    """Private key template, allow different keys to be used for each test to
    avoid collisions.
    """
    return escape_for_format(request.node.name) + ':{}'

def account_genesis_eth_balance() -> int:
    return DEFAULT_BALANCE

def token_amount(number_of_nodes: int, deposit: TokenAmount) -> int:
    total_per_node = 3 * (deposit + 1)
    total_token = total_per_node * number_of_nodes
    return total_token

def network_wait() -> float:
    """Time in seconds used to wait for network events."""
    return 10.0

def private_keys(number_of_nodes: int, privatekey_seed: str) -> List[str]:
    """Private keys for each raiden node."""
    result = [keccak(privatekey_seed.format(position).encode()) for position in range(number_of_nodes)]
    assert len(set(result)) == number_of_nodes, '`privatekey_seed` generate repeated keys'
    return result

def deploy_key(privatekey_seed: str) -> str:
    return keccak(privatekey_seed.format('deploykey').encode())

def blockchain_type(request) -> str:
    blockchain_type = request.config.option.blockchain_type
    if blockchain_type not in {client.value for client in EthClient}:
        raise ValueError(f'unknown blockchain_type {blockchain_type}')
    return blockchain_type

def blockchain_extra_config() -> Dict:
    return {}

def blockchain_number_of_nodes() -> int:
    """Number of nodes in the cluster, not the same as the number of raiden
    nodes. Used for all geth clusters.
    """
    return 1

def blockchain_key_seed(request) -> str:
    """Private key template for the nodes in the private blockchain, allows
    different keys to be used for each test to avoid collisions.
    """
    return escape_for_format(request.node.name) + 'cluster:{}'

def port_generator(request, worker_id) -> Iterator[int]:
    """count generator used to get a unique port number."""
    if worker_id == 'master':
        port_offset = 0
    else:
        port_offset = int(worker_id.replace('gw', '')) * 1000
    return get_free_port(request.config.getoption('base_port') + port_offset)

def eth_nodes_configuration(blockchain_number_of_nodes: int, blockchain_key_seed: str, port_generator: Iterator[int], blockchain_type: str, blockchain_extra_config: Dict) -> List[EthNodeDescription]:
    eth_nodes = []
    for position in range(blockchain_number_of_nodes):
        key = keccak(blockchain_key_seed.format(position).encode())
        eth_node = EthNodeDescription(private_key=key, rpc_port=next(port_generator), p2p_port=next(port_generator), miner=position == 0, extra_config=blockchain_extra_config, blockchain_type=blockchain_type)
        eth_nodes.append(eth_node)
    return eth_nodes

def environment_type() -> Environment:
    """Specifies the environment type"""
    return Environment.DEVELOPMENT

def unrecoverable_error_should_crash() -> bool:
    """For testing an UnrecoverableError should crash"""
    return True

def transport() -> str:
    """'all' replaced by parametrize in conftest.pytest_generate_tests"""
    return 'matrix'

def transport_protocol(transport: str) -> TransportProtocol:
    return TransportProtocol(transport)

def blockchain_query_interval() -> float:
    """
    Config setting (interval after which to check for new block.)  Set to this low value for the
    integration tests, where we use a block time of 1 second.
    """
    return 0.5

def skip_if_parity(blockchain_type: str):
    """Skip the test if it is run with a Parity node"""
    if blockchain_type == 'parity':
        pytest.skip('This test does not work with parity.')

def skip_if_not_parity(blockchain_type: str):
    """Skip the test if it is not run with a Parity node"""
    if blockchain_type != 'parity':
        pytest.skip('This test works only with parity.')

def skip_if_not_geth(blockchain_type: str):
    """Skip the test if it is run with a Geth node"""
    if blockchain_type != 'geth':
        pytest.skip('This test works only with geth.')

def start_raiden_apps() -> bool:
    """Determines if the raiden apps created at test setup should also be started"""
    return True

def enable_rest_api() -> bool:
    """Determines if the raiden apps created at test setup should also be started"""
    return False
