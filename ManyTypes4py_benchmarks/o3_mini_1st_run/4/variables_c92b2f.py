import random
from enum import Enum
from typing import Any, Dict, Iterator, List
import pytest
from eth_typing import HexStr
from eth_utils import keccak, remove_0x_prefix
from raiden.constants import Environment, EthClient
from raiden.network.utils import get_free_port
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS, DEFAULT_RETRY_TIMEOUT
from raiden.tests.fixtures.constants import DEFAULT_BALANCE
from raiden.tests.utils.ci import shortened_artifacts_storage
from raiden.tests.utils.eth_node import EthNodeDescription
from raiden.tests.utils.factories import UNIT_CHAIN_ID
from raiden.tests.utils.tests import unique_path
from raiden.utils.typing import Port, TokenAmount
from raiden_contracts.constants import TEST_SETTLE_TIMEOUT_MAX, TEST_SETTLE_TIMEOUT_MIN

DUPLICATED_BRACKETS = str.maketrans({'{': '{{', '}': '}}'})
__all__ = (
    'account_genesis_eth_balance', 'blockchain_extra_config', 'blockchain_key_seed',
    'blockchain_number_of_nodes', 'blockchain_query_interval', 'blockchain_type',
    'chain_id', 'channels_per_node', 'deploy_key', 'deposit', 'enable_rest_api',
    'environment_type', 'eth_nodes_configuration', 'logs_storage', 'network_wait',
    'number_of_nodes', 'number_of_tokens', 'port_generator', 'private_keys',
    'privatekey_seed', 'random_marker', 'register_tokens', 'retries_before_backoff',
    'retry_interval_initial', 'retry_interval_max', 'retry_timeout', 'reveal_timeout',
    'settle_timeout', 'settle_timeout_max', 'settle_timeout_min', 'skip_if_not_geth',
    'skip_if_not_parity', 'skip_if_parity', 'start_raiden_apps', 'token_amount',
    'transport', 'transport_protocol', 'unrecoverable_error_should_crash'
)

class TransportProtocol(Enum):
    MATRIX = 'matrix'


def escape_for_format(string: str) -> str:
    """Escape `string` so that it can be used with `.format()`.

    >>> escaped = escape_for_format('{}')
    >>> escaped + '{}'.format(0)
    '{}0'
    """
    return string.translate(DUPLICATED_BRACKETS)


@pytest.fixture
def settle_timeout(reveal_timeout: int) -> int:
    """
    NettingChannel default settle timeout for tests.
    If using geth we set it considerably lower since waiting for
    too many blocks to be mined is very costly time-wise.
    """
    return reveal_timeout * 3


@pytest.fixture
def chain_id() -> int:
    return UNIT_CHAIN_ID


@pytest.fixture
def settle_timeout_min() -> int:
    return TEST_SETTLE_TIMEOUT_MIN


@pytest.fixture
def settle_timeout_max() -> int:
    return TEST_SETTLE_TIMEOUT_MAX


@pytest.fixture
def reveal_timeout(number_of_nodes: int) -> int:
    """NettingChannel default reveal timeout for tests.

    If using geth we set it considerably lower since waiting for
    too many blocks to be mined is very costly time-wise.
    """
    return number_of_nodes * 4 + DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS


@pytest.fixture
def retry_timeout() -> int:
    return DEFAULT_RETRY_TIMEOUT


@pytest.fixture
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
    return remove_0x_prefix(HexStr(hex(random.getrandbits(100))))


@pytest.fixture
def logs_storage(request: pytest.FixtureRequest, tmpdir: Any) -> str:
    """Returns the path where debugging data should be saved.

    Use this to preserve the databases and logs necessary to debug test
    failures on the CI system.
    """
    short_path: Any = shortened_artifacts_storage(request.node) or str(tmpdir)
    return unique_path(short_path)


@pytest.fixture
def deposit() -> TokenAmount:
    """Raiden chain default deposit."""
    return TokenAmount(200)


@pytest.fixture
def number_of_tokens() -> int:
    """Number of tokens pre-registered in the test Registry."""
    return 1


@pytest.fixture
def register_tokens() -> bool:
    """Should fixture generated tokens be registered with raiden."""
    return True


@pytest.fixture
def number_of_nodes() -> int:
    """Number of raiden nodes in the test network."""
    return 3


@pytest.fixture
def channels_per_node() -> int:
    """Number of pre-created channels per test raiden node."""
    return 1


@pytest.fixture
def retry_interval_initial(transport_protocol: TransportProtocol) -> int:
    return 2


@pytest.fixture
def retry_interval_max(transport_protocol: TransportProtocol) -> int:
    return 2


@pytest.fixture
def retries_before_backoff() -> int:
    return 2


@pytest.fixture
def privatekey_seed(request: pytest.FixtureRequest) -> str:
    """Private key template, allow different keys to be used for each test to
    avoid collisions.
    """
    return escape_for_format(request.node.name) + ':{}'


@pytest.fixture
def account_genesis_eth_balance() -> int:
    return DEFAULT_BALANCE


@pytest.fixture
def token_amount(number_of_nodes: int, deposit: TokenAmount) -> TokenAmount:
    total_per_node: int = 3 * (deposit + 1)
    total_token: int = total_per_node * number_of_nodes
    return total_token  # type: ignore


@pytest.fixture
def network_wait() -> float:
    """Time in seconds used to wait for network events."""
    return 10.0


@pytest.fixture
def private_keys(number_of_nodes: int, privatekey_seed: str) -> List[bytes]:
    """Private keys for each raiden node."""
    result: List[bytes] = [
        keccak(privatekey_seed.format(position).encode()) for position in range(number_of_nodes)
    ]
    assert len(set(result)) == number_of_nodes, '`privatekey_seed` generate repeated keys'
    return result


@pytest.fixture
def deploy_key(privatekey_seed: str) -> bytes:
    return keccak(privatekey_seed.format('deploykey').encode())


@pytest.fixture(scope='session')
def blockchain_type(request: pytest.FixtureRequest) -> str:
    blockchain_type_value: str = request.config.option.blockchain_type
    if blockchain_type_value not in {client.value for client in EthClient}:
        raise ValueError(f'unknown blockchain_type {blockchain_type_value}')
    return blockchain_type_value


@pytest.fixture
def blockchain_extra_config() -> Dict[str, Any]:
    return {}


@pytest.fixture
def blockchain_number_of_nodes() -> int:
    """Number of nodes in the cluster, not the same as the number of raiden
    nodes. Used for all geth clusters.
    """
    return 1


@pytest.fixture
def blockchain_key_seed(request: pytest.FixtureRequest) -> str:
    """Private key template for the nodes in the private blockchain, allows
    different keys to be used for each test to avoid collisions.
    """
    return escape_for_format(request.node.name) + 'cluster:{}'


@pytest.fixture(scope='session')
def port_generator(request: pytest.FixtureRequest, worker_id: str) -> Iterator[int]:
    """count generator used to get a unique port number."""
    if worker_id == 'master':
        port_offset: int = 0
    else:
        port_offset = int(worker_id.replace('gw', '')) * 1000
    return get_free_port(request.config.getoption('base_port') + port_offset)


@pytest.fixture
def eth_nodes_configuration(
    blockchain_number_of_nodes: int,
    blockchain_key_seed: str,
    port_generator: Iterator[int],
    blockchain_type: str,
    blockchain_extra_config: Dict[str, Any]
) -> List[EthNodeDescription]:
    eth_nodes: List[EthNodeDescription] = []
    for position in range(blockchain_number_of_nodes):
        key: bytes = keccak(blockchain_key_seed.format(position).encode())
        eth_node = EthNodeDescription(
            private_key=key,
            rpc_port=next(port_generator),  # type: ignore
            p2p_port=next(port_generator),  # type: ignore
            miner=position == 0,
            extra_config=blockchain_extra_config,
            blockchain_type=blockchain_type
        )
        eth_nodes.append(eth_node)
    return eth_nodes


@pytest.fixture
def environment_type() -> Environment:
    """Specifies the environment type"""
    return Environment.DEVELOPMENT


@pytest.fixture
def unrecoverable_error_should_crash() -> bool:
    """For testing an UnrecoverableError should crash"""
    return True


@pytest.fixture
def transport() -> str:
    """'all' replaced by parametrize in conftest.pytest_generate_tests"""
    return 'matrix'


@pytest.fixture
def transport_protocol(transport: str) -> TransportProtocol:
    return TransportProtocol(transport)


@pytest.fixture
def blockchain_query_interval() -> float:
    """
    Config setting (interval after which to check for new block.)  Set to this low value for the
    integration tests, where we use a block time of 1 second.
    """
    return 0.5


@pytest.fixture
def skip_if_parity(blockchain_type: str) -> None:
    """Skip the test if it is run with a Parity node"""
    if blockchain_type == 'parity':
        pytest.skip('This test does not work with parity.')


@pytest.fixture
def skip_if_not_parity(blockchain_type: str) -> None:
    """Skip the test if it is not run with a Parity node"""
    if blockchain_type != 'parity':
        pytest.skip('This test works only with parity.')


@pytest.fixture
def skip_if_not_geth(blockchain_type: str) -> None:
    """Skip the test if it is run with a Geth node"""
    if blockchain_type != 'geth':
        pytest.skip('This test works only with geth.')


@pytest.fixture
def start_raiden_apps() -> bool:
    """Determines if the raiden apps created at test setup should also be started"""
    return True


@pytest.fixture
def enable_rest_api() -> bool:
    """Determines if the raiden apps created at test setup should also be started"""
    return False
