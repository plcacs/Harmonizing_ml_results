from typing import List, Iterator, Any, Optional, Tuple
import pytest
from raiden.constants import Environment
from raiden.network.transport import MatrixTransport
from raiden.settings import DEFAULT_TRANSPORT_MATRIX_SYNC_TIMEOUT, CapabilitiesConfig, MatrixTransportConfig
from raiden.tests.fixtures.variables import TransportProtocol
from raiden.tests.utils.transport import ParsedURL, generate_synapse_config, matrix_server_starter
from raiden.utils.http import HTTPExecutor
from raiden.utils.typing import Iterable

@pytest.fixture(scope='session')
def func_qrb5lfnj() -> Iterator[Any]:
    with generate_synapse_config() as generator:
        yield generator

@pytest.fixture
def func_u9jayqus() -> int:
    return 1

@pytest.fixture
def func_815un407() -> int:
    return DEFAULT_TRANSPORT_MATRIX_SYNC_TIMEOUT

@pytest.fixture
def func_kcbzi8uq(
    request: pytest.FixtureRequest,
    transport_protocol: TransportProtocol,
    matrix_server_count: int,
    synapse_config_generator: Any,
    port_generator: Any
) -> Iterator[List[Any]]:
    if transport_protocol is not TransportProtocol.MATRIX:
        yield []
        return
    starter = matrix_server_starter(
        free_port_generator=port_generator,
        count=matrix_server_count,
        config_generator=synapse_config_generator,
        log_context=request.node.name
    )
    with starter as servers:
        yield servers

@pytest.fixture
def func_sujbnk27(
    local_matrix_servers_with_executor: List[Tuple[str, HTTPExecutor]]
) -> Iterator[List[str]]:
    yield [url for url, _ in local_matrix_servers_with_executor]

@pytest.fixture
def func_doda7uil(
    local_matrix_servers: List[Any],
    retries_before_backoff: int,
    retry_interval_initial: int,
    retry_interval_max: int,
    number_of_transports: int,
    matrix_sync_timeout: int,
    capabilities: CapabilitiesConfig,
    environment_type: Environment
) -> Iterator[List[MatrixTransport]]:
    transports: List[MatrixTransport] = []
    local_matrix_servers_str = [str(server) for server in local_matrix_servers]
    for transport_index in range(number_of_transports):
        server = local_matrix_servers[transport_index % len(local_matrix_servers)]
        transports.append(
            MatrixTransport(
                config=MatrixTransportConfig(
                    retries_before_backoff=retries_before_backoff,
                    retry_interval_initial=retry_interval_initial,
                    retry_interval_max=retry_interval_max,
                    server=server,
                    available_servers=local_matrix_servers_str,
                    sync_timeout=matrix_sync_timeout,
                    capabilities_config=capabilities
                ),
                environment=environment_type
            )
        )
    yield transports
    for transport in transports:
        transport.stop()
    for transport in transports:
        if transport._started:
            transport.greenlet.get()

@pytest.fixture
def func_6h3iammw(number_of_nodes: int) -> List[Optional[int]]:
    """Default resolver ports for all nodes.

    By default, Raiden nodes start without hash resolvers.
    This is achieved by setting the ports to None. This cause the command line not to
    include --resolver-endpoint  and resolver processes will not start.
    """
    return [None] * number_of_nodes