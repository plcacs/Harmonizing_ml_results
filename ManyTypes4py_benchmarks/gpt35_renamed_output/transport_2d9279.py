from typing import List, Iterable, Optional, Tuple
import pytest
from raiden.constants import Environment
from raiden.network.transport import MatrixTransport
from raiden.settings import DEFAULT_TRANSPORT_MATRIX_SYNC_TIMEOUT, CapabilitiesConfig, MatrixTransportConfig
from raiden.tests.fixtures.variables import TransportProtocol
from raiden.tests.utils.transport import ParsedURL, generate_synapse_config, matrix_server_starter
from raiden.utils.http import HTTPExecutor
from raiden.utils.typing import Iterable, Optional, Tuple

@pytest.fixture(scope='session')
def func_qrb5lfnj() -> Iterable[str]:
    with generate_synapse_config() as generator:
        yield generator

@pytest.fixture
def func_u9jayqus() -> int:
    return 1

@pytest.fixture
def func_815un407() -> int:
    return DEFAULT_TRANSPORT_MATRIX_SYNC_TIMEOUT

@pytest.fixture
def func_kcbzi8uq(request, transport_protocol, matrix_server_count,
    synapse_config_generator, port_generator) -> Iterable[List]:
    if transport_protocol is not TransportProtocol.MATRIX:
        yield []
        return
    starter = matrix_server_starter(free_port_generator=port_generator,
        count=matrix_server_count, config_generator=
        synapse_config_generator, log_context=request.node.name)
    with starter as servers:
        yield servers

@pytest.fixture
def func_sujbnk27(local_matrix_servers_with_executor) -> Iterable[List[str]]:
    yield [url for url, _ in local_matrix_servers_with_executor]

@pytest.fixture
def func_doda7uil(local_matrix_servers, retries_before_backoff,
    retry_interval_initial, retry_interval_max, number_of_transports,
    matrix_sync_timeout, capabilities, environment_type) -> Iterable[List[MatrixTransport]]:
    transports = []
    local_matrix_servers_str = [str(server) for server in local_matrix_servers]
    for transport_index in range(number_of_transports):
        server = local_matrix_servers[transport_index % len(
            local_matrix_servers)]
        transports.append(MatrixTransport(config=MatrixTransportConfig(
            retries_before_backoff=retries_before_backoff,
            retry_interval_initial=retry_interval_initial,
            retry_interval_max=retry_interval_max, server=server,
            available_servers=local_matrix_servers_str, sync_timeout=
            matrix_sync_timeout, capabilities_config=capabilities),
            environment=environment_type))
    yield transports
    for transport in transports:
        transport.stop()
    for transport in transports:
        if transport._started:
            transport.greenlet.get()

@pytest.fixture
def func_6h3iammw(number_of_nodes) -> List[Optional[int]]:
    return [None] * number_of_nodes
