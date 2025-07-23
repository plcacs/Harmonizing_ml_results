from gevent import monkey
monkey.patch_all()
import logging.config
import os
import os.path
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from itertools import chain, count, product, repeat
from time import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, NewType, Optional, Tuple
import gevent
import gevent.os
import requests
import structlog
from eth_utils import is_checksum_address, to_canonical_address, to_checksum_address
from gevent.greenlet import Greenlet
from gevent.pool import Pool
from gevent.subprocess import DEVNULL, STDOUT, Popen
from greenlet import greenlet
from raiden.network.transport.matrix.rtc.utils import setup_asyncio_event_loop
from raiden.network.utils import get_free_port
from raiden.transfer.state import NetworkState
from raiden.utils.formatting import pex
from raiden.utils.nursery import Janitor, Nursery
from raiden.utils.typing import Address, Host, Port, TokenAmount

setup_asyncio_event_loop()

BaseURL = NewType('BaseURL', str)
Amount = NewType('Amount', int)
URL = NewType('URL', str)
TransferPath = List['RunningNode']
INITIATOR = 0
TARGET = -1
processors = [
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt='%Y-%m-%d %H:%M:%S.%f'),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info
]
structlog.reset_defaults()
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'colorized-formatter': {
            '()': structlog.stdlib.ProcessorFormatter,
            'processor': structlog.dev.ConsoleRenderer(colors=True),
            'foreign_pre_chain': processors
        }
    },
    'handlers': {
        'colorized-handler': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'colorized-formatter'
        }
    },
    'loggers': {
        '': {
            'handlers': ['colorized-handler'],
            'propagate': True
        }
    }
})
structlog.configure(
    processors=processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory()
)
log: structlog.BoundLogger = structlog.get_logger(__name__)
log.setLevel('DEBUG')
NO_ROUTE_ERROR: int = 409
UNBUFERRED: int = 0
FIRST_VALID_PAYMENT_ID: int = 1
WAIT_FOR_SOCKET_TO_BE_AVAILABLE: int = 60
PartialTransferPlan = Iterator[Amount]
TransferPlan = Iterator[Amount]
PartialTransferPlanGenerator = Callable[[Amount], Iterator[PartialTransferPlan]]
TransferPlanGenerator = Callable[[Amount], TransferPlan]
Scheduler = Callable[[List[TransferPath], TransferPlan], Iterator['Transfer']]

@dataclass
class InitialNodeConfig:
    """The configuration of a node provided by the user, this node is not yet
    running.
    """
    address: Address
    keystore_path: str
    password_file: str
    eth_rpc_endpoint: str
    network_id: int
    pathfinding_service_address: Optional[str] = None

@dataclass
class NodeConfig:
    """Configuration of a node after the address has been recovered, this
    contains the expected address of the node.
    """
    args: List[str]
    interface: Host
    address: Address
    data_dir: str

@dataclass
class RunningNode:
    """A running node, this has a Raiden instance running in the background
    in a separate process.
    """
    process: Popen
    config: NodeConfig
    url: URL
    starting_balances: Dict[Address, Amount]

@dataclass
class StressTestConfiguration:
    port_generator: Iterator[int]
    retry_timeout: int
    capacity_lower_bound: Amount
    token_address: Address
    iteration_counter: Iterator[int]
    profiler_data_directory: Optional[str]

@dataclass
class StressTestPlan:
    transfers: List[TransferPath]
    concurrency: List[int]
    planners: List[PartialTransferPlanGenerator]
    schedulers: List[Scheduler]

@dataclass
class Transfer:
    path: TransferPath
    amount: Amount

def is_ready(base_url: BaseURL) -> bool:
    try:
        result: Dict[str, Any] = requests.get(f'{base_url}/api/v1/status').json()
    except KeyError:
        log.info(f'Server {base_url} returned invalid json data.')
    except requests.ConnectionError:
        log.info(f'Waiting for the server {base_url} to start.')
    except requests.RequestException:
        log.exception(f'Request to server {base_url} failed.')
    else:
        if result.get('status') == 'ready':
            log.info(f'Server {base_url} ready.')
            return True
        log.info(f"Waiting for server {base_url} to become ready, status={result.get('status')}.")
    return False

def wait_for_status_ready(base_url: BaseURL, retry_timeout: int) -> None:
    """Keeps polling for the `/status` endpoint until the status is `ready`."""
    while not is_ready(base_url):
        gevent.sleep(retry_timeout)

def wait_for_reachable(transfers: List['Transfer'], token_address: Address, retry_timeout: int) -> None:
    """Wait until the nodes used for the transfers can see each other."""
    channels_not_reachable: set = set()
    for transfer in transfers:
        for payer, payee in zip(transfer.path, transfer.path[1:]):
            channel_url: str = f'{payer.url}/api/v1/channels/{token_address}/{payee.config.address}'
            channels_not_reachable.add(channel_url)
    while channels_not_reachable:
        log.info(f'Waiting for reachability of partner nodes: {channels_not_reachable}')
        for url in channels_not_reachable.copy():
            try:
                response: requests.Response = requests.get(url, headers={'Content-Type': 'application/json'})
                data: Dict[str, Any] = response.json()
                if data and data.get('network_state') == NetworkState.REACHABLE.value:
                    channels_not_reachable.remove(url)
            except requests.RequestException:
                log.exception(f'Failed to check channel status for {url}.')
        if channels_not_reachable:
            gevent.sleep(retry_timeout)

def start_and_wait_for_server(
    nursery: Nursery,
    port_generator: Iterator[int],
    node: NodeConfig,
    retry_timeout: int
) -> Optional[RunningNode]:
    """Start the Raiden node and waits for the REST API to be available,
    returns None if the script is being shutdown.
    """
    os.makedirs(os.path.expanduser(node.data_dir), exist_ok=True)
    stdout_path: str = os.path.join(node.data_dir, 'stress_test.out')
    stdout = open(stdout_path, 'a')
    port: int = next(port_generator)
    api_url: str = f'{node.interface}:{port}'
    running_url: URL = URL(f'http://{api_url}')
    process_args: List[str] = node.args + ['--api-address', api_url]
    process: Optional[Popen] = nursery.exec_under_watch(
        process_args,
        bufsize=UNBUFERRED,
        stdout=stdout,
        stderr=STDOUT
    )
    if process is not None:
        wait_for_status_ready(running_url, retry_timeout)
        return RunningNode(
            process=process,
            config=node,
            url=running_url,
            starting_balances=get_balance_for_node(running_url)
        )
    return None

def start_and_wait_for_all_servers(
    nursery: Nursery,
    port_generator: Iterator[int],
    nodes_config: List[NodeConfig],
    retry_timeout: int
) -> Optional[List[RunningNode]]:
    """Starts all nodes under the nursery, returns a list of `RunningNode`s or
    None if the script is shutting down.

    Important Note:

    `None` is not always returned if the script is shutting down! Due to race
    conditions it is possible for all processes to be spawned, and only
    afterwards the nursery is closed. IOW: At this stage `None` will only be
    returned if spawning the process fails (e.g. the binary name is wrong),
    however, if the subprocess is spawned and runs for some time, and *then*
    crashes, `None` will **not** be returned here (e.g. if the ethereum node is
    not available). For the second case, the `stop_event` will be set.

    Because of the above, for proper error handling, checking only the return
    value is **not** sufficient. The most reliable approach is to execute new
    logic in greenlets spawned with `spawn_under_watch` and let errors fall
    through.
    """
    greenlets: set = set(
        nursery.spawn_under_watch(
            start_and_wait_for_server,
            nursery,
            port_generator,
            node,
            retry_timeout
        )
        for node in nodes_config
    )
    all_running_nodes: List[RunningNode] = []
    for g in gevent.joinall(greenlets, raise_error=True):
        running_node: Optional[RunningNode] = g.get()
        if running_node is None:
            return None
        all_running_nodes.append(running_node)
    return all_running_nodes

def restart_and_wait_for_server(
    nursery: Nursery,
    port_generator: Iterator[int],
    node: NodeConfig,
    retry_timeout: int
) -> Optional[RunningNode]:
    """Stop `RunningNode` and start it again under the nursery, returns None if
    the script is shutting down.
    """
    node.process.send_signal(signal.SIGINT)
    exit_code: int = node.process.result.get()
    if exit_code != 0:
        raise Exception(f'Node did not shut down cleanly {node!r}')
    return start_and_wait_for_server(nursery, port_generator, node.config, retry_timeout)

def restart_network(
    nursery: Nursery,
    port_generator: Iterator[int],
    running_nodes: List[RunningNode],
    retry_timeout: int
) -> Optional[List[RunningNode]]:
    """Stop all `RunningNode`s and start them again under the nursery, returns
    None if the script is shutting down.
    """
    greenlets: set = set(
        nursery.spawn_under_watch(
            restart_and_wait_for_server,
            nursery,
            port_generator,
            node,
            retry_timeout
        )
        for node in running_nodes
    )
    all_running_nodes: List[RunningNode] = []
    for g in gevent.joinall(greenlets, raise_error=True):
        running_node: Optional[RunningNode] = g.get()
        if running_node is None:
            return None
        all_running_nodes.append(running_node)
    return all_running_nodes

def transfer_and_assert_successful(
    base_url: BaseURL,
    token_address: Address,
    target_address: Address,
    payment_identifier: int,
    amount: Amount
) -> None:
    post_url: str = f'{base_url}/api/v1/payments/{token_address}/{target_address}'
    json_data: Dict[str, Any] = {'amount': amount, 'identifier': payment_identifier}
    log.debug('Payment request', url=post_url, json=json_data)
    start_time: float = time()
    response: requests.Response = requests.post(post_url, json=json_data)
    elapsed: float = time() - start_time
    assert response is not None, 'request.post returned None'
    is_json: bool = response.headers.get('Content-Type') == 'application/json'
    assert is_json, (response.headers.get('Content-Type'), response.text)
    assert response.status_code == HTTPStatus.OK, response.json()
    log.debug('Payment done', url=post_url, json=json_data, time=elapsed)

def do_fifty_transfer_up_to(capacity_lower_bound: Amount) -> Iterator[Amount]:
    """Generates a plan with 50 transfers of the same value.

    >>> len(do_fifty_transfer_up_to(500))
    ... 50
    >>> list(do_fifty_transfer_up_to(500))
    ... [10, 10, 10 ..., 10]
    """
    qty_of_transfers: int = 50
    amount: Amount = Amount(capacity_lower_bound // qty_of_transfers)
    for _ in range(qty_of_transfers):
        yield amount

def do_transfers(
    transfers: List[Transfer],
    token_address: Address,
    identifier_generator: Iterator[int],
    pool_size: Optional[int] = None
) -> None:
    """Concurrently execute `transfers`.

    Note:
        To force serial transfers just provide `pool_size=1`.
    """
    pool: Pool = Pool(size=pool_size)
    current: greenlet = gevent.getcurrent()

    def propagate_error(result: gevent.event.AsyncResult) -> None:
        if result.exception:
            current.kill(result.exception)

    for transfer in transfers:
        task: Greenlet = pool.spawn(
            transfer_and_assert_successful,
            base_url=transfer.path[INITIATOR].url,
            token_address=token_address,
            target_address=transfer.path[TARGET].config.address,
            payment_identifier=next(identifier_generator),
            amount=transfer.amount
        )
        task.link_exception(propagate_error)
    pool.join(raise_error=True)

def paths_direct_transfers(running_nodes: List[RunningNode]) -> List[List[RunningNode]]:
    """Given the list of `running_nodes`, where each adjacent pair has a channel open,
    return a list of `[from, to]` which will do a direct transfer using each
    channel.
    """
    forward: List[List[RunningNode]] = [
        [from_, to_] for from_, to_ in zip(running_nodes[:-1], running_nodes[1:])
    ]
    backward: List[List[RunningNode]] = [
        [to_, from_] for from_, to_ in zip(running_nodes[:-1], running_nodes[1:])
    ]
    return forward + backward

def paths_for_mediated_transfers(running_nodes: List[RunningNode]) -> List[List[RunningNode]]:
    """Given the list of `running_nodes`, where each adjacent pair has a channel open,
    return the a list with the pair `[from, to]` which are the furthest apart.
    """
    msg: str = 'This function needs to be improved to generate all mediator paths for a chain with more than 3 running_nodes'
    assert len(running_nodes) == 3, msg
    return [list(running_nodes)] + [list(reversed(running_nodes))]

def scheduler_preserve_order(
    paths: List[TransferPath],
    plan: TransferPlan
) -> Iterator[Transfer]:
    """Execute the same plan for each path, in order.

    E.g.:

    >>> paths = [[a, b], [b, c]]
    >>> transfer_plan = [1,1]
    >>> list(scheduler_preserve_order(paths, transfer_plan))
    ... [Transfer([a, b], amount=1),
    ...  Transfer([a, b], amount=1),
    ...  Transfer([b, c], amount=1),
    ...  Transfer([b, c], amount=1)]
    """
    for path, transfer in product(paths, plan):
        yield Transfer(path=path, amount=transfer)

def run_profiler(
    nursery: Nursery,
    running_nodes: List[RunningNode],
    profiler_data_directory: str
) -> List[Popen]:
    os.makedirs(os.path.expanduser(profiler_data_directory), exist_ok=True)
    profiler_processes: List[Popen] = []
    for node in running_nodes:
        args: List[str] = [
            'py-spy', 'record',
            '--pid', str(node.process.pid),
            '--output', os.path.join(
                profiler_data_directory,
                f'{node.config.address}-{datetime.utcnow().isoformat()}.data'
            )
        ]
        profiler: Popen = Popen(args, stdout=DEVNULL, stderr=DEVNULL)
        nursery.exec_under_watch(profiler)
        profiler_processes.append(profiler)
    return profiler_processes

def get_balance_for_node(url: URL) -> Dict[Address, Amount]:
    response: requests.Response = requests.get(f'{url}/api/v1/channels')
    assert response.headers.get('Content-Type') == 'application/json', response.headers.get('Content-Type')
    assert response.status_code == HTTPStatus.OK, response.json()
    response_data: List[Dict[str, Any]] = response.json()
    return {channel['partner_address']: Amount(channel['balance']) for channel in response_data}

def wait_for_balance(running_nodes: List[RunningNode]) -> None:
    """Wait until the nodes have `starting_balance`, again

    This makes sure that we can run another iteration of the stress test
    """
    for node in running_nodes:
        balances: Dict[Address, Amount] = get_balance_for_node(node.url)
        while any(
            (bal < node.starting_balances.get(partner, 0))
            for partner, bal in balances.items()
        ):
            gevent.sleep(0.1)
            balances = get_balance_for_node(node.url)

def wait_for_user_input() -> None:
    print('All nodes are ready! Press Enter to continue and perform the stress tests.')
    gevent.os.tp_read(sys.stdin.fileno(), n=1)

def run_stress_test(
    nursery: Nursery,
    running_nodes: List[RunningNode],
    config: StressTestConfiguration
) -> None:
    identifier_generator: Iterator[int] = count(start=FIRST_VALID_PAYMENT_ID)
    profiler_processes: List[Popen] = []
    for iteration in config.iteration_counter:
        log.info(f'Starting iteration {iteration}')
        plan: StressTestPlan = StressTestPlan(
            transfers=paths_for_mediated_transfers(running_nodes),
            concurrency=[50],
            planners=[do_fifty_transfer_up_to],
            schedulers=[scheduler_preserve_order]
        )
        for concurrent_paths, concurrency, transfer_planner, scheduler in zip(
            repeat(plan.transfers),
            plan.concurrency,
            plan.planners,
            plan.schedulers
        ):
            log.info(f'Starting run {concurrent_paths}, {concurrency}, {transfer_planner}, {scheduler}')
            transfer_plan: TransferPlan = transfer_planner(config.capacity_lower_bound)
            transfers: List[Transfer] = list(scheduler(concurrent_paths, transfer_plan))
            if config.profiler_data_directory:
                profiler_processes = run_profiler(nursery, running_nodes, config.profiler_data_directory)
            wait_for_reachable(transfers, config.token_address, config.retry_timeout)
            do_transfers(
                transfers=transfers,
                token_address=config.token_address,
                identifier_generator=identifier_generator,
                pool_size=concurrency
            )
            wait_for_balance(running_nodes)
            restarted_nodes: Optional[List[RunningNode]] = restart_network(
                nursery,
                config.port_generator,
                running_nodes,
                config.retry_timeout
            )
            if restarted_nodes is None:
                return
            else:
                running_nodes = restarted_nodes
            for profiler in profiler_processes:
                profiler.send_signal(signal.SIGINT)

def main() -> None:
    import argparse
    import configparser
    import re

    NODE_SECTION_RE = re.compile('^node[0-9]+')
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes-data-dir', default=os.getcwd())
    parser.add_argument('--wait-after-first-sync', default=False, action='store_true')
    parser.add_argument('--profiler-data-directory', default=None)
    parser.add_argument('--interface', default='127.0.0.1')
    parser.add_argument('--iterations', default=5, type=int)
    parser.add_argument('config')
    args = parser.parse_args()

    if args.profiler_data_directory is not None and os.geteuid() != 0:
        raise RuntimeError('To enable profiling the script has to be executed with root.')

    config_parser: configparser.ConfigParser = configparser.ConfigParser()
    config_parser.read(args.config)
    datadir: str = args.nodes_data_dir
    interface: Host = Host(args.interface)
    port_generator: Iterator[int] = get_free_port(5000)
    retry_timeout: int = 1
    nodes_config: List[NodeConfig] = []
    token_address: Address = Address(to_checksum_address(config_parser.defaults()['token-address']))

    if not is_checksum_address(token_address):
        raise ValueError(f'Invalid token address {token_address}, check it is checksummed.')

    defaults: Dict[str, str] = {
        '--log-config': 'raiden:DEBUG',
        '--environment-type': 'development',
        '--datadir': datadir
    }

    for section in config_parser:
        if NODE_SECTION_RE.match(section):
            node_config = config_parser[section]
            address: Address = Address(to_checksum_address(node_config['address']))
            node_args: Dict[str, str] = {
                '--keystore-path': node_config['keystore-path'],
                '--password-file': node_config['password-file'],
                '--eth-rpc-endpoint': node_config['eth-rpc-endpoint'],
                '--network-id': node_config.getint('network-id'),
                '--address': address
            }
            pathfinding_url: Optional[str] = node_config.get('pathfinding-service-address')
            if pathfinding_url is not None:
                node_args['--pathfinding-service-address'] = pathfinding_url
            raiden_args: List[str] = [
                'raiden',
                '--accept-disclaimer',
                '--log-json',
                '--disable-debug-logfile',
                '--flat-fee', token_address,
                '0',
                '--proportional-fee', token_address,
                '0',
                '--proportional-imbalance-fee', token_address,
                '0'
            ]
            for key, value in node_args.items():
                raiden_args.extend([key, str(value)])
            nodedir: str = os.path.join(datadir, f'node_{pex(to_canonical_address(address))}')
            nodes_config.append(NodeConfig(
                args=raiden_args,
                interface=interface,
                address=address,
                data_dir=nodedir
            ))

    capacity_lower_bound: Amount = Amount(1130220)
    profiler_data_directory: Optional[str] = args.profiler_data_directory
    iterations: int = args.iterations
    if iterations is None:
        iteration_counter: Iterator[int] = count()
    else:
        iteration_counter: Iterator[int] = iter(range(iterations))

    test_config: StressTestConfiguration = StressTestConfiguration(
        port_generator=port_generator,
        retry_timeout=retry_timeout,
        capacity_lower_bound=capacity_lower_bound,
        token_address=token_address,
        iteration_counter=iteration_counter,
        profiler_data_directory=profiler_data_directory
    )

    with Janitor() as nursery:
        nodes_running: Optional[List[RunningNode]] = start_and_wait_for_all_servers(
            nursery,
            port_generator,
            nodes_config,
            retry_timeout
        )
        if nodes_running is None:
            return
        if args.wait_after_first_sync:
            nursery.spawn_under_watch(wait_for_user_input).get()
        nursery.spawn_under_watch(run_stress_test, nursery, nodes_running, test_config)
        nursery.wait(timeout=None)

if __name__ == '__main__':
    main()
