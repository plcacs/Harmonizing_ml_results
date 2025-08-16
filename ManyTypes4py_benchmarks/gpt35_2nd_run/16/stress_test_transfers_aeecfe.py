from typing import Any, Callable, Dict, Iterable, Iterator, List, NewType, Optional

BaseURL = NewType('BaseURL', str)
Amount = NewType('Amount', int)
URL = NewType('URL', str)
TransferPath = List['RunningNode']
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

@dataclass
class NodeConfig:
    """Configuration of a node after the address has been recovered, this
    contains the expected address of the node.
    """

@dataclass
class RunningNode:
    """A running node, this has a Raiden instance running in the background
    in a separate process.
    """

@dataclass
class StressTestConfiguration:
    pass

@dataclass
class StressTestPlan:
    pass

@dataclass
class Transfer:
    pass

def is_ready(base_url: BaseURL) -> bool:
    ...

def wait_for_status_ready(base_url: BaseURL, retry_timeout: int) -> None:
    ...

def wait_for_reachable(transfers: List[TransferPath], token_address: str, retry_timeout: int) -> None:
    ...

def start_and_wait_for_server(nursery: Any, port_generator: Any, node: Any, retry_timeout: int) -> Optional[RunningNode]:
    ...

def start_and_wait_for_all_servers(nursery: Any, port_generator: Any, nodes_config: List[NodeConfig], retry_timeout: int) -> Optional[List[RunningNode]]:
    ...

def restart_and_wait_for_server(nursery: Any, port_generator: Any, node: RunningNode, retry_timeout: int) -> Optional[RunningNode]:
    ...

def restart_network(nursery: Any, port_generator: Any, running_nodes: List[RunningNode], retry_timeout: int) -> Optional[List[RunningNode]]:
    ...

def transfer_and_assert_successful(base_url: BaseURL, token_address: str, target_address: str, payment_identifier: int, amount: int) -> None:
    ...

def do_fifty_transfer_up_to(capacity_lower_bound: int) -> PartialTransferPlan:
    ...

def do_transfers(transfers: List[Transfer], token_address: str, identifier_generator: Iterator[int], pool_size: Optional[int] = None) -> None:
    ...

def paths_direct_transfers(running_nodes: List[RunningNode]) -> List[TransferPath]:
    ...

def paths_for_mediated_transfers(running_nodes: List[RunningNode]) -> List[TransferPath]:
    ...

def scheduler_preserve_order(paths: List[TransferPath], plan: TransferPlan) -> Iterator[Transfer]:
    ...

def run_profiler(nursery: Any, running_nodes: List[RunningNode], profiler_data_directory: str) -> List[Any]:
    ...

def get_balance_for_node(url: URL) -> Dict[str, int]:
    ...

def wait_for_balance(running_nodes: List[RunningNode]) -> None:
    ...

def wait_for_user_input() -> None:
    ...

def run_stress_test(nursery: Any, running_nodes: List[RunningNode], config: StressTestConfiguration) -> None:
    ...

def main() -> None:
    ...

if __name__ == '__main__':
    main()
