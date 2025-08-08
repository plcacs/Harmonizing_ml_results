from typing import IO, NamedTuple, Protocol
from eth_typing import URI, HexStr
from eth_utils import denoms, remove_0x_prefix, to_canonical_address
from flask import Flask, jsonify
from gevent import sleep
from typing_extensions import Protocol
from web3 import HTTPProvider, Web3
from web3.contract import Contract
from raiden.accounts import AccountManager
from raiden.constants import BLOCK_ID_LATEST, EMPTY_ADDRESS, GENESIS_BLOCK_NUMBER, SECONDS_PER_DAY, UINT256_MAX, Environment, EthClient
from raiden.network.proxies.proxy_manager import ProxyManager, ProxyManagerMetadata
from raiden.network.proxies.user_deposit import UserDeposit
from raiden.network.rpc.client import JSONRPCClient, make_sane_poa_middleware
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS, RAIDEN_CONTRACT_VERSION
from raiden.tests.fixtures.constants import DEFAULT_BALANCE, DEFAULT_PASSPHRASE
from raiden.tests.utils.eth_node import AccountDescription, EthNodeDescription, GenesisDescription, eth_node_to_datadir, geth_keystore, parity_keystore, run_private_blockchain
from raiden.tests.utils.smartcontracts import deploy_token, is_tx_hash_bytes
from raiden.tests.utils.transport import make_requests_insecure
from raiden.transfer import channel, views
from raiden.transfer.state import ChannelState
from raiden.ui.app import run_raiden_service
from raiden.utils.formatting import to_checksum_address
from raiden.utils.http import HTTPExecutor, split_endpoint
from raiden.utils.keys import privatekey_to_address
from raiden.utils.typing import TYPE_CHECKING, Address, AddressHex, Any, Balance, BlockNumber, Callable, ChainID, Dict, Endpoint, Iterable, Iterator, List, MonitoringServiceAddress, OneToNAddress, Port, PrivateKey, ServiceRegistryAddress, TokenAddress, TokenAmount, TokenNetworkRegistryAddress, Tuple, UserDepositAddress
from raiden.waiting import wait_for_block
from raiden_contracts.constants import CHAINNAME_TO_ID, CONTRACT_CUSTOM_TOKEN, CONTRACT_MONITORING_SERVICE, CONTRACT_ONE_TO_N, CONTRACT_SECRET_REGISTRY, CONTRACT_SERVICE_REGISTRY, CONTRACT_TOKEN_NETWORK_REGISTRY, CONTRACT_USER_DEPOSIT, TEST_SETTLE_TIMEOUT_MAX, TEST_SETTLE_TIMEOUT_MIN
from raiden_contracts.contract_manager import ContractManager, contracts_precompiled_path
from raiden_contracts.deploy.contract_deployer import ContractDeployer

TEST_DEPOSIT_AMOUNT: TokenAmount = TokenAmount(5)
TEST_PRIVKEY: PrivateKey = PrivateKey(b'\xad\xd4\xd3\x10\xba\x04$hy\x1d\xd7\xbf\x7fn\xae\x85\xac\xc4\xdd\x14?\xfa\x81\x0e\xf1\x80\x9aj\x11\xf2\xbcD')
TEST_ACCOUNT_ADDRESS: Address = privatekey_to_address(TEST_PRIVKEY)

class StepPrinter(Protocol):

    def __call__(self, description: str, error: bool = False) -> None:
        ...

def _deploy_contract(deployer: ContractDeployer, name: str, args: List[Any]) -> Address:
    receipt = deployer.deploy(name, args=args)
    address = receipt['contractAddress']
    assert address is not None, 'must be a valid address'
    return to_canonical_address(address)

def ensure_executable(cmd: str) -> None:
    """look for the given command and make sure it can be executed"""
    if not shutil.which(cmd):
        raise ValueError('Error: unable to locate %s binary.\nMake sure it is installed and added to the PATH variable.' % cmd)

def deploy_smoketest_contracts(client: JSONRPCClient, chain_id: ChainID, contract_manager: ContractManager, token_address: TokenAddress) -> Dict[str, Address]:
    ...

def get_private_key(keystore: str) -> PrivateKey:
    ...

@contextmanager
def setup_testchain(eth_client: EthClient, free_port_generator: Iterator[int], base_datadir: str, base_logdir: str) -> Iterator[Dict[str, Any]]:
    ...

@contextmanager
def setup_matrix_for_smoketest(print_step: StepPrinter, free_port_generator: Iterator[int]) -> Iterator[ContextManager]:
    ...

@contextmanager
def setup_testchain_for_smoketest(eth_client: EthClient, print_step: StepPrinter, free_port_generator: Iterator[int], base_datadir: str, base_logdir: str) -> Iterator[Dict[str, Any]]:
    ...

class RaidenTestSetup(NamedTuple):
    pass

def setup_raiden(matrix_server: str, print_step: StepPrinter, contracts_version: str, eth_rpc_endpoint: URI, web3: Web3, base_datadir: str, keystore: str, free_port_generator: Iterator[int]) -> RaidenTestSetup:
    ...

def _start_dummy_pfs(url: str, token_network_registry_address: str, user_deposit_address: str) -> None:
    ...

def run_smoketest(print_step: StepPrinter, setup: RaidenTestSetup) -> None:
    ...

@contextmanager
def setup_smoketest(*, eth_client: EthClient, print_step: StepPrinter, free_port_generator: Iterator[int], debug: bool = False, stdout: IO = None, append_report: Callable[[str], None] = print) -> Iterator[ContextManager]:
    ...

@contextmanager
def step_printer(step_count: int, stdout: IO) -> Iterator[StepPrinter]:
    ...
