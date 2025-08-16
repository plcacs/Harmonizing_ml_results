from abc import abstractmethod
import pathlib
from typing import Tuple
from eth_utils import decode_hex, encode_hex
from web3 import Web3
from eth.chains.base import MiningChain
from eth.constants import CREATE_CONTRACT_ADDRESS
from eth.rlp.blocks import BaseBlock
from eth.tools.factories.transaction import new_transaction
from scripts.benchmark._utils.chain_plumbing import FUNDED_ADDRESS, FUNDED_ADDRESS_PRIVATE_KEY, SECOND_ADDRESS, SECOND_ADDRESS_PRIVATE_KEY, get_all_chains
from scripts.benchmark._utils.compile import get_compiled_contract
from scripts.benchmark._utils.reporting import DefaultStat
from .base_benchmark import BaseBenchmark

FIRST_TX_GAS_LIMIT: int = 1400000
SECOND_TX_GAS_LIMIT: int = 60000
TRANSFER_AMOUNT: int = 1000
TRANSER_FROM_AMOUNT: int = 1
CONTRACT_FILE: str = 'scripts/benchmark/contract_data/erc20.sol'
CONTRACT_NAME: str = 'SimpleToken'
W3_TX_DEFAULTS: dict = {'gas': 0, 'gasPrice': 0}

class BaseERC20Benchmark(BaseBenchmark):

    def __init__(self, num_blocks: int = 2, num_tx: int = 50) -> None:
        super().__init__()
        self.num_blocks: int = num_blocks
        self.num_tx: int = num_tx
        self.contract_interface: dict = get_compiled_contract(pathlib.Path(CONTRACT_FILE), CONTRACT_NAME)
        self.w3: Web3 = Web3()
        self.addr1: str = Web3.toChecksumAddress(FUNDED_ADDRESS)
        self.addr2: str = Web3.toChecksumAddress(SECOND_ADDRESS)

    def _setup_benchmark(self, chain: MiningChain) -> None:
        pass

    @abstractmethod
    def _next_transaction(self, chain: MiningChain) -> Tuple:
        pass

    def execute(self) -> DefaultStat:
        pass

    def mine_blocks(self, chain: MiningChain, num_blocks: int, num_tx: int) -> Tuple[int, int]:
        pass

    def mine_block(self, chain: MiningChain, block_number: int, num_tx: int) -> Tuple:
        pass

    def _deploy_simple_token(self, chain: MiningChain, nonce: int = None) -> Tuple:
        pass

    def _erc_transfer(self, addr: str, chain: MiningChain, nonce: int = None) -> Tuple:
        pass

    def _erc_approve(self, addr2: str, chain: MiningChain, nonce: int = None) -> Tuple:
        pass

    def _erc_transfer_from(self, addr1: str, addr2: str, chain: MiningChain, nonce: int = None) -> Tuple:
        pass

class ERC20DeployBenchmark(BaseERC20Benchmark):

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        pass

    def _setup_benchmark(self, chain: MiningChain) -> None:
        pass

    def _next_transaction(self, chain: MiningChain) -> Tuple:
        pass

class ERC20TransferBenchmark(BaseERC20Benchmark):

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        pass

    def _setup_benchmark(self, chain: MiningChain) -> None:
        pass

    def _next_transaction(self, chain: MiningChain) -> Tuple:
        pass

class ERC20ApproveBenchmark(BaseERC20Benchmark):

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        pass

    def _setup_benchmark(self, chain: MiningChain) -> None:
        pass

    def _next_transaction(self, chain: MiningChain) -> Tuple:
        pass

class ERC20TransferFromBenchmark(BaseERC20Benchmark):

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        pass

    def _setup_benchmark(self, chain: MiningChain) -> None:
        pass

    def _next_transaction(self, chain: MiningChain) -> Tuple:
        pass
