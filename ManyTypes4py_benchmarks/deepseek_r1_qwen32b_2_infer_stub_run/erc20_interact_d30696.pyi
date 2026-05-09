from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from eth_utils import decode_hex, encode_hex
from web3 import Web3
from eth.chains.base import MiningChain
from eth.rlp.blocks import BaseBlock
from eth.tools.factories.transaction import Transaction
from scripts.benchmark._utils.reporting import DefaultStat
from .base_benchmark import BaseBenchmark

FIRST_TX_GAS_LIMIT: int
SECOND_TX_GAS_LIMIT: int
TRANSFER_AMOUNT: int
TRANSER_FROM_AMOUNT: int
CONTRACT_FILE: str
CONTRACT_NAME: str
W3_TX_DEFAULTS: Dict[str, int]

class BaseERC20Benchmark(BaseBenchmark):
    def __init__(self, num_blocks: int = 2, num_tx: int = 50) -> None:
        ...
    
    def _setup_benchmark(self, chain: MiningChain) -> None:
        ...
    
    @abstractmethod
    def _next_transaction(self, chain: MiningChain) -> Tuple[Transaction, Any]:
        ...
    
    def execute(self) -> DefaultStat:
        ...
    
    def mine_blocks(self, chain: MiningChain, num_blocks: int, num_tx: int) -> Tuple[int, int]:
        ...
    
    def mine_block(self, chain: MiningChain, block_number: int, num_tx: int) -> Any:
        ...
    
    def _deploy_simple_token(self, chain: MiningChain, nonce: Optional[int] = None) -> Tuple[Transaction, Any]:
        ...
    
    def _erc_transfer(self, addr: str, chain: MiningChain, nonce: Optional[int] = None) -> Tuple[Transaction, Any]:
        ...
    
    def _erc_approve(self, addr2: str, chain: MiningChain, nonce: Optional[int] = None) -> Tuple[Transaction, Any]:
        ...
    
    def _erc_transfer_from(self, addr1: str, addr2: str, chain: MiningChain, nonce: Optional[int] = None) -> Tuple[Transaction, Any]:
        ...

class ERC20DeployBenchmark(BaseERC20Benchmark):
    @property
    def name(self) -> str:
        ...
    
    def _setup_benchmark(self, chain: MiningChain) -> None:
        ...
    
    def _next_transaction(self, chain: MiningChain) -> Tuple[Transaction, Any]:
        ...

class ERC20TransferBenchmark(BaseERC20Benchmark):
    @property
    def name(self) -> str:
        ...
    
    def _setup_benchmark(self, chain: MiningChain) -> None:
        ...
    
    def _next_transaction(self, chain: MiningChain) -> Tuple[Transaction, Any]:
        ...

class ERC20ApproveBenchmark(BaseERC20Benchmark):
    @property
    def name(self) -> str:
        ...
    
    def _setup_benchmark(self, chain: MiningChain) -> None:
        ...
    
    def _next_transaction(self, chain: MiningChain) -> Tuple[Transaction, Any]:
        ...

class ERC20TransferFromBenchmark(BaseERC20Benchmark):
    @property
    def name(self) -> str:
        ...
    
    def _setup_benchmark(self, chain: MiningChain) -> None:
        ...
    
    def _next_transaction(self, chain: MiningChain) -> Tuple[Transaction, Any]:
        ...