from typing import Type
from eth_bloom import BloomFilter
from eth.abc import BlockAPI, BlockHeaderAPI, ReceiptAPI, StateAPI, SignedTransactionAPI, ComputationAPI, TransactionFieldsAPI
from eth.constants import BLOCK_REWARD, UNCLE_DEPTH_PENALTY_FACTOR, ZERO_HASH32
from eth.rlp.logs import Log
from eth.rlp.receipts import Receipt
from eth.vm.base import VM
from .blocks import FrontierBlock
from .constants import MAX_REFUND_QUOTIENT
from .state import FrontierState
from .headers import create_frontier_header_from_parent, compute_frontier_difficulty, configure_frontier_header
from .validation import validate_frontier_transaction_against_header

def make_frontier_receipt(computation: Union[typing.Callable, str, bool], new_cumulative_gas_used: Union[bool, str, typing.Callable]) -> Receipt:
    logs = [Log(address, topics, data) for address, topics, data in computation.get_log_entries()]
    receipt = Receipt(state_root=ZERO_HASH32, gas_used=new_cumulative_gas_used, logs=logs)
    return receipt

class FrontierVM(VM):
    fork = 'frontier'
    block_class = FrontierBlock
    _state_class = FrontierState
    create_header_from_parent = staticmethod(create_frontier_header_from_parent)
    compute_difficulty = staticmethod(compute_frontier_difficulty)
    configure_header = configure_frontier_header
    validate_transaction_against_header = validate_frontier_transaction_against_header

    @staticmethod
    def get_block_reward():
        return BLOCK_REWARD

    @staticmethod
    def get_uncle_reward(block_number: Union[int, eth.abc.BlockHeaderAPI], uncle: Union[int, eth.abc.BlockHeaderAPI]):
        return BLOCK_REWARD * (UNCLE_DEPTH_PENALTY_FACTOR + uncle.block_number - block_number) // UNCLE_DEPTH_PENALTY_FACTOR

    @classmethod
    def get_nephew_reward(cls: Union[T, dict, str]) -> int:
        return cls.get_block_reward() // 32

    def add_receipt_to_header(self, old_header: Union[eth.abc.ReceiptAPI, eth.abc.BlockHeaderAPI, dict[str, typing.Any]], receipt: Union[eth.abc.ReceiptAPI, eth.abc.BlockHeaderAPI, dict[str, typing.Any]]) -> Union[bytes, str]:
        return old_header.copy(bloom=int(BloomFilter(old_header.bloom) | receipt.bloom), gas_used=receipt.gas_used, state_root=self.state.make_state_root())

    def increment_blob_gas_used(self, old_header: Union[BlockHeaderAPI, eth.rlp.headers.BlockHeader, SignedTransactionAPI], transaction: Union[BlockHeaderAPI, eth.rlp.headers.BlockHeader, SignedTransactionAPI]) -> Union[BlockHeaderAPI, eth.rlp.headers.BlockHeader, SignedTransactionAPI]:
        return old_header

    @classmethod
    def calculate_net_gas_refund(cls: Union[int, bytes, typing.Callable], consumed_gas: Union[int, eth.ChainGaps, float], gross_refund: Union[int, float, str]) -> Union[str, int, float]:
        max_refund = consumed_gas // MAX_REFUND_QUOTIENT
        return min(max_refund, gross_refund)

    @classmethod
    def finalize_gas_used(cls: Union[tuple[int], eth.abc.ComputationAPI, eth.abc.MiningChainAPI], transaction: eth.abc.DatabaseAPI, computation: Union[eth.abc.ComputationAPI, eth.abc.SignedTransactionAPI, typing.Type]) -> int:
        gas_remaining = computation.get_gas_remaining()
        consumed_gas = transaction.gas - gas_remaining
        gross_refund = computation.get_gas_refund()
        net_refund = cls.calculate_net_gas_refund(consumed_gas, gross_refund)
        return consumed_gas - net_refund

    @classmethod
    def make_receipt(cls: Union[eth.abc.BlockHeaderAPI, int, eth.rlp.transactions.BaseTransaction], base_header: Union[eth.abc.BlockHeaderAPI, int, eth.rlp.transactions.BaseTransaction], transaction: Union[eth.abc.BlockHeaderAPI, int, eth.rlp.transactions.BaseTransaction], computation: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI, bytes], state: Union[bool, nucypher.blockchain.eth.interfaces.BlockchainDeployerInterface, bytes]) -> Union[eth.abc.ChainAPI, str, dict[str, str]]:
        gas_used = base_header.gas_used + cls.finalize_gas_used(transaction, computation)
        receipt_without_state_root = make_frontier_receipt(computation, gas_used)
        return receipt_without_state_root.copy(state_root=state.make_state_root())