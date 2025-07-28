from typing import Type, List, Any
from eth_bloom import BloomFilter
from eth.abc import (
    BlockAPI,
    BlockHeaderAPI,
    ReceiptAPI,
    StateAPI,
    SignedTransactionAPI,
    ComputationAPI,
    TransactionFieldsAPI,
)
from eth.constants import BLOCK_REWARD, UNCLE_DEPTH_PENALTY_FACTOR, ZERO_HASH32
from eth.rlp.logs import Log
from eth.rlp.receipts import Receipt
from eth.vm.base import VM
from .blocks import FrontierBlock
from .constants import MAX_REFUND_QUOTIENT
from .state import FrontierState
from .headers import (
    create_frontier_header_from_parent,
    compute_frontier_difficulty,
    configure_frontier_header,
)
from .validation import validate_frontier_transaction_against_header


def make_frontier_receipt(
    computation: ComputationAPI, new_cumulative_gas_used: int
) -> ReceiptAPI:
    logs: List[Log] = [
        Log(address, topics, data)
        for address, topics, data in computation.get_log_entries()
    ]
    receipt: Receipt = Receipt(
        state_root=ZERO_HASH32, gas_used=new_cumulative_gas_used, logs=logs
    )
    return receipt


class FrontierVM(VM):
    fork: str = 'frontier'
    block_class: Type[BlockAPI] = FrontierBlock
    _state_class: Type[StateAPI] = FrontierState
    create_header_from_parent = staticmethod(create_frontier_header_from_parent)
    compute_difficulty = staticmethod(compute_frontier_difficulty)
    configure_header = configure_frontier_header
    validate_transaction_against_header = validate_frontier_transaction_against_header

    @staticmethod
    def get_block_reward() -> int:
        return BLOCK_REWARD

    @staticmethod
    def get_uncle_reward(block_number: int, uncle: BlockHeaderAPI) -> int:
        return (
            BLOCK_REWARD
            * (UNCLE_DEPTH_PENALTY_FACTOR + uncle.block_number - block_number)
            // UNCLE_DEPTH_PENALTY_FACTOR
        )

    @classmethod
    def get_nephew_reward(cls) -> int:
        return cls.get_block_reward() // 32

    def add_receipt_to_header(
        self, old_header: BlockHeaderAPI, receipt: ReceiptAPI
    ) -> BlockHeaderAPI:
        new_bloom_int = int(
            BloomFilter(old_header.bloom) | receipt.bloom  # type: ignore
        )
        new_gas_used = receipt.gas_used  # type: ignore
        new_state_root = self.state.make_state_root()
        return old_header.copy(
            bloom=new_bloom_int, gas_used=new_gas_used, state_root=new_state_root
        )

    def increment_blob_gas_used(
        self, old_header: BlockHeaderAPI, transaction: TransactionFieldsAPI
    ) -> BlockHeaderAPI:
        return old_header

    @classmethod
    def calculate_net_gas_refund(cls, consumed_gas: int, gross_refund: int) -> int:
        max_refund = consumed_gas // MAX_REFUND_QUOTIENT
        return min(max_refund, gross_refund)

    @classmethod
    def finalize_gas_used(
        cls, transaction: TransactionFieldsAPI, computation: ComputationAPI
    ) -> int:
        gas_remaining: int = computation.get_gas_remaining()
        consumed_gas: int = transaction.gas - gas_remaining  # type: ignore
        gross_refund: int = computation.get_gas_refund()
        net_refund: int = cls.calculate_net_gas_refund(consumed_gas, gross_refund)
        return consumed_gas - net_refund

    @classmethod
    def make_receipt(
        cls,
        base_header: BlockHeaderAPI,
        transaction: TransactionFieldsAPI,
        computation: ComputationAPI,
        state: StateAPI,
    ) -> ReceiptAPI:
        gas_used: int = base_header.gas_used + cls.finalize_gas_used(transaction, computation)  # type: ignore
        receipt_without_state_root: ReceiptAPI = make_frontier_receipt(computation, gas_used)
        return receipt_without_state_root.copy(state_root=state.make_state_root())