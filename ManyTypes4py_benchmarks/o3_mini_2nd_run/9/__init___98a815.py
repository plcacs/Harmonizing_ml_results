from typing import Type, Callable
from eth_utils.toolz import curry
from eth_bloom import BloomFilter
from eth_utils import encode_hex, ValidationError
from eth.abc import (
    BlockAPI,
    BlockHeaderAPI,
    ComputationAPI,
    ReceiptAPI,
    SignedTransactionAPI,
    StateAPI,
)
from eth.constants import MAX_UNCLE_DEPTH
from eth.validation import validate_lte
from eth.vm.forks.spurious_dragon import SpuriousDragonVM
from .blocks import ByzantiumBlock
from .constants import (
    EIP649_BLOCK_REWARD,
    EIP658_TRANSACTION_STATUS_CODE_FAILURE,
    EIP658_TRANSACTION_STATUS_CODE_SUCCESS,
)
from .headers import (
    create_byzantium_header_from_parent,
    configure_byzantium_header,
    compute_byzantium_difficulty,
)
from .state import ByzantiumState

@curry
def get_uncle_reward(
    block_reward: int, block_number: int, uncle: BlockHeaderAPI
) -> int:
    block_number_delta: int = block_number - uncle.block_number
    validate_lte(block_number_delta, MAX_UNCLE_DEPTH)
    return (8 - block_number_delta) * block_reward // 8

EIP658_STATUS_CODES = {
    EIP658_TRANSACTION_STATUS_CODE_SUCCESS,
    EIP658_TRANSACTION_STATUS_CODE_FAILURE,
}

class ByzantiumVM(SpuriousDragonVM):
    fork: str = 'byzantium'
    block_class: Type[BlockAPI] = ByzantiumBlock
    _state_class: Type[StateAPI] = ByzantiumState
    create_header_from_parent: Callable[..., BlockHeaderAPI] = staticmethod(
        create_byzantium_header_from_parent
    )
    compute_difficulty: Callable[..., int] = staticmethod(compute_byzantium_difficulty)
    configure_header: Callable[..., None] = configure_byzantium_header
    # After currying, get_uncle_reward accepts block_number and uncle.
    get_uncle_reward: Callable[[int, BlockHeaderAPI], int] = staticmethod(
        get_uncle_reward(EIP649_BLOCK_REWARD)
    )

    @classmethod
    def validate_receipt(cls, receipt: ReceiptAPI) -> None:
        super().validate_receipt(receipt)
        if receipt.state_root not in EIP658_STATUS_CODES:
            raise ValidationError(
                f"The receipt's `state_root` must be one of "
                f"[{encode_hex(EIP658_TRANSACTION_STATUS_CODE_SUCCESS)}, {encode_hex(EIP658_TRANSACTION_STATUS_CODE_FAILURE)}].  "
                f"Got: {encode_hex(receipt.state_root)}"
            )

    @staticmethod
    def get_block_reward() -> int:
        return EIP649_BLOCK_REWARD

    def add_receipt_to_header(
        self, old_header: BlockHeaderAPI, receipt: ReceiptAPI
    ) -> BlockHeaderAPI:
        new_bloom: int = int(BloomFilter(old_header.bloom) | receipt.bloom)
        return old_header.copy(bloom=new_bloom, gas_used=receipt.gas_used)

    @classmethod
    def make_receipt(
        cls,
        base_header: BlockHeaderAPI,
        transaction: SignedTransactionAPI,
        computation: ComputationAPI,
        state: StateAPI,
    ) -> ReceiptAPI:
        gas_used: int = base_header.gas_used + cls.finalize_gas_used(
            transaction, computation
        )
        if computation.is_error:
            status_code = EIP658_TRANSACTION_STATUS_CODE_FAILURE
        else:
            status_code = EIP658_TRANSACTION_STATUS_CODE_SUCCESS
        return transaction.make_receipt(
            status_code, gas_used, computation.get_log_entries()
        )