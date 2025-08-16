import logging
import operator
import random
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Type, Set
from eth_typing import Address, BlockNumber, Hash32
from eth_utils import ValidationError, encode_hex, to_set
from eth_utils.toolz import assoc, compose, concatv, groupby, iterate, sliding_window, take
from eth._utils.datatypes import Configurable
from eth._utils.db import apply_state_dict
from eth._utils.rlp import validate_imported_block_unchanged
from eth.abc import AtomicDatabaseAPI, BlockAndMetaWitness, BlockAPI, BlockHeaderAPI, BlockImportResult, BlockPersistResult, ChainAPI, ChainDatabaseAPI, ComputationAPI, ConsensusContextAPI, MiningChainAPI, ReceiptAPI, SignedTransactionAPI, StateAPI, UnsignedTransactionAPI, VirtualMachineAPI, WithdrawalAPI
from eth.consensus import ConsensusContext
from eth.constants import EMPTY_UNCLE_HASH, MAX_UNCLE_DEPTH
from eth.db.chain import ChainDB
from eth.db.header import HeaderDB
from eth.estimators import get_gas_estimator
from eth.exceptions import HeaderNotFound, TransactionNotFound, VMNotFound
from eth.rlp.headers import BlockHeader
from eth.typing import AccountState, HeaderParams, StaticMethod
from eth.validation import validate_block_number, validate_uint256, validate_vm_configuration, validate_word
from eth.vm.chain_context import ChainContext

class BaseChain(Configurable, ChainAPI):
    """
    The base class for all Chain objects
    """
    chaindb: Optional[ChainDatabaseAPI] = None
    chaindb_class: Optional[Type[ChainDB]] = None
    consensus_context_class: Optional[Type[ConsensusContextAPI]] = None
    vm_configuration: Optional[Sequence[Tuple[BlockNumber, Type[VirtualMachineAPI]]]] = None
    chain_id: Optional[int] = None

    @classmethod
    def get_vm_class_for_block_number(cls, block_number: BlockNumber) -> Type[VirtualMachineAPI]:
        ...

    @classmethod
    def get_vm_class(cls, header: BlockHeader) -> Type[VirtualMachineAPI]:
        ...

    def validate_chain(self, root: BlockHeader, descendants: Sequence[BlockHeader], seal_check_random_sample_rate: int = 1) -> None:
        ...

    def validate_chain_extension(self, headers: Sequence[BlockHeader]) -> None:
        ...

class Chain(BaseChain):
    logger: logging.Logger = logging.getLogger('eth.chain.chain.Chain')
    gas_estimator: Optional[Callable[[StateAPI, SignedTransactionAPI], int]] = None
    chaindb_class: Type[ChainDB] = ChainDB
    consensus_context_class: Type[ConsensusContext] = ConsensusContext

    def __init__(self, base_db: AtomicDatabaseAPI) -> None:
        ...

    @classmethod
    def from_genesis(cls, base_db: AtomicDatabaseAPI, genesis_params: Dict[str, Any], genesis_state: Optional[Dict[Address, AccountState]] = None) -> 'Chain':
        ...

    @classmethod
    def from_genesis_header(cls, base_db: AtomicDatabaseAPI, genesis_header: BlockHeader) -> 'Chain':
        ...

    def get_vm(self, at_header: Optional[BlockHeader] = None) -> VirtualMachineAPI:
        ...

    def create_header_from_parent(self, parent_header: BlockHeader, **header_params: Any) -> BlockHeader:
        ...

    def get_block_header_by_hash(self, block_hash: Hash32) -> BlockHeader:
        ...

    def get_canonical_block_header_by_number(self, block_number: BlockNumber) -> BlockHeader:
        ...

    def get_canonical_head(self) -> BlockHeader:
        ...

    def get_score(self, block_hash: Hash32) -> int:
        ...

    def ensure_header(self, header: Optional[BlockHeader] = None) -> BlockHeader:
        ...

    def get_ancestors(self, limit: int, header: BlockHeader) -> Tuple[BlockHeader, ...]:
        ...

    def get_block(self) -> BlockAPI:
        ...

    def get_block_by_hash(self, block_hash: Hash32) -> BlockAPI:
        ...

    def get_block_by_header(self, block_header: BlockHeader) -> BlockAPI:
        ...

    def get_canonical_block_by_number(self, block_number: BlockNumber) -> BlockAPI:
        ...

    def get_canonical_block_hash(self, block_number: BlockNumber) -> Hash32:
        ...

    def build_block_with_transactions_and_withdrawals(self, transactions: Sequence[SignedTransactionAPI], parent_header: Optional[BlockHeader] = None, withdrawals: Optional[Sequence[WithdrawalAPI]] = None) -> Tuple[BlockAPI, Sequence[ReceiptAPI], Sequence[ComputationAPI]]:
        ...

    def get_canonical_transaction_index(self, transaction_hash: Hash32) -> Tuple[BlockNumber, int]:
        ...

    def get_canonical_transaction(self, transaction_hash: Hash32) -> SignedTransactionAPI:
        ...

    def get_canonical_transaction_by_index(self, block_number: BlockNumber, index: int) -> SignedTransactionAPI:
        ...

    def create_transaction(self, *args: Any, **kwargs: Any) -> SignedTransactionAPI:
        ...

    def create_unsigned_transaction(self, *, nonce: int, gas_price: int, gas: int, to: Optional[Address], value: int, data: bytes) -> UnsignedTransactionAPI:
        ...

    def get_transaction_receipt(self, transaction_hash: Hash32) -> ReceiptAPI:
        ...

    def get_transaction_receipt_by_index(self, block_number: BlockNumber, index: int) -> ReceiptAPI:
        ...

    def get_transaction_result(self, transaction: SignedTransactionAPI, at_header: Optional[BlockHeader]) -> Any:
        ...

    def estimate_gas(self, transaction: SignedTransactionAPI, at_header: Optional[BlockHeader] = None) -> int:
        ...

    def import_block(self, block: BlockAPI, perform_validation: bool = True) -> BlockImportResult:
        ...

    def persist_block(self, block: BlockAPI, perform_validation: bool = True) -> BlockPersistResult:
        ...

    def validate_receipt(self, receipt: ReceiptAPI, at_header: BlockHeader) -> None:
        ...

    def validate_block(self, block: BlockAPI) -> None:
        ...

    def validate_seal(self, header: BlockHeader) -> None:
        ...

    def validate_uncles(self, block: BlockAPI) -> None:
        ...

@to_set
def _extract_uncle_hashes(blocks: Iterable[BlockAPI]) -> Set[Hash32]:
    ...

class MiningChain(Chain, MiningChainAPI):
    header: Optional[BlockHeader] = None

    def __init__(self, base_db: AtomicDatabaseAPI, header: Optional[BlockHeader] = None) -> None:
        ...

    def apply_transaction(self, transaction: SignedTransactionAPI) -> Tuple[BlockAPI, ReceiptAPI, ComputationAPI]:
        ...

    def import_block(self, block: BlockAPI, perform_validation: bool = True) -> BlockImportResult:
        ...

    def set_header_timestamp(self, timestamp: int) -> None:
        ...

    @staticmethod
    def _custom_header(base_header: BlockHeader, **kwargs: Any) -> BlockHeader:
        ...

    def mine_all(self, transactions: Sequence[SignedTransactionAPI], *args: Any, parent_header: Optional[BlockHeader] = None, withdrawals: Optional[Sequence[WithdrawalAPI]] = None, **kwargs: Any) -> Tuple[BlockImportResult, Sequence[ReceiptAPI], Sequence[ComputationAPI]]:
        ...

    def mine_block(self, *args: Any, **kwargs: Any) -> BlockAPI:
        ...

    def mine_block_extended(self, *args: Any, **kwargs: Any) -> BlockImportResult:
        ...

    def get_vm(self, at_header: Optional[BlockHeader] = None) -> VirtualMachineAPI:
        ...
