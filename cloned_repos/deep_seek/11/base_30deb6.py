import contextlib
import itertools
import logging
from typing import (
    TYPE_CHECKING, Any, ClassVar, Dict, Generator, Iterable, Iterator, List, 
    Optional, Sequence, Set, Tuple, Type, Union, cast
)
from cached_property import cached_property
from eth_hash.auto import keccak
from eth_typing import Address, Hash32
from eth_utils import ValidationError, encode_hex
import rlp
from eth._utils.datatypes import Configurable
from eth._utils.db import get_block_header_by_hash, get_parent_header
from eth.abc import (
    AtomicDatabaseAPI, BlockAndMetaWitness, BlockAPI, BlockHeaderAPI, 
    ChainContextAPI, ChainDatabaseAPI, ComputationAPI, ConsensusAPI, 
    ConsensusContextAPI, ExecutionContextAPI, ReceiptAPI, ReceiptBuilderAPI, 
    SignedTransactionAPI, StateAPI, TransactionBuilderAPI, 
    UnsignedTransactionAPI, VirtualMachineAPI, WithdrawalAPI
)
from eth.consensus.pow import PowConsensus
from eth.constants import GENESIS_PARENT_HASH, MAX_PREV_HEADER_DEPTH, MAX_UNCLES
from eth.db.trie import make_trie_root_and_nodes
from eth.exceptions import HeaderNotFound
from eth.rlp.sedes import uint32
from eth.validation import validate_gas_limit, validate_length_lte
from eth.vm.execution_context import ExecutionContext
from eth.vm.interrupt import EVMMissingData
from eth.vm.message import Message

if TYPE_CHECKING:
    from eth.typing import Block, JournalDBCheckpoint

class VM(Configurable, VirtualMachineAPI):
    block_class: ClassVar[Optional[Type[BlockAPI]]] = None
    consensus_class: ClassVar[Type[ConsensusAPI]] = PowConsensus
    extra_data_max_bytes: ClassVar[int] = 32
    fork: ClassVar[Optional[str]] = None
    chaindb: ChainDatabaseAPI
    _state_class: ClassVar[Optional[Type[StateAPI]] = None
    _state: Optional[StateAPI]
    _block: Optional[BlockAPI]
    cls_logger: ClassVar[logging.Logger] = logging.getLogger('eth.vm.base.VM')

    def __init__(
        self, 
        header: BlockHeaderAPI, 
        chaindb: ChainDatabaseAPI, 
        chain_context: ChainContextAPI, 
        consensus_context: ConsensusContextAPI
    ) -> None:
        self.chaindb = chaindb
        self.chain_context = chain_context
        self.consensus_context = consensus_context
        self._initial_header = header
        self._state: Optional[StateAPI] = None
        self._block: Optional[BlockAPI] = None

    def get_header(self) -> BlockHeaderAPI:
        if self._block is None:
            return self._initial_header
        else:
            return self._block.header

    def get_block(self) -> BlockAPI:
        if self._block is None:
            block_class = self.get_block_class()
            self._block = block_class.from_header(
                header=self._initial_header, 
                chaindb=self.chaindb
            )
        return self._block

    @property
    def state(self) -> StateAPI:
        if self._state is None:
            self._state = self.build_state(
                self.chaindb.db, 
                self.get_header(), 
                self.chain_context, 
                self.previous_hashes
            )
        return self._state

    @classmethod
    def build_state(
        cls, 
        db: AtomicDatabaseAPI, 
        header: BlockHeaderAPI, 
        chain_context: ChainContextAPI, 
        previous_hashes: Iterable[Hash32] = ()
    ) -> StateAPI:
        execution_context = cls.create_execution_context(
            header, 
            previous_hashes, 
            chain_context
        )
        return cls.get_state_class()(db, execution_context, header.state_root)

    @cached_property
    def _consensus(self) -> ConsensusAPI:
        return self.consensus_class(self.consensus_context)

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(f'eth.vm.base.VM.{self.__class__.__name__}')

    def apply_transaction(
        self, 
        header: BlockHeaderAPI, 
        transaction: SignedTransactionAPI
    ) -> Tuple[ReceiptAPI, ComputationAPI]:
        self.validate_transaction_against_header(header, transaction)
        self.state.lock_changes()
        computation = self.state.apply_transaction(transaction)
        receipt = self.make_receipt(header, transaction, computation, self.state)
        self.validate_receipt(receipt)
        return (receipt, computation)

    @classmethod
    def create_execution_context(
        cls, 
        header: BlockHeaderAPI, 
        prev_hashes: Iterable[Hash32], 
        chain_context: ChainContextAPI
    ) -> ExecutionContextAPI:
        fee_recipient = cls.consensus_class.get_fee_recipient(header)
        base_fee_per_gas = getattr(header, 'base_fee_per_gas', None)
        excess_blob_gas = getattr(header, 'excess_blob_gas', None)
        return ExecutionContext(
            coinbase=fee_recipient, 
            timestamp=header.timestamp, 
            block_number=header.block_number, 
            difficulty=header.difficulty, 
            mix_hash=header.mix_hash, 
            gas_limit=header.gas_limit, 
            prev_hashes=prev_hashes, 
            chain_id=chain_context.chain_id, 
            base_fee_per_gas=base_fee_per_gas, 
            excess_blob_gas=excess_blob_gas
        )

    def execute_bytecode(
        self,
        origin: Optional[Address],
        gas_price: int,
        gas: int,
        to: Address,
        sender: Address,
        value: int,
        data: bytes,
        code: bytes,
        code_address: Optional[Address] = None
    ) -> ComputationAPI:
        if origin is None:
            origin = sender
        message = Message(
            gas=gas,
            to=to,
            sender=sender,
            value=value,
            data=data,
            code=code,
            code_address=code_address
        )
        transaction_context = self.state.get_transaction_context_class()(
            gas_price=gas_price,
            origin=origin
        )
        return self.state.computation_class.apply_computation(
            self.state,
            message,
            transaction_context
        )

    def apply_all_transactions(
        self, 
        transactions: Sequence[SignedTransactionAPI], 
        base_header: BlockHeaderAPI
    ) -> Tuple[BlockHeaderAPI, Tuple[ReceiptAPI, ...], Tuple[ComputationAPI, ...]]:
        vm_header = self.get_header()
        if base_header.block_number != vm_header.block_number:
            raise ValidationError(
                f'This VM instance must only work on block #{self.get_header().block_number}, '
                f'but the target header has block #{base_header.block_number}'
            )
        receipts: List[ReceiptAPI] = []
        computations: List[ComputationAPI] = []
        previous_header = base_header
        result_header = base_header
        for transaction_index, transaction in enumerate(transactions):
            snapshot = self.state.snapshot()
            try:
                receipt, computation = self.apply_transaction(previous_header, transaction)
            except EVMMissingData:
                self.state.revert(snapshot)
                raise
            result_header = self.add_receipt_to_header(previous_header, receipt)
            previous_header = result_header
            result_header = self.increment_blob_gas_used(previous_header, transaction)
            previous_header = result_header
            receipts.append(receipt)
            computations.append(computation)
            self.transaction_applied_hook(
                transaction_index,
                transactions,
                vm_header,
                result_header,
                computation,
                receipt
            )
        receipts_tuple = tuple(receipts)
        computations_tuple = tuple(computations)
        return (result_header, receipts_tuple, computations_tuple)

    def apply_withdrawal(self, withdrawal: WithdrawalAPI) -> None:
        self.state.apply_withdrawal(withdrawal)

    def apply_all_withdrawals(self, withdrawals: Sequence[WithdrawalAPI]) -> None:
        touched_addresses: List[Address] = []
        for withdrawal in withdrawals:
            withdrawal.validate()
            self.apply_withdrawal(withdrawal)
            if withdrawal.address not in touched_addresses:
                touched_addresses.append(withdrawal.address)
        for address in touched_addresses:
            if self.state.account_is_empty(address):
                self.state.delete_account(address)

    def import_block(self, block: BlockAPI) -> BlockAndMetaWitness:
        if self.get_block().number != block.number:
            raise ValidationError(
                f'This VM can only import blocks at number #{self.get_block().number}, '
                f'the attempted block was #{block.number}'
            )
        block_header_fields: Dict[str, Any] = {
            'coinbase': block.header.coinbase,
            'difficulty': block.header.difficulty,
            'gas_limit': block.header.gas_limit,
            'timestamp': block.header.timestamp,
            'extra_data': block.header.extra_data,
            'mix_hash': block.header.mix_hash,
            'nonce': block.header.nonce,
            'uncles_hash': keccak(rlp.encode(block.uncles))
        }
        try:
            parent_beacon_block_root = block.header.parent_beacon_block_root
            block_header_fields['parent_beacon_block_root'] = parent_beacon_block_root
        except AttributeError:
            pass
        block_fields: Dict[str, Any] = {
            'header': self.configure_header(**block_header_fields),
            'uncles': block.uncles
        }
        if hasattr(block, 'withdrawals'):
            block_fields['withdrawals'] = block.withdrawals
        self._block = self.get_block().copy(**block_fields)
        execution_context = self.create_execution_context(
            block.header,
            self.previous_hashes,
            self.chain_context
        )
        header = self.get_header().copy(gas_used=0)
        self._state = self.get_state_class()(
            self.chaindb.db,
            execution_context,
            header.state_root
        )
        self.block_preprocessing(self._state, block.header)
        new_header, receipts, _ = self.apply_all_transactions(block.transactions, header)
        withdrawals = block.withdrawals if hasattr(block, 'withdrawals') else None
        if withdrawals:
            self.apply_all_withdrawals(block.withdrawals)
        filled_block = self.set_block_transactions_and_withdrawals(
            self.get_block(),
            new_header,
            block.transactions,
            receipts,
            withdrawals=withdrawals
        )
        return self.mine_block(filled_block)

    @classmethod
    def block_preprocessing(cls, state: StateAPI, header: BlockHeaderAPI) -> None:
        """
        Process any state changes before processing a block. Pre-processing does not
        become relevant until the Cancun network upgrade.
        """
        pass

    def mine_block(
        self, 
        block: BlockAPI, 
        *args: Any, 
        **kwargs: Any
    ) -> BlockAndMetaWitness:
        packed_block = self.pack_block(block, *args, **kwargs)
        block_result = self.finalize_block(packed_block)
        self.validate_block(block_result.block)
        return block_result

    def set_block_transactions_and_withdrawals(
        self,
        base_block: BlockAPI,
        new_header: BlockHeaderAPI,
        transactions: Sequence[SignedTransactionAPI],
        receipts: Sequence[ReceiptAPI],
        withdrawals: Optional[Sequence[WithdrawalAPI]] = None
    ) -> BlockAPI:
        tx_root_hash, tx_kv_nodes = make_trie_root_and_nodes(transactions)
        self.chaindb.persist_trie_data_dict(tx_kv_nodes)
        receipt_root_hash, receipt_kv_nodes = make_trie_root_and_nodes(receipts)
        self.chaindb.persist_trie_data_dict(receipt_kv_nodes)
        block_fields: Dict[str, Any] = {'transactions': transactions}
        block_header_fields: Dict[str, Any] = {
            'transaction_root': tx_root_hash,
            'receipt_root': receipt_root_hash
        }
        if withdrawals:
            withdrawals_root_hash, withdrawal_kv_nodes = make_trie_root_and_nodes(withdrawals)
            self.chaindb.persist_trie_data_dict(withdrawal_kv_nodes)
            block_fields['withdrawals'] = withdrawals
            block_header_fields['withdrawals_root'] = withdrawals_root_hash
        block_fields['header'] = new_header.copy(**block_header_fields)
        return base_block.copy(**block_fields)

    def _assign_block_rewards(self, block: BlockAPI) -> None:
        block_reward = self.get_block_reward() + len(block.uncles) * self.get_nephew_reward()
        self.state.delta_balance(block.header.coinbase, block_reward)
        self.logger.debug(
            f'BLOCK REWARD: {block_reward} -> {encode_hex(block.header.coinbase)}'
        )
        for uncle in block.uncles:
            uncle_reward = self.get_uncle_reward(block.number, uncle)
            self.logger.debug(
                f'UNCLE REWARD REWARD: {uncle_reward} -> {encode_hex(uncle.coinbase)}'
            )
            self.state.delta_balance(uncle.coinbase, uncle_reward)

    def finalize_block(self, block: BlockAPI) -> BlockAndMetaWitness:
        if block.number > 0:
            snapshot = self.state.snapshot()
            try:
                self._assign_block_rewards(block)
            except EVMMissingData:
                self.state.revert(snapshot)
                raise
            else:
                self.state.commit(snapshot)
        meta_witness = self.state.persist()
        final_block = block.copy(header=block.header.copy(state_root=self.state.state_root))
        self.logger.debug(
            f'{final_block} reads {len(meta_witness.hashes)} unique node hashes, '
            f'{len(meta_witness.accounts_queried)} addresses, '
            f'{len(meta_witness.account_bytecodes_queried)} bytecodes, and '
            f'{meta_witness.total_slots_queried} storage slots'
        )
        return BlockAndMetaWitness(final_block, meta_witness)

    def pack_block(
        self, 
        block: BlockAPI, 
        *args: Any, 
        **kwargs: Any
    ) -> BlockAPI:
        if 'uncles' in kwargs:
            uncles = kwargs.pop('uncles')
            kwargs.setdefault('uncles_hash', keccak(rlp.encode(uncles)))
        else:
            uncles = block.uncles
        provided_fields = set(kwargs.keys())
        known_fields = set(self.get_header()._meta.field_names)
        unknown_fields = provided_fields.difference(known_fields)
        if unknown_fields:
            raise AttributeError(
                f'Unable to set the field(s) {", ".join(known_fields)} on the '
                f'`BlockHeader` class. Received the following unexpected fields: '
                f'{", ".join(unknown_fields)}.'
            )
        header = block.header.copy(**kwargs)
        packed_block = block.copy(uncles=uncles, header=header)
        return packed_block

    @classmethod
    def generate_block_from_parent_header_and_coinbase(
        cls, 
        parent_header: BlockHeaderAPI, 
        coinbase: Address
    ) -> BlockAPI:
        block_header = cls.create_header_from_parent(parent_header, coinbase=coinbase)
        block = cls.get_block_class()(block_header, transactions=[], uncles=[])
        return block

    @classmethod
    def create_genesis_header(cls, **genesis_params: Any) -> BlockHeaderAPI:
        return cls.create_header_from_parent(None, **genesis_params)

    @classmethod
    def get_block_class(cls) -> Type[BlockAPI]:
        if cls.block_class is None:
            raise AttributeError('No `block_class` has been set for this VM')
        else:
            return cls.block_class

    @classmethod
    def get_prev_hashes(
        cls, 
        last_block_hash: Hash32, 
        chaindb: ChainDatabaseAPI
    ) -> Generator[Hash32, None, None]:
        if last_block_hash == GENESIS_PARENT_HASH:
            return
        block_header = get_block_header_by_hash(last_block_hash, chaindb)
        for _ in range(MAX_PREV_HEADER_DEPTH):
            yield block_header.hash
            try:
                block_header = get_parent_header(block_header, chaindb)
            except (IndexError, HeaderNotFound):
                break

    @property
    def previous_hashes(self) -> Generator[Hash32, None, None]:
        return self.get_prev_hashes(self.get_header().parent_hash, self.chaindb)

    def create_transaction(self, *args: Any, **kwargs: Any) -> SignedTransactionAPI:
        return self.get_transaction_builder().new_transaction(*args, **kwargs)

    @classmethod
    def create_unsigned_transaction(
        cls, 
        *, 
        nonce: int, 
        gas_price: int, 
        gas: int, 
        to: Address, 
        value: int, 
        data: bytes
    ) -> UnsignedTransactionAPI:
        return cls.get_transaction_builder().create_unsigned_transaction(
            nonce=nonce,
            gas_price=gas_price,
            gas=gas,
            to=to,
            value=value,
            data=data
        )

    @classmethod
    def get_transaction_builder(cls) -> TransactionBuilderAPI:
        return cls.get_block_class().get_transaction_builder()

    @classmethod
    def get_receipt_builder(cls) -> ReceiptBuilderAPI:
        return cls.get_block_class().get_receipt_builder()

    @classmethod
    def validate_receipt(cls, receipt: ReceiptAPI) -> None:
        already_checked: Set[Union[Address, int]] = set()
        for log_idx, log in enumerate(receipt.logs):
            if log.address in already_checked:
                continue
            elif log.address not in receipt.bloom_filter:
                raise ValidationError(
                    f'The address from the log entry at position {log_idx} is not '
                    f'present in the provided bloom filter