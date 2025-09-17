import logging
import operator
import random
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Type, Set, List
from eth_typing import Address, BlockNumber, Hash32
from eth_utils import ValidationError, encode_hex, to_set
from eth_utils.toolz import assoc, compose, concatv, groupby, iterate, sliding_window, take
from eth._utils.datatypes import Configurable
from eth._utils.db import apply_state_dict
from eth._utils.rlp import validate_imported_block_unchanged
from eth.abc import (
    AtomicDatabaseAPI,
    BlockAndMetaWitness,
    BlockAPI,
    BlockHeaderAPI,
    BlockImportResult,
    BlockPersistResult,
    ChainAPI,
    ChainDatabaseAPI,
    ComputationAPI,
    ConsensusContextAPI,
    MiningChainAPI,
    ReceiptAPI,
    SignedTransactionAPI,
    StateAPI,
    UnsignedTransactionAPI,
    VirtualMachineAPI,
    WithdrawalAPI,
)
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
    chaindb: Any = None
    chaindb_class: Optional[Type] = None
    consensus_context_class: Optional[Type[ConsensusContextAPI]] = None
    vm_configuration: Optional[Sequence[Tuple[BlockNumber, Type[VirtualMachineAPI]]]] = None
    chain_id: Optional[int] = None

    @classmethod
    def get_vm_class_for_block_number(cls, block_number: BlockNumber) -> Type[VirtualMachineAPI]:
        if cls.vm_configuration is None:
            raise AttributeError('Chain classes must define the VMs in vm_configuration')
        validate_block_number(block_number)
        for start_block, vm_class in reversed(cls.vm_configuration):
            if block_number >= start_block:
                return vm_class
        else:
            raise VMNotFound(f'No vm available for block #{block_number}')

    @classmethod
    def get_vm_class(cls, header: BlockHeaderAPI) -> Type[VirtualMachineAPI]:
        return cls.get_vm_class_for_block_number(header.block_number)

    def validate_chain(
        self,
        root: BlockHeaderAPI,
        descendants: Sequence[BlockHeaderAPI],
        seal_check_random_sample_rate: int = 1
    ) -> None:
        all_indices: Sequence[int] = list(range(len(descendants)))
        if seal_check_random_sample_rate == 1:
            indices_to_check_seal = set(all_indices)
        elif seal_check_random_sample_rate == 0:
            indices_to_check_seal = set()
        else:
            sample_size: int = len(all_indices) // seal_check_random_sample_rate
            indices_to_check_seal = set(random.sample(all_indices, sample_size))
        header_pairs: Iterable[Tuple[BlockHeaderAPI, BlockHeaderAPI]] = sliding_window(2, concatv([root], descendants))
        for index, (parent, child) in enumerate(header_pairs):
            if child.parent_hash != parent.hash:
                raise ValidationError(
                    f'Invalid header chain; {child} has parent {encode_hex(child.parent_hash)}, but expected {encode_hex(parent.hash)}'
                )
            vm: VirtualMachineAPI = self.get_vm(child)
            try:
                vm.validate_header(child, parent)
            except ValidationError as exc:
                raise ValidationError(f'{child} is not a valid child of {parent}: {exc}') from exc
            if index in indices_to_check_seal:
                vm.validate_seal(child)

    def validate_chain_extension(self, headers: Sequence[BlockHeaderAPI]) -> None:
        for index, header in enumerate(headers):
            vm: VirtualMachineAPI = self.get_vm(header)
            parents: Sequence[BlockHeaderAPI] = headers[:index]
            vm.validate_seal_extension(header, parents)


class Chain(BaseChain):
    logger: logging.Logger = logging.getLogger('eth.chain.chain.Chain')
    gas_estimator: Optional[Callable[..., int]] = None
    chaindb_class: Type[ChainDB] = ChainDB
    consensus_context_class: Type[ConsensusContext] = ConsensusContext

    def __init__(self, base_db: AtomicDatabaseAPI) -> None:
        if not self.vm_configuration:
            raise ValueError('The Chain class cannot be instantiated with an empty `vm_configuration`')
        else:
            validate_vm_configuration(self.vm_configuration)
        self.chaindb = self.get_chaindb_class()(base_db)
        self.consensus_context = self.consensus_context_class(self.chaindb.db)
        self.headerdb = HeaderDB(base_db)
        if self.gas_estimator is None:
            self.gas_estimator = get_gas_estimator()

    @classmethod
    def get_chaindb_class(cls) -> Type[ChainDB]:
        if cls.chaindb_class is None:
            raise AttributeError('`chaindb_class` not set')
        return cls.chaindb_class

    @classmethod
    def from_genesis(cls, base_db: AtomicDatabaseAPI, genesis_params: HeaderParams, genesis_state: Optional[AccountState] = None) -> "Chain":
        genesis_vm_class: Type[VirtualMachineAPI] = cls.get_vm_class_for_block_number(BlockNumber(0))
        pre_genesis_header = BlockHeader(difficulty=0, block_number=-1, gas_limit=0)  # type: ignore
        chain_context: ChainContext = ChainContext(cls.chain_id)
        state = genesis_vm_class.build_state(base_db, pre_genesis_header, chain_context)
        if genesis_state is None:
            genesis_state = {}
        apply_state_dict(state, genesis_state)
        state.persist()
        if 'state_root' not in genesis_params:
            genesis_params = assoc(genesis_params, 'state_root', state.state_root)
        elif genesis_params['state_root'] != state.state_root:
            raise ValidationError(
                f'The provided genesis state root does not match the computed genesis state root.  Got {state.state_root!r}.  Expected {genesis_params["state_root"]!r}'
            )
        genesis_header: BlockHeaderAPI = genesis_vm_class.create_genesis_header(**genesis_params)
        return cls.from_genesis_header(base_db, genesis_header)

    @classmethod
    def from_genesis_header(cls, base_db: AtomicDatabaseAPI, genesis_header: BlockHeaderAPI) -> "Chain":
        chaindb_instance: ChainDB = cls.get_chaindb_class()(base_db)
        chaindb_instance.persist_header(genesis_header)
        return cls(base_db)

    def get_vm(self, at_header: Optional[BlockHeaderAPI] = None) -> VirtualMachineAPI:
        header: BlockHeaderAPI = self.ensure_header(at_header)
        vm_class: Type[VirtualMachineAPI] = self.get_vm_class_for_block_number(header.block_number)
        chain_context: ChainContext = ChainContext(self.chain_id)
        return vm_class(header=header, chaindb=self.chaindb, chain_context=chain_context, consensus_context=self.consensus_context)  # type: ignore

    def create_header_from_parent(self, parent_header: BlockHeaderAPI, **header_params: Any) -> BlockHeaderAPI:
        return self.get_vm_class_for_block_number(block_number=BlockNumber(parent_header.block_number + 1)).create_header_from_parent(parent_header, **header_params)

    def get_block_header_by_hash(self, block_hash: Hash32) -> BlockHeaderAPI:
        validate_word(block_hash, title='Block Hash')
        return self.chaindb.get_block_header_by_hash(block_hash)

    def get_canonical_block_header_by_number(self, block_number: BlockNumber) -> BlockHeaderAPI:
        return self.chaindb.get_canonical_block_header_by_number(block_number)

    def get_canonical_head(self) -> BlockHeaderAPI:
        return self.chaindb.get_canonical_head()

    def get_score(self, block_hash: Hash32) -> int:
        return self.headerdb.get_score(block_hash)

    def ensure_header(self, header: Optional[BlockHeaderAPI] = None) -> BlockHeaderAPI:
        """
        Return ``header`` if it is not ``None``, otherwise return the header
        of the canonical head.
        """
        if header is None:
            head: BlockHeaderAPI = self.get_canonical_head()
            return self.create_header_from_parent(head)
        else:
            return header

    def get_ancestors(self, limit: int, header: BlockHeaderAPI) -> Tuple[BlockHeaderAPI, ...]:
        ancestor_count: int = min(header.block_number, limit)
        vm_class: Type[VirtualMachineAPI] = self.get_vm_class_for_block_number(header.block_number)
        block_class: Type[BlockAPI] = vm_class.get_block_class()  # type: ignore
        block: BlockAPI = block_class(header=header, uncles=[], transactions=[])
        ancestor_generator = iterate(compose(self.get_block_by_hash, operator.attrgetter('parent_hash'), operator.attrgetter('header')), block)
        next(ancestor_generator)
        return tuple(take(ancestor_count, ancestor_generator))

    def get_block(self) -> BlockAPI:
        return self.get_vm().get_block()

    def get_block_by_hash(self, block_hash: Hash32) -> BlockAPI:
        validate_word(block_hash, title='Block Hash')
        block_header: BlockHeaderAPI = self.get_block_header_by_hash(block_hash)
        return self.get_block_by_header(block_header)

    def get_block_by_header(self, block_header: BlockHeaderAPI) -> BlockAPI:
        vm: VirtualMachineAPI = self.get_vm(block_header)
        return vm.get_block()

    def get_canonical_block_by_number(self, block_number: BlockNumber) -> BlockAPI:
        validate_uint256(block_number, title='Block Number')
        return self.get_block_by_hash(self.chaindb.get_canonical_block_hash(block_number))

    def get_canonical_block_hash(self, block_number: BlockNumber) -> Hash32:
        return self.chaindb.get_canonical_block_hash(block_number)

    def build_block_with_transactions_and_withdrawals(
        self,
        transactions: Sequence[SignedTransactionAPI],
        parent_header: Optional[BlockHeaderAPI] = None,
        withdrawals: Optional[Sequence[WithdrawalAPI]] = None
    ) -> Tuple[BlockAPI, Sequence[ReceiptAPI], Sequence[ComputationAPI]]:
        base_header: BlockHeaderAPI = self.ensure_header(parent_header)
        vm: VirtualMachineAPI = self.get_vm(base_header)
        new_header, receipts, computations = vm.apply_all_transactions(transactions, base_header)  # type: ignore
        if withdrawals:
            vm.apply_all_withdrawals(withdrawals)
        new_block: BlockAPI = vm.set_block_transactions_and_withdrawals(vm.get_block(), new_header, transactions, receipts, withdrawals=withdrawals)
        return (new_block, receipts, computations)

    def get_canonical_transaction_index(self, transaction_hash: Hash32) -> Tuple[int, int]:
        return self.chaindb.get_transaction_index(transaction_hash)

    def get_canonical_transaction(self, transaction_hash: Hash32) -> SignedTransactionAPI:
        block_num, index = self.chaindb.get_transaction_index(transaction_hash)
        transaction: SignedTransactionAPI = self.get_canonical_transaction_by_index(block_num, index)
        if transaction.hash == transaction_hash:
            return transaction
        else:
            raise TransactionNotFound(
                f'Found transaction {encode_hex(transaction.hash)} instead of {encode_hex(transaction_hash)} in block {block_num} at {index}'
            )

    def get_canonical_transaction_by_index(self, block_number: BlockNumber, index: int) -> SignedTransactionAPI:
        VM_class: Type[VirtualMachineAPI] = self.get_vm_class_for_block_number(block_number)
        return self.chaindb.get_transaction_by_index(block_number, index, VM_class.get_transaction_builder())  # type: ignore

    def create_transaction(self, *args: Any, **kwargs: Any) -> SignedTransactionAPI:
        return self.get_vm().create_transaction(*args, **kwargs)

    def create_unsigned_transaction(
        self,
        *,
        nonce: int,
        gas_price: int,
        gas: int,
        to: Address,
        value: int,
        data: bytes
    ) -> UnsignedTransactionAPI:
        return self.get_vm().create_unsigned_transaction(nonce=nonce, gas_price=gas_price, gas=gas, to=to, value=value, data=data)

    def get_transaction_receipt(self, transaction_hash: Hash32) -> ReceiptAPI:
        transaction_block_number, transaction_index = self.chaindb.get_transaction_index(transaction_hash)
        return self.get_transaction_receipt_by_index(transaction_block_number, transaction_index)

    def get_transaction_receipt_by_index(self, block_number: BlockNumber, index: int) -> ReceiptAPI:
        vm_class: Type[VirtualMachineAPI] = self.get_vm_class_for_block_number(block_number)
        receipt: ReceiptAPI = self.chaindb.get_receipt_by_index(block_number, index, vm_class.get_receipt_builder())  # type: ignore
        return receipt

    def get_transaction_result(self, transaction: SignedTransactionAPI, at_header: BlockHeaderAPI) -> bytes:
        with self.get_vm(at_header).in_costless_state() as state:  # type: ignore
            computation: ComputationAPI = state.costless_execute_transaction(transaction)
        computation.raise_if_error()
        return computation.output

    def estimate_gas(self, transaction: SignedTransactionAPI, at_header: Optional[BlockHeaderAPI] = None) -> int:
        if at_header is None:
            at_header = self.get_canonical_head()
        with self.get_vm(at_header).in_costless_state() as state:  # type: ignore
            return self.gas_estimator(state, transaction)  # type: ignore

    def import_block(self, block: BlockAPI, perform_validation: bool = True) -> BlockImportResult:
        try:
            parent_header: BlockHeaderAPI = self.get_block_header_by_hash(block.header.parent_hash)  # type: ignore
        except HeaderNotFound:
            raise ValidationError(
                f'Attempt to import block #{block.number}.  Cannot import block {block.hash!r} before importing its parent block at {block.header.parent_hash!r}'
            )
        base_header_for_import: BlockHeaderAPI = self.create_header_from_parent(parent_header)
        annotated_header: BlockHeaderAPI = base_header_for_import.copy(gas_used=block.header.gas_used)  # type: ignore
        block_result = self.get_vm(annotated_header).import_block(block)
        imported_block: BlockAPI = block_result.block
        if perform_validation:
            try:
                validate_imported_block_unchanged(imported_block, block)
            except ValidationError:
                self.logger.warning("Proposed %s doesn't follow EVM rules, rejecting...", block)
                raise
        persist_result: BlockPersistResult = self.persist_block(imported_block, perform_validation)
        return BlockImportResult(*persist_result, block_result.meta_witness)

    def persist_block(self, block: BlockAPI, perform_validation: bool = True) -> BlockPersistResult:
        if perform_validation:
            self.validate_block(block)
        new_canonical_hashes, old_canonical_hashes = self.chaindb.persist_block(block)
        self.logger.debug('Persisted block: number %s | hash %s', block.number, encode_hex(block.hash))
        new_canonical_blocks: Tuple[BlockAPI, ...] = tuple((self.get_block_by_hash(header_hash) for header_hash in new_canonical_hashes))
        old_canonical_blocks: Tuple[BlockAPI, ...] = tuple((self.get_block_by_hash(header_hash) for header_hash in old_canonical_hashes))
        return BlockPersistResult(imported_block=block, new_canonical_blocks=new_canonical_blocks, old_canonical_blocks=old_canonical_blocks)

    def validate_receipt(self, receipt: ReceiptAPI, at_header: BlockHeaderAPI) -> None:
        VM_class: Type[VirtualMachineAPI] = self.get_vm_class(at_header)
        VM_class.validate_receipt(receipt)

    def validate_block(self, block: BlockAPI) -> None:
        if block.is_genesis:  # type: ignore
            raise ValidationError('Cannot validate genesis block this way')
        vm: VirtualMachineAPI = self.get_vm(block.header)
        parent_header: BlockHeaderAPI = self.get_block_header_by_hash(block.header.parent_hash)  # type: ignore
        vm.validate_header(block.header, parent_header)
        vm.validate_seal(block.header)
        vm.validate_seal_extension(block.header, ())
        self.validate_uncles(block)

    def validate_seal(self, header: BlockHeaderAPI) -> None:
        vm: VirtualMachineAPI = self.get_vm(header)
        vm.validate_seal(header)

    def validate_uncles(self, block: BlockAPI) -> None:
        has_uncles: bool = len(block.uncles) > 0  # type: ignore
        should_have_uncles: bool = block.header.uncles_hash != EMPTY_UNCLE_HASH  # type: ignore
        if not has_uncles and (not should_have_uncles):
            return
        elif has_uncles and (not should_have_uncles):
            raise ValidationError('Block has uncles but header suggests uncles should be empty')
        elif should_have_uncles and (not has_uncles):
            raise ValidationError('Header suggests block should have uncles but block has none')
        uncle_groups: Dict[Any, List[Any]] = groupby(operator.attrgetter('hash'), block.uncles)  # type: ignore
        duplicate_uncles: Tuple[str, ...] = tuple(
            sorted((str(hash) for hash, twins in uncle_groups.items() if len(twins) > 1))
        )
        if duplicate_uncles:
            raise ValidationError(f'Block contains duplicate uncles:\n - {" - ".join(duplicate_uncles)}')
        recent_ancestors: Tuple[BlockHeaderAPI, ...] = tuple((ancestor for ancestor in self.get_ancestors(MAX_UNCLE_DEPTH + 1, header=block.header)))  # type: ignore
        recent_ancestor_hashes: Set[Hash32] = {ancestor.hash for ancestor in recent_ancestors}
        recent_uncle_hashes: Set[Hash32] = _extract_uncle_hashes(recent_ancestors)
        for uncle in block.uncles:  # type: ignore
            if uncle.hash == block.hash:  # type: ignore
                raise ValidationError('Uncle has same hash as block')
            if uncle.hash in recent_uncle_hashes:
                raise ValidationError(f'Duplicate uncle: {encode_hex(uncle.hash)}')
            if uncle.hash in recent_ancestor_hashes:
                raise ValidationError(f'Uncle {encode_hex(uncle.hash)} cannot be an ancestor of {encode_hex(block.hash)}')
            if uncle.parent_hash not in recent_ancestor_hashes or uncle.parent_hash == block.header.parent_hash:  # type: ignore
                raise ValidationError(f"Uncle's parent {encode_hex(uncle.parent_hash)} is not an ancestor of {encode_hex(block.hash)}")
            self.validate_seal(uncle)
            try:
                uncle_parent: BlockHeaderAPI = self.get_block_header_by_hash(uncle.parent_hash)  # type: ignore
            except HeaderNotFound:
                raise ValidationError(f'Uncle ancestor not found: {uncle.parent_hash!r}')
            uncle_vm_class: Type[VirtualMachineAPI] = self.get_vm_class_for_block_number(uncle.block_number)  # type: ignore
            uncle_vm_class.validate_uncle(block, uncle, uncle_parent)

@to_set
def _extract_uncle_hashes(blocks: Iterable[BlockHeaderAPI]) -> Set[Hash32]:
    for block in blocks:
        for uncle in block.uncles:  # type: ignore
            yield uncle.hash


class MiningChain(Chain, MiningChainAPI):
    header: BlockHeaderAPI

    def __init__(self, base_db: AtomicDatabaseAPI, header: Optional[BlockHeaderAPI] = None) -> None:
        super().__init__(base_db)
        self.header = self.ensure_header(header)

    def apply_transaction(self, transaction: SignedTransactionAPI) -> Tuple[BlockAPI, ReceiptAPI, ComputationAPI]:
        vm: VirtualMachineAPI = self.get_vm(self.header)
        base_block: BlockAPI = vm.get_block()
        receipt, computation = vm.apply_transaction(base_block.header, transaction)  # type: ignore
        header_with_receipt: BlockHeaderAPI = vm.add_receipt_to_header(base_block.header, receipt)  # type: ignore
        vm.state.persist()
        new_header: BlockHeaderAPI = header_with_receipt.copy(state_root=vm.state.state_root)  # type: ignore
        transactions: Tuple[SignedTransactionAPI, ...] = base_block.transactions + (transaction,)  # type: ignore
        receipts: Tuple[ReceiptAPI, ...] = base_block.get_receipts(self.chaindb) + (receipt,)  # type: ignore
        new_block: BlockAPI = vm.set_block_transactions_and_withdrawals(base_block, new_header, transactions, receipts)
        self.header = new_block.header
        return (new_block, receipt, computation)

    def import_block(self, block: BlockAPI, perform_validation: bool = True) -> BlockImportResult:
        result: BlockImportResult = super().import_block(block, perform_validation)
        self.header = self.ensure_header()
        return result

    def set_header_timestamp(self, timestamp: int) -> None:
        self.header = self.header.copy(timestamp=timestamp)

    @staticmethod
    def _custom_header(base_header: BlockHeaderAPI, **kwargs: Any) -> BlockHeaderAPI:
        header_fields = {'coinbase'}
        header_params: Dict[str, Any] = {k: v for k, v in kwargs.items() if k in header_fields}
        return base_header.copy(**header_params)

    def mine_all(
        self,
        transactions: Sequence[SignedTransactionAPI],
        *args: Any,
        parent_header: Optional[BlockHeaderAPI] = None,
        withdrawals: Optional[Sequence[WithdrawalAPI]] = None,
        **kwargs: Any
    ) -> Tuple[BlockImportResult, Sequence[ReceiptAPI], Sequence[ComputationAPI]]:
        if parent_header is None:
            base_header: BlockHeaderAPI = self.header
        else:
            base_header = self.create_header_from_parent(parent_header)
        custom_header: BlockHeaderAPI = self._custom_header(base_header, **kwargs)
        vm: VirtualMachineAPI = self.get_vm(custom_header)
        new_header, receipts, computations = vm.apply_all_transactions(transactions, base_header)  # type: ignore
        if withdrawals:
            vm.apply_all_withdrawals(withdrawals)
        filled_block: BlockAPI = vm.set_block_transactions_and_withdrawals(vm.get_block(), new_header, transactions, receipts, withdrawals=withdrawals)
        block_result = vm.mine_block(filled_block, *args, **kwargs)
        imported_block: BlockAPI = block_result.block
        block_persist_result: BlockPersistResult = self.persist_block(imported_block)
        block_import_result: BlockImportResult = BlockImportResult(*block_persist_result, block_result.meta_witness)
        self.header = self.create_header_from_parent(imported_block.header)
        return (block_import_result, receipts, computations)

    def mine_block(self, *args: Any, **kwargs: Any) -> BlockAPI:
        """
        Mine whatever transactions have been incrementally applied so far.
        """
        return self.mine_block_extended(*args, **kwargs).block

    def mine_block_extended(self, *args: Any, **kwargs: Any) -> BlockImportResult:
        custom_header: BlockHeaderAPI = self._custom_header(self.header, **kwargs)
        vm: VirtualMachineAPI = self.get_vm(custom_header)
        current_block: BlockAPI = vm.get_block()
        mine_result = vm.mine_block(current_block, *args, **kwargs)
        mined_block: BlockAPI = mine_result.block
        self.validate_block(mined_block)
        self.chaindb.persist_block(mined_block)
        self.header = self.create_header_from_parent(mined_block.header)
        return mine_result

    def get_vm(self, at_header: Optional[BlockHeaderAPI] = None) -> VirtualMachineAPI:
        if at_header is None:
            at_header = self.header
        return super().get_vm(at_header)