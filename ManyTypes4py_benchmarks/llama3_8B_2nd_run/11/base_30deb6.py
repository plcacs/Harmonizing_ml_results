import contextlib
import itertools
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Type, Union
from cached_property import cached_property
from eth_hash.auto import keccak
from eth_typing import Address, Hash32
from eth_utils import ValidationError, encode_hex
import rlp
from eth._utils.datatypes import Configurable
from eth._utils.db import get_block_header_by_hash, get_parent_header
from eth.abc import AtomicDatabaseAPI, BlockAndMetaWitness, BlockAPI, BlockHeaderAPI, ChainContextAPI, ChainDatabaseAPI, ComputationAPI, ConsensusAPI, ConsensusContextAPI, ExecutionContextAPI, ReceiptAPI, ReceiptBuilderAPI, SignedTransactionAPI, StateAPI, TransactionBuilderAPI, UnsignedTransactionAPI, VirtualMachineAPI, WithdrawalAPI
from eth.consensus.pow import PowConsensus
from eth.constants import GENESIS_PARENT_HASH, MAX_PREV_HEADER_DEPTH, MAX_UNCLES
from eth.db.trie import make_trie_root_and_nodes
from eth.exceptions import HeaderNotFound
from eth.rlp.sedes import uint32
from eth.validation import validate_gas_limit, validate_length_lte
from eth.vm.execution_context import ExecutionContext
from eth.vm.interrupt import EVMMissingData
from eth.vm.message import Message

class VM(Configurable, VirtualMachineAPI):
    block_class: Type[BlockAPI]
    consensus_class: Type[ConsensusAPI]
    extra_data_max_bytes: int
    fork: Optional[str]
    chaindb: Optional[ChainDatabaseAPI]
    _state_class: Optional[Type[StateAPI]]
    _state: Optional[StateAPI]
    _block: Optional[BlockAPI]
    cls_logger: logging.Logger

    def __init__(self, header: BlockHeaderAPI, chaindb: ChainDatabaseAPI, chain_context: ChainContextAPI, consensus_context: ConsensusContextAPI) -> None:
        self.chaindb = chaindb
        self.chain_context = chain_context
        self.consensus_context = consensus_context
        self._initial_header = header

    def get_header(self) -> BlockHeaderAPI:
        if self._block is None:
            return self._initial_header
        else:
            return self._block.header

    def get_block(self) -> BlockAPI:
        if self._block is None:
            block_class = self.get_block_class()
            self._block = block_class.from_header(header=self._initial_header, chaindb=self.chaindb)
        return self._block

    @property
    def state(self) -> StateAPI:
        if self._state is None:
            self._state = self.build_state(self.chaindb.db, self.get_header(), self.chain_context, self.previous_hashes)
        return self._state

    @classmethod
    def build_state(cls, db: Any, header: BlockHeaderAPI, chain_context: ChainContextAPI, previous_hashes: Iterable[Hash32]) -> StateAPI:
        execution_context = cls.create_execution_context(header, previous_hashes, chain_context)
        return cls.get_state_class()(db, execution_context, header.state_root)

    # ... rest of the code ...
