    def __init__(self, header: BlockHeaderAPI, chaindb: ChainDatabaseAPI, chain_context: ChainContextAPI, consensus_context: ConsensusContextAPI) -> None:
    def get_header(self) -> BlockHeaderAPI:
    def get_block(self) -> BlockAPI:
    def build_state(cls, db: AtomicDatabaseAPI, header: BlockHeaderAPI, chain_context: ChainContextAPI, previous_hashes: Iterable[Hash32] = ()) -> StateAPI:
    def apply_transaction(self, header: BlockHeaderAPI, transaction: SignedTransactionAPI) -> Tuple[ReceiptAPI, ComputationAPI]:
    def create_execution_context(cls, header: BlockHeaderAPI, prev_hashes: Iterable[Hash32], chain_context: ChainContextAPI) -> ExecutionContextAPI:
    def execute_bytecode(self, origin: Optional[Address], gas_price: int, gas: int, to: Optional[Address], sender: Address, value: int, data: bytes, code: bytes, code_address: Optional[Address] = None) -> ComputationAPI:
    def apply_all_transactions(self, transactions: Sequence[SignedTransactionAPI], base_header: BlockHeaderAPI) -> Tuple[BlockHeaderAPI, Tuple[ReceiptAPI, ...], Tuple[ComputationAPI, ...]:
    def apply_withdrawal(self, withdrawal: WithdrawalAPI) -> None:
    def apply_all_withdrawals(self, withdrawals: Sequence[WithdrawalAPI]) -> None:
    def import_block(self, block: Block) -> BlockAndMetaWitness:
    def block_preprocessing(cls, state: StateAPI, header: BlockHeaderAPI) -> None:
    def mine_block(self, block: BlockAPI, *args, **kwargs) -> BlockAndMetaWitness:
    def set_block_transactions_and_withdrawals(self, base_block: BlockAPI, new_header: BlockHeaderAPI, transactions: Sequence[SignedTransactionAPI], receipts: Sequence[ReceiptAPI], withdrawals: Optional[Sequence[WithdrawalAPI]] = None) -> BlockAPI:
    def _assign_block_rewards(self, block: BlockAPI) -> None:
    def finalize_block(self, block: BlockAPI) -> BlockAndMetaWitness:
    def pack_block(self, block: BlockAPI, *args, **kwargs) -> BlockAPI:
    def generate_block_from_parent_header_and_coinbase(cls, parent_header: BlockHeaderAPI, coinbase: Address) -> BlockAPI:
    def create_genesis_header(cls, **genesis_params: Any) -> BlockHeaderAPI:
    def get_prev_hashes(cls, last_block_hash: Hash32, chaindb: ChainDatabaseAPI) -> Iterator[Hash32]:
    def create_transaction(self, *args, **kwargs) -> SignedTransactionAPI:
    def create_unsigned_transaction(cls, *, nonce: int, gas_price: int, gas: int, to: Optional[Address], value: int, data: bytes) -> UnsignedTransactionAPI:
    def validate_receipt(cls, receipt: ReceiptAPI) -> None:
    def validate_block(self, block: BlockAPI) -> None:
    def validate_header(cls, header: BlockHeaderAPI, parent_header: BlockHeaderAPI) -> None:
    def validate_gas(cls, header: BlockHeaderAPI, parent_header: BlockHeaderAPI) -> None:
    def validate_seal(self, header: BlockHeaderAPI) -> None:
    def validate_seal_extension(self, header: BlockHeaderAPI, parents: Sequence[BlockHeaderAPI]) -> None:
    def validate_uncle(cls, block: BlockAPI, uncle: BlockAPI, uncle_parent: BlockHeaderAPI) -> None:
