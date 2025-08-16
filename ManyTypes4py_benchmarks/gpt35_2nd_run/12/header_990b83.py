    def get_header_chain_gaps(self) -> ChainGaps:
    def _get_header_chain_gaps(cls, db: AtomicDatabaseAPI) -> ChainGaps:
    def _update_header_chain_gaps(cls, db: AtomicDatabaseAPI, persisted_header: BlockHeaderAPI, base_gaps: ChainGaps = None) -> Tuple[GapChange, ChainGaps]:
    def get_canonical_block_hash(self, block_number: BlockNumber) -> Hash32:
    def _get_canonical_block_hash(db: AtomicDatabaseAPI, block_number: BlockNumber) -> Hash32:
    def get_canonical_block_header_by_number(self, block_number: BlockNumber) -> BlockHeaderAPI:
    def _get_canonical_block_header_by_number(cls, db: AtomicDatabaseAPI, block_number: BlockNumber) -> BlockHeaderAPI:
    def get_canonical_head(self) -> BlockHeaderAPI:
    def _get_canonical_head(cls, db: AtomicDatabaseAPI) -> BlockHeaderAPI:
    def _get_canonical_head_hash(cls, db: AtomicDatabaseAPI) -> Hash32:
    def get_block_header_by_hash(self, block_hash: Hash32) -> BlockHeaderAPI:
    def _get_block_header_by_hash(db: AtomicDatabaseAPI, block_hash: Hash32) -> BlockHeaderAPI:
    def get_score(self, block_hash: Hash32) -> int:
    def _get_score(db: AtomicDatabaseAPI, block_hash: Hash32) -> int:
    def header_exists(self, block_hash: Hash32) -> bool:
    def _header_exists(db: AtomicDatabaseAPI, block_hash: Hash32) -> bool:
    def persist_header(self, header: BlockHeaderAPI) -> Tuple[Tuple[BlockHeaderAPI, ...], Tuple[BlockHeaderAPI, ...]]:
    def persist_header_chain(self, headers: Sequence[BlockHeaderAPI], genesis_parent_hash: Hash32 = GENESIS_PARENT_HASH) -> Tuple[Tuple[BlockHeaderAPI, ...], Tuple[BlockHeaderAPI, ...]:
    def persist_checkpoint_header(self, header: BlockHeaderAPI, score: int) -> Tuple[Tuple[BlockHeaderAPI, ...], Tuple[BlockHeaderAPI, ...]:
    def _set_hash_scores_to_db(cls, db: AtomicDatabaseAPI, header: BlockHeaderAPI, score: int) -> int:
    def _persist_checkpoint_header(cls, db: AtomicDatabaseAPI, header: BlockHeaderAPI, score: int) -> Tuple[Tuple[BlockHeaderAPI, ...], Tuple[BlockHeaderAPI, ...]:
    def _decanonicalize_descendant_orphans(cls, db: AtomicDatabaseAPI, header: BlockHeaderAPI, checkpoints: Tuple[Hash32, ...]) -> None:
    def _decanonicalize_single(cls, db: AtomicDatabaseAPI, block_num: BlockNumber, base_gaps: ChainGaps) -> ChainGaps:
    def _persist_header_chain(cls, db: AtomicDatabaseAPI, headers: Sequence[BlockHeaderAPI], genesis_parent_hash: Hash32) -> Tuple[Tuple[BlockHeaderAPI, ...], Tuple[BlockHeaderAPI, ...]:
    def _handle_gap_change(cls, db: AtomicDatabaseAPI, gap_info: Tuple[GapChange, ChainGaps], header: BlockHeaderAPI, genesis_parent_hash: Hash32) -> ChainGaps:
    def _canonicalize_header(cls, db: AtomicDatabaseAPI, header: BlockHeaderAPI, genesis_parent_hash: Hash32) -> Tuple[Tuple[BlockHeaderAPI, ...], Tuple[BlockHeaderAPI, ...]:
    def _set_as_canonical_chain_head(cls, db: AtomicDatabaseAPI, header: BlockHeaderAPI, genesis_parent_hash: Hash32) -> Tuple[Tuple[BlockHeaderAPI, ...], Tuple[BlockHeaderAPI, ...]:
    def _get_checkpoints(cls, db: AtomicDatabaseAPI) -> Tuple[Hash32, ...]:
    def _find_headers_to_decanonicalize(cls, db: AtomicDatabaseAPI, numbers_to_decanonicalize: Iterable[BlockNumber]) -> Tuple[BlockHeaderAPI, ...]:
    def _find_new_ancestors(cls, db: AtomicDatabaseAPI, header: BlockHeaderAPI, genesis_parent_hash: Hash32) -> Tuple[BlockHeaderAPI, ...]:
    def _add_block_number_to_hash_lookup(db: AtomicDatabaseAPI, header: BlockHeaderAPI) -> None:
