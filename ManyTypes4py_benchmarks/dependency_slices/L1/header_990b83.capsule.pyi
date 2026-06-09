# === Internal dependency: eth.abc ===
class BlockHeaderAPI(MiningHeaderAPI, BlockHeaderSedesAPI): ...
class HeaderDatabaseAPI(ABC):
    def __init__(self, db): ...
    def get_header_chain_gaps(self): ...
    def get_canonical_block_hash(self, block_number): ...
    def get_canonical_block_header_by_number(self, block_number): ...
    def get_canonical_head(self): ...
    def get_block_header_by_hash(self, block_hash): ...
    def get_score(self, block_hash): ...
    def header_exists(self, block_hash): ...
    def persist_checkpoint_header(self, header, score): ...
    def persist_header(self, header): ...
    def persist_header_chain(self, headers, genesis_parent_hash=...): ...

# === Internal dependency: eth.constants ===
ZERO_HASH32 = Hash32(...)
GENESIS_PARENT_HASH = ZERO_HASH32

# === Internal dependency: eth.db.chain_gaps ===
class GapChange(enum.Enum):
    GapFill = enum.auto(...)
    GapSplit = enum.auto(...)
    GapLeftShrink = enum.auto(...)
    GapRightShrink = enum.auto(...)
def reopen_gap(decanonicalized, base_gaps): ...
def fill_gap(newly_persisted, base_gaps): ...
GAP_WRITES = (GapChange.GapFill, GapChange.GapSplit, GapChange.GapLeftShrink, GapChange.GapRightShrink)
GENESIS_CHAIN_GAPS = ((), BlockNumber(...))
GapInfo = Tuple[GapChange, ChainGaps]

# === Internal dependency: eth.db.schema ===
class SchemaV1(SchemaAPI):
    ...

# === Internal dependency: eth.exceptions ===
class HeaderNotFound(PyEVMError): ...
class ParentNotFound(HeaderNotFound): ...
class CanonicalHeadNotFound(PyEVMError): ...
class CheckpointsMustBeCanonical(PyEVMError): ...

# === Internal dependency: eth.rlp.sedes ===
uint32 = BigEndianInt(...)
chain_gaps = rlp.sedes.List(...)

# === Internal dependency: eth.typing ===
BlockRange = Tuple[BlockNumber, BlockNumber]
ChainGaps = Tuple[Tuple[BlockRange, ...], BlockNumber]

# === Internal dependency: eth.validation ===
def validate_word(value, title=...): ...
def validate_block_number(block_number, title=...): ...

# === Internal dependency: eth.vm.header ===
from eth.vm.forks.prague.blocks import PragueBackwardsHeader
HeaderSedes = PragueBackwardsHeader

# === Third-party dependency: eth_typing ===
# Used symbols: BlockNumber, Hash32

# === Third-party dependency: eth_utils ===
# Used symbols: ValidationError, encode_hex, to_tuple

# === Third-party dependency: eth_utils.toolz ===
# Used symbols: concat, first, sliding_window

# === Third-party dependency: rlp ===
# Used symbols: decode, encode, sedes

# === Third-party dependency: rlp.sedes ===
# Used symbols: BigEndianInt