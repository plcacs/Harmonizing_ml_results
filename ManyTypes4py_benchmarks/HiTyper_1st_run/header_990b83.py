import functools
from typing import Iterable, Sequence, Tuple, cast
from eth_typing import BlockNumber, Hash32
from eth_utils import ValidationError, encode_hex, to_tuple
from eth_utils.toolz import concat, first, sliding_window
import rlp
from eth.abc import AtomicDatabaseAPI, BlockHeaderAPI, DatabaseAPI, HeaderDatabaseAPI
from eth.constants import GENESIS_PARENT_HASH
from eth.db.chain_gaps import GAP_WRITES, GENESIS_CHAIN_GAPS, GapChange, GapInfo, fill_gap, reopen_gap
from eth.db.schema import SchemaV1
from eth.exceptions import CanonicalHeadNotFound, CheckpointsMustBeCanonical, HeaderNotFound, ParentNotFound
from eth.rlp.sedes import chain_gaps
from eth.typing import ChainGaps
from eth.validation import validate_block_number, validate_word
from eth.vm.header import HeaderSedes

class HeaderDB(HeaderDatabaseAPI):

    def __init__(self, db: Union[str, solo.server.db.types.SQLEngine]) -> None:
        self.db = db

    def get_header_chain_gaps(self):
        return self._get_header_chain_gaps(self.db)

    @classmethod
    def _get_header_chain_gaps(cls: Union[AtomicDatabaseAPI, eth.rlp.headers.BlockHeader, BlockHeaderAPI], db: eth.abc.DatabaseAPI):
        try:
            encoded_gaps = db[SchemaV1.make_header_chain_gaps_lookup_key()]
        except KeyError:
            return GENESIS_CHAIN_GAPS
        else:
            return rlp.decode(encoded_gaps, sedes=chain_gaps)

    @classmethod
    def _update_header_chain_gaps(cls: Union[eth.abc.BlockHeaderAPI, eth.abc.DatabaseAPI, eth.ChainGaps], db: Union[eth.abc.DatabaseAPI, eth.ChainGaps, eth.abc.BlockHeaderAPI], persisted_header: Union[eth.abc.BlockHeaderAPI, eth.abc.DatabaseAPI, eth.ChainGaps], base_gaps: Union[None, eth.ChainGaps, eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI]=None) -> tuple[typing.Union[GapChange_@_NoChange,str,int,typing.Iterator]]:
        if base_gaps is None:
            base_gaps = cls._get_header_chain_gaps(db)
        gap_change, gaps = fill_gap(persisted_header.block_number, base_gaps)
        if gap_change is not GapChange.NoChange:
            db.set(SchemaV1.make_header_chain_gaps_lookup_key(), rlp.encode(gaps, sedes=chain_gaps))
        return (gap_change, gaps)

    def get_canonical_block_hash(self, block_number: int):
        return self._get_canonical_block_hash(self.db, block_number)

    @staticmethod
    def _get_canonical_block_hash(db: Union[typing.Iterable[int], int], block_number: int):
        validate_block_number(block_number)
        number_to_hash_key = SchemaV1.make_block_number_to_hash_lookup_key(block_number)
        try:
            encoded_key = db[number_to_hash_key]
        except KeyError:
            raise HeaderNotFound(f'No canonical header for block number #{block_number}')
        else:
            return rlp.decode(encoded_key, sedes=rlp.sedes.binary)

    def get_canonical_block_header_by_number(self, block_number: int):
        return self._get_canonical_block_header_by_number(self.db, block_number)

    @classmethod
    def _get_canonical_block_header_by_number(cls: Union[eth.abc.DatabaseAPI, eth.abc.BlockAPI, eth.chains.base.BaseChain], db: Union[eth.abc.DatabaseAPI, eth.abc.BlockAPI, eth.chains.base.BaseChain], block_number: Union[eth.abc.DatabaseAPI, int, float]) -> Union[bool, str, tuple[int]]:
        validate_block_number(block_number)
        canonical_block_hash = cls._get_canonical_block_hash(db, block_number)
        return cls._get_block_header_by_hash(db, canonical_block_hash)

    def get_canonical_head(self):
        return self._get_canonical_head(self.db)

    @classmethod
    def _get_canonical_head(cls: Union[eth.abc.DatabaseAPI, eth.abc.AtomicDatabaseAPI, eth.db.backends.base.BaseDB], db: Union[eth.abc.DatabaseAPI, eth.abc.AtomicDatabaseAPI, eth.db.backends.base.BaseDB]) -> Union[bytes, bool, str]:
        canonical_head_hash = cls._get_canonical_head_hash(db)
        return cls._get_block_header_by_hash(db, canonical_head_hash)

    @classmethod
    def _get_canonical_head_hash(cls: eth.abc.AtomicDatabaseAPI, db: Union[eth.abc.DatabaseAPI, eth.rlp.headers.BlockHeader, eth.abc.AtomicDatabaseAPI]) -> Hash32:
        try:
            return Hash32(db[SchemaV1.make_canonical_head_hash_lookup_key()])
        except KeyError:
            raise CanonicalHeadNotFound('No canonical head set for this chain')

    def get_block_header_by_hash(self, block_hash: Any):
        return self._get_block_header_by_hash(self.db, block_hash)

    @staticmethod
    def _get_block_header_by_hash(db: Union[str, bytes, eth.abc.DatabaseAPI], block_hash: Union[eth.abc.DatabaseAPI, eth.rlp.headers.BlockHeader]) -> Union[bytes, str, dict]:
        """
        Returns the requested block header as specified by block hash.

        Raises BlockNotFound if it is not present in the db.
        """
        validate_word(block_hash, title='Block Hash')
        try:
            header_rlp = db[block_hash]
        except KeyError:
            raise HeaderNotFound(f'No header with hash {encode_hex(block_hash)} found')
        return _decode_block_header(header_rlp)

    def get_score(self, block_hash: Union[bytes, typing.Iterable[eth.abc.BlockHeaderAPI], str, int]):
        return self._get_score(self.db, block_hash)

    @staticmethod
    def _get_score(db: Union[int, eth.abc.DatabaseAPI, eth.abc.BlockAPI], block_hash: Union[eth.abc.DatabaseAPI, eth.abc.BlockAPI, eth.abc.BlockHeaderAPI]):
        try:
            encoded_score = db[SchemaV1.make_block_hash_to_score_lookup_key(block_hash)]
        except KeyError:
            raise HeaderNotFound(f'No header with hash {encode_hex(block_hash)} found')
        return rlp.decode(encoded_score, sedes=rlp.sedes.big_endian_int)

    def header_exists(self, block_hash: bytes):
        return self._header_exists(self.db, block_hash)

    @staticmethod
    def _header_exists(db: Union[eth.abc.DatabaseAPI, eth.rlp.headers.BlockHeader, str], block_hash: Union[eth.abc.DatabaseAPI, eth.rlp.headers.BlockHeader]) -> bool:
        validate_word(block_hash, title='Block Hash')
        return block_hash in db

    def persist_header(self, header: Union[eth.rlp.headers.BlockHeader, eth.abc.BlockHeaderAPI, dict]) -> Union[str, dict]:
        return self.persist_header_chain((header,))

    def persist_header_chain(self, headers: Union[typing.Iterable[eth.abc.BlockHeaderAPI], eth.abc.BlockHeaderAPI, int], genesis_parent_hash: Any=GENESIS_PARENT_HASH):
        with self.db.atomic_batch() as db:
            return self._persist_header_chain(db, headers, genesis_parent_hash)

    def persist_checkpoint_header(self, header: Union[int, eth.abc.BlockHeaderAPI, dict], score: Union[int, eth.abc.BlockHeaderAPI, dict]):
        with self.db.atomic_batch() as db:
            return self._persist_checkpoint_header(db, header, score)

    @classmethod
    def _set_hash_scores_to_db(cls: Union[int, BlockHeaderAPI], db: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI], header: Union[int, eth.rlp.transactions.BaseTransaction, id3c.db.session.DatabaseSession], score: Union[int, bytes, str]):
        difficulty = header.difficulty
        new_score = score + difficulty if difficulty != 0 else score + header.block_number
        db.set(SchemaV1.make_block_hash_to_score_lookup_key(header.hash), rlp.encode(new_score, sedes=rlp.sedes.big_endian_int))
        return new_score

    @classmethod
    def _persist_checkpoint_header(cls: Union[int, eth.abc.DatabaseAPI, eth.ChainGaps], db: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI, eth.ChainGaps], header: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI], score: Union[int, eth.db.chain_gaps.GapInfo]) -> None:
        db.set(header.hash, rlp.encode(header))
        previous_checkpoints = cls._get_checkpoints(db)
        new_checkpoints = previous_checkpoints + (header.hash,)
        db.set(SchemaV1.make_checkpoint_headers_key(), b''.join(new_checkpoints))
        difficulty = header.difficulty
        previous_score = score - difficulty if difficulty != 0 else score - header.block_number
        cls._set_hash_scores_to_db(db, header, previous_score)
        cls._set_as_canonical_chain_head(db, header, GENESIS_PARENT_HASH)
        _, gaps = cls._update_header_chain_gaps(db, header)
        parent_block_num = BlockNumber(header.block_number - 1)
        try:
            parent_hash = cls._get_canonical_block_hash(db, parent_block_num)
        except HeaderNotFound:
            pass
        else:
            if parent_hash != header.parent_hash:
                try:
                    true_parent = cls._get_block_header_by_hash(db, header.parent_hash)
                except HeaderNotFound:
                    cls._decanonicalize_single(db, parent_block_num, gaps)
                else:
                    raise ValidationError(f'Why was a non-matching parent header {parent_hash!r} left as canonical after _set_as_canonical_chain_head() and {true_parent} is available?')
        cls._decanonicalize_descendant_orphans(db, header, new_checkpoints)

    @classmethod
    def _decanonicalize_descendant_orphans(cls: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI, eth.ChainGaps], db: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI, eth.ChainGaps], header: Union[eth.abc.BlockHeaderAPI, eth.abc.DatabaseAPI, eth.ChainGaps], checkpoints: eth.abc.DatabaseAPI) -> None:
        new_gaps = starting_gaps = cls._get_header_chain_gaps(db)
        child_number = BlockNumber(header.block_number + 1)
        try:
            child = cls._get_canonical_block_header_by_number(db, child_number)
        except HeaderNotFound:
            next_invalid_child = None
        else:
            if child.parent_hash != header.hash:
                if child.hash in checkpoints:
                    raise CheckpointsMustBeCanonical(f'Trying to decanonicalize {child} while making {header} the chain tip')
                else:
                    next_invalid_child = child
            else:
                next_invalid_child = None
        while next_invalid_child:
            db.delete(SchemaV1.make_block_number_to_hash_lookup_key(child_number))
            new_gaps = reopen_gap(child_number, new_gaps)
            child_number = BlockNumber(child_number + 1)
            try:
                next_invalid_child = cls._get_canonical_block_header_by_number(db, child_number)
            except HeaderNotFound:
                break
            else:
                if next_invalid_child.hash in checkpoints:
                    raise CheckpointsMustBeCanonical(f'Trying to decanonicalize {next_invalid_child} while making {header} the chain tip')
        if new_gaps != starting_gaps:
            db.set(SchemaV1.make_header_chain_gaps_lookup_key(), rlp.encode(new_gaps, sedes=chain_gaps))

    @classmethod
    def _decanonicalize_single(cls: Union[int, str, BlockHeaderAPI], db: Union[eth.abc.DatabaseAPI, eth.ChainGaps, eth.abc.BlockAPI], block_num: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI, eth.abc.BlockAPI], base_gaps: Union[eth.ChainGaps, int, eth.abc.BlockHeaderAPI]) -> Union[int, tuple[typing.Any]]:
        """
        A single block number was found to no longer be canonical. At doc-time,
        this only happens because it does not link up with a checkpoint header.
        So de-canonicalize this block number and insert a gap in the tracked
        chain gaps.
        """
        db.delete(SchemaV1.make_block_number_to_hash_lookup_key(block_num))
        new_gaps = reopen_gap(block_num, base_gaps)
        if new_gaps != base_gaps:
            db.set(SchemaV1.make_header_chain_gaps_lookup_key(), rlp.encode(new_gaps, sedes=chain_gaps))
        return new_gaps

    @classmethod
    def _persist_header_chain(cls: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI, eth.ChainGaps], db: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI, eth.ChainGaps], headers: Any, genesis_parent_hash: Union[eth.abc.DatabaseAPI, eth.ChainGaps, eth.abc.BlockHeaderAPI]) -> tuple[tuple]:
        headers_iterator = iter(headers)
        try:
            first_header = first(headers_iterator)
        except StopIteration:
            return ((), ())
        is_genesis = first_header.parent_hash == genesis_parent_hash
        if not is_genesis and (not cls._header_exists(db, first_header.parent_hash)):
            raise ParentNotFound(f'Cannot persist block header ({encode_hex(first_header.hash)}) with unknown parent ({encode_hex(first_header.parent_hash)})')
        if is_genesis:
            score = 0
        else:
            score = cls._get_score(db, first_header.parent_hash)
        curr_chain_head = first_header
        db.set(curr_chain_head.hash, rlp.encode(curr_chain_head))
        score = cls._set_hash_scores_to_db(db, curr_chain_head, score)
        base_gaps = cls._get_header_chain_gaps(db)
        gap_info = cls._update_header_chain_gaps(db, curr_chain_head, base_gaps)
        gaps = cls._handle_gap_change(db, gap_info, curr_chain_head, genesis_parent_hash)
        orig_headers_seq = concat([(first_header,), headers_iterator])
        for parent, child in sliding_window(2, orig_headers_seq):
            if parent.hash != child.parent_hash:
                raise ValidationError(f'Non-contiguous chain. Expected {encode_hex(child.hash)} to have {encode_hex(parent.hash)} as parent but was {encode_hex(child.parent_hash)}')
            curr_chain_head = child
            db.set(curr_chain_head.hash, rlp.encode(curr_chain_head))
            score = cls._set_hash_scores_to_db(db, curr_chain_head, score)
            gap_info = cls._update_header_chain_gaps(db, curr_chain_head, gaps)
            gaps = cls._handle_gap_change(db, gap_info, curr_chain_head, genesis_parent_hash)
        try:
            previous_canonical_head = cls._get_canonical_head_hash(db)
            head_score = cls._get_score(db, previous_canonical_head)
        except CanonicalHeadNotFound:
            return cls._set_as_canonical_chain_head(db, curr_chain_head, genesis_parent_hash)
        if score > head_score:
            return cls._set_as_canonical_chain_head(db, curr_chain_head, genesis_parent_hash)
        return ((), ())

    @classmethod
    def _handle_gap_change(cls: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI], db: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI], gap_info: Union[eth.db.chain_gaps.GapInfo, tuple[bool], eth.abc.BlockAPI], header: Union[eth.abc.BlockHeaderAPI, eth.db.chain_gaps.GapInfo, eth.abc.DatabaseAPI], genesis_parent_hash: Union[eth.abc.DatabaseAPI, typing.Iterable[eth.abc.BlockHeaderAPI], eth.abc.BlockHeaderAPI]) -> bool:
        gap_change, gaps = gap_info
        if gap_change not in GAP_WRITES:
            return gaps
        if gap_change in (GapChange.GapFill, GapChange.GapRightShrink):
            next_child_number = BlockNumber(header.block_number + 1)
            expected_child = cls._get_canonical_block_header_by_number(db, next_child_number)
            if header.hash != expected_child.parent_hash:
                checkpoints = cls._get_checkpoints(db)
                if expected_child.hash in checkpoints:
                    raise CheckpointsMustBeCanonical(f'Cannot make {header} canonical, because it is not the parent of declared checkpoint: {expected_child}')
                else:
                    gaps = cls._decanonicalize_single(db, expected_child.block_number, gaps)
        cls._canonicalize_header(db, header, genesis_parent_hash)
        return gaps

    @classmethod
    def _canonicalize_header(cls: Union[eth.abc.BlockHeaderAPI, eth.rlp.headers.BlockHeader, typing.Type], db: Union[eth.abc.BlockHeaderAPI, eth.rlp.headers.BlockHeader, typing.Type], header: Union[bytes, None], genesis_parent_hash: Union[bytes, None]) -> tuple[typing.Union[tuple,tuple[str],eth.rlp.headers.BlockHeader,dict,list]]:
        """
        Force this header to be canonical,
        and adjust its ancestors/descendants as necessary

        :raises CheckpointsMustBeCanonical: if trying to set a head that would
            de-canonicalize a checkpoint
        """
        new_canonical_headers = cast(Tuple[BlockHeaderAPI, ...], tuple(reversed(cls._find_new_ancestors(db, header, genesis_parent_hash))))
        old_canonical_headers = cls._find_headers_to_decanonicalize(db, [h.block_number for h in new_canonical_headers])
        checkpoints = cls._get_checkpoints(db)
        attempted_checkpoint_overrides = {old for old in old_canonical_headers if old.hash in checkpoints}
        if len(attempted_checkpoint_overrides):
            raise CheckpointsMustBeCanonical(f'Tried to switch chain away from checkpoint(s) {attempted_checkpoint_overrides!r} by inserting new canonical headers {new_canonical_headers}')
        for ancestor in new_canonical_headers:
            cls._add_block_number_to_hash_lookup(db, ancestor)
        if len(new_canonical_headers):
            cls._decanonicalize_descendant_orphans(db, new_canonical_headers[-1], checkpoints)
        return (new_canonical_headers, old_canonical_headers)

    @classmethod
    def _set_as_canonical_chain_head(cls: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI, eth.ChainGaps], db: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI, eth.ChainGaps], header: Union[eth.abc.BlockHeaderAPI, eth.rlp.headers.BlockHeader, eth.abc.DatabaseAPI], genesis_parent_hash: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI, eth.rlp.headers.BlockHeader]) -> tuple[typing.Union[eth.rlp.headers.BlockHeader,tuple,eth.abc.BlockHeaderAPI,tuple[typing.Union[eth.abc.BlockHeaderAPI,eth.rlp.headers.BlockHeader,eth.abc.DatabaseAPI]],eth.abc.HeaderChainAPI]]:
        """
        Sets the canonical chain HEAD to the block header as specified by the
        given block hash.

        :return: a tuple of the headers that are newly in the canonical chain, and the
            headers that are no longer in the canonical chain
        :raises CheckpointsMustBeCanonical: if trying to set a head that would
            de-canonicalize a checkpoint
        """
        try:
            current_canonical_head = cls._get_canonical_head_hash(db)
        except CanonicalHeadNotFound:
            current_canonical_head = None
        if current_canonical_head and header.parent_hash == current_canonical_head:
            new_canonical_headers = (header,)
            old_canonical_headers = ()
            cls._add_block_number_to_hash_lookup(db, header)
        else:
            new_canonical_headers, old_canonical_headers = cls._canonicalize_header(db, header, genesis_parent_hash)
        db.set(SchemaV1.make_canonical_head_hash_lookup_key(), header.hash)
        return (new_canonical_headers, old_canonical_headers)

    @classmethod
    def _get_checkpoints(cls: eth.abc.AtomicDatabaseAPI, db: id3c.db.session.DatabaseSession) -> Union[tuple, tuple[Hash32]]:
        concatenated_checkpoints = db.get(SchemaV1.make_checkpoint_headers_key())
        if concatenated_checkpoints is None:
            return ()
        else:
            return tuple((Hash32(concatenated_checkpoints[index:index + 32]) for index in range(0, len(concatenated_checkpoints), 32)))

    @classmethod
    @to_tuple
    def _find_headers_to_decanonicalize(cls: Union[eth.abc.DatabaseAPI, eth.ChainGaps], db: Union[eth.abc.DatabaseAPI, eth.ChainGaps], numbers_to_decanonicalize: str) -> typing.Generator:
        for block_number in numbers_to_decanonicalize:
            try:
                old_canonical_hash = cls._get_canonical_block_hash(db, block_number)
            except HeaderNotFound:
                continue
            else:
                yield cls._get_block_header_by_hash(db, old_canonical_hash)

    @classmethod
    @to_tuple
    def _find_new_ancestors(cls: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI, eth.ChainGaps], db: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI, eth.ChainGaps], header: Union[eth.abc.BlockHeaderAPI, eth.rlp.headers.BlockHeader, eth.abc.DatabaseAPI], genesis_parent_hash: eth.abc.DatabaseAPI) -> typing.Generator[typing.Union[eth.abc.BlockHeaderAPI,eth.rlp.headers.BlockHeader,eth.abc.DatabaseAPI]]:
        """
        Returns the chain leading up from the given header until (but not including)
        the first ancestor it has in common with our canonical chain.

        If D is the canonical head in the following chain, and F is the new header,
        then this function returns (F, E).

        A - B - C - D
                               E - F
        """
        h = header
        while True:
            try:
                orig = cls._get_canonical_block_header_by_number(db, h.block_number)
            except HeaderNotFound:
                pass
            else:
                if orig.hash == h.hash:
                    break
            yield h
            if h.parent_hash == genesis_parent_hash:
                break
            else:
                try:
                    h = cls._get_block_header_by_hash(db, h.parent_hash)
                except HeaderNotFound:
                    break

    @staticmethod
    def _add_block_number_to_hash_lookup(db: Union[eth.abc.DatabaseAPI, eth.abc.BlockHeaderAPI, eth.rlp.headers.BlockHeader], header: Union[eth.abc.BlockHeaderAPI, eth.rlp.headers.BlockHeader, eth.abc.DatabaseAPI]) -> None:
        """
        Sets a record in the database to allow looking up this header by its
        block number.
        """
        block_number_to_hash_key = SchemaV1.make_block_number_to_hash_lookup_key(header.block_number)
        db.set(block_number_to_hash_key, rlp.encode(header.hash, sedes=rlp.sedes.binary))

@functools.lru_cache(128)
def _decode_block_header(header_rlp: bytes):
    return rlp.decode(header_rlp, sedes=HeaderSedes)