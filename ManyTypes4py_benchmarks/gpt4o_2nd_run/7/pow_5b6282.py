from collections import OrderedDict
from typing import Dict, Tuple, Optional
from eth_typing import Address, Hash32
from eth_utils import ValidationError, big_endian_to_int, encode_hex
from eth.abc import AtomicDatabaseAPI, BlockHeaderAPI, ConsensusAPI
from eth.consensus.ethash import get_dataset_full_size, hashimoto_light, mkcache
from eth.constants import EPOCH_LENGTH
from eth.validation import validate_length, validate_lte

cache_by_epoch: Dict[int, bytes] = OrderedDict()
CACHE_MAX_ITEMS: int = 10

def get_cache(block_number: int) -> bytes:
    epoch_index = block_number // EPOCH_LENGTH
    if epoch_index in cache_by_epoch:
        c = cache_by_epoch.pop(epoch_index)
        cache_by_epoch[epoch_index] = c
        return c
    block_number = epoch_index * EPOCH_LENGTH
    c = mkcache(block_number)
    cache_by_epoch[epoch_index] = c
    if len(cache_by_epoch) > CACHE_MAX_ITEMS:
        cache_by_epoch.popitem(last=False)
    return c

def check_pow(block_number: int, mining_hash: Hash32, mix_hash: Hash32, nonce: bytes, difficulty: int) -> None:
    validate_length(mix_hash, 32, title='Mix Hash')
    validate_length(mining_hash, 32, title='Mining Hash')
    validate_length(nonce, 8, title='POW Nonce')
    cache = get_cache(block_number)
    mining_output = hashimoto_light(get_dataset_full_size(block_number), cache, mining_hash, nonce)
    if mining_output['mix_digest'] != mix_hash:
        raise ValidationError(f'mix hash mismatch; expected: {encode_hex(mining_output["mix_digest"])} != actual: {encode_hex(mix_hash)}.\n    Mix hash calculated from block #{block_number},\n    mine hash: {encode_hex(mining_hash)},\n    nonce: {encode_hex(nonce)},\n    difficulty: {difficulty}')
    result = big_endian_to_int(mining_output['result'])
    validate_lte(result, 2 ** 256 // difficulty, title='POW Difficulty')

MAX_TEST_MINE_ATTEMPTS: int = 1000

def mine_pow_nonce(block_number: int, mining_hash: Hash32, difficulty: int) -> Tuple[bytes, Hash32]:
    cache = get_cache(block_number)
    for nonce in range(MAX_TEST_MINE_ATTEMPTS):
        mining_output = hashimoto_light(get_dataset_full_size(block_number), cache, mining_hash, nonce.to_bytes(8, 'big'))
        result = big_endian_to_int(mining_output['result'])
        result_cap = 2 ** 256 // difficulty
        if result <= result_cap:
            return (nonce.to_bytes(8, 'big'), mining_output['mix_digest'])
    raise Exception('Too many attempts at POW mining, giving up')

class PowConsensus(ConsensusAPI):
    """
    Modify a set of VMs to validate blocks via Proof of Work (POW)
    """

    def __init__(self, base_db: AtomicDatabaseAPI) -> None:
        pass

    def validate_seal(self, header: BlockHeaderAPI) -> None:
        """
        Validate the seal on the given header by checking the proof of work.
        """
        check_pow(header.block_number, header.mining_hash, header.mix_hash, header.nonce, header.difficulty)

    def validate_seal_extension(self, header: BlockHeaderAPI, parents: Iterable[BlockHeaderAPI]) -> None:
        pass

    @classmethod
    def get_fee_recipient(cls, header: BlockHeaderAPI) -> Address:
        """
        Return the ``coinbase`` of the passed ``header`` as the recipient for any
        rewards for the block.
        """
        return header.coinbase
