from eth_keys import keys
from eth_utils import ValidationError, decode_hex, to_tuple
from eth.chains.base import MiningChain
from eth.chains.goerli import GOERLI_GENESIS_HEADER
from eth.consensus.clique import NONCE_AUTH, NONCE_DROP, CliqueApplier, CliqueConsensus, CliqueConsensusContext, VoteAction
from eth.consensus.clique._utils import get_block_signer, sign_block_header
from eth.consensus.clique.constants import SIGNATURE_LENGTH, VANITY_LENGTH
from eth.constants import ZERO_ADDRESS
from eth.rlp.headers import BlockHeader
from eth.tools.factories.keys import PublicKeyFactory
from eth.tools.factories.transaction import new_transaction
from eth.vm.forks.istanbul import IstanbulVM
from eth.vm.forks.petersburg import PetersburgVM
from typing import List, Tuple

ALICE_PK: keys.PrivateKey = keys.PrivateKey(decode_hex('0x45a915e4d060149eb4365960e6a7a45f334393093061116b197e3240065ff2d8'))
ALICE: bytes = ALICE_PK.public_key.to_canonical_address()
ALICE_INITIAL_BALANCE: int = 21000000
BOB_PK: keys.PrivateKey = keys.PrivateKey(decode_hex('0x15a915e4d060149eb4365960e6a7a45f334393093061116b197e3240065ff2d8'))
BOB: bytes = BOB_PK.public_key.to_canonical_address()
BOB_INITIAL_BALANCE: int = 21000000
RON_PK: keys.PrivateKey = keys.PrivateKey(decode_hex('0x25a915e4d060149eb4365960e6a7a45f334393093061116b197e3240065ff2d8'))
RON: bytes = RON_PK.public_key.to_canonical_address()
PARAGON_GENESIS_HEADER: BlockHeader = sign_block_header(GOERLI_GENESIS_HEADER.copy(extra_data=VANITY_LENGTH * b'0' + ALICE + SIGNATURE_LENGTH * b'0', state_root=b'\xce]\x98X"Xm\xaf\xab\xc7\xf8\x91\xc0{\xfc\x0eNKf9uu\xd8\xe2\x0e\x81@g68\x1a\xa3'), ALICE_PK)
PARAGON_GENESIS_PARAMS: dict = {'coinbase': PARAGON_GENESIS_HEADER.coinbase, 'difficulty': PARAGON_GENESIS_HEADER.difficulty, 'timestamp': PARAGON_GENESIS_HEADER.timestamp, 'gas_limit': PARAGON_GENESIS_HEADER.gas_limit, 'extra_data': PARAGON_GENESIS_HEADER.extra_data, 'nonce': PARAGON_GENESIS_HEADER.nonce}
PARAGON_GENESIS_STATE: dict = {ALICE: {'balance': ALICE_INITIAL_BALANCE, 'code': b'', 'nonce': 0, 'storage': {}}, BOB: {'balance': BOB_INITIAL_BALANCE, 'code': b'', 'nonce': 0, 'storage': {}}}
GOERLI_GENESIS_HASH: bytes = decode_hex('0xbf7e331f7f7c1dd2e05159666b3bf8bc7a8a3a9eb1d518969eab529dd9b88c1a')
GOERLI_GENESIS_ALLOWED_SIGNER: bytes = decode_hex('0xe0a2bd4258d2768837baa26a28fe71dc079f84c7')
DAPOWERPLAY_SIGNER: bytes = decode_hex('0xa8e8f14732658e4b51e8711931053a8a69baf2b1')
GOERLI_HEADER_ONE: BlockHeader = BlockHeader(difficulty=2, block_number=1, gas_limit=10475521, timestamp=1548947453, coinbase=decode_hex('0x0000000000000000000000000000000000000000'), parent_hash=decode_hex('0xbf7e331f7f7c1dd2e05159666b3bf8bc7a8a3a9eb1d518969eab529dd9b88c1a'), uncles_hash=decode_hex('0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347'), state_root=decode_hex('0x5d6cded585e73c4e322c30c2f782a336316f17dd85a4863b9d838d2d4b8b3008'), transaction_root=decode_hex('0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421'), receipt_root=decode_hex('0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421'), bloom=0, gas_used=0, extra_data=decode_hex('0x506172697479205465636820417574686f7269747900000000000000000000002bbf886181970654ed46e3fae0ded41ee53fec702c47431988a7ae80e6576f3552684f069af80ba11d36327aaf846d470526e4a1c461601b2fd4ebdcdc2b734a01'), mix_hash=decode_hex('0x0000000000000000000000000000000000000000000000000000000000000000'), nonce=decode_hex('0x0000000000000000'))

def has_vote_to(subject: bytes, votes: List) -> bool:
    return any((vote.subject == subject for vote in votes))

def has_vote_from(signer: bytes, votes: List) -> bool:
    return any((vote.signer == signer for vote in votes))

def make_next_header(chain: MiningChain, previous_header: BlockHeader, signer_private_key: keys.PrivateKey, coinbase: bytes = ZERO_ADDRESS, nonce: int = NONCE_DROP, difficulty: int = 2) -> BlockHeader:
    unsigned_header = chain.create_header_from_parent(previous_header, coinbase=coinbase, nonce=nonce, timestamp=previous_header.timestamp + 1, gas_limit=previous_header.gas_limit, difficulty=difficulty, extra_data=VANITY_LENGTH * b'0' + SIGNATURE_LENGTH * b'0')
    return sign_block_header(unsigned_header, signer_private_key)

@to_tuple
def alice_nominates_bob_and_ron_then_they_kick_her(chain: MiningChain) -> Tuple[BlockHeader]:
    header = PARAGON_GENESIS_HEADER
    header = make_next_header(chain, header, ALICE_PK)
    yield header
    header = make_next_header(chain, header, ALICE_PK, BOB, NONCE_AUTH)
    yield header
    header = make_next_header(chain, header, BOB_PK, RON, NONCE_AUTH)
    yield header
    header = make_next_header(chain, header, ALICE_PK, RON, NONCE_AUTH)
    yield header
    header = make_next_header(chain, header, BOB_PK, ALICE, NONCE_DROP)
    yield header
    header = make_next_header(chain, header, RON_PK, ALICE, NONCE_DROP)
    yield header
    header = make_next_header(chain, header, BOB_PK)
    yield header

def validate_seal_and_get_snapshot(clique: CliqueConsensus, header: BlockHeader) -> Snapshot:
    clique.validate_seal_extension(header, ())
    return clique.get_snapshot(header)

def get_clique(chain: MiningChain, header: BlockHeader = None) -> CliqueConsensus:
    if header:
        vm = chain.get_vm(header)
    else:
        vm = chain.get_vm()
    clique = vm._consensus
    assert isinstance(clique, CliqueConsensus)
    return clique
