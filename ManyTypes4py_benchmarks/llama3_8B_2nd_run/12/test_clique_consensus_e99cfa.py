import pytest
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

@to_tuple
def alice_nominates_bob_and_ron_then_they_kick_her(chain: MiningChain) -> tuple:
    ...

def has_vote_to(subject: bytes, votes: list) -> bool:
    return any((vote.subject == subject for vote in votes))

def has_vote_from(signer: bytes, votes: list) -> bool:
    return any((vote.signer == signer for vote in votes))

@to_tuple
def make_next_header(chain: MiningChain, previous_header: BlockHeader, signer_private_key: keys.PrivateKey, coinbase: bytes = ZERO_ADDRESS, nonce: int = NONCE_DROP, difficulty: int = 2) -> BlockHeader:
    ...

@to_tuple
def test_can_retrieve_root_snapshot(paragon_chain: MiningChain) -> None:
    ...

def test_raises_unknown_ancestor_error(paragon_chain: MiningChain) -> None:
    ...

def test_validate_chain_works_across_forks(paragon_chain: MiningChain) -> None:
    ...

def test_import_block(paragon_chain: MiningChain) -> None:
    ...

def test_reapplies_headers_without_snapshots(paragon_chain: MiningChain) -> None:
    ...

def test_can_persist_and_restore_snapshot_from_db(paragon_chain: MiningChain) -> None:
    ...

def test_revert_previous_nominate(paragon_chain: MiningChain) -> None:
    ...

def test_revert_previous_kick(paragon_chain: MiningChain) -> None:
    ...

def test_does_not_count_multiple_kicks(paragon_chain: MiningChain) -> None:
    ...

def test_does_not_count_multiple_nominates(paragon_chain: MiningChain) -> None:
    ...

def test_alice_votes_in_bob_and_ron_then_gets_kicked(paragon_chain: MiningChain) -> None:
    ...

def test_removes_all_pending_votes_after_nomination(paragon_chain: MiningChain) -> None:
    ...

def test_removes_all_pending_votes_after_kick(paragon_chain: MiningChain) -> None:
    ...
