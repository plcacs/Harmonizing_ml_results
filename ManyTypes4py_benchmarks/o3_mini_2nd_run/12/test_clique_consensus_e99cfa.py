#!/usr/bin/env python3
from typing import Any, Tuple, Sequence, Optional

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

PARAGON_GENESIS_HEADER: BlockHeader = sign_block_header(
    GOERLI_GENESIS_HEADER.copy(
        extra_data=VANITY_LENGTH * b'0' + ALICE + SIGNATURE_LENGTH * b'0',
        state_root=b'\xce]\x98X"Xm\xaf\xab\xc7\xf8\x91\xc0{\xfc\x0eNKf9uu\xd8\xe2\x0e\x81@g68\x1a\xa3'
    ),
    ALICE_PK
)
PARAGON_GENESIS_PARAMS: dict = {
    'coinbase': PARAGON_GENESIS_HEADER.coinbase,
    'difficulty': PARAGON_GENESIS_HEADER.difficulty,
    'timestamp': PARAGON_GENESIS_HEADER.timestamp,
    'gas_limit': PARAGON_GENESIS_HEADER.gas_limit,
    'extra_data': PARAGON_GENESIS_HEADER.extra_data,
    'nonce': PARAGON_GENESIS_HEADER.nonce
}
PARAGON_GENESIS_STATE: dict = {
    ALICE: {'balance': ALICE_INITIAL_BALANCE, 'code': b'', 'nonce': 0, 'storage': {}},
    BOB: {'balance': BOB_INITIAL_BALANCE, 'code': b'', 'nonce': 0, 'storage': {}}
}
GOERLI_GENESIS_HASH: bytes = decode_hex('0xbf7e331f7f7c1dd2e05159666b3bf8bc7a8a3a9eb1d518969eab529dd9b88c1a')
GOERLI_GENESIS_ALLOWED_SIGNER: bytes = decode_hex('0xe0a2bd4258d2768837baa26a28fe71dc079f84c7')
DAPOWERPLAY_SIGNER: bytes = decode_hex('0xa8e8f14732658e4b51e8711931053a8a69baf2b1')
GOERLI_HEADER_ONE: BlockHeader = BlockHeader(
    difficulty=2,
    block_number=1,
    gas_limit=10475521,
    timestamp=1548947453,
    coinbase=decode_hex('0x0000000000000000000000000000000000000000'),
    parent_hash=decode_hex('0xbf7e331f7f7c1dd2e05159666b3bf8bc7a8a3a9eb1d518969eab529dd9b88c1a'),
    uncles_hash=decode_hex('0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347'),
    state_root=decode_hex('0x5d6cded585e73c4e322c30c2f782a336316f17dd85a4863b9d838d2d4b8b3008'),
    transaction_root=decode_hex('0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421'),
    receipt_root=decode_hex('0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421'),
    bloom=0,
    gas_used=0,
    extra_data=decode_hex(
        '0x506172697479205465636820417574686f7269747900000000000000000000002bbf886181970654ed46e3fae0ded41ee53fec702c47431988a7ae80e6576f3552684f069af80ba11d36327aaf846d470526e4a1c461601b2fd4ebdcdc2b734a01'
    ),
    mix_hash=decode_hex('0x0000000000000000000000000000000000000000000000000000000000000000'),
    nonce=decode_hex('0x0000000000000000')
)


def has_vote_to(subject: bytes, votes: Sequence[Any]) -> bool:
    return any((vote.subject == subject for vote in votes))


def has_vote_from(signer: bytes, votes: Sequence[Any]) -> bool:
    return any((vote.signer == signer for vote in votes))


def make_next_header(chain: MiningChain,
                     previous_header: BlockHeader,
                     signer_private_key: keys.PrivateKey,
                     coinbase: bytes = ZERO_ADDRESS,
                     nonce: bytes = NONCE_DROP,
                     difficulty: int = 2) -> BlockHeader:
    unsigned_header: BlockHeader = chain.create_header_from_parent(
        previous_header,
        coinbase=coinbase,
        nonce=nonce,
        timestamp=previous_header.timestamp + 1,
        gas_limit=previous_header.gas_limit,
        difficulty=difficulty,
        extra_data=VANITY_LENGTH * b'0' + SIGNATURE_LENGTH * b'0'
    )
    return sign_block_header(unsigned_header, signer_private_key)


@to_tuple
def alice_nominates_bob_and_ron_then_they_kick_her(chain: MiningChain) -> Tuple[BlockHeader, ...]:
    header: BlockHeader = PARAGON_GENESIS_HEADER
    header = make_next_header(chain, header, ALICE_PK)
    yield header
    header = make_next_header(chain, header, ALICE_PK, BOB, NONCE_AUTH)
    yield header
    header = make_next_header(chain, header, BOB_PK, RON, NONCE_AUTH)
    yield header
    header = make_next_header(chain, header, ALICE_PK, RON, NONCE_AUTH)
    yield header
    header = make_next_header(chain, header, ALICE_PK, ALICE, NONCE_DROP)
    yield header
    header = make_next_header(chain, header, RON_PK, ALICE, NONCE_DROP)
    yield header
    header = make_next_header(chain, header, BOB_PK)
    yield header


def validate_seal_and_get_snapshot(clique: Any, header: BlockHeader) -> Any:
    clique.validate_seal_extension(header, ())
    return clique.get_snapshot(header)


@pytest.fixture
def paragon_chain(base_db: Any) -> MiningChain:
    vms: Tuple[Tuple[int, Any], ...] = ((0, PetersburgVM), (2, IstanbulVM))
    clique_vms: Any = CliqueApplier().amend_vm_configuration(vms)
    chain: MiningChain = MiningChain.configure(
        vm_configuration=clique_vms,
        consensus_context_class=CliqueConsensusContext,
        chain_id=5
    ).from_genesis(base_db, PARAGON_GENESIS_PARAMS, PARAGON_GENESIS_STATE)
    return chain


def get_clique(chain: MiningChain, header: Optional[BlockHeader] = None) -> CliqueConsensus:
    if header:
        vm = chain.get_vm(header)
    else:
        vm = chain.get_vm()
    clique: Any = vm._consensus
    assert isinstance(clique, CliqueConsensus)
    return clique


def test_can_retrieve_root_snapshot(paragon_chain: MiningChain) -> None:
    head: BlockHeader = paragon_chain.get_canonical_head()
    snapshot: Any = get_clique(paragon_chain, head).get_snapshot(head)
    assert snapshot.get_sorted_signers() == [ALICE]


def test_raises_unknown_ancestor_error(paragon_chain: MiningChain) -> None:
    head: BlockHeader = paragon_chain.get_canonical_head()
    next_header: BlockHeader = make_next_header(paragon_chain, head, ALICE_PK, RON, NONCE_AUTH)
    clique: Any = get_clique(paragon_chain, head)
    with pytest.raises(ValidationError, match='Unknown ancestor'):
        clique.get_snapshot(next_header)


def test_validate_chain_works_across_forks(paragon_chain: MiningChain) -> None:
    voting_chain: Tuple[BlockHeader, ...] = alice_nominates_bob_and_ron_then_they_kick_her(paragon_chain)
    paragon_chain.validate_chain_extension((PARAGON_GENESIS_HEADER,) + voting_chain)


def test_import_block(paragon_chain: MiningChain) -> None:
    vm: Any = paragon_chain.get_vm()
    tx: Any = new_transaction(vm, ALICE, BOB, 10, ALICE_PK, gas_price=10)
    assert vm.state.get_balance(ALICE) == ALICE_INITIAL_BALANCE
    assert vm.state.get_balance(BOB) == BOB_INITIAL_BALANCE
    assert vm.state.get_balance(vm.get_block().header.coinbase) == 0
    signed_header: BlockHeader = sign_block_header(
        vm.get_block().header.copy(
            extra_data=VANITY_LENGTH * b'0' + SIGNATURE_LENGTH * b'0',
            state_root=b'\x02g\xd5{\xf9\x9f\x9e\xab)\x06\x1eY\x9a\xb7W\x95\xfd\xae\x9a:\x83m%\xbb\xcc\xe1\xca\xe3\x85d\xa7\x05',
            transaction_root=b'\xd1\t\xc4\x150\x9f\xb0\xb4H{\xfd$?Q\x16\x90\xac\xb2L[f\x98\xdd\xc6*\xf7\n\x84f\xafg\xb3',
            nonce=NONCE_DROP,
            gas_used=21000,
            difficulty=2,
            receipt_root=b'\x05k#\xfb\xbaH\x06\x96\xb6_\xe5\xa5\x9b\x8f!H\xa1)\x91\x03\xc4\xf5}\xf89#:\xf2\xcfL\xa2\xd2'
        ),
        ALICE_PK
    )
    block: Any = vm.get_block_class()(header=signed_header, transactions=[tx])
    assert get_block_signer(block.header) == ALICE
    paragon_chain.import_block(block)
    assert paragon_chain.get_vm().state.get_balance(ALICE) == 20999990
    assert paragon_chain.get_vm().state.get_balance(BOB) == 21000010
    assert paragon_chain.get_vm().state.get_balance(vm.get_block().header.coinbase) == 0


def test_reapplies_headers_without_snapshots(paragon_chain: MiningChain) -> None:
    voting_chain: Tuple[BlockHeader, ...] = alice_nominates_bob_and_ron_then_they_kick_her(paragon_chain)
    for i in range(5):
        paragon_chain.chaindb.persist_header(voting_chain[i])
    clique: Any = get_clique(paragon_chain)
    snapshot: Any = validate_seal_and_get_snapshot(clique, voting_chain[5])
    assert snapshot.signers == {BOB, RON}


def test_can_persist_and_restore_snapshot_from_db(paragon_chain: MiningChain) -> None:
    clique: Any = get_clique(paragon_chain)
    snapshot: Any = validate_seal_and_get_snapshot(clique, PARAGON_GENESIS_HEADER)
    clique._snapshot_manager.persist_snapshot(snapshot)
    revived: Any = clique._snapshot_manager.get_snapshot_from_db(PARAGON_GENESIS_HEADER.hash)
    assert snapshot == revived


def test_revert_previous_nominate(paragon_chain: MiningChain) -> None:
    head: BlockHeader = paragon_chain.get_canonical_head()
    clique: Any = get_clique(paragon_chain)
    snapshot: Any = validate_seal_and_get_snapshot(clique, head)
    assert len(snapshot.tallies) == 0
    alice_votes_bob: BlockHeader = make_next_header(paragon_chain, head, ALICE_PK, coinbase=BOB, nonce=NONCE_AUTH)
    snapshot = validate_seal_and_get_snapshot(clique, alice_votes_bob)
    assert snapshot.get_sorted_signers() == [ALICE, BOB]
    alice_votes_ron: BlockHeader = make_next_header(paragon_chain, alice_votes_bob, ALICE_PK, coinbase=RON, nonce=NONCE_AUTH)
    snapshot = validate_seal_and_get_snapshot(clique, alice_votes_ron)
    assert snapshot.get_sorted_signers() == [ALICE, BOB]
    assert snapshot.tallies[RON].action == VoteAction.NOMINATE
    assert snapshot.tallies[RON].votes == 1
    alice_votes_against_ron: BlockHeader = make_next_header(paragon_chain, alice_votes_ron, ALICE_PK, coinbase=RON, nonce=NONCE_DROP, difficulty=1)
    snapshot = validate_seal_and_get_snapshot(clique, alice_votes_against_ron)
    assert snapshot.get_sorted_signers() == [ALICE, BOB]
    assert RON not in snapshot.tallies


def test_revert_previous_kick(paragon_chain: MiningChain) -> None:
    head: BlockHeader = paragon_chain.get_canonical_head()
    clique: Any = get_clique(paragon_chain)
    snapshot: Any = validate_seal_and_get_snapshot(clique, head)
    assert len(snapshot.tallies) == 0
    alice_votes_bob: BlockHeader = make_next_header(paragon_chain, head, ALICE_PK, coinbase=BOB, nonce=NONCE_AUTH)
    snapshot = validate_seal_and_get_snapshot(clique, alice_votes_bob)
    assert snapshot.get_sorted_signers() == [ALICE, BOB]
    alice_kicks_bob: BlockHeader = make_next_header(paragon_chain, alice_votes_bob, ALICE_PK, coinbase=BOB, nonce=NONCE_DROP)
    snapshot = validate_seal_and_get_snapshot(clique, alice_kicks_bob)
    assert snapshot.get_sorted_signers() == [ALICE, BOB]
    assert snapshot.tallies[BOB].action == VoteAction.KICK
    assert snapshot.tallies[BOB].votes == 1
    alice_votes_bob = make_next_header(paragon_chain, alice_kicks_bob, ALICE_PK, coinbase=BOB, nonce=NONCE_AUTH, difficulty=1)
    snapshot = validate_seal_and_get_snapshot(clique, alice_votes_bob)
    assert snapshot.get_sorted_signers() == [ALICE, BOB]
    assert BOB not in snapshot.tallies


def test_does_not_count_multiple_kicks(paragon_chain: MiningChain) -> None:
    head: BlockHeader = paragon_chain.get_canonical_head()
    clique: Any = get_clique(paragon_chain)
    snapshot: Any = validate_seal_and_get_snapshot(clique, head)
    assert len(snapshot.tallies) == 0
    alice_votes_bob: BlockHeader = make_next_header(paragon_chain, head, ALICE_PK, coinbase=BOB, nonce=NONCE_AUTH)
    snapshot = validate_seal_and_get_snapshot(clique, alice_votes_bob)
    assert snapshot.get_sorted_signers() == [ALICE, BOB]
    alice_kicks_bob: BlockHeader = make_next_header(paragon_chain, alice_votes_bob, ALICE_PK, coinbase=BOB, nonce=NONCE_DROP)
    snapshot = validate_seal_and_get_snapshot(clique, alice_kicks_bob)
    assert snapshot.get_sorted_signers() == [ALICE, BOB]
    assert snapshot.tallies[BOB].action == VoteAction.KICK
    assert snapshot.tallies[BOB].votes == 1
    alice_kicks_bob_again: BlockHeader = make_next_header(paragon_chain, alice_kicks_bob, ALICE_PK, coinbase=BOB, nonce=NONCE_DROP, difficulty=1)
    snapshot = validate_seal_and_get_snapshot(clique, alice_kicks_bob_again)
    assert snapshot.get_sorted_signers() == [ALICE, BOB]
    assert snapshot.tallies[BOB].action == VoteAction.KICK
    assert snapshot.tallies[BOB].votes == 1


def test_does_not_count_multiple_nominates(paragon_chain: MiningChain) -> None:
    head: BlockHeader = paragon_chain.get_canonical_head()
    clique: Any = get_clique(paragon_chain)
    snapshot: Any = validate_seal_and_get_snapshot(clique, head)
    assert len(snapshot.tallies) == 0
    alice_votes_bob: BlockHeader = make_next_header(paragon_chain, head, ALICE_PK, coinbase=BOB, nonce=NONCE_AUTH)
    snapshot = validate_seal_and_get_snapshot(clique, alice_votes_bob)
    assert snapshot.get_sorted_signers() == [ALICE, BOB]
    alice_votes_ron: BlockHeader = make_next_header(paragon_chain, alice_votes_bob, ALICE_PK, coinbase=RON, nonce=NONCE_AUTH)
    snapshot = validate_seal_and_get_snapshot(clique, alice_votes_ron)
    assert snapshot.get_sorted_signers() == [ALICE, BOB]
    assert snapshot.tallies[RON].action == VoteAction.NOMINATE
    assert snapshot.tallies[RON].votes == 1
    alice_votes_ron_again: BlockHeader = make_next_header(paragon_chain, alice_votes_ron, ALICE_PK, coinbase=RON, nonce=NONCE_AUTH, difficulty=1)
    snapshot = validate_seal_and_get_snapshot(clique, alice_votes_ron_again)
    assert snapshot.get_sorted_signers() == [ALICE, BOB]
    assert snapshot.tallies[RON].action == VoteAction.NOMINATE
    assert snapshot.tallies[RON].votes == 1


def test_alice_votes_in_bob_and_ron_then_gets_kicked(paragon_chain: MiningChain) -> None:
    clique: Any = get_clique(paragon_chain)
    voting_chain: Tuple[BlockHeader, ...] = alice_nominates_bob_and_ron_then_they_kick_her(paragon_chain)
    snapshot: Any = validate_seal_and_get_snapshot(clique, voting_chain[0])
    assert snapshot.signers == {ALICE}
    snapshot = validate_seal_and_get_snapshot(clique, voting_chain[1])
    assert snapshot.signers == {ALICE, BOB}
    snapshot = validate_seal_and_get_snapshot(clique, voting_chain[2])
    assert snapshot.tallies[RON].action == VoteAction.NOMINATE
    assert snapshot.tallies[RON].votes == 1
    assert snapshot.signers == {ALICE, BOB}
    snapshot = validate_seal_and_get_snapshot(clique, voting_chain[3])
    assert snapshot.signers == {ALICE, BOB, RON}
    assert RON not in snapshot.tallies
    snapshot = validate_seal_and_get_snapshot(clique, voting_chain[4])
    assert snapshot.tallies[ALICE].action == VoteAction.KICK
    assert snapshot.tallies[ALICE].votes == 1
    assert snapshot.signers == {ALICE, BOB, RON}
    snapshot = validate_seal_and_get_snapshot(clique, voting_chain[5])
    assert snapshot.signers == {BOB, RON}
    assert ALICE not in snapshot.tallies


def test_removes_all_pending_votes_after_nomination(paragon_chain: MiningChain) -> None:
    clique: Any = get_clique(paragon_chain)
    voting_chain: Tuple[BlockHeader, ...] = alice_nominates_bob_and_ron_then_they_kick_her(paragon_chain)
    snapshot: Any = None
    for i in range(3):
        snapshot = validate_seal_and_get_snapshot(clique, voting_chain[i])
    assert snapshot.signers == {ALICE, BOB}
    assert has_vote_to(RON, snapshot.votes)
    assert has_vote_from(BOB, snapshot.votes)
    assert not has_vote_from(ALICE, snapshot.votes)
    snapshot = validate_seal_and_get_snapshot(clique, voting_chain[3])
    assert snapshot.signers == {ALICE, BOB, RON}
    assert RON not in snapshot.tallies
    assert not has_vote_to(RON, snapshot.votes)
    assert not has_vote_from(BOB, snapshot.votes)
    assert not has_vote_from(ALICE, snapshot.votes)


def test_removes_all_pending_votes_after_kick(paragon_chain: MiningChain) -> None:
    clique: Any = get_clique(paragon_chain)
    ALICE_FRIEND: bytes = PublicKeyFactory().to_canonical_address()
    voting_chain: Tuple[BlockHeader, ...] = alice_nominates_bob_and_ron_then_they_kick_her(paragon_chain)
    snapshot: Any = None
    for i in range(4):
        snapshot = validate_seal_and_get_snapshot(clique, voting_chain[i])
    assert snapshot.signers == {ALICE, BOB, RON}
    alices_nominates_friend: BlockHeader = make_next_header(paragon_chain, voting_chain[3], ALICE_PK, coinbase=ALICE_FRIEND, nonce=NONCE_AUTH, difficulty=1)
    snapshot = validate_seal_and_get_snapshot(clique, alices_nominates_friend)
    assert ALICE_FRIEND in snapshot.tallies
    assert has_vote_to(ALICE_FRIEND, snapshot.votes)
    assert has_vote_from(ALICE, snapshot.votes)
    bob_kicks_alice: BlockHeader = make_next_header(paragon_chain, alices_nominates_friend, BOB_PK, coinbase=ALICE, nonce=NONCE_DROP, difficulty=1)
    snapshot = validate_seal_and_get_snapshot(clique, bob_kicks_alice)
    ron_kicks_alice: BlockHeader = make_next_header(paragon_chain, bob_kicks_alice, RON_PK, coinbase=ALICE, nonce=NONCE_DROP, difficulty=1)
    snapshot = validate_seal_and_get_snapshot(clique, ron_kicks_alice)
    assert snapshot.signers == {BOB, RON}
    assert not has_vote_from(ALICE, snapshot.votes)
    assert not has_vote_to(ALICE_FRIEND, snapshot.votes)
    assert ALICE_FRIEND not in snapshot.tallies
