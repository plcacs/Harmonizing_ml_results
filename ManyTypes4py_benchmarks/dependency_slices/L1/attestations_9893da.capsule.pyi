from typing import Any

# === Internal dependency: eth2spec.test.context ===
def expect_assertion_error(fn): ...

# === Internal dependency: eth2spec.test.helpers.block ===
def build_empty_block_for_next_slot(spec, state, proposer_index=...): ...

# === Internal dependency: eth2spec.test.helpers.forks ===
def is_post_altair(spec): ...
def is_post_deneb(spec): ...
def is_post_electra(spec): ...

# === Internal dependency: eth2spec.test.helpers.keys ===
privkeys = [i + 1 for i in range(32 * 256)]

# === Internal dependency: eth2spec.test.helpers.state ===
def next_slot(spec, state): ...
def next_epoch(spec, state): ...
def state_transition_and_sign_block(spec, state, block, expect_fail=...): ...

# === Internal dependency: eth2spec.utils ===
bls: Any

# === Internal dependency: eth2spec.utils.ssz.ssz_typing ===
from remerkleable.bitfields import Bitlist

# === Extension dependency: lru ===
# Used symbols: LRU