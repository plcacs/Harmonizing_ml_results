from typing import Any

# === Internal dependency: eth2spec.test.context ===
def single_phase(fn): ...
def spec_test(fn): ...
def spec_state_test(fn): ...
def always_bls(fn): ...
def with_all_phases(fn): ...
def with_phases(phases, other_phases=...): ...

# === Internal dependency: eth2spec.test.helpers.attestations ===
def build_attestation_data(spec, state, slot, index, beacon_block_root=..., shard=...): ...
def get_valid_attestation(spec, state, slot=..., index=..., filter_participant_set=..., beacon_block_root=..., signed=...): ...

# === Internal dependency: eth2spec.test.helpers.block ===
def build_empty_block(spec, state, slot=..., proposer_index=...): ...

# === Internal dependency: eth2spec.test.helpers.constants ===
PHASE0 = SpecForkName(...)

# === Internal dependency: eth2spec.test.helpers.deposits ===
def prepare_state_and_deposit(spec, state, validator_index, amount, pubkey=..., privkey=..., withdrawal_credentials=..., signed=...): ...

# === Internal dependency: eth2spec.test.helpers.keys ===
privkeys = [i + 1 for i in range(32 * 256)]
pubkeys = [bls.SkToPk(privkey) for privkey in privkeys]

# === Internal dependency: eth2spec.test.helpers.state ===
def next_epoch(spec, state): ...

# === Internal dependency: eth2spec.test.helpers.typing ===
SpecForkName = NewType(...)

# === Internal dependency: eth2spec.utils ===
bls: Any

# === Internal dependency: eth2spec.utils.ssz.ssz_typing ===
from remerkleable.bitfields import Bitlist