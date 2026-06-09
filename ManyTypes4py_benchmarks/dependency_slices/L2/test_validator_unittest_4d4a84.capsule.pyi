from typing import Any

# === Internal dependency: eth2spec.test.context ===
def single_phase(fn) -> Any: ...
def spec_test(fn) -> Any: ...
def spec_state_test(fn) -> Any: ...
def always_bls(fn) -> Any: ...
def with_all_phases(fn) -> Any: ...
def with_phases(phases, other_phases = ...) -> Any: ...

# === Internal dependency: eth2spec.test.helpers.attestations ===
def build_attestation_data(spec, state, slot, index, beacon_block_root = ..., shard = ...) -> Any: ...
def get_valid_attestation(spec, state, slot = ..., index = ..., filter_participant_set = ..., beacon_block_root = ..., signed = ...) -> Any: ...

# === Internal dependency: eth2spec.test.helpers.block ===
def build_empty_block(spec, state, slot = ..., proposer_index = ...) -> Any: ...

# === Internal dependency: eth2spec.test.helpers.constants ===
PHASE0: SpecForkName

# === Internal dependency: eth2spec.test.helpers.deposits ===
def prepare_state_and_deposit(spec, state, validator_index, amount, pubkey = ..., privkey = ..., withdrawal_credentials = ..., signed = ...) -> Any: ...

# === Internal dependency: eth2spec.test.helpers.keys ===
privkeys: Any
pubkeys: Any

# === Internal dependency: eth2spec.test.helpers.state ===
def next_epoch(spec, state) -> Any: ...

# === Internal dependency: eth2spec.test.helpers.typing ===
SpecForkName: NewType

# === Internal dependency: eth2spec.utils ===
bls: Any

# === Internal dependency: eth2spec.utils.ssz.ssz_typing ===
# re-export: from remerkleable.bitfields import Bitlist