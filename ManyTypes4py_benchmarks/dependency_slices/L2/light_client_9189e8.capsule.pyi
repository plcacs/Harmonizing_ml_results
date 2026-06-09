from typing import Any

# === Internal dependency: eth2spec.test.helpers.constants ===
CAPELLA: SpecForkName
DENEB: SpecForkName
ELECTRA: SpecForkName

# === Internal dependency: eth2spec.test.helpers.fork_transition ===
def transition_across_forks(spec, state, to_slot, phases = ..., with_block = ..., sync_aggregate = ...) -> Any: ...

# === Internal dependency: eth2spec.test.helpers.forks ===
def is_post_capella(spec) -> Any: ...
def is_post_deneb(spec) -> Any: ...
def is_post_electra(spec) -> Any: ...

# === Internal dependency: eth2spec.test.helpers.sync_committee ===
def compute_aggregate_sync_committee_signature(spec, state, slot, participants, block_root = ..., domain_type = ...) -> Any: ...
def compute_committee_indices(state, committee = ...) -> Any: ...

# === Internal dependency: eth2spec.test.helpers.typing ===
SpecForkName: NewType