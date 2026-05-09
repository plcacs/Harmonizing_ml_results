from eth2spec.lightclient import (
    Spec,
    State,
    Block,
    LightClientUpdate,
    SyncAggregate,
    LightClientHeader,
    LightClientBootstrap,
    LightClientFinalityUpdate,
    LightClientStore,
)
from eth2spec.test.helpers.forks import ForkPhases

def latest_finalized_root_gindex(spec: Spec) -> int: ...

def latest_current_sync_committee_gindex(spec: Spec) -> int: ...

def latest_next_sync_committee_gindex(spec: Spec) -> int: ...

def latest_normalize_merkle_branch(spec: Spec, branch: bytes, gindex: int) -> bytes: ...

def compute_start_slot_at_sync_committee_period(spec: Spec, sync_committee_period: int) -> int: ...

def compute_start_slot_at_next_sync_committee_period(spec: Spec, state: State) -> int: ...

def get_sync_aggregate(
    spec: Spec,
    state: State,
    num_participants: int | None = None,
    signature_slot: int | None = None,
    phases: ForkPhases | None = None,
) -> tuple[SyncAggregate, int]: ...

def create_update(
    spec: Spec,
    attested_state: State,
    attested_block: Block,
    finalized_block: Block,
    with_next: bool,
    with_finality: bool,
    participation_rate: float,
    signature_slot: int | None = None,
) -> LightClientUpdate: ...

def needs_upgrade_to_capella(spec: Spec, new_spec: Spec) -> bool: ...

def needs_upgrade_to_deneb(spec: Spec, new_spec: Spec) -> bool: ...

def needs_upgrade_to_electra(spec: Spec, new_spec: Spec) -> bool: ...

def check_merkle_branch_equal(
    spec: Spec,
    new_spec: Spec,
    data: bytes,
    upgraded: bytes,
    gindex: int,
) -> None: ...

def check_lc_header_equal(
    spec: Spec,
    new_spec: Spec,
    data: LightClientHeader,
    upgraded: LightClientHeader,
) -> None: ...

def upgrade_lc_header_to_new_spec(
    spec: Spec,
    new_spec: Spec,
    data: LightClientHeader,
    phases: ForkPhases,
) -> LightClientHeader: ...

def check_lc_bootstrap_equal(
    spec: Spec,
    new_spec: Spec,
    data: LightClientBootstrap,
    upgraded: LightClientBootstrap,
) -> None: ...

def upgrade_lc_bootstrap_to_new_spec(
    spec: Spec,
    new_spec: Spec,
    data: LightClientBootstrap,
    phases: ForkPhases,
) -> LightClientBootstrap: ...

def check_lc_update_equal(
    spec: Spec,
    new_spec: Spec,
    data: LightClientUpdate,
    upgraded: LightClientUpdate,
) -> None: ...

def upgrade_lc_update_to_new_spec(
    spec: Spec,
    new_spec: Spec,
    data: LightClientUpdate,
    phases: ForkPhases,
) -> LightClientUpdate: ...

def check_lc_finality_update_equal(
    spec: Spec,
    new_spec: Spec,
    data: LightClientFinalityUpdate,
    upgraded: LightClientFinalityUpdate,
) -> None: ...

def upgrade_lc_finality_update_to_new_spec(
    spec: Spec,
    new_spec: Spec,
    data: LightClientFinalityUpdate,
    phases: ForkPhases,
) -> LightClientFinalityUpdate: ...

def check_lc_store_equal(
    spec: Spec,
    new_spec: Spec,
    data: LightClientStore,
    upgraded: LightClientStore,
) -> None: ...

def upgrade_lc_store_to_new_spec(
    spec: Spec,
    new_spec: Spec,
    data: LightClientStore,
    phases: ForkPhases,
) -> LightClientStore: ...