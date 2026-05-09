from math import floor
from typing import Any, Optional, Tuple, Union
from eth2spec.types import (
    LightClientUpdate,
    SyncAggregate,
    LightClientHeader,
    LightClientBootstrap,
    LightClientFinalityUpdate,
    LightClientStore,
)

CAPELLA: str
DENEB: str
ELECTRA: str

def latest_finalized_root_gindex(spec: Any) -> int:
    ...

def latest_current_sync_committee_gindex(spec: Any) -> int:
    ...

def latest_next_sync_committee_gindex(spec: Any) -> int:
    ...

def latest_normalize_merkle_branch(spec: Any, branch: Any, gindex: int) -> Any:
    ...

def compute_start_slot_at_sync_committee_period(spec: Any, sync_committee_period: int) -> int:
    ...

def compute_start_slot_at_next_sync_committee_period(spec: Any, state: Any) -> int:
    ...

def get_sync_aggregate(
    spec: Any,
    state: Any,
    num_participants: Optional[int] = None,
    signature_slot: Optional[int] = None,
    phases: Optional[Any] = None,
) -> Tuple[SyncAggregate, int]:
    ...

def create_update(
    spec: Any,
    attested_state: Any,
    attested_block: Any,
    finalized_block: Any,
    with_next: bool,
    with_finality: bool,
    participation_rate: float,
    signature_slot: Optional[int] = None,
) -> LightClientUpdate:
    ...

def needs_upgrade_to_capella(spec: Any, new_spec: Any) -> bool:
    ...

def needs_upgrade_to_deneb(spec: Any, new_spec: Any) -> bool:
    ...

def needs_upgrade_to_electra(spec: Any, new_spec: Any) -> bool:
    ...

def check_merkle_branch_equal(
    spec: Any,
    new_spec: Any,
    data: Any,
    upgraded: Any,
    gindex: int,
) -> None:
    ...

def check_lc_header_equal(
    spec: Any,
    new_spec: Any,
    data: LightClientHeader,
    upgraded: LightClientHeader,
) -> None:
    ...

def upgrade_lc_header_to_new_spec(
    spec: Any,
    new_spec: Any,
    data: LightClientHeader,
    phases: Any,
) -> LightClientHeader:
    ...

def check_lc_bootstrap_equal(
    spec: Any,
    new_spec: Any,
    data: LightClientBootstrap,
    upgraded: LightClientBootstrap,
) -> None:
    ...

def upgrade_lc_bootstrap_to_new_spec(
    spec: Any,
    new_spec: Any,
    data: LightClientBootstrap,
    phases: Any,
) -> LightClientBootstrap:
    ...

def check_lc_update_equal(
    spec: Any,
    new_spec: Any,
    data: LightClientUpdate,
    upgraded: LightClientUpdate,
) -> None:
    ...

def upgrade_lc_update_to_new_spec(
    spec: Any,
    new_spec: Any,
    data: LightClientUpdate,
    phases: Any,
) -> LightClientUpdate:
    ...

def check_lc_finality_update_equal(
    spec: Any,
    new_spec: Any,
    data: LightClientFinalityUpdate,
    upgraded: LightClientFinalityUpdate,
) -> None:
    ...

def upgrade_lc_finality_update_to_new_spec(
    spec: Any,
    new_spec: Any,
    data: LightClientFinalityUpdate,
    phases: Any,
) -> LightClientFinalityUpdate:
    ...

def check_lc_store_equal(
    spec: Any,
    new_spec: Any,
    data: LightClientStore,
    upgraded: LightClientStore,
) -> None:
    ...

def upgrade_lc_store_to_new_spec(
    spec: Any,
    new_spec: Any,
    data: LightClientStore,
    phases: Any,
) -> LightClientStore:
    ...