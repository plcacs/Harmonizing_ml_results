from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
from eth_utils import encode_hex
from eth2spec.utils.ssz.ssz_typing import Bytes32

class PayloadStatusV1StatusAlias(Enum):
    NOT_VALIDATED: str = 'NOT_VALIDATED'
    INVALIDATED: str = 'INVALIDATED'

class PayloadStatusV1Status(Enum):
    VALID: str = 'VALID'
    INVALID: str = 'INVALID'
    SYNCING: str = 'SYNCING'
    ACCEPTED: str = 'ACCEPTED'
    INVALID_BLOCK_HASH: str = 'INVALID_BLOCK_HASH'

    @property
    def alias(self) -> PayloadStatusV1StatusAlias:
        ...

@dataclass
class PayloadStatusV1:
    status: PayloadStatusV1Status = PayloadStatusV1Status.VALID
    latest_valid_hash: Optional[Bytes32] = None
    validation_error: Optional[str] = None

    @property
    def formatted_output(self) -> Dict[str, Optional[str]]:
        ...

class MegaStore:
    spec: Optional = None
    fc_store: Optional = None
    opt_store: Optional = None
    block_payload_statuses: Dict[Bytes32, PayloadStatusV1] = dict()

    def __init__(self, spec, fc_store, opt_store):
        ...

def get_optimistic_store(spec, anchor_state, anchor_block) -> OptimisticStore:
    ...

def get_valid_flag_value(status: PayloadStatusV1Status) -> bool:
    ...

def add_optimistic_block(spec, mega_store, signed_block, test_steps, payload_status=None, status=PayloadStatusV1Status.SYNCING):
    ...

def get_opt_head_block_root(spec, mega_store) -> Bytes32:
    ...

def is_invalidated(mega_store, block_root: Bytes32) -> bool:
    ...

def get_formatted_optimistic_head_output(mega_store) -> Dict[str, Union[int, str]]:
    ...

def clean_up_store(mega_store):
    ...
