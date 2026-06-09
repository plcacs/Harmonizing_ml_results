from typing import Any

# === Internal dependency: eth2spec.test.context ===
def spec_state_test(fn) -> Any: ...
def expect_assertion_error(fn) -> Any: ...
def with_presets(preset_bases, reason = ...) -> Any: ...
with_electra_and_later: Any

# === Internal dependency: eth2spec.test.helpers.constants ===
MINIMAL: PresetBaseName

# === Internal dependency: eth2spec.test.helpers.state ===
def get_validator_index_by_pubkey(state, pubkey) -> Any: ...

# === Internal dependency: eth2spec.test.helpers.typing ===
PresetBaseName: NewType

# === Internal dependency: eth2spec.test.helpers.withdrawals ===
def set_eth1_withdrawal_credential_with_balance(spec, state, index, balance = ..., address = ...) -> Any: ...
def set_compounding_withdrawal_credential(spec, state, index, address = ...) -> Any: ...