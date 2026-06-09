from typing import Any

# === Internal dependency: eth2spec.utils ===
bls: Any

# === Extension dependency: lru ===
# Used symbols: LRU

# === Third-party dependency: pytest ===
# Used symbols: skip

# === Internal dependency: tests.core.pyspec.eth2spec.test.exceptions ===
class SkippedTest(Exception): ...

# === Internal dependency: tests.core.pyspec.eth2spec.test.helpers.constants ===
PHASE0: SpecForkName
ALTAIR: SpecForkName
BELLATRIX: SpecForkName
CAPELLA: SpecForkName
DENEB: SpecForkName
ELECTRA: SpecForkName
WHISK: SpecForkName
EIP7594: SpecForkName
ALL_PHASES: Any
LIGHT_CLIENT_TESTING_FORKS: Any
ALLOWED_TEST_RUNNER_FORKS: Any
POST_FORK_OF: Any
MINIMAL: PresetBaseName

# === Internal dependency: tests.core.pyspec.eth2spec.test.helpers.forks ===
def is_post_fork(a, b) -> bool: ...
def is_post_electra(spec) -> Any: ...

# === Internal dependency: tests.core.pyspec.eth2spec.test.helpers.genesis ===
def create_genesis_state(spec, validator_balances, activation_threshold) -> Any: ...

# === Internal dependency: tests.core.pyspec.eth2spec.test.helpers.specs ===
spec_targets: Dict[PresetBaseName, Dict[SpecForkName, Spec]]

# === Internal dependency: tests.core.pyspec.eth2spec.test.helpers.typing ===
SpecForkName: NewType
PresetBaseName: NewType

# === Internal dependency: tests.core.pyspec.eth2spec.test.utils ===
# re-export: from .utils import vector_test
# re-export: from .utils import with_meta_tags