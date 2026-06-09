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
PHASE0 = SpecForkName(...)
ALTAIR = SpecForkName(...)
BELLATRIX = SpecForkName(...)
CAPELLA = SpecForkName(...)
DENEB = SpecForkName(...)
ELECTRA = SpecForkName(...)
WHISK = SpecForkName(...)
EIP7594 = SpecForkName(...)
EIP7732 = SpecForkName(...)
MAINNET_FORKS = (PHASE0, ALTAIR, BELLATRIX, CAPELLA, DENEB)
ALL_PHASES = (*MAINNET_FORKS, ELECTRA, EIP7594)
LIGHT_CLIENT_TESTING_FORKS = (*[item for item in MAINNET_FORKS if item != PHASE0], ELECTRA)
ALLOWED_TEST_RUNNER_FORKS = (*ALL_PHASES, WHISK, EIP7732)
POST_FORK_OF = {PHASE0: ALTAIR, ALTAIR: BELLATRIX, BELLATRIX: CAPELLA, CAPELLA: DENEB, DENEB: ELECTRA}
MINIMAL = PresetBaseName(...)

# === Internal dependency: tests.core.pyspec.eth2spec.test.helpers.forks ===
def is_post_fork(a, b): ...
def is_post_electra(spec): ...

# === Internal dependency: tests.core.pyspec.eth2spec.test.helpers.genesis ===
def create_genesis_state(spec, validator_balances, activation_threshold): ...

# === Internal dependency: tests.core.pyspec.eth2spec.test.helpers.specs ===
ALL_EXECUTABLE_SPEC_NAMES = ALL_PHASES + (WHISK,)
spec_targets = {MINIMAL: {fork: eval(f'spec_{fork}_minimal') for fork in ALL_EXECUTABLE_SPEC_NAMES}, MAINNET: {fork: eval(f'spec_{fork}_mainnet') for fork in ALL_EXECUTABLE_SPEC_NAMES}}

# === Internal dependency: tests.core.pyspec.eth2spec.test.helpers.typing ===
SpecForkName = NewType(...)
PresetBaseName = NewType(...)

# === Internal dependency: tests.core.pyspec.eth2spec.test.utils ===
from .utils import vector_test
from .utils import with_meta_tags