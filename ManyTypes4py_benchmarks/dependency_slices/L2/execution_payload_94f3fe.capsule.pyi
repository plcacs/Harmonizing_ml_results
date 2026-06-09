from typing import Any

# === Internal dependency: eth2spec.debug.random_value ===
def get_random_bytes_list(rng: Random, length: int) -> bytes: ...

# === Internal dependency: eth2spec.test.helpers.forks ===
def is_post_capella(spec) -> Any: ...
def is_post_deneb(spec) -> Any: ...
def is_post_electra(spec) -> Any: ...
def is_post_eip7732(spec) -> Any: ...

# === Internal dependency: eth2spec.test.helpers.keys ===
privkeys: Any

# === Internal dependency: eth2spec.test.helpers.withdrawals ===
def get_expected_withdrawals(spec, state) -> Any: ...

# === Internal dependency: eth2spec.utils.ssz.ssz_impl ===
def hash_tree_root(obj: View) -> Bytes32: ...

# === Third-party dependency: eth_hash.auto ===
keccak: Keccak256

# === Third-party dependency: eth_hash.main ===
class Keccak256: ...

# === Third-party dependency: rlp ===
# Used symbols: encode

# === Third-party dependency: rlp.sedes ===
# Used symbols: Binary, List, big_endian_int

# === Third-party dependency: trie ===
# Used symbols: HexaryTrie