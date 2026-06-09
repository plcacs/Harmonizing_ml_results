# === Internal dependency: eth2spec.debug.random_value ===
def get_random_bytes_list(rng, length): ...

# === Internal dependency: eth2spec.test.helpers.forks ===
def is_post_capella(spec): ...
def is_post_deneb(spec): ...
def is_post_electra(spec): ...
def is_post_eip7732(spec): ...

# === Internal dependency: eth2spec.test.helpers.keys ===
privkeys = [i + 1 for i in range(32 * 256)]

# === Internal dependency: eth2spec.test.helpers.withdrawals ===
def get_expected_withdrawals(spec, state): ...

# === Internal dependency: eth2spec.utils.ssz.ssz_impl ===
def hash_tree_root(obj): ...

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