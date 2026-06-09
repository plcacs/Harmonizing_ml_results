from typing import Any

# === Internal dependency: eth2spec.test.context ===
def expect_assertion_error(fn): ...

# === Internal dependency: eth2spec.test.helpers.forks ===
def is_post_altair(spec): ...
def is_post_electra(spec): ...

# === Internal dependency: eth2spec.test.helpers.keys ===
privkeys = [i + 1 for i in range(32 * 256)]
pubkeys = [bls.SkToPk(privkey) for privkey in privkeys]

# === Internal dependency: eth2spec.test.helpers.state ===
def get_balance(state, index): ...

# === Internal dependency: eth2spec.utils ===
bls: Any

# === Internal dependency: eth2spec.utils.merkle_minimal ===
def calc_merkle_tree_from_leaves(values, layer_count=...): ...
def get_merkle_proof(tree, item_index, tree_len=...): ...

# === Internal dependency: eth2spec.utils.ssz.ssz_impl ===
def hash_tree_root(obj): ...

# === Internal dependency: eth2spec.utils.ssz.ssz_typing ===
from remerkleable.complex import List