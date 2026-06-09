from typing import Any

# === Internal dependency: eth2spec.test.context ===
def expect_assertion_error(fn) -> Any: ...

# === Internal dependency: eth2spec.test.helpers.forks ===
def is_post_altair(spec) -> Any: ...
def is_post_electra(spec) -> Any: ...

# === Internal dependency: eth2spec.test.helpers.keys ===
privkeys: Any
pubkeys: Any

# === Internal dependency: eth2spec.test.helpers.state ===
def get_balance(state, index) -> Any: ...

# === Internal dependency: eth2spec.utils ===
bls: Any

# === Internal dependency: eth2spec.utils.merkle_minimal ===
def calc_merkle_tree_from_leaves(values, layer_count = ...) -> Any: ...
def get_merkle_proof(tree, item_index, tree_len = ...) -> Any: ...

# === Internal dependency: eth2spec.utils.ssz.ssz_impl ===
def hash_tree_root(obj: View) -> Bytes32: ...

# === Internal dependency: eth2spec.utils.ssz.ssz_typing ===
# re-export: from remerkleable.complex import List