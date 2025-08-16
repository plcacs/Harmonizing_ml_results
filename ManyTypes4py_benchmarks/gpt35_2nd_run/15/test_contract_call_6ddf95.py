from typing import Any

def uint256_to_bytes(uint: int) -> bytes:
    return to_bytes(uint).rjust(32, b'\x00')

def test_get_transaction_result(chain: Any, simple_contract_address: bytes, signature: str, gas_price: int, expected: bytes) -> None:
    ...

def test_get_transaction_result_revert(vm: Any, chain_from_vm: Any, simple_contract_address: bytes, signature: str, expected: Any) -> None:
    ...
