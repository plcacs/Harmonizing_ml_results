from typing import Any

def _create_message(gas: int = 1, to: bytes = ADDRESS_A, sender: bytes = ADDRESS_B, value: int = 0, data: bytes = b'', code: bytes = b'', **kwargs: Any) -> Message:

def test_parameter_validation(init_kwargs: dict, is_valid: bool):

def test_code_address_defaults_to_to_address():

def test_code_address_uses_provided_address():

def test_storage_address_defaults_to_to_address():

def test_storage_address_uses_provided_address():

def test_is_create_computed_property():
