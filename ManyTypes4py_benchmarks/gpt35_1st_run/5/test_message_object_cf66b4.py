from typing import Any, Dict

ADDRESS_A: bytes = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0'
ADDRESS_B: bytes = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1'
ADDRESS_C: bytes = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf2'

def _create_message(gas: int = 1, to: bytes = ADDRESS_A, sender: bytes = ADDRESS_B, value: int = 0, data: bytes = b'', code: bytes = b'', **kwargs: Any) -> Message:
    return Message(gas=gas, to=to, sender=sender, value=value, data=data, code=code, **kwargs)

def test_parameter_validation(init_kwargs: Dict[str, Any], is_valid: bool) -> None:
    if is_valid:
        _create_message(**init_kwargs)
    else:
        with pytest.raises(ValidationError):
            _create_message(**init_kwargs)

def test_code_address_defaults_to_to_address() -> None:
    message = _create_message()
    assert message.code_address == message.to

def test_code_address_uses_provided_address() -> None:
    message = _create_message(code_address=ADDRESS_C)
    assert message.code_address == ADDRESS_C

def test_storage_address_defaults_to_to_address() -> None:
    message = _create_message()
    assert message.storage_address == message.to

def test_storage_address_uses_provided_address() -> None:
    message = _create_message(create_address=ADDRESS_C)
    assert message.storage_address == ADDRESS_C

def test_is_create_computed_property() -> None:
    create_message = _create_message(to=CREATE_CONTRACT_ADDRESS)
    assert create_message.is_create is True
    not_create_message = _create_message(to=ADDRESS_B)
    assert not_create_message.is_create is False
