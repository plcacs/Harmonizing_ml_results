import pytest
from typing import Any, Dict, Union
from eth_utils import ValidationError, to_normalized_address
from eth.constants import CREATE_CONTRACT_ADDRESS
from eth.vm.message import Message
ADDRESS_A: bytes = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0'
ADDRESS_B: bytes = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1'
ADDRESS_C: bytes = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf2'

def _create_message(gas=1, to=ADDRESS_A, sender=ADDRESS_B, value=0, data=b'', code=b'', **kwargs: Any):
    return Message(gas=gas, to=to, sender=sender, value=value, data=data, code=code, **kwargs)

@pytest.mark.parametrize('init_kwargs,is_valid', (({}, True), ({'gas': True}, False), ({'gas': -1}, False), ({'gas': 1.0}, False), ({'gas': '1'}, False), ({'value': True}, False), ({'value': -1}, False), ({'value': 1.0}, False), ({'value': '1'}, False), ({'sender': to_normalized_address(ADDRESS_A)}, False), ({'to': to_normalized_address(ADDRESS_A)}, False), ({'create_address': to_normalized_address(ADDRESS_A)}, False), ({'code_address': to_normalized_address(ADDRESS_A)}, False), ({'should_transfer_value': 1}, False), ({'should_transfer_value': 0}, False)))
def test_parameter_validation(init_kwargs, is_valid):
    if is_valid:
        _create_message(**init_kwargs)
    else:
        with pytest.raises(ValidationError):
            _create_message(**init_kwargs)

def test_code_address_defaults_to_to_address():
    message: Message = _create_message()
    assert message.code_address == message.to

def test_code_address_uses_provided_address():
    message: Message = _create_message(code_address=ADDRESS_C)
    assert message.code_address == ADDRESS_C

def test_storage_address_defaults_to_to_address():
    message: Message = _create_message()
    assert message.storage_address == message.to

def test_storage_address_uses_provided_address():
    message: Message = _create_message(create_address=ADDRESS_C)
    assert message.storage_address == ADDRESS_C

def test_is_create_computed_property():
    create_message: Message = _create_message(to=CREATE_CONTRACT_ADDRESS)
    assert create_message.is_create is True
    not_create_message: Message = _create_message(to=ADDRESS_B)
    assert not_create_message.is_create is False