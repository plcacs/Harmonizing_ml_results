import logging
from typing import Optional
from eth_typing import Address
from eth.abc import MessageAPI
from eth.constants import CREATE_CONTRACT_ADDRESS
from eth.typing import BytesOrView
from eth.validation import validate_canonical_address, validate_gte, validate_is_boolean, validate_is_bytes_or_view, validate_is_integer, validate_uint256


class Message(MessageAPI):
    __slots__ = ['to', 'sender', 'value', 'data', 'depth', 'gas', 'code',
        '_code_address', 'create_address', 'should_transfer_value',
        'is_static', '_storage_address']
    logger: logging.Logger = logging.getLogger('eth.vm.message.Message')
    to: Address
    sender: Address
    value: int
    data: BytesOrView
    depth: int
    gas: int
    code: bytes
    _code_address: Optional[Address]
    create_address: Optional[Address]
    should_transfer_value: bool
    is_static: bool
    _storage_address: Optional[Address]

    def __init__(self, gas, to, sender, value, data, code, depth=0,
        create_address=None, code_address=None, should_transfer_value=True,
        is_static=False):
        validate_uint256(gas, title='Message.gas')
        self.gas: int = gas
        if to != CREATE_CONTRACT_ADDRESS:
            validate_canonical_address(to, title='Message.to')
        self.to = to
        validate_canonical_address(sender, title='Message.sender')
        self.sender = sender
        validate_uint256(value, title='Message.value')
        self.value = value
        validate_is_bytes_or_view(data, title='Message.data')
        self.data = data
        validate_is_integer(depth, title='Message.depth')
        validate_gte(depth, minimum=0, title='Message.depth')
        self.depth = depth
        validate_is_bytes(code, title='Message.code')
        self.code = code
        if create_address is not None:
            validate_canonical_address(create_address, title=
                'Message.storage_address')
        self.storage_address = create_address
        if code_address is not None:
            validate_canonical_address(code_address, title=
                'Message.code_address')
        self.code_address = code_address
        validate_is_boolean(should_transfer_value, title=
            'Message.should_transfer_value')
        self.should_transfer_value = should_transfer_value
        validate_is_boolean(is_static, title='Message.is_static')
        self.is_static = is_static

    @property
    def code_address(self):
        if self._code_address is not None:
            return self._code_address
        else:
            return self.to

    @code_address.setter
    def code_address(self, value):
        self._code_address = value

    @property
    def storage_address(self):
        if self._storage_address is not None:
            return self._storage_address
        else:
            return self.to

    @storage_address.setter
    def storage_address(self, value):
        self._storage_address = value

    @property
    def is_create(self):
        return self.to == CREATE_CONTRACT_ADDRESS

    @property
    def data_as_bytes(self):
        return bytes(self.data)
