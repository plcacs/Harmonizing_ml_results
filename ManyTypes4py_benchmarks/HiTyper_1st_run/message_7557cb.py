import logging
from eth_typing import Address
from eth.abc import MessageAPI
from eth.constants import CREATE_CONTRACT_ADDRESS
from eth.typing import BytesOrView
from eth.validation import validate_canonical_address, validate_gte, validate_is_boolean, validate_is_bytes, validate_is_bytes_or_view, validate_is_integer, validate_uint256

class Message(MessageAPI):
    __slots__ = ['to', 'sender', 'value', 'data', 'depth', 'gas', 'code', '_code_address', 'create_address', 'should_transfer_value', 'is_static', '_storage_address']
    logger = logging.getLogger('eth.vm.message.Message')

    def __init__(self, gas: Union[bytes, int], to: Union[int, raiden.utils.Address], sender: Union[str, Address, None, list["Address"]], value: Union[int, bytes], data: Union[bytes, None, str], code: bytes, depth: int=0, create_address: Union[None, str, raiden.utils.Address, int]=None, code_address: Union[None, str, bytes, raiden.utils.Address]=None, should_transfer_value: bool=True, is_static: bool=False) -> None:
        validate_uint256(gas, title='Message.gas')
        self.gas = gas
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
            validate_canonical_address(create_address, title='Message.storage_address')
        self.storage_address = create_address
        if code_address is not None:
            validate_canonical_address(code_address, title='Message.code_address')
        self.code_address = code_address
        validate_is_boolean(should_transfer_value, title='Message.should_transfer_value')
        self.should_transfer_value = should_transfer_value
        validate_is_boolean(is_static, title='Message.is_static')
        self.is_static = is_static

    @property
    def code_address(self) -> Union[str, set]:
        if self._code_address is not None:
            return self._code_address
        else:
            return self.to

    @code_address.setter
    def code_address(self, value) -> Union[str, set]:
        self._code_address = value

    @property
    def storage_address(self) -> Union[str, dict[typing.Any, dict[str, str]], tuple[typing.Any]]:
        if self._storage_address is not None:
            return self._storage_address
        else:
            return self.to

    @storage_address.setter
    def storage_address(self, value) -> Union[str, dict[typing.Any, dict[str, str]], tuple[typing.Any]]:
        self._storage_address = value

    @property
    def is_create(self) -> bool:
        return self.to == CREATE_CONTRACT_ADDRESS

    @property
    def data_as_bytes(self) -> bytes:
        return bytes(self.data)