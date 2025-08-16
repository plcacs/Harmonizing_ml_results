from typing import Optional, Union, Any

class Message(MessageAPI):
    __slots__: list[str]

    def __init__(self, gas: int, to: Address, sender: Address, value: int, data: BytesOrView, code: bytes, depth: int = 0, create_address: Optional[Address] = None, code_address: Optional[Address] = None, should_transfer_value: bool = True, is_static: bool = False) -> None:
        ...

    @property
    def code_address(self) -> Address:
        ...

    @code_address.setter
    def code_address(self, value: Address) -> None:
        ...

    @property
    def storage_address(self) -> Address:
        ...

    @storage_address.setter
    def storage_address(self, value: Address) -> None:
        ...

    @property
    def is_create(self) -> bool:
        ...

    @property
    def data_as_bytes(self) -> bytes:
        ...
