```python
from dataclasses import dataclass, field
from typing import Any, ClassVar

def assert_envelope_values(
    nonce: Any,
    channel_identifier: Any,
    transferred_amount: Any,
    locked_amount: Any,
    locksroot: Any
) -> None: ...

def assert_transfer_values(
    payment_identifier: Any,
    token: Any,
    recipient: Any
) -> None: ...

@dataclass(repr=False, eq=False)
class Lock:
    amount: Any
    expiration: Any
    secrethash: Any
    
    def __post_init__(self) -> None: ...
    
    @property
    def as_bytes(self) -> Any: ...
    
    @property
    def lockhash(self) -> Any: ...
    
    @classmethod
    def from_bytes(cls, serialized: Any) -> Any: ...

@dataclass(repr=False, eq=False)
class EnvelopeMessage(SignedRetrieableMessage):
    def __post_init__(self) -> None: ...
    
    @property
    def message_hash(self) -> Any: ...
    
    def _data_to_sign(self) -> Any: ...

@dataclass(repr=False, eq=False)
class SecretRequest(SignedRetrieableMessage):
    cmdid: ClassVar[Any]
    
    @classmethod
    def from_event(cls, event: Any) -> Any: ...
    
    def _data_to_sign(self) -> Any: ...

@dataclass(repr=False, eq=False)
class Unlock(EnvelopeMessage):
    cmdid: ClassVar[Any]
    secret: Any = field(repr=False)
    
    def __post_init__(self) -> None: ...
    
    @property
    def secrethash(self) -> Any: ...
    
    @classmethod
    def from_event(cls, event: Any) -> Any: ...
    
    @property
    def message_hash(self) -> Any: ...

@dataclass(repr=False, eq=False)
class RevealSecret(SignedRetrieableMessage):
    cmdid: ClassVar[Any]
    secret: Any = field(repr=False)
    
    @property
    def secrethash(self) -> Any: ...
    
    @classmethod
    def from_event(cls, event: Any) -> Any: ...
    
    def _data_to_sign(self) -> Any: ...

@dataclass(repr=False, eq=False)
class LockedTransferBase(EnvelopeMessage):
    def __post_init__(self) -> None: ...
    
    @classmethod
    def from_event(cls, event: Any) -> Any: ...
    
    def _packed_data(self) -> Any: ...
    
    @classmethod
    def _pack_locked_transfer_data(
        cls,
        cmdid: Any,
        message_identifier: Any,
        payment_identifier: Any,
        token: Any,
        recipient: Any,
        target: Any,
        initiator: Any,
        lock: Any
    ) -> Any: ...

@dataclass(repr=False, eq=False)
class LockedTransfer(LockedTransferBase):
    cmdid: ClassVar[Any]
    
    @property
    def message_hash(self) -> Any: ...

@dataclass(repr=False, eq=False)
class RefundTransfer(LockedTransferBase):
    cmdid: ClassVar[Any]
    
    @property
    def message_hash(self) -> Any: ...

@dataclass(repr=False, eq=False)
class LockExpired(EnvelopeMessage):
    cmdid: ClassVar[Any]
    
    @classmethod
    def from_event(cls, event: Any) -> Any: ...
    
    @property
    def message_hash(self) -> Any: ...
```