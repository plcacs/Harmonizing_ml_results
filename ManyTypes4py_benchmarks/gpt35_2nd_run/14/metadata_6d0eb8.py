from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional
from raiden.utils.typing import Address, AddressMetadata, Dict, EncryptedSecret, MetadataHash
from raiden.utils.validation import MetadataValidation

@dataclass
class RouteMetadata(MetadataValidation):
    address_metadata: Dict[Address, AddressMetadata] = None

    def __post_init__(self):
        self.address_metadata = deepcopy(self.address_metadata)
        self._validate_address_metadata()

    def _validate_address_metadata(self):
        validation_errors = self.validate_address_metadata()
        if self.address_metadata is not None:
            for address in validation_errors:
                del self.address_metadata[address]

    def get_metadata(self) -> Dict[Address, AddressMetadata]:
        return self.address_metadata

    def __repr__(self) -> str:
        return f'RouteMetadata: {" -> ".join([to_checksum_address(a) for a in self.route])}'

@dataclass(frozen=True)
class Metadata:
    _original_data: Any = None
    secret: Optional[EncryptedSecret] = None

    @classmethod
    def from_event(cls, event) -> 'Metadata':
        transfer = event.transfer
        routes = [RouteMetadata(route=r.route, address_metadata=r.address_to_metadata) for r in transfer.route_states]
        target_metadata = get_address_metadata(Address(transfer.target), transfer.route_states)
        encrypted_secret = encrypt_secret(transfer.secret, target_metadata, event.transfer.lock.amount, event.transfer.payment_identifier)
        return cls(routes=routes, _original_data=transfer.metadata, secret=encrypted_secret)

    @cached_property
    def hash(self) -> MetadataHash:
        return MetadataHash(keccak(self._serialize_canonical()))

    def __repr__(self) -> str:
        return f'Metadata: routes: {[repr(route) for route in self.routes]}'

    def to_dict(self) -> Dict[str, Any]:
        data = serializer.DictSerializer.serialize(self)
        serializer.remove_type_inplace(data)
        return data

    def _serialize_canonical(self) -> str:
        data = self.to_dict()
        return canonicaljson.encode_canonical_json(data)
