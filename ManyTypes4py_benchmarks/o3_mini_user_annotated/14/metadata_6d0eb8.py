from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import canonicaljson
from eth_utils import keccak, to_checksum_address
from marshmallow import EXCLUDE, post_dump, post_load

from raiden.messages.abstract import cached_property
from raiden.storage.serialization import serializer
from raiden.transfer.mediated_transfer.events import SendLockedTransfer
from raiden.transfer.state import get_address_metadata
from raiden.transfer.utils.secret import encrypt_secret
from raiden.utils.typing import (
    Address,
    AddressMetadata,
    EncryptedSecret,
    MetadataHash,
)


@dataclass
class RouteMetadata:
    route: List[Address]
    address_metadata: Optional[Dict[Address, AddressMetadata]] = None

    class Meta:
        """
        Sets meta-options for the Schema as defined in
        raiden.storage.serialization.schema.BaseSchemaOpts and the standard marshmallow options.
        """
        unknown = EXCLUDE
        # Don't include optional fields that are None during dumping
        serialize_missing = False

    def __post_init__(self) -> None:
        # don't use the original object, since this would result in mutated state
        self.address_metadata = deepcopy(self.address_metadata)
        self._validate_address_metadata()

    def _validate_address_metadata(self) -> None:
        validation_errors: Dict[Address, Any] = self.validate_address_metadata()
        if self.address_metadata is not None:
            for address in validation_errors:
                del self.address_metadata[address]

    def validate_address_metadata(self) -> Dict[Address, Any]:
        # Dummy implementation for validation; should be replaced with actual validation logic.
        return {}

    def get_metadata(self) -> Optional[Dict[Address, AddressMetadata]]:
        return self.address_metadata

    def __repr__(self) -> str:
        return f"RouteMetadata: {' -> '.join([to_checksum_address(a) for a in self.route])}"


@dataclass(frozen=True)
class Metadata:
    routes: List[RouteMetadata]
    _original_data: Optional[Any] = None
    secret: Optional[EncryptedSecret] = None

    class Meta:
        """
        Sets meta-options for the Schema as defined in
        raiden.storage.serialisation.schema.BaseSchemaOpts and the standard marshmallow options.
        """
        unknown = EXCLUDE
        serialize_missing = False

    @classmethod
    def from_event(cls, event: SendLockedTransfer) -> "Metadata":
        transfer = event.transfer
        routes: List[RouteMetadata] = [
            RouteMetadata(route=r.route, address_metadata=r.address_to_metadata)
            for r in transfer.route_states
        ]
        target_metadata = get_address_metadata(Address(transfer.target), transfer.route_states)
        encrypted_secret: EncryptedSecret = encrypt_secret(
            transfer.secret,
            target_metadata,
            event.transfer.lock.amount,
            event.transfer.payment_identifier,
        )
        return cls(routes=routes, _original_data=transfer.metadata, secret=encrypted_secret)

    @cached_property
    def hash(self) -> MetadataHash:
        return MetadataHash(keccak(self._serialize_canonical()))

    @post_load(pass_original=True, pass_many=True)
    def _post_load(
        self,
        data: Dict[str, Any],
        original_data: Dict[str, Any],
        many: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        data["_original_data"] = original_data
        return data

    @post_dump(pass_many=True)
    def _post_dump(self, data: Dict[str, Any], many: bool) -> Dict[str, Any]:
        dumped_data = data.pop("_original_data", None)
        if dumped_data is not None:
            return dumped_data
        return data

    def __repr__(self) -> str:
        return f"Metadata: routes: {[repr(route) for route in self.routes]}"

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = serializer.DictSerializer.serialize(self)
        serializer.remove_type_inplace(data)
        return data

    def _serialize_canonical(self) -> bytes:
        data: Dict[str, Any] = self.to_dict()
        return canonicaljson.encode_canonical_json(data)