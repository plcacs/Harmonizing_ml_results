from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional
import canonicaljson
from eth_utils import keccak, to_checksum_address
from marshmallow import EXCLUDE, post_dump, post_load
from raiden.messages.abstract import cached_property
from raiden.storage.serialization import serializer
from raiden.transfer.mediated_transfer.events import SendLockedTransfer
from raiden.transfer.state import get_address_metadata
from raiden.transfer.utils.secret import encrypt_secret
from raiden.utils.typing import Address, AddressMetadata, Dict, EncryptedSecret, List, MetadataHash, Optional
from raiden.utils.validation import MetadataValidation

@dataclass
class RouteMetadata(MetadataValidation):
    address_metadata: Optional[Dict[Address, AddressMetadata]] = None

    class Meta:
        unknown = EXCLUDE
        serialize_missing = False

    def __post_init__(self) -> None:
        self.address_metadata = deepcopy(self.address_metadata)
        self._validate_address_metadata()

    def _validate_address_metadata(self) -> None:
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
    """
    Metadata is used by nodes to provide following hops in a transfer with additional information,
    e.g. to make additonal optimizations possible.
    It can contain arbitrary data and should be considered a read-only datastructure
    for mediating nodes.

    For mediating / target nodes, the original datastructure as sent out by a previous node
    is memorized in ``_original_data``.

    All other attributes are attributes that our node can use internally.
    If an additional attribute is required (non-`Optional`) this means that our node requires
    this information from the previous node in order to successfully continue the payment.
    Therefore a strategy for error recovery has to be implemented upon deserialization, in the
    case that these attributes are lacking.

    Note for serialization:
    If there is ``_original_data`` present in the object, as per the ``_post_dump`` other
    attributes on the object (e.g. ``routes``) will not get dumped, and thus only the
    originally received data will get persisted, e.g. upon dumping to the WAL!
    """
    _original_data: Optional[Any] = None
    secret: Optional[EncryptedSecret] = None

    class Meta:
        unknown = EXCLUDE
        serialize_missing = False

    @classmethod
    def from_event(cls, event: SendLockedTransfer) -> 'Metadata':
        transfer = event.transfer
        routes = [RouteMetadata(route=r.route, address_metadata=r.address_to_metadata) for r in transfer.route_states]
        target_metadata = get_address_metadata(Address(transfer.target), transfer.route_states)
        encrypted_secret = encrypt_secret(transfer.secret, target_metadata, event.transfer.lock.amount, event.transfer.payment_identifier)
        return cls(routes=routes, _original_data=transfer.metadata, secret=encrypted_secret)

    @cached_property
    def hash(self) -> MetadataHash:
        return MetadataHash(keccak(self._serialize_canonical()))

    @post_load(pass_original=True, pass_many=True)
    def _post_load(self, data: Dict, original_data: Any, many: bool, **kwargs: Any) -> Dict:
        data['_original_data'] = original_data
        return data

    @post_dump(pass_many=True)
    def _post_dump(self, data: Dict, many: bool) -> Any:
        """
        If ``_original_data`` are present we received the metadata (we are Mediator/Target) and
        only deserialize to the Metadata object for internal processing, so once we pass them to
        the next node just dump the data exactly as we received them.

        If no ``_original_data`` are present we are the Initiator and initially created the
        Metadata, so dump them as per the Schema.
        """
        dumped_data = data.pop('_original_data', None)
        if dumped_data is not None:
            return dumped_data
        return data

    def __repr__(self) -> str:
        return f'Metadata: routes: {[repr(route) for route in self.routes]}'

    def to_dict(self) -> Dict:
        data = serializer.DictSerializer.serialize(self)
        serializer.remove_type_inplace(data)
        return data

    def _serialize_canonical(self) -> str:
        data = self.to_dict()
        return canonicaljson.encode_canonical_json(data)
