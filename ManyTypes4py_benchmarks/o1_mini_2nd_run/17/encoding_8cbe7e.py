import binascii
from typing import Any, Dict, Optional, List, Union
from urllib.parse import parse_qs
from eth_utils import (
    is_0x_prefixed,
    is_checksum_address,
    to_bytes,
    to_canonical_address,
    to_checksum_address,
    to_hex,
)
from marshmallow import INCLUDE, Schema, SchemaOpts, fields, post_dump, post_load, pre_load, validate
from werkzeug.exceptions import NotFound
from werkzeug.routing import BaseConverter
from werkzeug.urls import url_encode, url_parse
from raiden.api.objects import Address, AddressList, PartnersPerToken, PartnersPerTokenList
from raiden.constants import (
    NULL_ADDRESS_BYTES,
    NULL_ADDRESS_HEX,
    SECRET_LENGTH,
    SECRETHASH_LENGTH,
    UINT256_MAX,
)
from raiden.settings import DEFAULT_INITIAL_CHANNEL_TARGET, DEFAULT_JOINABLE_FUNDS_TARGET
from raiden.storage.serialization.schemas import IntegerToStringField
from raiden.storage.utils import TimestampedEvent
from raiden.transfer import channel
from raiden.transfer.state import (
    ChainState,
    ChannelState,
    NettingChannelState,
    RouteState,
)
from raiden.transfer.views import get_token_network_by_address
from raiden.utils.capabilities import _bool_to_binary, int_bool
from raiden.utils.typing import Address as AddressBytes, AddressHex


class InvalidEndpoint(NotFound):
    """
    Exception to be raised instead of ValidationError if we want to skip the remaining
    endpoint matching rules and give a reason why the endpoint is invalid.
    """


class HexAddressConverter(BaseConverter):
    @staticmethod
    def to_python(value: str) -> bytes:
        if not is_0x_prefixed(value):
            raise InvalidEndpoint("Not a valid hex address, 0x prefix missing.")
        if not is_checksum_address(value):
            raise InvalidEndpoint("Not a valid EIP55 encoded address.")
        try:
            value_bytes: bytes = to_canonical_address(value)
        except ValueError:
            raise InvalidEndpoint("Could not decode hex.")
        return value_bytes

    @staticmethod
    def to_url(value: bytes) -> str:
        return to_checksum_address(value)


class AddressField(fields.Field):
    default_error_messages = {
        "missing_prefix": "Not a valid hex encoded address, must be 0x prefixed.",
        "invalid_checksum": "Not a valid EIP55 encoded address",
        "invalid_data": "Not a valid hex encoded address, contains invalid characters.",
        "invalid_size": "Not a valid hex encoded address, decoded address is not 20 bytes long.",
        "null_address": f"The {NULL_ADDRESS_HEX} address is not accepted",
    }

    @staticmethod
    def _serialize(value: AddressBytes, attr: str, obj: Any, **kwargs: Any) -> str:
        return to_checksum_address(value)

    def _deserialize(
        self, value: str, attr: str, data: Any, **kwargs: Any
    ) -> bytes:
        if not is_0x_prefixed(value):
            raise self.make_error("missing_prefix")
        if not is_checksum_address(value):
            raise self.make_error("invalid_checksum")
        try:
            value_bytes: bytes = to_canonical_address(value)
        except ValueError:
            raise self.make_error("invalid_data")
        if len(value_bytes) != 20:
            raise self.make_error("invalid_size")
        if value_bytes == NULL_ADDRESS_BYTES:
            raise self.make_error("null_address")
        return value_bytes


class SecretField(fields.Field):
    default_error_messages = {
        "missing_prefix": "Not a valid hex encoded value, must be 0x prefixed.",
        "invalid_data": "Not a valid hex formated string, contains invalid characters.",
        "invalid_size": f"Not a valid hex encoded secret, it is not {SECRET_LENGTH} characters long.",
    }

    @staticmethod
    def _serialize(value: bytes, attr: str, obj: Any, **kwargs: Any) -> str:
        return to_hex(value)

    def _deserialize(
        self, value: str, attr: str, data: Any, **kwargs: Any
    ) -> bytes:
        if not is_0x_prefixed(value):
            raise self.make_error("missing_prefix")
        try:
            value_bytes: bytes = to_bytes(hexstr=value)
        except binascii.Error:
            raise self.make_error("invalid_data")
        if len(value_bytes) != SECRET_LENGTH:
            raise self.make_error("invalid_size")
        return value_bytes


class SecretHashField(fields.Field):
    default_error_messages = {
        "missing_prefix": "Not a valid hex encoded value, must be 0x prefixed.",
        "invalid_data": "Not a valid hex formated string, contains invalid characters.",
        "invalid_size": f"Not a valid secrethash, decoded value is not {SECRETHASH_LENGTH} bytes long.",
    }

    @staticmethod
    def _serialize(value: bytes, attr: str, obj: Any, **kwargs: Any) -> str:
        return to_hex(value)

    def _deserialize(
        self, value: str, attr: str, data: Any, **kwargs: Any
    ) -> bytes:
        if not is_0x_prefixed(value):
            raise self.make_error("missing_prefix")
        try:
            value_bytes: bytes = to_bytes(hexstr=value)
        except binascii.Error:
            raise self.make_error("invalid_data")
        if len(value_bytes) != SECRETHASH_LENGTH:
            raise self.make_error("invalid_size")
        return value_bytes


class CapabilitiesField(fields.Field):
    @staticmethod
    def _serialize(
        value: Optional[Dict[str, bool]], attr: str, obj: Any, **kwargs: Any
    ) -> str:
        capdict: Dict[str, str] = {}
        if value:
            for key, bool_val in value.items():
                capdict[key] = _bool_to_binary(bool_val)
        return f"mxc://raiden.network/cap?{url_encode(capdict)}"

    def _deserialize(
        self, value: str, attr: str, data: Any, **kwargs: Any
    ) -> Dict[str, Union[str, bool]]:
        capstring = url_parse(value)
        capdict = parse_qs(capstring.query)
        capabilities: Dict[str, Union[str, bool]] = {}
        for key, values in capdict.items():
            if len(values) == 1:
                capabilities[key] = int_bool(values[0])
            else:
                capabilities[key] = values
        return capabilities


class BaseOpts(SchemaOpts):
    """
    This allows for having the Object the Schema encodes to inside of the class Meta
    """

    def __init__(self, meta: Any, ordered: bool) -> None:
        super().__init__(meta, ordered=ordered)
        self.decoding_class = getattr(meta, "decoding_class", None)


class BaseSchema(Schema):
    OPTIONS_CLASS = BaseOpts

    @post_load
    def make_object(self, data: Dict[str, Any], **kwargs: Any) -> Any:
        decoding_class = self.opts.decoding_class
        if decoding_class is None:
            return data
        return decoding_class(**data)


class BaseListSchema(Schema):
    OPTIONS_CLASS = BaseOpts

    @pre_load
    def wrap_data_envelope(self, data: Any, **kwargs: Any) -> Dict[str, Any]:
        data_wrapped = {"data": data}
        return data_wrapped

    @post_dump
    def unwrap_data_envelope(self, data: Dict[str, Any], **kwargs: Any) -> Any:
        return data["data"]

    @post_load
    def make_object(self, data: Dict[str, Any], **kwargs: Any) -> Any:
        decoding_class = self.opts.decoding_class
        list_data: List[Any] = data["data"]
        return decoding_class(list_data)


class RaidenEventsRequestSchema(BaseSchema):
    limit: Optional[str] = IntegerToStringField(missing=None)
    offset: Optional[str] = IntegerToStringField(missing=None)


class AddressSchema(BaseSchema):
    address: bytes = AddressField()

    class Meta:
        decoding_class = Address


class AddressListSchema(BaseListSchema):
    data: List[bytes] = fields.List(AddressField())

    class Meta:
        decoding_class = AddressList


class PartnersPerTokenSchema(BaseSchema):
    partner_address: bytes = AddressField()
    channel: str = fields.String()

    class Meta:
        decoding_class = PartnersPerToken


class PartnersPerTokenListSchema(BaseListSchema):
    data: List[PartnersPerToken] = fields.Nested(PartnersPerTokenSchema, many=True)

    class Meta:
        decoding_class = PartnersPerTokenList


class MintTokenSchema(BaseSchema):
    to: bytes = AddressField(required=True)
    value: str = IntegerToStringField(
        required=True, validate=validate.Range(min=1, max=UINT256_MAX)
    )


class ChannelStateSchema(BaseSchema):
    channel_identifier: str = IntegerToStringField(attribute="identifier")
    token_network_address: bytes = AddressField()
    token_address: bytes = AddressField()
    partner_address: str = fields.Method("get_partner_address")
    settle_timeout: str = IntegerToStringField()
    reveal_timeout: str = IntegerToStringField()
    balance: str = fields.Method("get_balance")
    state: str = fields.Method("get_state")
    total_deposit: str = fields.Method("get_total_deposit")
    total_withdraw: str = fields.Method("get_total_withdraw")

    @staticmethod
    def get_partner_address(channel_state: ChannelState) -> str:
        return to_checksum_address(channel_state.partner_state.address)

    @staticmethod
    def get_balance(channel_state: ChannelState) -> str:
        return str(channel.get_balance(channel_state.our_state, channel_state.partner_state))

    @staticmethod
    def get_state(channel_state: ChannelState) -> str:
        return channel.get_status(channel_state).value

    @staticmethod
    def get_total_deposit(channel_state: ChannelState) -> str:
        """Return our total deposit in the contract for this channel"""
        return str(channel_state.our_total_deposit)

    @staticmethod
    def get_total_withdraw(channel_state: ChannelState) -> str:
        """Return our total withdraw from this channel"""
        return str(channel_state.our_total_withdraw)


class ChannelPutSchema(BaseSchema):
    token_address: bytes = AddressField(required=True)
    partner_address: bytes = AddressField(required=True)
    reveal_timeout: Optional[str] = IntegerToStringField(missing=None)
    settle_timeout: Optional[str] = IntegerToStringField(missing=None)
    total_deposit: Optional[str] = IntegerToStringField(default=None, missing=None)


class ChannelPatchSchema(BaseSchema):
    total_deposit: Optional[str] = IntegerToStringField(default=None, missing=None)
    total_withdraw: Optional[str] = IntegerToStringField(default=None, missing=None)
    reveal_timeout: Optional[str] = IntegerToStringField(default=None, missing=None)
    state: Optional[str] = fields.String(
        default=None,
        missing=None,
        validate=validate.OneOf(
            [
                ChannelState.STATE_CLOSED.value,
                ChannelState.STATE_OPENED.value,
                ChannelState.STATE_SETTLED.value,
            ]
        ),
    )


class RouteMetadataSchema(BaseSchema):
    route: List[bytes] = fields.List(AddressField(), required=True)
    address_to_metadata: Dict[bytes, Any] = fields.Dict(
        keys=AddressField(), required=True, data_key="address_metadata"
    )
    estimated_fee: Optional[int] = fields.Integer(required=False, data_key="fee")

    class Meta:
        decoding_class = RouteState


class PaymentSchema(BaseSchema):
    initiator_address: Optional[bytes] = AddressField(missing=None)
    target_address: Optional[bytes] = AddressField(missing=None)
    token_address: Optional[bytes] = AddressField(missing=None)
    amount: str = IntegerToStringField(required=True)
    identifier: Optional[str] = IntegerToStringField(missing=None)
    secret: Optional[bytes] = SecretField(missing=None)
    secret_hash: Optional[bytes] = SecretHashField(missing=None)
    lock_timeout: Optional[str] = IntegerToStringField(missing=None)
    paths: Optional[List[RouteState]] = fields.List(
        fields.Nested(RouteMetadataSchema()), missing=None
    )


class ConnectionsConnectSchema(BaseSchema):
    funds: str = IntegerToStringField(required=True)
    initial_channel_target: Optional[str] = IntegerToStringField(
        missing=DEFAULT_INITIAL_CHANNEL_TARGET
    )
    joinable_funds_target: float = fields.Decimal(missing=DEFAULT_JOINABLE_FUNDS_TARGET)


class EventPaymentSchema(BaseSchema):
    block_number: str = IntegerToStringField()
    identifier: str = IntegerToStringField()
    log_time: Any = fields.DateTime()  # Ideally datetime.datetime
    token_address: Optional[bytes] = AddressField(missing=None)

    def serialize(
        self, chain_state: ChainState, event: TimestampedEvent
    ) -> Dict[str, Any]:
        serialized_event: Dict[str, Any] = self.dump(event)
        token_network = get_token_network_by_address(
            chain_state=chain_state, token_network_address=event.event.token_network_address
        )
        assert token_network, "Token network object should be registered if we got events with it"
        serialized_event["token_address"] = to_checksum_address(token_network.token_address)
        return serialized_event


class EventPaymentSentFailedSchema(EventPaymentSchema):
    event: str = fields.Constant("EventPaymentSentFailed")
    reason: str = fields.Str()
    target: bytes = AddressField()

    class Meta:
        fields = ("block_number", "event", "reason", "target", "log_time", "token_address")


class EventPaymentSentSuccessSchema(EventPaymentSchema):
    event: str = fields.Constant("EventPaymentSentSuccess")
    amount: str = IntegerToStringField()
    target: bytes = AddressField()

    class Meta:
        fields = ("block_number", "event", "amount", "target", "identifier", "log_time", "token_address")


class EventPaymentReceivedSuccessSchema(EventPaymentSchema):
    event: str = fields.Constant("EventPaymentReceivedSuccess")
    amount: str = IntegerToStringField()
    initiator: bytes = AddressField()

    class Meta:
        fields = (
            "block_number",
            "event",
            "amount",
            "initiator",
            "identifier",
            "log_time",
            "token_address",
        )


class UserDepositPostSchema(BaseSchema):
    total_deposit: Optional[str] = IntegerToStringField(default=None, missing=None)
    planned_withdraw_amount: Optional[str] = IntegerToStringField(default=None, missing=None)
    withdraw_amount: Optional[str] = IntegerToStringField(default=None, missing=None)


class NotificationSchema(BaseSchema):
    id: str = fields.String()
    summary: str = fields.String()
    body: str = fields.String()
    urgency: Optional[str] = fields.String(
        default=None,
        missing=None,
        validate=validate.OneOf(["normal", "low", "critical"]),
    )


class CapabilitiesSchema(BaseSchema):
    capabilities: Dict[str, Union[str, bool]] = CapabilitiesField(missing="mxc://")

    class Meta:
        unknown = INCLUDE
