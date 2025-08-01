from random import Random
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import marshmallow
from eth_utils import to_bytes, to_canonical_address, to_hex
from marshmallow import EXCLUDE, Schema, SchemaOpts, post_dump, pre_load
from marshmallow_dataclass import class_schema
from marshmallow_polyfield import PolyField
from raiden.transfer.architecture import BalanceProofSignedState, BalanceProofUnsignedState, ContractSendEvent, TransferTask
from raiden.transfer.events import ContractSendChannelBatchUnlock, ContractSendChannelClose, ContractSendChannelSettle, ContractSendChannelUpdateTransfer, ContractSendChannelWithdraw, ContractSendSecretReveal, SendMessageEvent, SendProcessed, SendWithdrawConfirmation, SendWithdrawExpired, SendWithdrawRequest
from raiden.transfer.identifiers import CanonicalIdentifier, QueueIdentifier
from raiden.transfer.mediated_transfer.events import SendLockedTransfer, SendLockExpired, SendSecretRequest, SendSecretReveal, SendUnlock
from raiden.transfer.mediated_transfer.tasks import InitiatorTask, MediatorTask, TargetTask
from raiden.utils.formatting import to_hex_address
from raiden.utils.typing import AdditionalHash, Address, Any, BalanceHash, BlockExpiration, BlockGasLimit, BlockHash, BlockNumber, BlockTimeout, ChainID, ChannelID, EncodedData, EncryptedSecret, FeeAmount, InitiatorAddress, LockedAmount, Locksroot, MessageID, MetadataHash, MonitoringServiceAddress, Nonce, OneToNAddress, Optional as TypingOptional, PaymentAmount, PaymentID, PaymentWithFeeAmount, ProportionalFeeAmount, Secret, SecretHash, SecretRegistryAddress, Signature, TargetAddress, TokenAddress, TokenAmount, TokenNetworkAddress, TokenNetworkRegistryAddress, TransactionHash, TransferID, Tuple, Union, UserDepositAddress, WithdrawAmount

MESSAGE_DATA_KEY = 'message_data'


class IntegerToStringField(marshmallow.fields.Integer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(as_string=True, **kwargs)


class OptionalIntegerToStringField(marshmallow.fields.Integer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(as_string=True, required=False, **kwargs)


class BytesField(marshmallow.fields.Field):
    """Used for `bytes` in the dataclass, serialize to hex encoding"""

    def _serialize(self, value: Optional[bytes], attr: str, obj: Any, **kwargs: Any) -> Optional[str]:
        if value is None:
            return value
        return to_hex(value)

    def _deserialize(self, value: Optional[str], attr: str, data: Any, **kwargs: Any) -> Optional[bytes]:
        if value is None:
            return value
        try:
            return to_bytes(hexstr=value)
        except (TypeError, ValueError):
            raise self.make_error('validator_failed', input=value)


class AddressField(marshmallow.fields.Field):
    """Converts addresses from bytes to hex and vice versa"""

    def _serialize(self, value: bytes, attr: str, obj: Any, **kwargs: Any) -> str:
        return to_hex_address(value)

    def _deserialize(self, value: str, attr: str, data: Any, **kwargs: Any) -> bytes:
        try:
            return to_canonical_address(value)
        except (TypeError, ValueError):
            raise self.make_error('validator_failed', input=value)


class QueueIdentifierField(marshmallow.fields.Field):
    """Converts QueueIdentifier objects to a tuple"""

    @staticmethod
    def _canonical_id_from_string(string: str) -> CanonicalIdentifier:
        try:
            chain_id_str, token_network_address_hex, channel_id_str = string.split('|')
            return CanonicalIdentifier(
                chain_identifier=ChainID(int(chain_id_str)),
                token_network_address=TokenNetworkAddress(to_canonical_address(token_network_address_hex)),
                channel_identifier=ChannelID(int(channel_id_str))
            )
        except ValueError:
            raise ValueError(f'Could not reconstruct canonical identifier from string: {string}')

    @staticmethod
    def _canonical_id_to_string(canonical_id: CanonicalIdentifier) -> str:
        return f'{canonical_id.chain_identifier}|{to_hex_address(canonical_id.token_network_address)}|{canonical_id.channel_identifier}'

    def _serialize(self, value: QueueIdentifier, attr: str, obj: Any, **kwargs: Any) -> str:
        return f'{to_hex_address(value.recipient)}-{self._canonical_id_to_string(value.canonical_identifier)}'

    def _deserialize(self, value: str, attr: str, data: Any, **kwargs: Any) -> QueueIdentifier:
        try:
            str_recipient, str_canonical_id = value.split('-')
            return QueueIdentifier(to_canonical_address(str_recipient), self._canonical_id_from_string(str_canonical_id))
        except (TypeError, ValueError, AttributeError):
            raise self.make_error('validator_failed', input=value)


class PRNGField(marshmallow.fields.Field):
    """Serialization for instances of random.Random."""

    @staticmethod
    def pseudo_random_generator_from_json(data: Dict[str, Any]) -> Random:
        pseudo_random_generator: Random = Random()
        state = list(data['pseudo_random_generator'])
        state[1] = tuple(state[1])
        pseudo_random_generator.setstate(tuple(state))
        return pseudo_random_generator

    def _serialize(self, value: Random, attr: str, obj: Any, **kwargs: Any) -> Any:
        return value.getstate()

    def _deserialize(self, value: Any, attr: str, data: Dict[str, Any], **kwargs: Any) -> Random:
        try:
            return self.pseudo_random_generator_from_json(data)
        except (TypeError, ValueError):
            raise self.make_error('validator_failed', input=value)


class CallablePolyField(PolyField):
    def __init__(self, allowed_classes: Iterable[type], many: bool = False, **metadata: Any) -> None:
        super().__init__(many=many, **metadata)
        self._class_of_classname: Dict[str, type] = {cls.__name__: cls for cls in allowed_classes}

    @staticmethod
    def serialization_schema_selector(obj: Any, parent: Any) -> Schema:
        return class_schema(obj.__class__, base_schema=BaseSchema)()

    def deserialization_schema_selector(self, deserializable_dict: Dict[str, Any], parent: Any) -> Schema:
        type_: str = deserializable_dict['_type'].split('.')[-1]
        return class_schema(self._class_of_classname[type_], base_schema=BaseSchema)()

    def __call__(self, **metadata: Any) -> "CallablePolyField":
        self.metadata = metadata
        return self


class BaseSchemaOpts(SchemaOpts):
    """
    This class defines additional, custom options for the `class Meta` options.
    (https://marshmallow.readthedocs.io/en/stable/api_reference.html#marshmallow.Schema.Meta)
    They can be set per Schema definition.

    For more info, see:
    https://marshmallow.readthedocs.io/en/stable/extending.html#custom-class-meta-options
    """
    def __init__(self, meta: Any, **kwargs: Any) -> None:
        SchemaOpts.__init__(self, meta, **kwargs)
        self.serialize_missing: bool = getattr(meta, 'serialize_missing', True)
        self.add_class_types: bool = getattr(meta, 'add_class_types', True)


class BaseSchema(marshmallow.Schema):
    OPTIONS_CLASS = BaseSchemaOpts

    class Meta:
        unknown = EXCLUDE
        serialize_missing: bool = True
        add_class_types: bool = True

    TYPE_MAPPING: Dict[Any, Any] = {
        Address: AddressField,
        InitiatorAddress: AddressField,
        MonitoringServiceAddress: AddressField,
        OneToNAddress: AddressField,
        TokenNetworkRegistryAddress: AddressField,
        SecretRegistryAddress: AddressField,
        TargetAddress: AddressField,
        TokenAddress: AddressField,
        TokenNetworkAddress: AddressField,
        UserDepositAddress: AddressField,
        EncodedData: BytesField,
        AdditionalHash: BytesField,
        BalanceHash: BytesField,
        BlockHash: BytesField,
        Locksroot: BytesField,
        MetadataHash: BytesField,
        Secret: BytesField,
        SecretHash: BytesField,
        Signature: BytesField,
        TransactionHash: BytesField,
        EncryptedSecret: BytesField,
        BlockExpiration: IntegerToStringField,
        BlockNumber: IntegerToStringField,
        BlockTimeout: IntegerToStringField,
        TokenAmount: IntegerToStringField,
        FeeAmount: IntegerToStringField,
        ProportionalFeeAmount: IntegerToStringField,
        LockedAmount: IntegerToStringField,
        BlockGasLimit: IntegerToStringField,
        MessageID: IntegerToStringField,
        Nonce: IntegerToStringField,
        PaymentAmount: IntegerToStringField,
        PaymentID: IntegerToStringField,
        PaymentWithFeeAmount: IntegerToStringField,
        TransferID: IntegerToStringField,
        WithdrawAmount: IntegerToStringField,
        TypingOptional[BlockNumber]: OptionalIntegerToStringField,
        ChainID: IntegerToStringField,
        ChannelID: IntegerToStringField,
        TransferTask: CallablePolyField(allowed_classes=[InitiatorTask, MediatorTask, TargetTask]),
        Union[BalanceProofUnsignedState, BalanceProofSignedState]:
            CallablePolyField(allowed_classes=[BalanceProofUnsignedState, BalanceProofSignedState]),
        Optional[Union[BalanceProofUnsignedState, BalanceProofSignedState]]:
            CallablePolyField(allowed_classes=[BalanceProofUnsignedState, BalanceProofSignedState], allow_none=True),
        SendMessageEvent:
            CallablePolyField(allowed_classes=[SendLockExpired, SendLockedTransfer, SendSecretReveal, SendUnlock, SendSecretRequest, SendWithdrawRequest, SendWithdrawConfirmation, SendWithdrawExpired, SendProcessed], allow_none=True),
        ContractSendEvent:
            CallablePolyField(allowed_classes=[ContractSendChannelWithdraw, ContractSendChannelClose, ContractSendChannelSettle, ContractSendChannelUpdateTransfer, ContractSendChannelBatchUnlock, ContractSendSecretReveal], allow_none=False),
        QueueIdentifier: QueueIdentifierField,
        Random: PRNGField
    }

    @pre_load()
    def remove_envelope(self, data: Dict[str, Any], many: bool, **kwargs: Any) -> Dict[str, Any]:
        if MESSAGE_DATA_KEY in data:
            return data[MESSAGE_DATA_KEY]
        return data

    @post_dump(pass_original=True)
    def __post_dump(self, data: Dict[str, Any], original_data: Any, many: bool) -> Dict[str, Any]:
        if self.opts.serialize_missing is False:
            data = self.remove_missing(data)
        if data and self.opts.add_class_types:
            data = self.add_class_type(data, original_data)
        return data

    def add_class_type(self, data: Dict[str, Any], original_data: Any) -> Dict[str, Any]:
        data['_type'] = class_type(original_data)
        return data

    def remove_missing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for field_name, value in list(data.items()):
            field = self.declared_fields.get(field_name)
            if not field:
                continue
            if value is None and field.required is False and (field.allow_none is True):
                del data[field_name]
        return data


def class_type(instance: Any) -> str:
    return f'{instance.__class__.__module__}.{instance.__class__.__name__}'