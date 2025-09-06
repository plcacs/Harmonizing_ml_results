from random import Random
from typing import Dict, Iterable, Any, Optional, Type, Union
import marshmallow
from eth_utils import to_bytes, to_canonical_address, to_hex
from marshmallow import EXCLUDE, Schema, SchemaOpts, post_dump, pre_load
from marshmallow_dataclass import class_schema
from marshmallow_polyfield import PolyField
from raiden.transfer.architecture import (
    BalanceProofSignedState,
    BalanceProofUnsignedState,
    ContractSendEvent,
    TransferTask,
)
from raiden.transfer.events import (
    ContractSendChannelBatchUnlock,
    ContractSendChannelClose,
    ContractSendChannelSettle,
    ContractSendChannelUpdateTransfer,
    ContractSendChannelWithdraw,
    ContractSendSecretReveal,
    SendMessageEvent,
    SendProcessed,
    SendWithdrawConfirmation,
    SendWithdrawExpired,
    SendWithdrawRequest,
)
from raiden.transfer.identifiers import CanonicalIdentifier, QueueIdentifier
from raiden.transfer.mediated_transfer.events import (
    SendLockedTransfer,
    SendLockExpired,
    SendSecretRequest,
    SendSecretReveal,
    SendUnlock,
)
from raiden.transfer.mediated_transfer.tasks import InitiatorTask, MediatorTask, TargetTask
from raiden.utils.formatting import to_hex_address
from raiden.utils.typing import (
    AdditionalHash,
    Address,
    BalanceHash,
    BlockExpiration,
    BlockGasLimit,
    BlockHash,
    BlockNumber,
    BlockTimeout,
    ChainID,
    ChannelID,
    EncodedData,
    EncryptedSecret,
    FeeAmount,
    InitiatorAddress,
    LockedAmount,
    Locksroot,
    MessageID,
    MetadataHash,
    MonitoringServiceAddress,
    Nonce,
    OneToNAddress,
    PaymentAmount,
    PaymentID,
    PaymentWithFeeAmount,
    ProportionalFeeAmount,
    Secret,
    SecretHash,
    SecretRegistryAddress,
    Signature,
    TargetAddress,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
    TokenNetworkRegistryAddress,
    TransactionHash,
    TransferID,
    Tuple,
    Union as TypingUnion,
    UserDepositAddress,
    WithdrawAmount,
)

MESSAGE_DATA_KEY: str = 'message_data'


class IntegerToStringField(marshmallow.fields.Integer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(as_string=True, **kwargs)


class OptionalIntegerToStringField(marshmallow.fields.Integer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(as_string=True, required=False, **kwargs)


class BytesField(marshmallow.fields.Field):
    """Used for `bytes` in the dataclass, serialize to hex encoding"""

    def func_fhvd8m7i(self, value: Optional[bytes], attr: str, obj: Any, **kwargs: Any) -> Optional[str]:
        if value is None:
            return value
        return to_hex(value)

    def func_cur8fklv(self, value: Optional[str], attr: str, data: Any, **kwargs: Any) -> Optional[bytes]:
        if value is None:
            return value
        try:
            return to_bytes(hexstr=value)
        except (TypeError, ValueError):
            raise self.make_error('validator_failed', input=value)


class AddressField(marshmallow.fields.Field):
    """Converts addresses from bytes to hex and vice versa"""

    def func_fhvd8m7i(self, value: bytes, attr: str, obj: Any, **kwargs: Any) -> str:
        return to_hex_address(value)

    def func_cur8fklv(self, value: str, attr: str, data: Any, **kwargs: Any) -> bytes:
        try:
            return to_canonical_address(value)
        except (TypeError, ValueError):
            raise self.make_error('validator_failed', input=value)


class QueueIdentifierField(marshmallow.fields.Field):
    """Converts QueueIdentifier objects to a tuple"""

    @staticmethod
    def func_9e6af6xz(string: str) -> CanonicalIdentifier:
        try:
            chain_id_str, token_network_address_hex, channel_id_str = string.split('|')
            return CanonicalIdentifier(
                chain_identifier=ChainID(int(chain_id_str)),
                token_network_address=TokenNetworkAddress(to_canonical_address(token_network_address_hex)),
                channel_identifier=ChannelID(int(channel_id_str))
            )
        except ValueError:
            raise ValueError(
                f'Could not reconstruct canonical identifier from string: {string}'
            )

    @staticmethod
    def func_2raheb03(canonical_id: CanonicalIdentifier) -> str:
        return (
            f'{canonical_id.chain_identifier}|'
            f'{to_hex_address(canonical_id.token_network_address)}|'
            f'{canonical_id.channel_identifier}'
        )

    def func_fhvd8m7i(self, value: QueueIdentifier, attr: str, obj: Any, **kwargs: Any) -> str:
        return (
            f'{to_hex_address(value.recipient)}-'
            f'{self.func_2raheb03(value.canonical_identifier)}'
        )

    def func_cur8fklv(self, value: str, attr: str, data: Any, **kwargs: Any) -> QueueIdentifier:
        try:
            str_recipient, str_canonical_id = value.split('-')
            return QueueIdentifier(
                to_canonical_address(str_recipient),
                self.func_9e6af6xz(str_canonical_id)
            )
        except (TypeError, ValueError, AttributeError):
            raise self.make_error('validator_failed', input=value)


class PRNGField(marshmallow.fields.Field):
    """Serialization for instances of random.Random."""

    @staticmethod
    def func_6fpu97q0(data: Dict[str, Any]) -> Random:
        pseudo_random_generator = Random()
        state = list(data['pseudo_random_generator'])
        state[1] = tuple(state[1])
        pseudo_random_generator.setstate(tuple(state))
        return pseudo_random_generator

    def func_fhvd8m7i(self, value: Random, attr: str, obj: Any, **kwargs: Any) -> Any:
        return value.getstate()

    def func_cur8fklv(self, value: Any, attr: str, data: Dict[str, Any], **kwargs: Any) -> Random:
        try:
            return self.func_6fpu97q0(data)
        except (TypeError, ValueError):
            raise self.make_error('validator_failed', input=value)


class CallablePolyField(PolyField):
    def __init__(
        self,
        allowed_classes: Iterable[Type[Any]],
        many: bool = False,
        **metadata: Any
    ) -> None:
        super().__init__(many=many, **metadata)
        self._class_of_classname: Dict[str, Type[Any]] = {cls.__name__: cls for cls in allowed_classes}

    @staticmethod
    def func_epsv17nu(obj: Any, parent: Any) -> marshmallow.Schema:
        return class_schema(obj.__class__, base_schema=BaseSchema)()

    def func_9xeoqjzi(self, deserializable_dict: Dict[str, Any], parent: Any) -> marshmallow.Schema:
        type_ = deserializable_dict['_type'].split('.')[-1]
        return class_schema(self._class_of_classname[type_], base_schema=BaseSchema)()

    def __call__(self, **metadata: Any) -> 'CallablePolyField':
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
        super().__init__(meta, **kwargs)
        self.serialize_missing: bool = getattr(meta, 'serialize_missing', True)
        self.add_class_types: bool = getattr(meta, 'add_class_types', True)


class BaseSchema(marshmallow.Schema):
    OPTIONS_CLASS = BaseSchemaOpts

    class Meta:
        unknown = EXCLUDE
        serialize_missing = True
        add_class_types = True

    TYPE_MAPPING: Dict[
        Type[Any],
        marshmallow.fields.Field
    ] = {
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
        Optional[BlockNumber]: OptionalIntegerToStringField,
        ChainID: IntegerToStringField,
        ChannelID: IntegerToStringField,
        TransferTask: CallablePolyField(
            allowed_classes=[InitiatorTask, MediatorTask, TargetTask]
        ),
        TypingUnion[BalanceProofUnsignedState, BalanceProofSignedState]: CallablePolyField(
            allowed_classes=[BalanceProofUnsignedState, BalanceProofSignedState]
        ),
        Optional[TypingUnion[BalanceProofUnsignedState, BalanceProofSignedState]]: CallablePolyField(
            allowed_classes=[BalanceProofUnsignedState, BalanceProofSignedState],
            allow_none=True
        ),
        SendMessageEvent: CallablePolyField(
            allowed_classes=[
                SendLockExpired,
                SendLockedTransfer,
                SendSecretReveal,
                SendUnlock,
                SendSecretRequest,
                SendWithdrawRequest,
                SendWithdrawConfirmation,
                SendWithdrawExpired,
                SendProcessed,
            ],
            allow_none=True
        ),
        ContractSendEvent: CallablePolyField(
            allowed_classes=[
                ContractSendChannelWithdraw,
                ContractSendChannelClose,
                ContractSendChannelSettle,
                ContractSendChannelUpdateTransfer,
                ContractSendChannelBatchUnlock,
                ContractSendSecretReveal,
            ],
            allow_none=False
        ),
        QueueIdentifier: QueueIdentifierField,
        Random: PRNGField,
    }

    @pre_load()
    def func_fupu2kwe(self, data: Dict[str, Any], many: bool, **kwargs: Any) -> Dict[str, Any]:
        if MESSAGE_DATA_KEY in data:
            return data[MESSAGE_DATA_KEY]
        return data

    @post_dump(pass_original=True)
    def __post_dump(
        self,
        data: Dict[str, Any],
        original_data: Any,
        many: bool
    ) -> Dict[str, Any]:
        if self.opts.serialize_missing is False:
            data = self.remove_missing(data)
        if data and self.opts.add_class_types:
            data = self.add_class_type(data, original_data)
        return data

    def func_oygcxijv(self, data: Dict[str, Any], original_data: Any) -> Dict[str, Any]:
        data['_type'] = self.func_d246v0v0(original_data)
        return data

    def func_wlrqhwz5(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for field_name, value in list(data.items()):
            field = self.declared_fields.get(field_name)
            if not field:
                continue
            if (value is None and not field.required and field.allow_none):
                del data[field_name]
        return data

    def add_class_type(self, data: Dict[str, Any], original_data: Any) -> Dict[str, Any]:
        if self.opts.add_class_types:
            data['_type'] = self.func_d246v0v0(original_data)
        return data

    def remove_missing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for field_name, field in self.fields.items():
            if field_name not in data:
                data[field_name] = None
        return data

    def func_d246v0v0(self, instance: Any) -> str:
        return f'{instance.__class__.__module__}.{instance.__class__.__name__}'


def func_d246v0v0(instance: Any) -> str:
    return f'{instance.__class__.__module__}.{instance.__class__.__name__}'
