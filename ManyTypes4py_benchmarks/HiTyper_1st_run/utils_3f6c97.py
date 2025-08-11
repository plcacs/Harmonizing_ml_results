from dataclasses import dataclass, replace
from hypothesis.strategies import builds, composite, integers, sampled_from
from raiden.messages.decode import lockedtransfersigned_from_message
from raiden.messages.encode import message_from_sendevent
from raiden.messages.transfers import LockedTransfer
from raiden.tests.utils import factories
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.transfer.mediated_transfer.events import SendLockedTransfer, SendSecretRequest, SendSecretReveal, SendUnlock
from raiden.transfer.mediated_transfer.initiator import send_lockedtransfer
from raiden.transfer.mediated_transfer.state import LockedTransferSignedState, TransferDescriptionWithSecretState
from raiden.transfer.mediated_transfer.state_change import ActionInitInitiator, ActionInitTarget, ReceiveSecretRequest, ReceiveSecretReveal
from raiden.transfer.state import HopState, NettingChannelState, RouteState
from raiden.transfer.state_change import ReceiveUnlock
from raiden.utils.signer import LocalSigner
from raiden.utils.typing import MYPY_ANNOTATION, Address, Any, BlockNumber, List, MessageID, PrivateKey

def signed_transfer_from_description(private_key: Union[raiden.utils.PrivateKey, int, str], description: Union[raiden.transfer.mediated_transfer.state.TransferDescriptionWithSecretState, raiden.utils.MessageID], channel: Union[raiden.transfer.mediated_transfer.state.TransferDescriptionWithSecretState, raiden.utils.MessageID], message_id: Union[raiden.transfer.mediated_transfer.state.TransferDescriptionWithSecretState, raiden.utils.MessageID], block_number: Union[raiden.transfer.mediated_transfer.state.TransferDescriptionWithSecretState, raiden.utils.MessageID], route_state: Union[raiden.transfer.mediated_transfer.state.TransferDescriptionWithSecretState, raiden.utils.MessageID], route_states: Union[raiden.transfer.mediated_transfer.state.TransferDescriptionWithSecretState, raiden.utils.MessageID]) -> Union[str, bytes, raiden.utils.Address]:
    send_locked_transfer = send_lockedtransfer(transfer_description=description, channel_state=channel, message_identifier=message_id, block_number=block_number, route_state=route_state, route_states=route_states)
    message = message_from_sendevent(send_locked_transfer)
    assert isinstance(message, LockedTransfer), MYPY_ANNOTATION
    message.sign(LocalSigner(private_key))
    return lockedtransfersigned_from_message(message)

def action_init_initiator_to_action_init_target(action: Union[raiden.utils.PrivateKey, raiden.utils.BlockNumber, raiden.transfer.state.NettingChannelState], channel: Union[raiden.utils.PrivateKey, raiden.utils.BlockNumber, raiden.transfer.state.NettingChannelState], block_number: Union[raiden.utils.PrivateKey, raiden.utils.BlockNumber, raiden.transfer.state.NettingChannelState], route_state: Union[raiden.utils.PrivateKey, raiden.utils.BlockNumber, raiden.transfer.state.NettingChannelState], address: Union[raiden.utils.Address, raiden.transfer.state.NettingChannelState, raiden.utils.BlockNumber], private_key: Union[raiden.utils.PrivateKey, raiden.utils.BlockNumber, raiden.transfer.state.NettingChannelState]) -> ActionInitTarget:
    transfer = signed_transfer_from_description(private_key=private_key, description=action.transfer, channel=channel, message_id=factories.make_message_identifier(), block_number=block_number, route_state=route_state, route_states=action.routes)
    from_hop = HopState(node_address=address, channel_identifier=channel.identifier)
    return ActionInitTarget(from_hop=from_hop, transfer=transfer, sender=address, balance_proof=transfer.balance_proof)

@dataclass(frozen=True)
class SendSecretRequestInNode:
    pass

def send_secret_request_to_receive_secret_request(source: Union[raiden.tests.fuzz.utils.SendLockedTransferInNode, raiden.tests.fuzz.utils.SendSecretRevealInNode, raiden.tests.fuzz.utils.SendUnlockInNode]) -> ReceiveSecretRequest:
    return ReceiveSecretRequest(sender=source.node, payment_identifier=source.event.payment_identifier, amount=source.event.amount, expiration=source.event.expiration, secrethash=source.event.secrethash)

@dataclass(frozen=True)
class SendSecretRevealInNode:
    pass

def send_secret_reveal_to_recieve_secret_reveal(source: Union[raiden.tests.fuzz.utils.SendSecretRevealInNode, raiden.tests.fuzz.utils.SendLockedTransferInNode, raiden.tests.fuzz.utils.SendUnlockInNode]) -> ReceiveSecretReveal:
    return ReceiveSecretReveal(sender=source.node, secrethash=source.event.secrethash, secret=source.event.secret)

@dataclass(frozen=True)
class SendLockedTransferInNode:
    pass

def send_lockedtransfer_to_locked_transfer(source: Union[bytes, bytearray, str]):
    locked_transfer = message_from_sendevent(source.event)
    assert isinstance(locked_transfer, LockedTransfer), MYPY_ANNOTATION
    locked_transfer.sign(LocalSigner(source.private_key))
    return locked_transfer

def locked_transfer_to_action_init_target(locked_transfer: Union[raiden.messages.transfers.LockedTransfer, raiden.raiden_service.RaidenService, raiden.messages.SecretRequest]) -> ActionInitTarget:
    from_transfer = lockedtransfersigned_from_message(locked_transfer)
    channel_id = from_transfer.balance_proof.channel_identifier
    from_hop = HopState(node_address=Address(locked_transfer.initiator), channel_identifier=channel_id)
    init_target_statechange = ActionInitTarget(from_hop=from_hop, transfer=from_transfer, balance_proof=from_transfer.balance_proof, sender=from_transfer.balance_proof.sender)
    return init_target_statechange

@dataclass(frozen=True)
class SendUnlockInNode:
    pass

def send_unlock_to_receive_unlock(source: Union[raiden.transfer.identifiers.CanonicalIdentifier, raiden.utils.Address, raiden.tests.fuzz.utils.SendSecretRequestInNode], canonical_identifier: Union[raiden.transfer.identifiers.CanonicalIdentifier, raiden.utils.Address, raiden.utils.BlockIdentifier]) -> ReceiveUnlock:
    mirrored_balance_proof = replace(source.event.balance_proof, canonical_identifier=canonical_identifier)
    signed_balance_proof = factories.make_signed_balance_proof_from_unsigned(unsigned=mirrored_balance_proof, signer=LocalSigner(source.private_key))
    return ReceiveUnlock(sender=source.node, message_identifier=source.event.message_identifier, secret=source.event.secret, secrethash=source.event.secrethash, balance_proof=signed_balance_proof)

@dataclass
class Scrambling:

    @property
    def kwargs(self) -> dict:
        return {self.field: self.value}

@composite
def scrambling(draw: Union[list[str], typing.Collection], fields: Any) -> Scrambling:
    field = draw(sampled_from(list(fields.keys())))
    value = draw(fields[field])
    return Scrambling(field, value)

@composite
def balance_proof_scrambling(draw: Any) -> Union[bool, bytes, Address]:
    fields = {'nonce': builds(factories.make_nonce), 'transferred_amount': integers(min_value=0), 'locked_amount': integers(min_value=0), 'locksroot': builds(factories.make_locksroot), 'canonical_identifier': builds(factories.make_canonical_identifier), 'balance_hash': builds(factories.make_transaction_hash)}
    return draw(scrambling(fields))

@composite
def hash_time_lock_scrambling(draw: Union[set[tuple[int]], str, int]) -> str:
    fields = {'amount': integers(min_value=0), 'expiration': integers(min_value=1), 'secrethash': builds(factories.make_secret_hash)}
    return draw(scrambling(fields))

@composite
def locked_transfer_scrambling(draw: Union[set[tuple[int]], tuple[frozenset], bytes]) -> Union[str, typing.Mapping]:
    fields = {'token': builds(factories.make_token_address), 'token_network_address': builds(factories.make_token_network_address), 'channel_identifier': builds(factories.make_channel_identifier), 'chain_id': builds(factories.make_chain_id)}
    return draw(scrambling(fields))