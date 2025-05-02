from eth_utils import decode_hex, encode_hex
from raiden.constants import LOCKSROOT_OF_NO_LOCKS, UINT256_MAX
from raiden.transfer.utils import hash_balance_data
from raiden.utils.typing import BalanceHash, ChannelID, LockedAmount, Locksroot, TokenAmount, TokenNetworkAddress
from raiden_contracts.constants import MessageTypeId
LOCKSROOT_OF_NO_LOCKS_AS_STRING = encode_hex(LOCKSROOT_OF_NO_LOCKS)

class BalanceProof:
    """A Balance Proof

    If transferred_amount, locked_amount and locksroot are set, balance_proof hash is
    computed using these values. Otherwise a value stored in _balance_hash is returned.

    Serialization will also add these items only if each of transferred_amount, locked_amount
    and locksroot is set.
    """

    def __init__(self, channel_identifier, token_network_address, balance_hash=None, nonce=0, additional_hash='0x%064x' % 0, chain_id=1, signature=None, transferred_amount=None, locked_amount=LockedAmount(0), locksroot=LOCKSROOT_OF_NO_LOCKS):
        self.channel_identifier = channel_identifier
        self.token_network_address = token_network_address
        self._balance_hash = balance_hash
        self.additional_hash = additional_hash
        self.nonce = nonce
        self.chain_id = chain_id
        self.signature = signature
        if transferred_amount and locked_amount and locksroot and balance_hash:
            assert 0 <= transferred_amount <= UINT256_MAX
            assert 0 <= locked_amount <= UINT256_MAX
            assert self.hash_balance_data(transferred_amount, locked_amount, locksroot) == balance_hash
        self.transferred_amount = transferred_amount
        self.locked_amount = locked_amount
        self.locksroot = locksroot

    def serialize_bin(self, msg_type=MessageTypeId.BALANCE_PROOF):
        return self.token_network_address + self.chain_id.to_bytes(32, byteorder='big') + msg_type.value.to_bytes(32, byteorder='big') + self.channel_identifier.to_bytes(32, byteorder='big') + self.balance_hash + self.nonce.to_bytes(32, byteorder='big') + decode_hex(self.additional_hash)

    @property
    def balance_hash(self):
        if self._balance_hash:
            return self._balance_hash
        if None not in (self.transferred_amount, self.locked_amount, self.locksroot):
            assert isinstance(self.transferred_amount, int)
            return self.hash_balance_data(self.transferred_amount, self.locked_amount, self.locksroot)
        raise ValueError("Can't compute balance hash")

    @balance_hash.setter
    def balance_hash(self, value):
        self._balance_hash = value

    @staticmethod
    def hash_balance_data(transferred_amount, locked_amount, locksroot):
        return hash_balance_data(transferred_amount, locked_amount, locksroot)