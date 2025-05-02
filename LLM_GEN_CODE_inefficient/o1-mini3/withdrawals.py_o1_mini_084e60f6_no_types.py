from typing import cast, Optional, List, Tuple
from cached_property import cached_property
from eth_typing import Address, Hash32
from eth_utils import keccak
import rlp
from rlp.sedes import big_endian_int
from eth.abc import WithdrawalAPI
from eth.rlp.sedes import address
from eth.validation import validate_canonical_address, validate_uint64

class Withdrawal(rlp.Serializable):
    fields: List[Tuple[str, rlp.sedes.Serializable]] = [('index', big_endian_int), ('validator_index', big_endian_int), ('address', address), ('amount', big_endian_int)]

    def __init__(self, index=0, validator_index=0, address=None, amount=0):
        super().__init__(index=index, validator_index=validator_index, address=address, amount=amount)

    def validate(self):
        validate_uint64(self.index, 'Withdrawal.index')
        validate_uint64(self.validator_index, 'Withdrawal.validator_index')
        validate_canonical_address(self.address, 'Withdrawal.address')
        validate_uint64(self.amount, 'Withdrawal.amount')

    @classmethod
    def decode(cls, encoded):
        return rlp.decode(encoded, sedes=cls)

    def encode(self):
        return rlp.encode(self)

    @cached_property
    def hash(self):
        return cast(Hash32, keccak(self.encode()))