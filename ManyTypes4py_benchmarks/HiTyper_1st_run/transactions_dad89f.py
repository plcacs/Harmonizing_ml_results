from typing import Optional
from eth_keys.datatypes import PrivateKey
from eth_typing import Address
from eth_utils import int_to_big_endian
import rlp
from eth._utils.numeric import is_even
from eth._utils.transactions import create_transaction_signature, extract_chain_id, is_eip_155_signed_transaction
from eth.vm.forks.homestead.transactions import HomesteadTransaction, HomesteadUnsignedTransaction

class SpuriousDragonTransaction(HomesteadTransaction):

    @property
    def y_parity(self) -> int:
        if is_eip_155_signed_transaction(self):
            return int(is_even(self.v))
        else:
            return super().y_parity

    def get_message_for_signing(self):
        if is_eip_155_signed_transaction(self):
            txn_parts = rlp.decode(rlp.encode(self))
            txn_parts_for_signing = txn_parts[:-3] + [int_to_big_endian(self.chain_id), b'', b'']
            return rlp.encode(txn_parts_for_signing)
        else:
            return rlp.encode(SpuriousDragonUnsignedTransaction(nonce=self.nonce, gas_price=self.gas_price, gas=self.gas, to=self.to, value=self.value, data=self.data))

    @classmethod
    def create_unsigned_transaction(cls: Union[int, bytes], *, nonce: Union[int, bytes], gas_price: Union[int, bytes], gas: Union[int, bytes], to: Union[int, bytes], value: Union[int, bytes], data: Union[int, bytes]) -> SpuriousDragonUnsignedTransaction:
        return SpuriousDragonUnsignedTransaction(nonce, gas_price, gas, to, value, data)

    @property
    def chain_id(self) -> None:
        if is_eip_155_signed_transaction(self):
            return extract_chain_id(self.v)
        else:
            return None

    @property
    def v_min(self) -> int:
        if is_eip_155_signed_transaction(self):
            return 35 + 2 * self.chain_id
        else:
            return 27

    @property
    def v_max(self) -> int:
        if is_eip_155_signed_transaction(self):
            return 36 + 2 * self.chain_id
        else:
            return 28

class SpuriousDragonUnsignedTransaction(HomesteadUnsignedTransaction):

    def as_signed_transaction(self, private_key: Union[eth_keys.datatypes.PrivateKey, int, bytes], chain_id: Union[None, eth_keys.datatypes.PrivateKey, int, bytes]=None) -> SpuriousDragonTransaction:
        v, r, s = create_transaction_signature(self, private_key, chain_id=chain_id)
        return SpuriousDragonTransaction(nonce=self.nonce, gas_price=self.gas_price, gas=self.gas, to=self.to, value=self.value, data=self.data, v=v, r=r, s=s)