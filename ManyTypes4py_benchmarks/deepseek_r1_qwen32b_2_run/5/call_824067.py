from abc import ABC, abstractmethod
from typing import Tuple, Union
from eth_typing import Address
from eth import constants
from eth._utils.address import force_bytes_to_address
from eth.abc import ComputationAPI
from eth.exceptions import OutOfGas, WriteProtection
from eth.vm.opcode import Opcode

CallParams = Tuple[int, int, Address, Address, Address, int, int, int, int, bool, bool]

class BaseCall(Opcode, ABC):
    def __call__(self, computation: ComputationAPI) -> None:
        pass

    @abstractmethod
    def compute_msg_extra_gas(self, computation: ComputationAPI, gas: int, to: Address, value: int) -> int:
        pass

    @abstractmethod
    def get_call_params(self, computation: ComputationAPI) -> CallParams:
        pass

    def compute_msg_gas(self, computation: ComputationAPI, gas: int, to: Address, value: int) -> Tuple[int, int]:
        pass

    def get_account_load_fee(self, computation: ComputationAPI, code_address: Address) -> int:
        pass

class Call(BaseCall):
    def compute_msg_extra_gas(self, computation: ComputationAPI, gas: int, to: Address, value: int) -> int:
        pass

    def get_call_params(self, computation: ComputationAPI) -> CallParams:
        pass

class CallCode(BaseCall):
    def compute_msg_extra_gas(self, computation: ComputationAPI, gas: int, to: Address, value: int) -> int:
        pass

    def get_call_params(self, computation: ComputationAPI) -> CallParams:
        pass

class DelegateCall(BaseCall):
    def compute_msg_gas(self, computation: ComputationAPI, gas: int, to: Address, value: int) -> Tuple[int, int]:
        pass

    def compute_msg_extra_gas(self, computation: ComputationAPI, gas: int, to: Address, value: int) -> int:
        pass

    def get_call_params(self, computation: ComputationAPI) -> CallParams:
        pass

class CallEIP150(Call):
    def compute_msg_gas(self, computation: ComputationAPI, gas: int, to: Address, value: int) -> Tuple[int, int]:
        pass

class CallCodeEIP150(CallCode):
    def compute_msg_gas(self, computation: ComputationAPI, gas: int, to: Address, value: int) -> Tuple[int, int]:
        pass

class DelegateCallEIP150(DelegateCall):
    def compute_msg_gas(self, computation: ComputationAPI, gas: int, to: Address, value: int) -> Tuple[int, int]:
        pass

def max_child_gas_eip150(gas: int) -> int:
    pass

def compute_eip150_msg_gas(*, computation: ComputationAPI, gas: int, extra_gas: int, value: int, mnemonic: str, callstipend: int) -> Tuple[int, int]:
    pass

class CallEIP161(CallEIP150):
    def compute_msg_extra_gas(self, computation: ComputationAPI, gas: int, to: Address, value: int) -> int:
        pass

class StaticCall(CallEIP161):
    def get_call_params(self, computation: ComputationAPI) -> CallParams:
        pass

class CallByzantium(CallEIP161):
    def get_call_params(self, computation: ComputationAPI) -> CallParams:
        pass