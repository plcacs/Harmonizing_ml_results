from bisect import bisect_right
from copy import copy
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple, TypeVar, Union
from raiden.exceptions import UndefinedMediationFee
from raiden.transfer.architecture import State
from raiden.utils.typing import Balance, FeeAmount, PaymentWithFeeAmount, ProportionalFeeAmount, TokenAmount, typecheck

NUM_DISCRETISATION_POINTS: int = 21

class Interpolate:
    def __init__(self, x_list: List[Fraction], y_list: List[Fraction]) -> None:
    def __call__(self, x: Fraction) -> Fraction:
    def __repr__(self) -> str:

def sign(x: Fraction) -> int:

def _collect_x_values(penalty_func_in, penalty_func_out, balance_in, balance_out, max_x) -> List[Fraction]:

def _cap_fees(x_list: List[Fraction], y_list: List[Fraction]) -> Tuple[List[Fraction], List[Fraction]:

def _mediation_fee_func(schedule_in, schedule_out, balance_in, balance_out, receivable, amount_with_fees, amount_without_fees, cap_fees) -> Interpolate:

T = TypeVar('T', bound='FeeScheduleState')

@dataclass
class FeeScheduleState(State):
    cap_fees: bool = True
    flat: FeeAmount = FeeAmount(0)
    proportional: ProportionalFeeAmount = ProportionalFeeAmount(0)
    imbalance_penalty: Optional[List[Tuple[Fraction, Fraction]]] = None
    _penalty_func: Optional[Interpolate] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
    def _update_penalty_func(self) -> None:
    def fee(self, balance: TokenAmount, amount: TokenAmount) -> Fraction:
    @staticmethod
    def mediation_fee_func(schedule_in, schedule_out, balance_in, balance_out, receivable, amount_with_fees, cap_fees) -> Interpolate:
    @staticmethod
    def mediation_fee_backwards_func(schedule_in, schedule_out, balance_in, balance_out, receivable, amount_without_fees, cap_fees) -> Interpolate:

def linspace(start: TokenAmount, stop: TokenAmount, num: int) -> List[TokenAmount]:

def calculate_imbalance_fees(channel_capacity: TokenAmount, proportional_imbalance_fee: int) -> Optional[List[Tuple[TokenAmount, FeeAmount]]]:
