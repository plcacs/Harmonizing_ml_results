from bisect import bisect, bisect_right
from copy import copy
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, List, Optional, Sequence, Tuple, TypeVar, Union
from raiden.exceptions import UndefinedMediationFee
from raiden.transfer.architecture import State
from raiden.utils.typing import Balance, FeeAmount, PaymentWithFeeAmount, ProportionalFeeAmount, TokenAmount, typecheck

NUM_DISCRETISATION_POINTS: int

class Interpolate:
    x_list: List[Fraction]
    y_list: List[Fraction]
    slopes: List[Fraction]
    def __init__(self, x_list: Sequence[Union[Fraction, int, float]], y_list: Sequence[Union[Fraction, int, float]]) -> None: ...
    def __call__(self, x: Union[int, float]) -> Fraction: ...
    def __repr__(self) -> str: ...

def sign(x: Union[int, float, Fraction]) -> int: ...

def _collect_x_values(
    penalty_func_in: Interpolate,
    penalty_func_out: Interpolate,
    balance_in: Balance,
    balance_out: Balance,
    max_x: Balance,
) -> List[Fraction]: ...

def _cap_fees(x_list: List[Fraction], y_list: List[Fraction]) -> Tuple[List[Fraction], List[Fraction]]: ...

def _mediation_fee_func(
    schedule_in: 'FeeScheduleState',
    schedule_out: 'FeeScheduleState',
    balance_in: Balance,
    balance_out: Balance,
    receivable: Balance,
    amount_with_fees: Optional[PaymentWithFeeAmount],
    amount_without_fees: Optional[TokenAmount],
    cap_fees: bool,
) -> Interpolate: ...

T = TypeVar('T', bound='FeeScheduleState')

@dataclass
class FeeScheduleState(State):
    cap_fees: bool
    flat: FeeAmount
    proportional: ProportionalFeeAmount
    imbalance_penalty: Optional[Any]
    _penalty_func: Optional[Interpolate] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None: ...
    def _update_penalty_func(self) -> None: ...
    def fee(self, balance: Balance, amount: TokenAmount) -> Fraction: ...
    @staticmethod
    def mediation_fee_func(
        schedule_in: 'FeeScheduleState',
        schedule_out: 'FeeScheduleState',
        balance_in: Balance,
        balance_out: Balance,
        receivable: Balance,
        amount_with_fees: PaymentWithFeeAmount,
        cap_fees: bool,
    ) -> Interpolate: ...
    @staticmethod
    def mediation_fee_backwards_func(
        schedule_in: 'FeeScheduleState',
        schedule_out: 'FeeScheduleState',
        balance_in: Balance,
        balance_out: Balance,
        receivable: Balance,
        amount_without_fees: TokenAmount,
        cap_fees: bool,
    ) -> Interpolate: ...

def linspace(start: TokenAmount, stop: TokenAmount, num: int) -> List[TokenAmount]: ...

def calculate_imbalance_fees(
    channel_capacity: TokenAmount,
    proportional_imbalance_fee: ProportionalFeeAmount,
) -> Optional[List[Tuple[TokenAmount, FeeAmount]]]: ...