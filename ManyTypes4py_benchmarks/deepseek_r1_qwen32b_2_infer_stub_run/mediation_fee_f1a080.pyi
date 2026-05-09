from __future__ import annotations
from fractions import Fraction
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from bisect import bisect_right
from dataclasses import dataclass
from raiden.exceptions import UndefinedMediationFee
from raiden.transfer.architecture import State
from raiden.utils.typing import (
    Balance,
    FeeAmount,
    PaymentWithFeeAmount,
    ProportionalFeeAmount,
    TokenAmount,
)

T = TypeVar('T', bound='FeeScheduleState')

class Interpolate:
    def __init__(self, x_list: List[Fraction], y_list: List[Fraction]) -> None:
        ...
    
    def __call__(self, x: Fraction) -> Fraction:
        ...
    
    def __repr__(self) -> str:
        ...

def sign(x: Union[int, float]) -> int:
    ...

def _collect_x_values(
    penalty_func_in: Interpolate,
    penalty_func_out: Interpolate,
    balance_in: TokenAmount,
    balance_out: TokenAmount,
    max_x: int,
) -> List[Fraction]:
    ...

def _cap_fees(
    x_list: List[Fraction],
    y_list: List[Fraction],
) -> Tuple[List[Fraction], List[Fraction]]:
    ...

def _mediation_fee_func(
    schedule_in: FeeScheduleState,
    schedule_out: FeeScheduleState,
    balance_in: TokenAmount,
    balance_out: TokenAmount,
    receivable: TokenAmount,
    amount_with_fees: Optional[Union[TokenAmount, Fraction]],
    amount_without_fees: Optional[Union[TokenAmount, Fraction]],
    cap_fees: bool,
) -> Interpolate:
    ...

@dataclass
class FeeScheduleState(State):
    cap_fees: bool
    flat: FeeAmount
    proportional: ProportionalFeeAmount
    imbalance_penalty: Optional[List[Tuple[TokenAmount, FeeAmount]]]
    _penalty_func: Optional[Interpolate] = None

    def fee(self, balance: TokenAmount, amount: TokenAmount) -> Fraction:
        ...

    @staticmethod
    def mediation_fee_func(
        schedule_in: FeeScheduleState,
        schedule_out: FeeScheduleState,
        balance_in: TokenAmount,
        balance_out: TokenAmount,
        receivable: TokenAmount,
        amount_with_fees: Optional[TokenAmount],
        cap_fees: bool,
    ) -> Interpolate:
        ...

    @staticmethod
    def mediation_fee_backwards_func(
        schedule_in: FeeScheduleState,
        schedule_out: FeeScheduleState,
        balance_in: TokenAmount,
        balance_out: TokenAmount,
        receivable: TokenAmount,
        amount_without_fees: Optional[TokenAmount],
        cap_fees: bool,
    ) -> Interpolate:
        ...

def linspace(
    start: TokenAmount,
    stop: TokenAmount,
    num: int,
) -> List[TokenAmount]:
    ...

def calculate_imbalance_fees(
    channel_capacity: TokenAmount,
    proportional_imbalance_fee: int,
) -> Optional[List[Tuple[TokenAmount, FeeAmount]]]:
    ...