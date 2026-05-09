from bisect import bisect_right
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple, TypeVar, Union
from raiden.transfer.architecture import State
from raiden.utils.typing import (
    Balance,
    FeeAmount,
    PaymentWithFeeAmount,
    ProportionalFeeAmount,
    TokenAmount,
)

class Interpolate:
    def __init__(self, x_list: List[Fraction], y_list: List[Fraction]) -> None: ...
    def __call__(self, x: Fraction) -> Fraction: ...
    def __repr__(self) -> str: ...

def sign(x: Union[int, float]) -> int: ...

def _collect_x_values(
    penalty_func_in: Interpolate,
    penalty_func_out: Interpolate,
    balance_in: Balance,
    balance_out: Balance,
    max_x: Balance,
) -> List[Fraction]: ...

def _cap_fees(
    x_list: List[Fraction],
    y_list: List[Fraction],
) -> Tuple[List[Fraction], List[Fraction]]: ...

T = TypeVar('T', bound='FeeScheduleState')

@dataclass
class FeeScheduleState(State):
    cap_fees: bool
    flat: FeeAmount
    proportional: ProportionalFeeAmount
    imbalance_penalty: Optional[List[Tuple[TokenAmount, FeeAmount]]]
    _penalty_func: Optional[Interpolate]

    def __post_init__(self) -> None: ...
    def _update_penalty_func(self) -> None: ...
    def fee(self, balance: Fraction, amount: Fraction) -> Fraction: ...
    @staticmethod
    def mediation_fee_func(
        schedule_in: FeeScheduleState,
        schedule_out: FeeScheduleState,
        balance_in: Balance,
        balance_out: Balance,
        receivable: TokenAmount,
        amount_with_fees: Optional[Union[int, float]],
        cap_fees: bool,
    ) -> Interpolate: ...
    @staticmethod
    def mediation_fee_backwards_func(
        schedule_in: FeeScheduleState,
        schedule_out: FeeScheduleState,
        balance_in: Balance,
        balance_out: Balance,
        receivable: TokenAmount,
        amount_without_fees: Optional[Union[int, float]],
        cap_fees: bool,
    ) -> Interpolate: ...

def linspace(
    start: TokenAmount,
    stop: TokenAmount,
    num: int,
) -> List[TokenAmount]: ...

def calculate_imbalance_fees(
    channel_capacity: TokenAmount,
    proportional_imbalance_fee: int,
) -> Optional[List[Tuple[TokenAmount, FeeAmount]]]: ...