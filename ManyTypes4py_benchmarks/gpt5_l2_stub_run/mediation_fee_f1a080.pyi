from fractions import Fraction
from typing import List, Optional, Sequence, Tuple, TypeVar, Union
from raiden.transfer.architecture import State
from raiden.utils.typing import FeeAmount, ProportionalFeeAmount, TokenAmount

NUM_DISCRETISATION_POINTS: int = ...

class Interpolate:
    x_list: List[Fraction]
    y_list: List[Fraction]
    slopes: List[Fraction]

    def __init__(self, x_list: Sequence[Union[int, Fraction]], y_list: Sequence[Union[int, Fraction]]) -> None: ...
    def __call__(self, x: Union[int, Fraction]) -> Fraction: ...
    def __repr__(self) -> str: ...

def sign(x: Union[int, Fraction]) -> int: ...
def _collect_x_values(
    penalty_func_in: Interpolate,
    penalty_func_out: Interpolate,
    balance_in: int,
    balance_out: int,
    max_x: int,
) -> List[Fraction]: ...
def _cap_fees(x_list: Sequence[Fraction], y_list: Sequence[Fraction]) -> Tuple[List[Fraction], List[Fraction]]: ...
def _mediation_fee_func(
    schedule_in: "FeeScheduleState",
    schedule_out: "FeeScheduleState",
    balance_in: int,
    balance_out: int,
    receivable: int,
    amount_with_fees: Optional[int],
    amount_without_fees: Optional[int],
    cap_fees: bool,
) -> Interpolate: ...

T = TypeVar("T", bound="FeeScheduleState")

class FeeScheduleState(State):
    cap_fees: bool
    flat: FeeAmount
    proportional: ProportionalFeeAmount
    imbalance_penalty: Optional[List[Tuple[TokenAmount, FeeAmount]]]
    _penalty_func: Optional[Interpolate]

    def __post_init__(self) -> None: ...
    def _update_penalty_func(self) -> None: ...
    def fee(self, balance: int, amount: Union[int, Fraction]) -> Fraction: ...
    @staticmethod
    def mediation_fee_func(
        schedule_in: "FeeScheduleState",
        schedule_out: "FeeScheduleState",
        balance_in: int,
        balance_out: int,
        receivable: int,
        amount_with_fees: int,
        cap_fees: bool,
    ) -> Interpolate: ...
    @staticmethod
    def mediation_fee_backwards_func(
        schedule_in: "FeeScheduleState",
        schedule_out: "FeeScheduleState",
        balance_in: int,
        balance_out: int,
        receivable: int,
        amount_without_fees: int,
        cap_fees: bool,
    ) -> Interpolate: ...

def linspace(start: int, stop: int, num: int) -> List[TokenAmount]: ...
def calculate_imbalance_fees(
    channel_capacity: int, proportional_imbalance_fee: ProportionalFeeAmount
) -> Optional[List[Tuple[TokenAmount, FeeAmount]]]: ...