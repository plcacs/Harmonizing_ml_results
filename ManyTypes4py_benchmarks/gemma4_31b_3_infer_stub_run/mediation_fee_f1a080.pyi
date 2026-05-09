from fractions import Fraction
from typing import List, Optional, Sequence, Tuple, TypeVar, Union
from raiden.transfer.architecture import State
from raiden.utils.typing import Balance, FeeAmount, ProportionalFeeAmount, TokenAmount

NUM_DISCRETISATION_POINTS: int = ...

class Interpolate:
    """Linear interpolation of a function with given points

    Based on https://stackoverflow.com/a/7345691/114926
    """
    x_list: List[Fraction]
    y_list: List[Fraction]
    slopes: List[Fraction]

    def __init__(self, x_list: Sequence[Union[int, float, Fraction, TokenAmount]], y_list: Sequence[Union[int, float, Fraction, FeeAmount]]) -> None: ...
    def __call__(self, x: Union[int, float, Fraction, TokenAmount]) -> Fraction: ...
    def __repr__(self) -> str: ...

def sign(x: Union[int, float, Fraction]) -> int: ...

def _collect_x_values(
    penalty_func_in: Interpolate,
    penalty_func_out: Interpolate,
    balance_in: Union[int, float, Fraction, TokenAmount],
    balance_out: Union[int, float, Fraction, TokenAmount],
    max_x: Union[int, float, Fraction, TokenAmount]
) -> List[Fraction]: ...

def _cap_fees(x_list: List[Fraction], y_list: List[Fraction]) -> Tuple[List[Fraction], List[Fraction]]: ...

def _mediation_fee_func(
    schedule_in: 'FeeScheduleState',
    schedule_out: 'FeeScheduleState',
    balance_in: Union[int, float, Fraction, TokenAmount],
    balance_out: Union[int, float, Fraction, TokenAmount],
    receivable: Union[int, float, Fraction, TokenAmount],
    amount_with_fees: Optional[Union[int, float, Fraction, TokenAmount]],
    amount_without_fees: Optional[Union[int, float, Fraction, TokenAmount]],
    cap_fees: bool
) -> Interpolate: ...

T = TypeVar('T', bound='FeeScheduleState')

class FeeScheduleState(State):
    cap_fees: bool
    flat: FeeAmount
    proportional: ProportionalFeeAmount
    imbalance_penalty: Optional[List[Tuple[TokenAmount, FeeAmount]]]
    _penalty_func: Optional[Interpolate]

    def __post_init__(self) -> None: ...
    def _update_penalty_func(self) -> None: ...
    def fee(self, balance: Union[int, float, Fraction, TokenAmount], amount: Union[int, float, Fraction, TokenAmount]) -> Fraction: ...

    @staticmethod
    def mediation_fee_func(
        schedule_in: 'FeeScheduleState',
        schedule_out: 'FeeScheduleState',
        balance_in: Union[int, float, Fraction, TokenAmount],
        balance_out: Union[int, float, Fraction, TokenAmount],
        receivable: Union[int, float, Fraction, TokenAmount],
        amount_with_fees: Optional[Union[int, float, Fraction, TokenAmount]],
        cap_fees: bool
    ) -> Interpolate: ...

    @staticmethod
    def mediation_fee_backwards_func(
        schedule_in: 'FeeScheduleState',
        schedule_out: 'FeeScheduleState',
        balance_in: Union[int, float, Fraction, TokenAmount],
        balance_out: Union[int, float, Fraction, TokenAmount],
        receivable: Union[int, float, Fraction, TokenAmount],
        amount_without_fees: Optional[Union[int, float, Fraction, TokenAmount]],
        cap_fees: bool
    ) -> Interpolate: ...

def linspace(start: Union[int, float, Fraction, TokenAmount], stop: Union[int, float, Fraction, TokenAmount], num: int) -> List[TokenAmount]: ...

def calculate_imbalance_fees(channel_capacity: Union[int, float, Fraction, TokenAmount], proportional_imbalance_fee: Union[int, float, Fraction, ProportionalFeeAmount]) -> Optional[List[Tuple[TokenAmount, FeeAmount]]]: ...