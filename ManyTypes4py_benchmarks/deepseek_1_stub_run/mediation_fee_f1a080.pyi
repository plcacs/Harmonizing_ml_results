```python
from fractions import Fraction
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

from raiden.exceptions import UndefinedMediationFee
from raiden.transfer.architecture import State
from raiden.utils.typing import Balance, FeeAmount, PaymentWithFeeAmount, ProportionalFeeAmount, TokenAmount

NUM_DISCRETISATION_POINTS: int = ...

class Interpolate:
    x_list: List[Fraction]
    y_list: List[Fraction]
    slopes: List[Fraction]
    
    def __init__(self, x_list: Sequence[Any], y_list: Sequence[Any]) -> None: ...
    def __call__(self, x: Any) -> Fraction: ...
    def __repr__(self) -> str: ...

def sign(x: Any) -> int: ...

def _collect_x_values(
    penalty_func_in: Interpolate,
    penalty_func_out: Interpolate,
    balance_in: Any,
    balance_out: Any,
    max_x: Any
) -> List[Fraction]: ...

def _cap_fees(x_list: List[Fraction], y_list: List[Fraction]) -> Tuple[List[Fraction], List[Fraction]]: ...

def _mediation_fee_func(
    schedule_in: Any,
    schedule_out: Any,
    balance_in: Any,
    balance_out: Any,
    receivable: Any,
    amount_with_fees: Optional[Any],
    amount_without_fees: Optional[Any],
    cap_fees: bool
) -> Interpolate: ...

T = TypeVar('T', bound='FeeScheduleState')

class FeeScheduleState(State):
    cap_fees: bool
    flat: FeeAmount
    proportional: ProportionalFeeAmount
    imbalance_penalty: Optional[List[Tuple[Any, Any]]]
    _penalty_func: Optional[Interpolate]
    
    def __post_init__(self) -> None: ...
    def _update_penalty_func(self) -> None: ...
    def fee(self, balance: Any, amount: Any) -> Fraction: ...
    
    @staticmethod
    def mediation_fee_func(
        schedule_in: 'FeeScheduleState',
        schedule_out: 'FeeScheduleState',
        balance_in: Any,
        balance_out: Any,
        receivable: Any,
        amount_with_fees: Any,
        cap_fees: bool
    ) -> Interpolate: ...
    
    @staticmethod
    def mediation_fee_backwards_func(
        schedule_in: 'FeeScheduleState',
        schedule_out: 'FeeScheduleState',
        balance_in: Any,
        balance_out: Any,
        receivable: Any,
        amount_without_fees: Any,
        cap_fees: bool
    ) -> Interpolate: ...

def linspace(start: TokenAmount, stop: TokenAmount, num: int) -> List[TokenAmount]: ...

def calculate_imbalance_fees(
    channel_capacity: TokenAmount,
    proportional_imbalance_fee: ProportionalFeeAmount
) -> Optional[List[Tuple[TokenAmount, FeeAmount]]]: ...
```