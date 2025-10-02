from bisect import bisect, bisect_right
from copy import copy
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple, TypeVar, Union

from raiden.exceptions import UndefinedMediationFee
from raiden.transfer.architecture import State
from raiden.utils.typing import (
    Balance,
    FeeAmount,
    PaymentWithFeeAmount,
    ProportionalFeeAmount,
    TokenAmount,
    typecheck,
)

NUM_DISCRETISATION_POINTS: int = 21


class Interpolate:
    """Linear interpolation of a function with given points

    Based on https://stackoverflow.com/a/7345691/114926
    """

    def __init__(self, x_list: Sequence[Union[int, float, Fraction]], y_list: Sequence[Union[int, float, Fraction]]) -> None:
        if any((y - x <= 0 for x, y in zip(x_list, x_list[1:]))):
            raise ValueError("x_list must be in strictly ascending order!")
        self.x_list: List[Fraction] = [Fraction(x) for x in x_list]
        self.y_list: List[Fraction] = [Fraction(y) for y in y_list]
        intervals = zip(self.x_list, self.x_list[1:], y_list, y_list[1:])
        self.slopes: List[Fraction] = [
            (Fraction(y2) - Fraction(y1)) / (Fraction(x2) - Fraction(x1))
            for x1, x2, y1, y2 in intervals
        ]

    def __call__(self, x: Union[int, float, Fraction]) -> Fraction:
        if not self.x_list[0] <= x <= self.x_list[-1]:
            raise ValueError("x out of bounds!")
        if x == self.x_list[-1]:
            return self.y_list[-1]
        i: int = bisect_right(self.x_list, x) - 1
        return self.y_list[i] + self.slopes[i] * (Fraction(x) - self.x_list[i])

    def __repr__(self) -> str:
        return f"Interpolate({self.x_list}, {self.y_list})"


def sign(x: Union[int, float]) -> int:
    """Sign of input, returns zero on zero input"""
    if x == 0:
        return 0
    else:
        return 1 if x > 0 else -1


def _collect_x_values(
    penalty_func_in: Interpolate,
    penalty_func_out: Interpolate,
    balance_in: Fraction,
    balance_out: Fraction,
    max_x: Fraction,
) -> List[Fraction]:
    """Normalizes the x-axis of the penalty functions around the amount of
    tokens being transferred.

    A penalty function maps the participant's balance to a fee. These functions
    are then used to penalize transfers that unbalance the node's channels, and
    as a consequence incentivize transfers that re-balance the channels.

    Here the x-axis of the penalty functions normalized around the current
    channel's capacity. So that instead of computing:

        penalty(current_capacity + amount_being_transferred)

    One can simply compute:
        penalty(amount_being_transferred)
    To find the penalty fee for the current transfer.
    """
    all_x_vals: List[Fraction] = [
        x - balance_in for x in penalty_func_in.x_list
    ] + [
        balance_out - x for x in penalty_func_out.x_list
    ]
    limited_x_vals = ( max(Fraction(0), min(x, balance_out, max_x)) for x in all_x_vals )
    return sorted(set(limited_x_vals))


def _cap_fees(x_list: List[Fraction], y_list: List[Fraction]) -> Tuple[List[Fraction], List[Fraction]]:
    """Insert extra points for intersections with x-axis, see `test_fee_capping`"""
    x_list = copy(x_list)
    y_list = copy(y_list)
    for i in range(len(x_list) - 1):
        y1, y2 = y_list[i : i + 2]
        if sign(float(y1)) * sign(float(y2)) == -1:
            x1, x2 = x_list[i : i + 2]
            new_x = x1 + abs(y1) / abs(y2 - y1) * (x2 - x1)
            new_index = bisect(x_list, new_x)
            x_list.insert(new_index, new_x)
            y_list.insert(new_index, Fraction(0))
    y_list = [max(y, Fraction(0)) for y in y_list]
    return (x_list, y_list)


def _mediation_fee_func(
    schedule_in: "FeeScheduleState",
    schedule_out: "FeeScheduleState",
    balance_in: Fraction,
    balance_out: Fraction,
    receivable: Fraction,
    amount_with_fees: Optional[Fraction],
    amount_without_fees: Optional[Fraction],
    cap_fees: bool,
) -> Interpolate:
    """Returns a function which calculates total_mediation_fee(x)

    Either `amount_with_fees` or `amount_without_fees` must be given while the
    other one is None. The returned function will depend on the value that is
    not given.
    """
    assert (
        amount_with_fees is None or amount_without_fees is None
    ), "Must be called with either amount_with_fees or amount_without_fees as None"
    if balance_out == 0 or receivable == 0:
        raise UndefinedMediationFee()
    if not schedule_in._penalty_func:
        schedule_in = copy(schedule_in)
        schedule_in._penalty_func = Interpolate([0, balance_in + receivable], [0, 0])
    if not schedule_out._penalty_func:
        schedule_out = copy(schedule_out)
        schedule_out._penalty_func = Interpolate([0, balance_out], [0, 0])
    x_list = _collect_x_values(
        penalty_func_in=schedule_in._penalty_func,
        penalty_func_out=schedule_out._penalty_func,
        balance_in=balance_in,
        balance_out=balance_out,
        max_x=receivable if amount_with_fees is None else balance_out,
    )
    try:
        y_list: List[Fraction] = [
            schedule_in.fee(balance_in, x if amount_with_fees is None else Fraction(amount_with_fees))
            + schedule_out.fee(balance_out, -x if amount_without_fees is None else -Fraction(amount_without_fees))
            for x in x_list
        ]
    except ValueError:
        raise UndefinedMediationFee()
    if cap_fees:
        x_list, y_list = _cap_fees(x_list, y_list)
    return Interpolate(x_list, y_list)


T = TypeVar("T", bound="FeeScheduleState")


@dataclass
class FeeScheduleState(State):
    cap_fees: bool = True
    flat: FeeAmount = FeeAmount(0)
    proportional: ProportionalFeeAmount = ProportionalFeeAmount(0)
    imbalance_penalty: Optional[List[Tuple[TokenAmount, FeeAmount]]] = None
    _penalty_func: Optional[Interpolate] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._update_penalty_func()

    def _update_penalty_func(self) -> None:
        if self.imbalance_penalty:
            typecheck(self.imbalance_penalty, list)
            x_list, y_list = tuple(zip(*self.imbalance_penalty))
            self._penalty_func = Interpolate(list(x_list), list(y_list))

    def fee(self, balance: Fraction, amount: Fraction) -> Fraction:
        penalty = (
            self._penalty_func(balance + amount) - self._penalty_func(balance)
            if self._penalty_func
            else Fraction(0)
        )
        return (
            Fraction(self.flat)
            + Fraction(self.proportional, 1_000_000) * Fraction(abs(amount))
            + penalty
        )

    @staticmethod
    def mediation_fee_func(
        schedule_in: "FeeScheduleState",
        schedule_out: "FeeScheduleState",
        balance_in: Fraction,
        balance_out: Fraction,
        receivable: Fraction,
        amount_with_fees: Fraction,
        cap_fees: bool,
    ) -> Interpolate:
        """Returns a function which calculates total_mediation_fee(amount_without_fees)"""
        return _mediation_fee_func(
            schedule_in=schedule_in,
            schedule_out=schedule_out,
            balance_in=balance_in,
            balance_out=balance_out,
            receivable=receivable,
            amount_with_fees=Fraction(amount_with_fees),
            amount_without_fees=None,
            cap_fees=cap_fees,
        )

    @staticmethod
    def mediation_fee_backwards_func(
        schedule_in: "FeeScheduleState",
        schedule_out: "FeeScheduleState",
        balance_in: Fraction,
        balance_out: Fraction,
        receivable: Fraction,
        amount_without_fees: Fraction,
        cap_fees: bool,
    ) -> Interpolate:
        """Returns a function which calculates total_mediation_fee(amount_with_fees)"""
        return _mediation_fee_func(
            schedule_in=schedule_in,
            schedule_out=schedule_out,
            balance_in=balance_in,
            balance_out=balance_out,
            receivable=receivable,
            amount_with_fees=None,
            amount_without_fees=Fraction(amount_without_fees),
            cap_fees=cap_fees,
        )


def linspace(start: TokenAmount, stop: TokenAmount, num: int) -> List[TokenAmount]:
    """Returns a list of num numbers from start to stop (inclusive)."""
    assert num > 1, "Must generate at least one step"
    assert start <= stop, "start must be smaller than stop"
    step: Fraction = Fraction(stop - start, num - 1)
    result: List[TokenAmount] = []
    for i in range(num):
        result.append(TokenAmount(start + round(i * step)))
    return result


def calculate_imbalance_fees(
    channel_capacity: TokenAmount, proportional_imbalance_fee: ProportionalFeeAmount
) -> Optional[List[Tuple[TokenAmount, FeeAmount]]]:
    """Calculates a U-shaped imbalance curve

    The penalty term takes the following value at the extrema:
    channel_capacity * (proportional_imbalance_fee / 1_000_000)
    """
    assert channel_capacity >= 0, "channel_capacity must be larger than zero"
    assert proportional_imbalance_fee >= 0, "prop. imbalance fee must be larger than zero"
    if proportional_imbalance_fee == 0:
        return None
    if channel_capacity == 0:
        return None
    MAXIMUM_SLOPE: float = 0.1
    max_imbalance_fee: float = channel_capacity * float(proportional_imbalance_fee) / 1_000_000.0
    assert (
        float(proportional_imbalance_fee) / 1_000_000.0 <= MAXIMUM_SLOPE / 2
    ), "Too high imbalance fee"
    s: float = MAXIMUM_SLOPE
    c: float = max_imbalance_fee
    o: float = channel_capacity / 2
    b: float = s * o / c
    b = min(b, 10)
    a: float = c / (o ** b)

    def f(x: TokenAmount) -> FeeAmount:
        return FeeAmount(int(round(a * abs(x - o) ** b))

        )

    num_base_points: int = min(NUM_DISCRETISATION_POINTS, channel_capacity + 1)
    x_values: List[TokenAmount] = linspace(TokenAmount(0), channel_capacity, num_base_points)
    y_values: List[FeeAmount] = [f(x) for x in x_values]
    return list(zip(x_values, y_values))
