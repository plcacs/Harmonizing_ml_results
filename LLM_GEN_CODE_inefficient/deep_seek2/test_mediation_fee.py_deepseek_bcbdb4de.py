from fractions import Fraction
from typing import List, Optional, Tuple

import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis.strategies import integers

from raiden.tests.unit.transfer.test_channel import make_hash_time_lock_state
from raiden.tests.utils import factories
from raiden.tests.utils.factories import (
    NettingChannelEndStateProperties,
    NettingChannelStateProperties,
)
from raiden.tests.utils.mediation_fees import (
    get_amount_with_fees,
    get_initial_amount_for_amount_after_fees,
)
from raiden.transfer.mediated_transfer.initiator import calculate_safe_amount_with_fee
from raiden.transfer.mediated_transfer.mediation_fee import (
    NUM_DISCRETISATION_POINTS,
    FeeScheduleState,
    Interpolate,
    calculate_imbalance_fees,
    linspace,
)
from raiden.transfer.mediated_transfer.mediator import get_amount_without_fees
from raiden.transfer.state import NettingChannelState
from raiden.utils.mediation_fees import ppm_fee_per_channel
from raiden.utils.typing import (
    Balance,
    FeeAmount,
    PaymentAmount,
    PaymentWithFeeAmount,
    ProportionalFeeAmount,
    TokenAmount,
)


def test_interpolation() -> None:
    interp = Interpolate((0, 100), (0, 100))
    for i in range(101):
        assert interp(i) == i

    interp = Interpolate((0, 50, 100), (0, 100, 200))
    for i in range(101):
        assert interp(i) == 2 * i

    interp = Interpolate((0, 50, 100), (0, -50, 50))
    assert interp(40) == -40
    assert interp(60) == -30
    assert interp(90) == 30
    assert interp(99) == 48

    interp = Interpolate((0, 100), (Fraction("12.35"), Fraction("67.2")))
    assert interp(0) == Fraction("12.35")
    assert interp(50) == pytest.approx((12.35 + 67.2) / 2)
    assert interp(100) == Fraction("67.2")


def test_imbalance_penalty() -> None:
    v_schedule = FeeScheduleState(
        imbalance_penalty=[
            (TokenAmount(0), FeeAmount(10)),
            (TokenAmount(50), FeeAmount(0)),
            (TokenAmount(100), FeeAmount(20)),
        ]
    )
    reverse_schedule = FeeScheduleState(
        imbalance_penalty=[
            (TokenAmount(0), FeeAmount(20)),
            (TokenAmount(50), FeeAmount(0)),
            (TokenAmount(100), FeeAmount(10)),
        ]
    )

    for cap_fees, x1, amount, expected_fee_in, expected_fee_out in [
        (False, 0, 50, -8, -10),
        (False, 50, 30, 20, 12),
        (False, 0, 10, -2, -2),
        (False, 10, 10, -2, -2),
        (False, 0, 20, -3, -4),
        (False, 40, 15, 0, 0),
        (False, 50, 31, None, 12),
        (False, 100, 1, None, None),
        (True, 0, 50, 0, 0),
        (True, 50, 30, 20, 12),
        (True, 0, 10, 0, 0),
        (True, 10, 10, 0, 0),
        (True, 0, 20, 0, 0),
        (True, 40, 15, 0, 0),
    ]:
        v_schedule.cap_fees = cap_fees
        amount_with_fees = get_amount_with_fees(
            amount_without_fees=PaymentWithFeeAmount(amount),
            balance_in=Balance(x1),
            balance_out=Balance(100),
            schedule_in=v_schedule,
            schedule_out=FeeScheduleState(cap_fees=cap_fees),
            receivable_amount=TokenAmount(100 - x1),
        )
        if expected_fee_in is None:
            assert amount_with_fees is None
        else:
            assert amount_with_fees is not None
            assert amount_with_fees - amount == FeeAmount(expected_fee_in)

        reverse_schedule.cap_fees = cap_fees
        amount_with_fees = get_amount_with_fees(
            amount_without_fees=PaymentWithFeeAmount(amount),
            balance_in=Balance(0),
            balance_out=Balance(100 - x1),
            schedule_in=FeeScheduleState(cap_fees=cap_fees),
            schedule_out=reverse_schedule,
            receivable_amount=TokenAmount(100),
        )
        if expected_fee_out is None:
            assert amount_with_fees is None
        else:
            assert amount_with_fees is not None
            assert amount_with_fees - amount == FeeAmount(expected_fee_out)


def test_fee_capping() -> None:
    schedule = FeeScheduleState(
        imbalance_penalty=[(TokenAmount(0), FeeAmount(0)), (TokenAmount(100), FeeAmount(20))],
        flat=FeeAmount(5),
    )
    fee_func = FeeScheduleState.mediation_fee_func(
        schedule_in=FeeScheduleState(),
        schedule_out=schedule,
        balance_in=Balance(0),
        balance_out=Balance(100),
        receivable=TokenAmount(100),
        amount_with_fees=PaymentWithFeeAmount(5),
        cap_fees=True,
    )
    assert fee_func(30) == 0
    assert fee_func(20) == 5 - 4


def test_linspace() -> None:
    assert linspace(TokenAmount(0), TokenAmount(4), 5) == [0, 1, 2, 3, 4]
    assert linspace(TokenAmount(0), TokenAmount(4), 4) == [0, 1, 3, 4]
    assert linspace(TokenAmount(0), TokenAmount(4), 3) == [0, 2, 4]
    assert linspace(TokenAmount(0), TokenAmount(4), 2) == [0, 4]
    assert linspace(TokenAmount(0), TokenAmount(0), 3) == [0, 0, 0]

    with pytest.raises(AssertionError):
        assert linspace(TokenAmount(0), TokenAmount(4), 1)

    with pytest.raises(AssertionError):
        assert linspace(TokenAmount(4), TokenAmount(0), 2)


def test_rebalancing_fee_calculation() -> None:
    sample = calculate_imbalance_fees(TokenAmount(200), ProportionalFeeAmount(50_000))
    assert sample is not None
    assert len(sample) == NUM_DISCRETISATION_POINTS
    assert all(0 <= x <= 200 for x, _ in sample)
    assert max(x for x, _ in sample) == 200
    assert all(0 <= y <= 10 for _, y in sample)
    assert max(y for _, y in sample) == 10

    sample = calculate_imbalance_fees(TokenAmount(100), ProportionalFeeAmount(20_000))
    assert sample is not None
    assert len(sample) == NUM_DISCRETISATION_POINTS
    assert all(0 <= x <= 100 for x, _ in sample)
    assert max(x for x, _ in sample) == 100
    assert all(0 <= y <= 2 for _, y in sample)
    assert max(y for _, y in sample) == 2

    sample = calculate_imbalance_fees(TokenAmount(15), ProportionalFeeAmount(50_000))
    assert sample is not None
    assert len(sample) == 16
    assert all(0 <= x <= 16 for x, _ in sample)
    assert max(x for x, _ in sample) == 15
    assert all(0 <= y <= 1 for _, y in sample)
    assert max(y for _, y in sample) == 1

    sample = calculate_imbalance_fees(TokenAmount(1000), ProportionalFeeAmount(5_490))
    assert sample is not None
    assert len(sample) == NUM_DISCRETISATION_POINTS
    assert all(0 <= x <= 1000 for x, _ in sample)
    assert max(x for x, _ in sample) == 1000
    assert all(0 <= y <= 5 for _, y in sample)
    assert max(y for _, y in sample) == 5

    sample = calculate_imbalance_fees(TokenAmount(1000), ProportionalFeeAmount(5_500))
    assert sample is not None
    assert len(sample) == NUM_DISCRETISATION_POINTS
    assert all(0 <= x <= 1000 for x, _ in sample)
    assert max(x for x, _ in sample) == 1000
    assert all(0 <= y <= 6 for _, y in sample)
    assert max(y for _, y in sample) == 6

    assert calculate_imbalance_fees(TokenAmount(0), ProportionalFeeAmount(1)) is None
    assert calculate_imbalance_fees(TokenAmount(10), ProportionalFeeAmount(0)) is None


@pytest.mark.parametrize(
    "flat_fee, prop_fee, initial_amount, expected_amount",
    [
        (50, 0, 1000, 1000 - 50 - 50),
        (0, 1_000_000, 2000, 1000),
        (0, 100_000, 1100, 1000),
        (0, 50_000, 1050, 1000),
        (0, 10_000, 1010, 1000),
        (0, 10_000, 101, 100),
        (0, 4_990, 100, 100),
        (1, 500_000, 1000 + 500 + 2, 1000),
        (10, 500_000, 1000 + 500 + 20, 997),
        (100, 500_000, 1000 + 500 + 200, 967),
        (1, 100_000, 1000 + 100 + 2, 1000),
        (10, 100_000, 1000 + 100 + 20, 999),
        (100, 100_000, 1000 + 100 + 200, 991),
        (1, 10_000, 1000 + 10 + 2, 1000),
        (10, 10_000, 1000 + 10 + 20, 1000),
        (100, 10_000, 1000 + 10 + 200, 999),
        (100, 500_000, 1000 + 750, 1000),
        (0, 200_000, 47 + 9, 47),
        (0, 200_000, 39 + 8, 39),
    ],
)
def test_get_lock_amount_after_fees(
    flat_fee: int, prop_fee: int, initial_amount: int, expected_amount: int
) -> None:
    prop_fee_per_channel = ppm_fee_per_channel(ProportionalFeeAmount(prop_fee))
    lock = make_hash_time_lock_state(amount=initial_amount)
    channel_in = factories.create(
        NettingChannelStateProperties(
            partner_state=NettingChannelEndStateProperties(balance=TokenAmount(2000)),
            fee_schedule=FeeScheduleState(flat=flat_fee, proportional=prop_fee_per_channel),
        )
    )
    channel_out = factories.create(
        NettingChannelStateProperties(
            our_state=NettingChannelEndStateProperties(balance=TokenAmount(2000)),
            fee_schedule=FeeScheduleState(flat=flat_fee, proportional=prop_fee_per_channel),
        )
    )

    locked_after_fees = get_amount_without_fees(
        amount_with_fees=lock.amount, channel_in=channel_in, channel_out=channel_out
    )
    assert locked_after_fees == expected_amount


@pytest.mark.parametrize(
    "cap_fees, flat_fee, prop_fee, imbalance_fee, initial_amount, expected_amount",
    [
        (False, 0, 0, 10_000, 50_000, 50_000 + 2_000),
        (False, 0, 0, 20_000, 50_000, 50_000 + 3_995),
        (False, 0, 0, 30_000, 50_000, 50_000 + 5_910),
        (False, 0, 0, 40_000, 50_000, 50_000 + 7_613),
        (False, 0, 0, 50_000, 50_000, 50_000 + 9_091),
        (True, 0, 0, 10_000, 50_000, 50_000),
        (True, 0, 0, 20_000, 50_000, 50_000),
        (True, 0, 0, 30_000, 50_000, 50_000),
        (True, 0, 0, 40_000, 50_000, 50_000),
        (True, 0, 0, 50_000, 50_000, 50_000),
    ],
)
def test_get_lock_amount_after_fees_imbalanced_channel(
    cap_fees: bool,
    flat_fee: int,
    prop_fee: int,
    imbalance_fee: int,
    initial_amount: int,
    expected_amount: int,
) -> None:
    balance = TokenAmount(100_000)
    prop_fee_per_channel = ppm_fee_per_channel(ProportionalFeeAmount(prop_fee))
    imbalance_fee = calculate_imbalance_fees(
        channel_capacity=balance, proportional_imbalance_fee=ProportionalFeeAmount(imbalance_fee)
    )
    lock = make_hash_time_lock_state(amount=initial_amount)
    channel_in = factories.create(
        NettingChannelStateProperties(
            our_state=NettingChannelEndStateProperties(balance=TokenAmount(0)),
            partner_state=NettingChannelEndStateProperties(balance=balance),
            fee_schedule=FeeScheduleState(
                cap_fees=cap_fees,
                flat=FeeAmount(flat_fee),
                proportional=prop_fee_per_channel,
                imbalance_penalty=imbalance_fee,
            ),
        )
    )
    channel_out = factories.create(
        NettingChannelStateProperties(
            our_state=NettingChannelEndStateProperties(balance=balance),
            partner_state=NettingChannelEndStateProperties(balance=TokenAmount(0)),
            fee_schedule=FeeScheduleState(
                cap_fees=cap_fees,
                flat=FeeAmount(flat_fee),
                proportional=prop_fee_per_channel,
                imbalance_penalty=imbalance_fee,
            ),
        )
    )

    locked_after_fees = get_amount_without_fees(
        amount_with_fees=lock.amount, channel_in=channel_in, channel_out=channel_out
    )
    assert locked_after_fees == expected_amount


@given(
    integers(min_value=0, max_value=100),
    integers(min_value=0, max_value=10_000),
    integers(min_value=0, max_value=50_000),
    integers(min_value=1, max_value=90_000_000_000_000_000),
    integers(min_value=1, max_value=100_000_000_000_000_000),
    integers(min_value=1, max_value=100_000_000_000_000_000),
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_fee_round_trip(
    flat_fee: int, prop_fee: int, imbalance_fee: int, amount: int, balance1: int, balance2: int
) -> None:
    amount = int(min(amount, balance1 * 0.95 - 1, balance2 * 0.95 - 1))
    assume(amount > 0)

    total_balance = TokenAmount(100_000_000_000_000_000_000)
    prop_fee_per_channel = ppm_fee_per_channel(ProportionalFeeAmount(prop_fee))
    imbalance_fee = calculate_imbalance_fees(
        channel_capacity=total_balance,
        proportional_imbalance_fee=ProportionalFeeAmount(imbalance_fee),
    )
    channel_in = factories.create(
        NettingChannelStateProperties(
            our_state=NettingChannelEndStateProperties(balance=total_balance - balance1),
            partner_state=NettingChannelEndStateProperties(balance=balance1),
            fee_schedule=FeeScheduleState(
                cap_fees=False,
                flat=FeeAmount(flat_fee),
                proportional=prop_fee_per_channel,
                imbalance_penalty=imbalance_fee,
            ),
        )
    )
    channel_out = factories.create(
        NettingChannelStateProperties(
            our_state=NettingChannelEndStateProperties(balance=balance2),
            partner_state=NettingChannelEndStateProperties(balance=total_balance - balance2),
            fee_schedule=FeeScheduleState(
                cap_fees=False,
                flat=FeeAmount(flat_fee),
                proportional=prop_fee_per_channel,
                imbalance_penalty=imbalance_fee,
            ),
        )
    )

    fee_calculation = get_initial_amount_for_amount_after_fees(
        amount_after_fees=PaymentAmount(amount), channels=[(channel_in, channel_out)]
    )
    assume(fee_calculation)
    assert fee_calculation

    amount_with_margin = calculate_safe_amount_with_fee(
        fee_calculation.amount_without_fees, FeeAmount(sum(fee_calculation.mediation_fees))
    )
    amount_without_margin_after_fees = get_amount_without_fees(
        amount_with_fees=fee_calculation.total_amount,
        channel_in=channel_in,
        channel_out=channel_out,
    )
    assume(amount_without_margin_after_fees)
    assert abs(amount - amount_without_margin_after_fees) <= 1

    amount_with_fee_and_margin = calculate_safe_amount_with_fee(
        fee_calculation.amount_without_fees, FeeAmount(sum(fee_calculation.mediation_fees))
    )
    amount_with_margin_after_fees = get_amount_without_fees(
        amount_with_fees=amount_with_fee_and_margin, channel_in=channel_in, channel_out=channel_out
    )
    assume(amount_with_margin_after_fees)
    assert amount_with_margin_after_fees >= amount


@example(flat_fee=0, prop_fee=0, imbalance_fee=1277, amount=1, balance1=33, balance2=481)
@given(
    integers(min_value=0, max_value=100),
    integers(min_value=0, max_value=10_000),
    integers(min_value=0, max_value=50_000),
    integers(min_value=1, max_value=90_000_000_000_000_000_000),
    integers(min_value=1, max_value=100_000_000_000_000_000_000),
    integers(min_value=1, max_value=100_000_000_000_000_000_000),
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_fee_add_remove_invariant(
    flat_fee: int, prop_fee: int, imbalance_fee: int, amount: int, balance1: int, balance2: int
) -> None:
    total_balance = TokenAmount(100_000_000_000_000_000_000)
    prop_fee_per_channel = ppm_fee_per_channel(ProportionalFeeAmount(prop_fee))
    imbalance_fee = calculate_imbalance_fees(
        channel_capacity=total_balance,
        proportional_imbalance_fee=ProportionalFeeAmount(imbalance_fee),
    )
    fee_schedule = FeeScheduleState(
        cap_fees=False,
        flat=FeeAmount(flat_fee),
        proportional=prop_fee_per_channel,
        imbalance_penalty=imbalance_fee,
    )
    channel_in = factories.create(
        NettingChannelStateProperties(
            our_state=NettingChannelEndStateProperties(balance=total_balance - balance1),
            partner_state=NettingChannelEndStateProperties(balance=balance1),
            fee_schedule=fee_schedule,
        )
    )
    channel_out = factories.create(
        NettingChannelStateProperties(
            our_state=NettingChannelEndStateProperties(balance=balance2),
            partner_state=NettingChannelEndStateProperties(balance=total_balance - balance2),
            fee_schedule=fee_schedule,
        )
    )

    amount_with_fees = get_amount_with_fees(
        amount_without_fees=amount,
        schedule_in=channel_in.fee_schedule,
        schedule_out=channel_out.fee_schedule,
        receivable_amount=balance1,
        balance_in=total_balance - balance1,
        balance_out=balance2,
    )
    assume(amount_with_fees)
    assert amount_with_fees
    amount_without_fees = get_amount_without_fees(
        amount_with_fees=amount_with_fees, channel_in=channel_in, channel_out=channel_out
    )
    assume(amount_without_fees)
    assert amount - 1 <= amount_without_fees <= amount + 1


def running_sum(a: List[int]) -> List[int]:
    total = 0
    for item in a:
        total += item
        yield total


def make_channel_pair(
    fee_schedule: FeeScheduleState, balance1: int = 0, balance2: int = 0
) -> Tuple[NettingChannelState, NettingChannelState]:
    balance1 = TokenAmount(balance1)
    balance2 = TokenAmount(balance2)
    return (
        factories.create(
            NettingChannelStateProperties(
                our_state=NettingChannelEndStateProperties(balance=balance2),
                partner_state=NettingChannelEndStateProperties(balance=balance1),
                fee_schedule=fee_schedule,
            )
        ),
        factories.create(
            NettingChannelStateProperties(
                our_state=NettingChannelEndStateProperties(balance=balance1),
                partner_state=NettingChannelEndStateProperties(balance=balance2),
                fee_schedule=fee_schedule,
            )
        ),
    )


def test_mfee1() -> None:
    amount = 10_000
    deposit = 100_000
    flat_fee = 100 // 2
    fee_schedule = FeeScheduleState(flat=FeeAmount(flat_fee))
    channels = make_channel_pair(fee_schedule, deposit)

    fee_calculation = get_initial_amount_for_amount_after_fees(
        amount_after_fees=PaymentAmount(amount), channels=[channels, channels]
    )
    assert fee_calculation
    amount_with_margin = calculate_safe_amount_with_fee(
        fee_calculation.amount_without_fees, FeeAmount(sum(fee_calculation.mediation_fees))
    )
    assert amount_with_margin == 10_211

    print(deposit - amount_with_margin, amount_with_margin)
    for med_fee in running_sum(fee_calculation.mediation_fees):
        print(deposit - amount_with_margin + med_fee, amount_with_margin - med_fee)


def test_mfee2() -> None:
    amount = 10_000
    deposit = 100_000
    prop_fee = ppm_fee_per_channel(ProportionalFeeAmount(10_000))
    fee_schedule = FeeScheduleState(proportional=ProportionalFeeAmount(prop_fee))
    channels = make_channel_pair(fee_schedule, deposit)

    fee_calculation = get_initial_amount_for_amount_after_fees(
        amount_after_fees=PaymentAmount(amount), channels=[channels, channels]
    )
    assert fee_calculation
    amount_with_margin = calculate_safe_amount_with_fee(
        fee_calculation.amount_without_fees, FeeAmount(sum(fee_calculation.mediation_fees))
    )
    assert amount_with_margin == 10_213

    print(deposit - amount_with_margin, amount_with_margin)
    for med_fee in running_sum(fee_calculation.mediation_fees):
        print(deposit - amount_with_margin + med_fee, amount_with_margin - med_fee)


def test_mfee3() -> None:
    amount = 500_000_000_000_000_000
    deposit = TokenAmount(1_000_000_000_000_000_000)
    imbalance_penalty = calculate_imbalance_fees(deposit, ProportionalFeeAmount(10_000))
    fee_schedule = FeeScheduleState(imbalance_penalty=imbalance_penalty, cap_fees=False)
    channels = make_channel_pair(fee_schedule, deposit)

    fee_calculation = get_initial_amount_for_amount_after_fees(
        amount_after_fees=PaymentAmount(amount), channels=[channels]
    )
    assert fee_calculation
    amount_with_margin = calculate_safe_amount_with_fee(
        fee_calculation.amount_without_fees, FeeAmount(sum(fee_calculation.mediation_fees))
    )
    assert amount_with_margin == 480_850_038_799_922_400

    print("{:_} {:_}".format(deposit - amount_with_margin, amount_with_margin))
    for med_fee in running_sum(fee_calculation.mediation_fees):
        print(
            "{:_} {:_}".format(
                deposit - amount_with_margin + med_fee, amount_with_margin - med_fee
            )
        )


def test_mfee4() -> None:
    amount = PaymentAmount(500_000_000_000_000_000)
    deposit = 1_000_000_000_000_000_000
    prop_fee = ppm_fee_per_channel(ProportionalFeeAmount(10_000))
    imbalance_penalty = calculate_imbalance_fees(
        TokenAmount(deposit * 2), ProportionalFeeAmount(20_000)
    )
    fee_schedule = FeeScheduleState(
        flat=FeeAmount(100 // 2),
        proportional=prop_fee,
        imbalance_penalty=imbalance_penalty,
        cap_fees=False,
    )
    channels = make_channel_pair(fee_schedule, deposit, deposit)

    fee_calculation = get_initial_amount_for_amount_after_fees(
        amount_after_fees=PaymentAmount(amount), channels=[channels, channels]
    )
    assert fee_calculation

    amount_with_margin = calculate_safe_amount_with_fee(
        amount, FeeAmount(sum(fee_calculation.mediation_fees))
    )

    med_fees = []
    incoming_amount = amount_with_margin
    for _ in range(2):
        outgoing_amount = get_amount_without_fees(
            amount_with_fees=incoming_amount, channel_in=channels[0], channel_out=channels[1]
        )
        assert outgoing_amount
        med_fees.append(incoming_amount - outgoing_amount)
        incoming_amount = outgoing_amount

    assert amount_with_margin == 543_503_066_141_505_551

    print("{:_} {:_}".format(deposit - amount_with_margin, deposit + amount_with_margin))
    for med_fee in running_sum(med_fees):
        print(
            "{:_} {:_}".format(
                deposit - amount_with_margin + med_fee, deposit + amount_with_margin - med_fee
            )
        )
