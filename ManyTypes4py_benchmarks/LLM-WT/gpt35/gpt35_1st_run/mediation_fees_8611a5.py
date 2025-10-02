from typing import Tuple, List, NamedTuple, Optional
from raiden.exceptions import UndefinedMediationFee
from raiden.transfer.state import NettingChannelState
from raiden.utils.typing import Balance, FeeAmount, PaymentAmount, PaymentWithFeeAmount, TokenAmount

def imbalance_fee_receiver(fee_schedule, amount, balance) -> FeeAmount:
    ...

def imbalance_fee_sender(fee_schedule, amount, balance) -> FeeAmount:
    ...

class FeesCalculation(NamedTuple):
    total_amount: PaymentAmount
    mediation_fees: List[FeeAmount]

def get_amount_with_fees(amount_without_fees: PaymentAmount, balance_in: Balance, balance_out: Balance, schedule_in, schedule_out, receivable_amount: TokenAmount) -> Optional[PaymentWithFeeAmount]:
    ...

def get_initial_amount_for_amount_after_fees(amount_after_fees: int, channels: List[Tuple[NettingChannelState, NettingChannelState]]) -> Optional[FeesCalculation]:
    ...

class PaymentAmountCalculation(NamedTuple):
    amount_to_send: PaymentAmount
    mediation_fees: List[FeeAmount]
    amount_with_fees: PaymentWithFeeAmount

def get_amount_for_sending_before_and_after_fees(amount_to_leave_initiator: int, channels: List[Tuple[NettingChannelState, NettingChannelState]]) -> Optional[PaymentAmountCalculation]:
    ...
