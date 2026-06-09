# === Internal dependency: raiden.exceptions ===
class UndefinedMediationFee(RaidenError): ...

# === Internal dependency: raiden.transfer.architecture ===
class State:
    ...

# === Internal dependency: raiden.utils.typing ===
def typecheck(value, expected): ...
from raiden_contracts.utils.type_aliases import TokenAmount
T_FeeAmount = int
FeeAmount = NewType(...)
T_ProportionalFeeAmount = int
ProportionalFeeAmount = NewType(...)