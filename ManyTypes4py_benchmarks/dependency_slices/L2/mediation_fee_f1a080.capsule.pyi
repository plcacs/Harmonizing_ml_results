from typing import Any

# === Internal dependency: raiden.exceptions ===
class UndefinedMediationFee(RaidenError): ...

# === Internal dependency: raiden.transfer.architecture ===
class State:
    ...

# === Internal dependency: raiden.utils.typing ===
def typecheck(value: Any, expected: Union[Type, Tuple[Type, ...]]) -> None: ...
# re-export: from raiden_contracts.utils.type_aliases import TokenAmount
FeeAmount: NewType
ProportionalFeeAmount: NewType