import numpy as np
from skopt.space import Integer
from typing import Any, List, Optional, Union


class SKDecimal(Integer):
    def __init__(
        self,
        low: float,
        high: float,
        decimals: int = 3,
        prior: str = "uniform",
        base: int = 10,
        transform: Optional[Any] = None,
        name: Optional[str] = None,
        dtype: Any = np.int64,
    ) -> None:
        self.decimals: int = decimals

        self.pow_dot_one: float = pow(0.1, self.decimals)
        self.pow_ten: float = pow(10, self.decimals)

        _low: int = int(low * self.pow_ten)
        _high: int = int(high * self.pow_ten)
        # trunc to precision to avoid points out of space
        self.low_orig: float = round(_low * self.pow_dot_one, self.decimals)
        self.high_orig: float = round(_high * self.pow_dot_one, self.decimals)

        super().__init__(_low, _high, prior, base, transform, name, dtype)

    def __repr__(self) -> str:
        return (
            f"Decimal(low={self.low_orig}, high={self.high_orig}, decimals={self.decimals}, "
            f"prior='{self.prior}', transform='{self.transform_}')"
        )

    def __contains__(self, point: Union[float, int, List[float], np.ndarray]) -> bool:
        if isinstance(point, list):
            point = np.array(point)
        return self.low_orig <= point <= self.high_orig

    def transform(self, Xt: List[float]) -> Any:
        int_converted: List[int] = [int(v * self.pow_ten) for v in Xt]
        return super().transform(int_converted)

    def inverse_transform(self, Xt: List[int]) -> List[float]:
        res: Any = super().inverse_transform(Xt)
        # equivalent to [round(x * pow(0.1, self.decimals), self.decimals) for x in res]
        return [int(v) / self.pow_ten for v in res]