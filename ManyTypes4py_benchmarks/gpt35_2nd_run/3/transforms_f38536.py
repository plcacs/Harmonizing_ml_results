from typing import Optional, Union, ArrayLike

BoundType = Optional[Union[ArrayLike, float]]

class Transform:
    def __init__(self) -> None:
        self.name: str = uuid.uuid4().hex

    def forward(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def backward(self, y: ArrayLike) -> ArrayLike

    def reverted(self) -> 'Reverted':
        return Reverted(self)

class Reverted(Transform):
    def __init__(self, transform: Transform) -> None:
        super().__init__()
        self.transform: Transform = transform
        self.name: str = f'Rv({self.transform.name})'

    def forward(self, x: ArrayLike) -> ArrayLike:
        return self.transform.backward(x)

    def backward(self, y: ArrayLike) -> ArrayLike:
        return self.transform.forward(y)

class Affine(Transform):
    def __init__(self, a: BoundType, b: BoundType) -> None:
        super().__init__()
        self.a: ArrayLike = bound_to_array(a)
        self.b: ArrayLike = bound_to_array(b)

    def forward(self, x: ArrayLike) -> ArrayLike:
        return self.a * x + self.b

    def backward(self, y: ArrayLike) -> ArrayLike:
        return (y - self.b) / self.a

class Exponentiate(Transform):
    def __init__(self, base: float = 10.0, coeff: float = 1.0) -> None:
        super().__init__()
        self.base: float = base
        self.coeff: float = coeff

    def forward(self, x: float) -> float:
        return self.base ** (float(self.coeff) * x)

    def backward(self, y: float) -> float:
        return np.log(y) / (float(self.coeff) * np.log(self.base)

class BoundTransform(Transform):
    def __init__(self, a_min: Optional[float] = None, a_max: Optional[float] = None) -> None:
        self.a_min: Optional[ArrayLike] = None
        self.a_max: Optional[ArrayLike] = None

    def _check_shape(self, x: ArrayLike) -> None:

class TanhBound(BoundTransform):
    def __init__(self, a_min: float, a_max: float) -> None:

    def forward(self, x: ArrayLike) -> ArrayLike:

    def backward(self, y: ArrayLike) -> ArrayLike:

class Clipping(BoundTransform):
    def __init__(self, a_min: Optional[float] = None, a_max: Optional[float] = None, bounce: bool = False) -> None:

    def forward(self, x: ArrayLike) -> ArrayLike:

    def backward(self, y: ArrayLike) -> ArrayLike:

class ArctanBound(BoundTransform):
    def __init__(self, a_min: float, a_max: float) -> None:

    def forward(self, x: ArrayLike) -> ArrayLike:

    def backward(self, y: ArrayLike) -> ArrayLike:

class CumulativeDensity(BoundTransform):
    def __init__(self, lower: float = 0.0, upper: float = 1.0, eps: float = 1e-09, scale: float = 1.0, density: str = 'gaussian') -> None:

    def forward(self, x: ArrayLike) -> ArrayLike:

    def backward(self, y: ArrayLike) -> ArrayLike:

class Fourrier(Transform):
    def __init__(self, axes: Union[int, ArrayLike]) -> None:

    def forward(self, x: ArrayLike) -> ArrayLike:

    def backward(self, y: ArrayLike) -> ArrayLike:
