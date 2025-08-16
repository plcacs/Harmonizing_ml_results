from typing import Union, Callable, TypeVar, Any
from typing_extensions import Literal
import eagerpy as ep
from functools import wraps
FuncType = Callable[..., None]
F = TypeVar('F', bound=FuncType)

def maybenoop(f: F) -> F:

class TensorBoard:
    def __init__(self, logdir: Union[str, None]) -> None:

    @maybenoop
    def close(self) -> None:

    @maybenoop
    def scalar(self, tag: str, x: ep.Tensor, step: int) -> None:

    @maybenoop
    def mean(self, tag: str, x: ep.Tensor, step: int) -> None:

    @maybenoop
    def probability(self, tag: str, x: ep.Tensor, step: int) -> None:

    @maybenoop
    def conditional_mean(self, tag: str, x: ep.Tensor, cond: ep.Tensor, step: int) -> None:

    @maybenoop
    def probability_ratio(self, tag: str, x: ep.Tensor, y: ep.Tensor, step: int) -> None:

    @maybenoop
    def histogram(self, tag: str, x: ep.Tensor, step: int, *, first: bool = True) -> None:
