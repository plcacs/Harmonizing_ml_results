from typing import Union, Callable, TypeVar, Any, Optional, cast
from typing_extensions import Literal
import eagerpy as ep
from functools import wraps
from tensorboardX import SummaryWriter

FuncType = Callable[..., None]
F = TypeVar("F", bound=FuncType)

def maybenoop(f: F) -> F:
    @wraps(f)
    def wrapper(self: "TensorBoard", *args: Any, **kwds: Any) -> None:
        if self.writer is None:
            return
        return f(self, *args, **kwds)
    return cast(F, wrapper)

class TensorBoard:
    """
    A custom TensorBoard class that accepts EagerPy tensors and that
    can be disabled by turning it into a noop by passing logdir=False.

    This makes it possible to add tensorboard logging without any if
    statements and without any computational overhead if it's disabled.
    """

    def __init__(self, logdir: Union[str, Literal[False], None]) -> None:
        if logdir or logdir is None:
            self.writer: Optional[SummaryWriter] = SummaryWriter(logdir=logdir)
        else:
            self.writer = None

    @maybenoop
    def close(self) -> None:
        self.writer.close()  # type: ignore

    @maybenoop
    def scalar(self, tag: str, x: ep.Tensor, step: int) -> None:
        self.writer.add_scalar(tag, x, step)  # type: ignore

    @maybenoop
    def mean(self, tag: str, x: ep.Tensor, step: int) -> None:
        self.writer.add_scalar(tag, x.mean(axis=0).item(), step)  # type: ignore

    @maybenoop
    def probability(self, tag: str, x: ep.Tensor, step: int) -> None:
        self.writer.add_scalar(tag, x.float32().mean(axis=0).item(), step)  # type: ignore

    @maybenoop
    def conditional_mean(self, tag: str, x: ep.Tensor, cond: ep.Tensor, step: int) -> None:
        cond_ = cond.numpy()
        if ~cond_.any():
            return
        x_ = x.numpy()
        x_ = x_[cond_]
        self.writer.add_scalar(tag, x_.mean(axis=0).item(), step)  # type: ignore

    @maybenoop
    def probability_ratio(self, tag: str, x: ep.Tensor, y: ep.Tensor, step: int) -> None:
        x_ = x.float32().mean(axis=0).item()
        y_ = y.float32().mean(axis=0).item()
        if y_ == 0:
            return
        self.writer.add_scalar(tag, x_ / y_, step)  # type: ignore

    @maybenoop
    def histogram(self, tag: str, x: ep.Tensor, step: int, *, first: bool = True) -> None:
        x_np = x.numpy()
        self.writer.add_histogram(tag, x_np, step)  # type: ignore
        if first:
            self.writer.add_scalar(tag + "/0", x_np[0].item(), step)  # type: ignore