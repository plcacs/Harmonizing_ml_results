import logging
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple, Type
import torch
import sockeye.constants as C
from sockeye.utils import check_condition

logger: logging.Logger = logging.getLogger(__name__)

class LearningRateScheduler:
    def __init__(self, optimizer: Optional[Any] = None, base_lr: float = 1.0, warmup: int = 0) -> None:
    def __call__(self, optimizer: Any) -> 'LearningRateScheduler':
    def __repr__(self) -> str:
    def state_dict(self) -> Dict[str, Any]:
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    def get_lr(self) -> List[float]:
    def get_last_lr(self) -> List[float]:
    def step(self, t: Optional[int] = None) -> None:
    def _warmup(self, t: int) -> float:

class AdaptiveLearningRateScheduler(LearningRateScheduler):
    def new_evaluation_result(self, has_improved: bool) -> bool:

class LearningRateSchedulerInvSqrtDecay(LearningRateScheduler):
    def get_lr(self) -> List[float]:

class LearningRateSchedulerLinearDecay(LearningRateScheduler):
    def __init__(self, optimizer: Any, base_lr: float, total_steps: int, warmup: int = 0) -> None:
    def get_lr(self) -> List[float]:

class LearningRateSchedulerPlateauReduce(AdaptiveLearningRateScheduler):
    def __init__(self, optimizer: Any, base_lr: float, reduce_factor: float, reduce_num_not_improved: int, warmup: int = 0) -> None:
    def __repr__(self) -> str:
    def new_evaluation_result(self, has_improved: bool) -> bool:
    def get_lr(self) -> List[float]:

def get_lr_scheduler(scheduler_type: Optional[str], base_learning_rate: float, learning_rate_reduce_factor: Optional[float], learning_rate_reduce_num_not_improved: Optional[int], learning_rate_warmup: int = 0, max_updates: Optional[int] = None) -> Tuple[Optional[Type[LearningRateScheduler]], Dict[str, Any]]:
