from typing import Dict, Any, List, Tuple, Optional, Union
import torch
from allennlp.common.lazy import Lazy
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler


@LearningRateScheduler.register("combined")
class CombinedLearningRateScheduler(LearningRateScheduler):
    """
    This `LearningRateScheduler` can be used to apply an arbitrary number of other schedulers
    one after the other.

    These schedulers are defined though the `schedulers` parameter, which takes
    a list of `Tuple[int, Lazy[LearningRateScheduler]]`. The first field of the
    tuple, the `int`, specifies how many epochs the corresponding scheduler will
     be used before the next scheduler takes its place.

    While it usually makes sense for the sum

    