import logging
from time import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from pandas import DataFrame
from torch.nn import functional as F
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.base_models.BasePyTorchModel import BasePyTorchModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from torch import Tensor

logger = logging.getLogger(__name__)

class BasePyTorchClassifier(BasePyTorchModel):
    """
    A PyTorch implementation of a classifier.
    User must implement fit method

    Important!

    - User must declare the target class names in the strategy,
    under IStrategy.set_freqai_targets method.

    for example, in your strategy:
    