from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

def multi_perspective_match(vector1: torch.Tensor, vector2: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ...

def multi_perspective_match_pairwise(vector1: torch.Tensor, vector2: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    ...

class BiMpmMatching(nn.Module):
    def __init__(self, hidden_dim: int = 100, num_perspectives: int = 20, share_weights_between_directions: bool = True, is_forward: bool = None, with_full_match: bool = True, with_maxpool_match: bool = True, with_attentive_match: bool = True, with_max_attentive_match: bool = True):
        ...

    def get_output_dim(self) -> int:
        ...

    def forward(self, context_1: torch.Tensor, mask_1: torch.BoolTensor, context_2: torch.Tensor, mask_2: torch.BoolTensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        ...
