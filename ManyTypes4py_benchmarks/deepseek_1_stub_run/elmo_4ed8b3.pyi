```python
import torch
import numpy
from typing import Any, Dict, List, Optional, Tuple, Union

class Elmo(torch.nn.Module):
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        num_output_representations: int,
        requires_grad: bool = ...,
        do_layer_norm: bool = ...,
        dropout: float = ...,
        vocab_to_cache: Optional[List[str]] = ...,
        keep_sentence_boundaries: bool = ...,
        scalar_mix_parameters: Optional[List[float]] = ...,
        module: Optional[torch.nn.Module] = ...
    ) -> None: ...
    def get_output_dim(self) -> int: ...
    def forward(
        self,
        inputs: torch.Tensor,
        word_inputs: Optional[torch.Tensor] = ...
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]: ...

def batch_to_ids(batch: List[List[str]]) -> torch.Tensor: ...

class _ElmoCharacterEncoder(torch.nn.Module):
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        requires_grad: bool = ...
    ) -> None: ...
    def get_output_dim(self) -> int: ...
    def forward(
        self,
        inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]: ...

class _ElmoBiLm(torch.nn.Module):
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        requires_grad: bool = ...,
        vocab_to_cache: Optional[List[str]] = ...
    ) -> None: ...
    def get_output_dim(self) -> int: ...
    def forward(
        self,
        inputs: torch.Tensor,
        word_inputs: Optional[torch.Tensor] = ...
    ) -> Dict[str, Any]: ...
    def create_cached_cnn_embeddings(self, tokens: List[str]) -> None: ...
```