import os
from os import PathLike
from typing import Union, Dict, Optional
import torch
from torch.cuda import amp
from torch.testing import assert_allclose
import pytest
from allennlp.common.testing import AllenNlpTestCase, run_distributed_test, requires_multi_gpu
from allennlp.nn.util import load_state_dict_distributed
from allennlp.nn.parallel import FairScaleFsdpAccelerator, FairScaleFsdpWrappedModel, ShardedModuleMixin

class EncoderDecoderModel(torch.nn.Module):
    def __init__(self, fsdp_wrapper: FairScaleFsdpAccelerator):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 4)
        self.emb_proj = fsdp_wrapper.wrap_module(torch.nn.Linear(4, 4))
        self.encoder = fsdp_wrapper.wrap_module(Encoder())
        self.decoder = Decoder(self.embedding, fsdp_wrapper)
        self.register_buffer('buffer', torch.randn(4, 4))

    def tie_weights(self) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ff1 = FeedForward()
        self.ff2 = FeedForward()
        self.register_buffer('buffer', torch.randn(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

class Decoder(torch.nn.Module):
    def __init__(self, embedding: torch.nn.Embedding, fsdp_wrapper: FairScaleFsdpAccelerator):
        super().__init__()
        self.ff = fsdp_wrapper.wrap_module(FeedForward())
        self.linear = torch.nn.Linear(4, 12, bias=False)
        self.linear.weight = embedding.weight
        self.register_buffer('buffer', torch.randn(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

class FeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

def _dist_load_and_train(global_rank: int, world_size: int, gpu_id: int, test_dir: str, mixed_precision: bool, **kwargs) -> None:
    ...

class TestFairScaleFsdpAccelerator(AllenNlpTestCase):
    def test_distributed_loading_and_training(self, mixed_precision: bool, flatten_parameters: bool) -> None:
        ...
