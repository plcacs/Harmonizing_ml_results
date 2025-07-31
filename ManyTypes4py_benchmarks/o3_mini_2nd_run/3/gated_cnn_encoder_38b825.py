from typing import Sequence, List
import math
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        layers: Sequence[Sequence[int]],
        direction: str,
        do_weight_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dropout: float = dropout
        self._convolutions: torch.nn.ModuleList = torch.nn.ModuleList()
        last_dim: int = input_dim
        for k, layer in enumerate(layers):
            if len(layer) == 2:
                conv = torch.nn.Conv1d(last_dim, layer[1] * 2, layer[0], stride=1, padding=layer[0] - 1, bias=True)
            elif len(layer) == 3:
                assert layer[0] == 2, 'only support kernel = 2 for now'
                conv = torch.nn.Conv1d(last_dim, layer[1] * 2, layer[0], stride=1, padding=layer[2], dilation=layer[2], bias=True)
            else:
                raise ValueError('each layer must have length 2 or 3')
            conv_dropout: float = dropout if k == 0 else 0.0
            std: float = math.sqrt(4 * (1.0 - conv_dropout) / (layer[0] * last_dim))
            conv.weight.data.normal_(0, std=std)
            conv.bias.data.zero_()
            if do_weight_norm:
                conv = torch.nn.utils.weight_norm(conv, name='weight', dim=0)
            self._convolutions.append(conv)
            last_dim = layer[1]
        assert last_dim == input_dim
        if direction not in ('forward', 'backward'):
            raise ConfigurationError(f'invalid direction: {direction}')
        self._direction: str = direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = x
        timesteps: int = x.size(2)
        for k, convolution in enumerate(self._convolutions):
            if k == 0 and self.dropout > 0:
                out = torch.nn.functional.dropout(out, self.dropout, self.training)
            conv_out: torch.Tensor = convolution(out)
            dims_to_remove: int = conv_out.size(2) - timesteps
            if dims_to_remove > 0:
                if self._direction == 'forward':
                    conv_out = conv_out.narrow(2, 0, timesteps)
                else:
                    conv_out = conv_out.narrow(2, dims_to_remove, timesteps)
            out = torch.nn.functional.glu(conv_out, dim=1)
        return (out + x) * math.sqrt(0.5)

@Seq2SeqEncoder.register('gated-cnn-encoder')
class GatedCnnEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        layers: Sequence[Sequence[Sequence[int]]],
        dropout: float = 0.0,
        return_all_layers: bool = False,
    ) -> None:
        super().__init__()
        self._forward_residual_blocks: torch.nn.ModuleList = torch.nn.ModuleList()
        self._backward_residual_blocks: torch.nn.ModuleList = torch.nn.ModuleList()
        self._input_dim: int = input_dim
        self._output_dim: int = input_dim * 2
        for layer in layers:
            self._forward_residual_blocks.append(ResidualBlock(input_dim, layer, 'forward', dropout=dropout))
            self._backward_residual_blocks.append(ResidualBlock(input_dim, layer, 'backward', dropout=dropout))
        self._return_all_layers: bool = return_all_layers

    def forward(self, token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        transposed_embeddings: torch.Tensor = torch.transpose(token_embeddings, 1, 2)
        mask_for_fill: torch.Tensor = ~mask.unsqueeze(1)
        if self._return_all_layers:
            layer_outputs: List[List[torch.Tensor]] = [[], []]
        else:
            outputs: List[torch.Tensor] = []
        for k, blocks in enumerate([self._forward_residual_blocks, self._backward_residual_blocks]):
            out: torch.Tensor = transposed_embeddings
            for block in blocks:
                out = block(out.masked_fill(mask_for_fill, 0.0))
                if self._return_all_layers:
                    layer_outputs[k].append(out)
            if not self._return_all_layers:
                outputs.append(out)
        if self._return_all_layers:
            # zip(*layer_outputs) will give tuples of corresponding layers from forward and backward
            return [torch.cat([fwd, bwd], dim=1).transpose(1, 2) for fwd, bwd in zip(*layer_outputs)]
        else:
            return torch.cat(outputs, dim=1).transpose(1, 2)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def is_bidirectional(self) -> bool:
        return True