```python
from typing import Optional

import torch
from torch import nn

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.util import add_positional_features


@Seq2SeqEncoder.register("pytorch_transformer")
class PytorchTransformer(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        feedforward_hidden_dim: int = 2048,
        num_attention_heads: int = 8,
        positional_encoding: Optional[str] = None,
        positional_embedding_size: int = 512,
        dropout_prob: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_hidden_dim,
            dropout=dropout_prob,
            activation=activation,
        )
        self._transformer: nn.TransformerEncoder = nn.TransformerEncoder(layer, num_layers)
        self._input_dim: int = input_dim

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if positional_encoding is None:
            self._sinusoidal_positional_encoding: bool = False
            self._positional_embedding: Optional[nn.Embedding] = None
        elif positional_encoding == "sinusoidal":
            self._sinusoidal_positional_encoding: bool = True
            self._positional_embedding: Optional[nn.Embedding] = None
        elif positional_encoding == "embedding":
            self._sinusoidal_positional_encoding: bool = False
            self._positional_embedding: nn.Embedding = nn.Embedding(positional_embedding_size, input_dim)
        else:
            raise ValueError(
                "positional_encoding must be one of None, 'sinusoidal', or 'embedding'"
            )

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def is_bidirectional(self) -> bool:
        return False

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        output: torch.Tensor = inputs
        if self._sinusoidal_positional_encoding:
            output = add_positional_features(output)
        if self._positional_embedding is not None:
            position_ids: torch.Tensor = torch.arange(inputs.size(1), dtype=torch.long, device=output.device)
            position_ids = position_ids.unsqueeze(0).expand(inputs.shape[:-1])
            output = output + self._positional_embedding(position_ids)

        output = output.permute(1, 0, 2)
        mask = ~mask
        output = self._transformer(output, src_key_padding_mask=mask)
        output = output.permute(1, 0, 2)

        return output
```