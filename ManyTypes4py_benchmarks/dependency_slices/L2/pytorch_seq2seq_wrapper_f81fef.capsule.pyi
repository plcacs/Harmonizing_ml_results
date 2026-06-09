from typing import Any

# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...

# === Internal dependency: allennlp.modules.augmented_lstm ===
class AugmentedLstm(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, go_forward: bool = ..., recurrent_dropout_probability: float = ..., use_highway: bool = ..., use_input_projection_bias: bool = ...) -> Any: ...

# === Internal dependency: allennlp.modules.seq2seq_encoders.seq2seq_encoder ===
class Seq2SeqEncoder(_EncoderBase, Registrable):
    ...

# === Internal dependency: allennlp.modules.stacked_alternating_lstm ===
class StackedAlternatingLstm(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, recurrent_dropout_probability: float = ..., use_highway: bool = ..., use_input_projection_bias: bool = ...) -> None: ...

# === Internal dependency: allennlp.modules.stacked_bidirectional_lstm ===
class StackedBidirectionalLstm(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, recurrent_dropout_probability: float = ..., layer_dropout_probability: float = ..., use_highway: bool = ...) -> None: ...

# === Third-party dependency: torch ===
# Used symbols: Tensor, cat, nn

# === Third-party dependency: torch.nn.utils.rnn ===
def pad_packed_sequence(sequence: PackedSequence, batch_first: bool = ..., padding_value: float = ..., total_length: int | None = ...) -> tuple[Tensor, Tensor]: ...