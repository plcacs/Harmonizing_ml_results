# === Internal dependency: allennlp.common ===
from allennlp.common.from_params import FromParams

# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...

# === Internal dependency: allennlp.modules.matrix_attention.matrix_attention ===
class MatrixAttention(torch.nn.Module, Registrable): ...

# === Internal dependency: allennlp.modules.transformer.transformer_module ===
class TransformerModule(Module):
    ...

# === Internal dependency: allennlp.modules.transformer.util ===
def apply_mask(values, mask): ...
FloatT = Union[torch.FloatTensor]
IntT = Union[torch.IntTensor]

# === Third-party dependency: torch ===
# Used symbols: Tensor, abs, arange, cat, full_like, log, long, matmul, min, nn, where, zeros, zeros_like

# === Third-party dependency: torch.nn.functional ===
def dropout(input: Tensor, p: float = ..., training: bool = ..., inplace: bool = ...) -> Tensor: ...