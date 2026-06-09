from typing import Any

# === Internal dependency: allennlp.common ===
# re-export: from allennlp.common.from_params import FromParams

# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...

# === Internal dependency: allennlp.common.file_utils ===
def cached_path(url_or_filename: Union[str, PathLike], cache_dir: Union[str, Path] = ..., extract_archive: bool = ..., force_extract: bool = ...) -> str: ...

# === Internal dependency: allennlp.common.util ===
def lazy_groups_of(iterable: Iterable[A], group_size: int) -> Iterator[List[A]]: ...

# === Internal dependency: allennlp.data.batch ===
class Batch(Iterable):
    def __init__(self, instances: Iterable[Instance]) -> None: ...
    def as_tensor_dict(self, padding_lengths: Dict[str, Dict[str, int]] = ..., verbose: bool = ...) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]: ...
    def index_instances(self, vocab: Vocabulary) -> None: ...

# === Internal dependency: allennlp.data.fields ===
# re-export: from allennlp.data.fields.text_field import TextField

# === Internal dependency: allennlp.data.instance ===
class Instance(Mapping[str, Field]):
    def __init__(self, fields: MutableMapping[str, Field]) -> None: ...

# === Internal dependency: allennlp.data.token_indexers.elmo_indexer ===
class ELMoCharacterMapper: ...
class ELMoTokenCharactersIndexer(TokenIndexer):
    def __init__(self, namespace: str = ..., tokens_to_add: Dict[str, int] = ..., token_min_padding_length: int = ...) -> None: ...

# === Internal dependency: allennlp.data.tokenizers.token_class ===
class Token: ...

# === Internal dependency: allennlp.data.vocabulary ===
class Vocabulary(Registrable):
    def __init__(self, counter: Dict[str, Dict[str, int]] = ..., min_count: Dict[str, int] = ..., max_vocab_size: Union[int, Dict[str, int]] = ..., non_padded_namespaces: Iterable[str] = ..., pretrained_files: Optional[Dict[str, str]] = ..., only_include_pretrained_words: bool = ..., tokens_to_add: Dict[str, List[str]] = ..., min_pretrained_embeddings: Dict[str, int] = ..., padding_token: Optional[str] = ..., oov_token: Optional[str] = ...) -> None: ...

# === Internal dependency: allennlp.modules.elmo_lstm ===
class ElmoLstm(_EncoderBase):
    def __init__(self, input_size: int, hidden_size: int, cell_size: int, num_layers: int, requires_grad: bool = ..., recurrent_dropout_probability: float = ..., memory_cell_clip_value: Optional[float] = ..., state_projection_clip_value: Optional[float] = ...) -> None: ...

# === Internal dependency: allennlp.modules.highway ===
class Highway(torch.nn.Module):
    def __init__(self, input_dim: int, num_layers: int = ..., activation: Callable[[torch.Tensor], torch.Tensor] = ...) -> None: ...

# === Internal dependency: allennlp.modules.scalar_mix ===
class ScalarMix(torch.nn.Module):
    def __init__(self, mixture_size: int, do_layer_norm: bool = ..., initial_scalar_parameters: List[float] = ..., trainable: bool = ...) -> None: ...

# === Internal dependency: allennlp.nn.util ===
def get_device_of(tensor: torch.Tensor) -> int: ...
def add_sentence_boundary_token_ids(tensor: torch.Tensor, mask: torch.BoolTensor, sentence_begin_token: Any, sentence_end_token: Any) -> Tuple[torch.Tensor, torch.BoolTensor]: ...
def remove_sentence_boundaries(tensor: torch.Tensor, mask: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]: ...

# === Third-party dependency: h5py ===
# Used symbols: File

# === Third-party dependency: numpy ===
# Used symbols: array, concatenate, transpose, zeros

# === Third-party dependency: torch ===
# Used symbols: FloatTensor, Tensor, cat, chunk, from_numpy, max, nn, tanh, transpose

# === Third-party dependency: torch.nn.modules ===
# Used symbols: Dropout