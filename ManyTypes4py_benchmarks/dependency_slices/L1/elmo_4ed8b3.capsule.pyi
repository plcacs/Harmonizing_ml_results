# === Internal dependency: allennlp.common ===
from allennlp.common.from_params import FromParams

# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...

# === Internal dependency: allennlp.common.file_utils ===
def cached_path(url_or_filename, cache_dir=..., extract_archive=..., force_extract=...): ...

# === Internal dependency: allennlp.common.util ===
def lazy_groups_of(iterable, group_size): ...

# === Internal dependency: allennlp.data.batch ===
class Batch(Iterable):
    def __init__(self, instances): ...
    def as_tensor_dict(self, padding_lengths=..., verbose=...): ...
    def index_instances(self, vocab): ...

# === Internal dependency: allennlp.data.fields ===
from allennlp.data.fields.text_field import TextField

# === Internal dependency: allennlp.data.instance ===
class Instance(Mapping[str, Field]):
    def __init__(self, fields): ...

# === Internal dependency: allennlp.data.token_indexers.elmo_indexer ===
class ELMoCharacterMapper: ...
class ELMoTokenCharactersIndexer(TokenIndexer):
    def __init__(self, namespace=..., tokens_to_add=..., token_min_padding_length=...): ...

# === Internal dependency: allennlp.data.tokenizers.token_class ===
class Token: ...

# === Internal dependency: allennlp.data.vocabulary ===
class Vocabulary(Registrable):
    def __init__(self, counter=..., min_count=..., max_vocab_size=..., non_padded_namespaces=..., pretrained_files=..., only_include_pretrained_words=..., tokens_to_add=..., min_pretrained_embeddings=..., padding_token=..., oov_token=...): ...

# === Internal dependency: allennlp.modules.elmo_lstm ===
class ElmoLstm(_EncoderBase):
    def __init__(self, input_size, hidden_size, cell_size, num_layers, requires_grad=..., recurrent_dropout_probability=..., memory_cell_clip_value=..., state_projection_clip_value=...): ...

# === Internal dependency: allennlp.modules.highway ===
class Highway(torch.nn.Module):
    def __init__(self, input_dim, num_layers=..., activation=...): ...

# === Internal dependency: allennlp.modules.scalar_mix ===
class ScalarMix(torch.nn.Module):
    def __init__(self, mixture_size, do_layer_norm=..., initial_scalar_parameters=..., trainable=...): ...

# === Internal dependency: allennlp.nn.util ===
def get_device_of(tensor): ...
def add_sentence_boundary_token_ids(tensor, mask, sentence_begin_token, sentence_end_token): ...
def remove_sentence_boundaries(tensor, mask): ...

# === Third-party dependency: h5py ===
# Used symbols: File

# === Third-party dependency: numpy ===
# Used symbols: array, concatenate, transpose, zeros

# === Third-party dependency: torch ===
# Used symbols: FloatTensor, Tensor, cat, chunk, from_numpy, max, nn, tanh, transpose

# === Third-party dependency: torch.nn.modules ===
# Used symbols: Dropout