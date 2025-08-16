def get_transformer_encoder(config: EncoderConfig, inference_only: bool = False, dtype: Optional[str] = None, clamp_to_dtype: bool = False) -> TransformerEncoder:
    return TransformerEncoder(config=config, inference_only=inference_only, dtype=dtype, clamp_to_dtype=clamp_to_dtype)

def get_encoder(config: EncoderConfig, inference_only: bool = False, dtype: Optional[str] = None, clamp_to_dtype: bool = False) -> TransformerEncoder:
    return TransformerEncoder(config=config, inference_only=inference_only, dtype=dtype, clamp_to_dtype=clamp_to_dtype)

def get_num_hidden(self) -> int:
    ...

def get_encoded_seq_len(self, seq_len: int) -> int:
    ...

def get_max_seq_len(self) -> Optional[int]:
    ...

@dataclass
class FactorConfig(config.Config):
    pass

@dataclass
class EmbeddingConfig(config.Config):
    num_factors: int = field(init=False)
    factor_configs: Optional[List[FactorConfig]] = None
    allow_sparse_grad: bool = False

    def __post_init__(self):
        ...

class Embedding(Encoder):
    def __init__(self, config: EmbeddingConfig, embedding: Optional[pt.nn.Module] = None, dtype: Optional[str] = None):
        ...

    def forward(self, data: pt.Tensor) -> pt.Tensor:
        ...

    def get_num_hidden(self) -> int:
        ...

class TransformerEncoder(Encoder):
    def __init__(self, config: TransformerConfig, inference_only: bool = False, dtype: Optional[str] = None, clamp_to_dtype: bool = False):
        ...

    def forward(self, data: pt.Tensor, valid_length: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        ...

    def get_num_hidden(self) -> int:
        ...
