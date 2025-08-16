    def __init__(self, bias_direction: BiasDirectionWrapper, embedding_layer: torch.nn.Embedding, equalize_word_pairs_file: Union[PathLike, str], tokenizer: Tokenizer, mitigator_vocab: Optional[Vocabulary] = None, namespace: str = 'tokens', requires_grad: bool = True) -> None:

    def __call__(self, module, module_in, module_out) -> torch.Tensor:

    def train(self, mode: bool = True) -> None:

    def __init__(self, bias_direction: BiasDirectionWrapper, embedding_layer: torch.nn.Embedding, requires_grad: bool = True) -> None:

    def __call__(self, module, module_in, module_out) -> torch.Tensor:

    def train(self, mode: bool = True) -> None:

    def __init__(self, embedding_layer: torch.nn.Embedding, seed_word_pairs_file: Union[PathLike, str], tokenizer: Tokenizer, mitigator_vocab: Optional[Vocabulary] = None, namespace: str = 'tokens') -> None:

    def __call__(self, module, module_in, module_out) -> torch.Tensor:

    def train(self, mode: bool = True) -> None:

    def __init__(self, bias_direction1: BiasDirectionWrapper, bias_direction2: BiasDirectionWrapper, embedding_layer: torch.nn.Embedding, requires_grad: bool = True) -> None:

    def __call__(self, module, module_in, module_out) -> torch.Tensor:

    def train(self, mode: bool = True) -> None:
