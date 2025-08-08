    def __init__(self, input_dim: int, num_layers: int, feedforward_hidden_dim: int = 2048, num_attention_heads: int = 8, positional_encoding: Optional[str] = None, positional_embedding_size: int = 512, dropout_prob: float = 0.1, activation: str = 'relu') -> None:
    
    def get_input_dim(self) -> int:
    
    def get_output_dim(self) -> int:
    
    def is_bidirectional(self) -> bool:
    
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
