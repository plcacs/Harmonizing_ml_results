    def __init__(self, embed_dim: int, lstm_dim: int, use_highway: bool = True, use_bias: bool = True) -> None:
    def reset_parameters(self) -> None:
    def forward(self, x: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor], variational_dropout_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    def forward(self, inputs: PackedSequence, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
    def __init__(self, input_size: int, hidden_size: int, go_forward: bool = True, recurrent_dropout_probability: float = 0.0, use_highway: bool = True, use_input_projection_bias: bool = True) -> None:
    def forward(self, inputs: PackedSequence, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, recurrent_dropout_probability: float = 0.0, bidirectional: bool = False, padding_value: float = 0.0, use_highway: bool = True) -> None:
    def forward(self, inputs: PackedSequence, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
