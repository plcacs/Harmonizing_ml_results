    def setup_method(self) -> None:
    def test_get_valid_tokens_mask(self, device: Any) -> None:
    def test_ngrams(self, device: Any) -> None:
    def test_bleu_computed_correctly(self, device: Any) -> None:
    def test_bleu_computed_with_zero_counts(self, device: Any) -> None:
    def test_distributed_bleu(self) -> None:
    def test_multiple_distributed_runs(global_rank: int, world_size: int, gpu_id: int, metric: BLEU, metric_kwargs: Dict[str, List[torch.Tensor]], desired_values: Dict[str, float], exact: bool = True) -> None:
