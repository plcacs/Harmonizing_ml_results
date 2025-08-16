    def test_accuracy_computation(self, device: Union[str, torch.device]):
    def test_skips_completely_masked_instances(self, device: Union[str, torch.device]):
    def test_incorrect_gold_labels_shape_catches_exceptions(self, device: Union[str, torch.device]):
    def test_incorrect_mask_shape_catches_exceptions(self, device: Union[str, torch.device]):
    def test_does_not_divide_by_zero_with_no_count(self, device: Union[str, torch.device]):
    def test_distributed_accuracy(self):
    def test_distributed_accuracy_unequal_batches(self):
    def test_multiple_distributed_runs(global_rank: int, world_size: int, gpu_id: int, metric: BooleanAccuracy, metric_kwargs: Dict[str, List[torch.Tensor]], desired_values: float, exact: bool = True):
