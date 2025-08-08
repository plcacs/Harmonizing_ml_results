    def setup_method(self) -> None:
    def tearDown(self) -> None:
    def test_evalb_correctly_scores_identical_trees(self) -> None:
    def test_evalb_correctly_scores_imperfect_trees(self) -> None:
    def test_evalb_correctly_calculates_bracketing_metrics_over_multiple_trees(self) -> None:
    def test_evalb_with_terrible_trees_handles_nan_f1(self) -> None:
    def test_distributed_evalb(self) -> None:
    def test_multiple_distributed_runs(self) -> None:
def multiple_runs(global_rank: int, world_size: int, gpu_id: int, metric: EvalbBracketingScorer, metric_kwargs: Dict[str, List[Tree]], desired_values: Dict[str, float], exact: bool) -> None:
