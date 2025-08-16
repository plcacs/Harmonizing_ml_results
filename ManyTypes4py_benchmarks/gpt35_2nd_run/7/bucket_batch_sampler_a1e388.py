    def add_noise_to_value(value: float, noise_param: float) -> float:
        noise_value = value * noise_param
        noise = random.uniform(-noise_value, noise_value)
        return value + noise

    def __init__(self, batch_size: int, sorting_keys: Optional[List[str]] = None, padding_noise: float = 0.1, drop_last: bool = False, shuffle: bool = True) -> None:

    def _argsort_by_padding(self, instances: List[Instance]) -> Tuple[List[Instance], List[List[float]]]:

    def get_batch_indices(self, instances: List[Instance]) -> Iterable[List[int]]:

    def _guess_sorting_keys(self, instances: Iterable[Instance], num_instances: int = 10) -> None:

    def get_num_batches(self, instances: List[Instance]) -> int:

    def get_batch_size(self) -> int:
