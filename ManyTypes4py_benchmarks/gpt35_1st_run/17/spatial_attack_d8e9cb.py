    def __init__(self, max_translation: int = 3, max_rotation: int = 30, num_translations: int = 5, num_rotations: int = 5, grid_search: bool = True, random_steps: int = 100) -> None:
    def __call__(self, model: Model, inputs: Any, criterion: Union[Criterion, Tuple[Criterion, Any]], **kwargs: Any) -> Tuple[Any, Any, Any]:
    def run(self, model: Model, inputs: Any, criterion: Union[Criterion, Tuple[Criterion, Any]], **kwargs: Any) -> Any:
    def repeat(self, times: int) -> 'SpatialAttack':
