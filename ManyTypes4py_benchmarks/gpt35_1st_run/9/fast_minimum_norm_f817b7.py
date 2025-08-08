def best_other_classes(logits: T, exclude: T) -> T:
    ...

def project_onto_l1_ball(x: T, eps: float) -> T:
    ...

class FMNAttackLp(MinimizationAttack, ABC):
    def __init__(self, *, steps: int = 100, max_stepsize: float = 1.0, min_stepsize: Optional[float] = None, gamma: float = 0.05, init_attack: Optional[Any] = None, binary_search_steps: int = 10):
        ...

    def run(self, model: Model, inputs: T, criterion: Any, *, starting_points: Optional[T] = None, early_stop: Optional[Any] = None):
        ...

    def normalize(self, gradients: T) -> T:
        ...

    @abstractmethod
    def project(self, x: T, x0: T, epsilon: T) -> T:
        ...

    @abstractmethod
    def mid_points(self, x0: T, x1: T, epsilons: T, bounds: Tuple[T, T]) -> T:

class L1FMNAttack(FMNAttackLp):
    def get_random_start(self, x0: T, epsilon: float) -> T:
        ...

    def project(self, x: T, x0: T, epsilon: T) -> T:
        ...

    def mid_points(self, x0: T, x1: T, epsilons: T, bounds: Tuple[T, T]) -> T:

class L2FMNAttack(FMNAttackLp):
    def get_random_start(self, x0: T, epsilon: float) -> T:
        ...

    def project(self, x: T, x0: T, epsilon: T) -> T:
        ...

    def mid_points(self, x0: T, x1: T, epsilons: T, bounds: Tuple[T, T]) -> T:

class LInfFMNAttack(FMNAttackLp):
    def get_random_start(self, x0: T, epsilon: float) -> T:
        ...

    def project(self, x: T, x0: T, epsilon: T) -> T:
        ...

    def mid_points(self, x0: T, x1: T, epsilons: T, bounds: Tuple[T, T]) -> T:

class L0FMNAttack(FMNAttackLp):
    def project(self, x: T, x0: T, epsilon: T) -> T:
        ...

    def mid_points(self, x0: T, x1: T, epsilons: T, bounds: Tuple[T, T]) -> T:
