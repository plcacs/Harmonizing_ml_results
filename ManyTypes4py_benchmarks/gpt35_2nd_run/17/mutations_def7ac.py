    def __init__(self, random_state: np.random.RandomState) -> None:
    
    def significantly_mutate(self, v: float, arity: int) -> float:
    
    def doerr_discrete_mutation(self, parent: tp.ArrayLike, arity: int = 2) -> tp.ArrayLike:
    
    def doubledoerr_discrete_mutation(self, parent: tp.ArrayLike, max_ratio: float = 1.0, arity: int = 2) -> tp.ArrayLike:
    
    def rls_mutation(self, parent: tp.ArrayLike, arity: int = 2) -> tp.ArrayLike:
    
    def portfolio_discrete_mutation(self, parent: tp.ArrayLike, intensity: tp.Optional[int] = None, arity: int = 2) -> tp.ArrayLike:
    
    def coordinatewise_mutation(self, parent: tp.ArrayLike, velocity: float, boolean_vector: np.ndarray, arity: int) -> tp.ArrayLike:
    
    def discrete_mutation(self, parent: tp.ArrayLike, arity: int = 2) -> tp.ArrayLike:
    
    def crossover(self, parent: tp.ArrayLike, donor: tp.ArrayLike, rotation: bool = False, crossover_type: str = 'none') -> tp.ArrayLike:
    
    def get_roulette(self, archive, num: tp.Optional[int] = None) -> np.ndarray:
