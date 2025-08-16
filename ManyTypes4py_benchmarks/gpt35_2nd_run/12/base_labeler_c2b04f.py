    def __init__(self, cardinality: int = 2, **kwargs: Any) -> None:
    
    def predict_proba(self, L: np.ndarray) -> np.ndarray:
    
    def predict(self, L: np.ndarray, return_probs: bool = False, tie_break_policy: str = 'abstain') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    
    def score(self, L: np.ndarray, Y: np.ndarray, metrics: List[str] = ['accuracy'], tie_break_policy: str = 'abstain') -> Dict[str, float]:
    
    def save(self, destination: str) -> None:
    
    def load(self, source: str) -> None:
