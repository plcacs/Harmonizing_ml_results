from typing import Optional, Union
from hypothesis.internal.conjecture.data import ConjectureResult, Status
from hypothesis.internal.conjecture.engine import ConjectureRunner
from hypothesis.internal.conjecture.junkdrawer import bits_to_bytes, find_integer
from hypothesis.internal.conjecture.pareto import NO_SCORE

class Optimiser:
    def __init__(self, engine: ConjectureRunner, data: ConjectureResult, target: str, max_improvements: int = 100):
    def score_function(self, data: ConjectureResult) -> float:
    @property
    def current_score(self) -> float:
    def consider_new_data(self, data: ConjectureResult) -> bool:
    def hill_climb(self):
    def attempt_replace(k: int) -> bool:
