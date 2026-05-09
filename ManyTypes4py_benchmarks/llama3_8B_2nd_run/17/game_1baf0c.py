import functools
import numpy as np
from nevergrad.functions import ExperimentFunction
from nevergrad.parametrization import parameter as p

class _Game:
    def __init__(self) -> None:
        self.verbose: bool = False
        self.history1: list = []
        self.history2: list = []
        self.batawaf: bool = False
        self.converter: dict = {'flip': self.flip_play_game, 'batawaf': functools.partial(self.war_play_game, batawaf=True), 'war': self.war_play_game, 'guesswho': self.guesswho_play_game, 'bigguesswho': functools.partial(self.guesswho_play_game, init=96)}

    def get_list_of_games(self) -> list:
        return list(self.converter.keys())

    def play_game(self, game: str, policy1: p.Array = None, policy2: p.Array = None) -> int:
        self.history1 = []
        self.history2 = []
        if game not in self.converter.keys():
            raise NotImplementedError(f'{game} is not implemented, choose among: {list(self.converter.keys())}')
        return self.converter[game](policy1, policy2)

    # ... (rest of the class remains the same)

class Game(ExperimentFunction):
    """
    Parameters
    ----------
    nint intaum_stocks: number of stocks to be managed
    depth: number of layers in the neural networks
    width: number of neurons per hidden layer
    """

    def __init__(self, game: str = 'war') -> None:
        self.game: str = game
        self.game_object: _Game = _Game()
        dimension: int = self.game_object.play_game(self.game) * 2
        super().__init__(self._simulate_game, p.Array(shape=(dimension,)))
        self.parametrization.function.deterministic: bool = False
        self.parametrization.function.metrizable: bool = game not in ['war', 'batawaf']

    def _simulate_game(self, x: p.Array) -> float:
        p1: p.Array = x[:self.dimension // 2]
        p2: p.Array = np.random.normal(size=self.dimension // 2)
        r: int = self.game_object.play_game(self.game, p1, p2)
        result: float = 0.0 if r == 1 else 0.5 if r == 0 else 1.0
        p1: p.Array = np.random.normal(size=self.dimension // 2)
        p2: p.Array = x[self.dimension // 2:]
        r: int = self.game_object.play_game(self.game, p1, p2)
        return (result + (0.0 if r == 2 else 0.5 if r == 0 else 1.0)) / 2

    def evaluation_function(self, *recommendations: list) -> float:
        assert len(recommendations) == 1, 'Should not be a pareto set for a singleobjective function'
        x: p.Array = recommendations[0].value
        loss: float = sum([self.function(x) for _ in range(42)]) / 42.0
        assert isinstance(loss, float)
        return loss
