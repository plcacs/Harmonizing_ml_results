import functools
from typing import Optional, List, Dict, Callable, Any
import numpy as np
from nevergrad.functions import ExperimentFunction
from nevergrad.parametrization import parameter as p


class _Game:
    def __init__(self) -> None:
        self.verbose: bool = False
        self.history1: List[Any] = []
        self.history2: List[Any] = []
        self.batawaf: bool = False
        self.converter: Dict[str, Callable[..., int]] = {
            "flip": self.flip_play_game,
            "batawaf": functools.partial(self.war_play_game, batawaf=True),
            "war": self.war_play_game,
            "guesswho": self.guesswho_play_game,
            "bigguesswho": functools.partial(self.guesswho_play_game, init=96),
        }

    def get_list_of_games(self) -> List[str]:
        return list(self.converter.keys())

    def play_game(
        self, game: str, policy1: Optional[np.ndarray] = None, policy2: Optional[np.ndarray] = None
    ) -> int:
        self.history1 = []
        self.history2 = []
        if game not in self.converter.keys():
            raise NotImplementedError(
                f"{game} is not implemented, choose among: {list(self.converter.keys())}"
            )
        return self.converter[game](policy1, policy2)  # type: ignore

    def guesswho_play_noturn(self, decks: List[int], policy: Optional[np.ndarray]) -> int:
        assert decks[0] > 0
        assert decks[1] > 0
        baseline: int = decks[0] // 2  # The default dichotomic policy.
        if policy is None:
            return baseline
        difference: int = decks[0] - decks[1]
        late: float = max(0.0, decks[0] - decks[1])  # How late I am.
        try:
            return int(
                0.5
                + baseline
                + policy[0] * difference
                + policy[1] * late
                + policy[2] * late * difference / (1 + decks[0])
                + policy[3] * late * difference / (1 + decks[1])
            )
        except Exception:
            return baseline

    def guesswho_play(self, policy: Optional[np.ndarray], decks: List[int], turn: int) -> List[int]:
        if turn == 0:
            choice: int = self.guesswho_play_noturn(decks, policy)
        else:
            choice = self.guesswho_play_noturn([decks[1], decks[0]], policy)
        choice = max(1, min(choice, decks[turn] - 1))
        decks_copy: List[int] = list(decks)
        # Simulating a probabilistic move.
        decks_copy[turn] = choice if np.random.randint(decks_copy[turn]) <= choice else decks_copy[turn] - choice
        return decks_copy

    def guesswho_play_game(
        self, policy1: Optional[np.ndarray], policy2: Optional[np.ndarray], init: int = 24
    ) -> int:
        if policy1 is None and policy2 is None:
            return 4
        remaining_cards: List[int] = [init, init]
        if np.random.uniform(0.0, 1.0) > 0.5:
            remaining_cards = self.guesswho_play(policy1, remaining_cards, 0)
            if min(remaining_cards) <= 1:
                return 1
        while True:
            remaining_cards = self.guesswho_play(policy2, remaining_cards, 1)
            if min(remaining_cards) <= 1:
                return 2
            remaining_cards = self.guesswho_play(policy1, remaining_cards, 0)
            if min(remaining_cards) <= 1:
                return 1

    def flip_play_game(self, policy1: Optional[np.ndarray], policy2: Optional[np.ndarray]) -> int:
        if policy1 is None and policy2 is None:
            return 57 * 57
        if np.random.uniform(0.0, 1.0) > 0.5:
            r: int = self.flip_play_game_nosym(policy1=policy2, policy2=policy1)
            return 1 if r == 2 else 2 if r == 1 else 0
        return self.flip_play_game_nosym(policy1, policy2)

    def flip_match(self, a: int, b: int) -> bool:
        return abs((a - b) % 13) <= 1

    def flip_play_game_nosym(
        self, policy1: Optional[np.ndarray], policy2: Optional[np.ndarray]
    ) -> int:
        cards: List[int] = [i // 4 for i in range(32)]
        np.random.shuffle(cards)
        stack: List[int] = sorted(cards[:2])
        visible1: List[int] = sorted(cards[2:7])
        visible2: List[int] = sorted(cards[7:12])
        cards1: List[int] = [cards[i] for i in range(12, 22)]
        cards2: List[int] = [cards[i] for i in range(22, 32)]
        nan: float = float("nan")
        something_moves: bool = True
        while something_moves:
            if self.verbose:
                print("==========")
                print(visible1 + [nan] + [len(visible1) + len(cards1)])
                print(stack)
                print(visible2 + [nan] + [len(visible2) + len(cards2)])
                print("==========")
            something_moves = False

            bestvalue: float = self.flip_value(
                visible1, visible2, len(visible1) + len(cards1), len(visible2) + len(cards2), stack, policy1
            )
            we_play: bool = False
            next_visible1: Optional[List[int]] = None
            next_stack: Optional[List[int]] = None
            next_visible2: Optional[List[int]] = None
            for i in range(len(visible1)):
                for location in range(2):
                    if self.flip_match(visible1[i], stack[location]):
                        candidate_visible1: List[int] = visible1[:i] + visible1[i + 1 :]
                        candidate_stack: List[int] = sorted([visible1[i], stack[1 - location]])
                        value: float = self.flip_value(
                            candidate_visible1,
                            visible2,
                            len(cards1) - 1 + len(visible1),
                            len(cards2) + len(visible2),
                            candidate_stack,
                            policy1,
                        )
                        if value < bestvalue:
                            next_visible1 = candidate_visible1
                            bestvalue = value
                            next_stack = candidate_stack
                            we_play = True
            if we_play and next_visible1 is not None and next_stack is not None:
                something_moves = True
                visible1 = sorted(next_visible1 + ([cards1[0]] if cards1 else []))  # type: ignore
                stack = sorted(next_stack)
                if cards1:
                    del cards1[0]
                if not visible1:
                    return 1
            bestvalue = self.flip_value(
                visible2, visible1, len(cards2) + len(visible2), len(cards1) + len(visible1), stack, policy2
            )
            we_play = False
            for i in range(len(visible2)):
                for location in range(2):
                    if self.flip_match(visible2[i], stack[location]):
                        candidate_visible2: List[int] = visible2[:i] + visible2[i + 1 :]
                        candidate_stack = sorted([visible2[i], stack[1 - location]])
                        value = self.flip_value(
                            candidate_visible2,
                            visible1,
                            len(visible2) + len(cards2) - 1,
                            len(visible1) + len(cards1),
                            candidate_stack,
                            policy2,
                        )
                        if value < bestvalue:
                            next_visible2 = candidate_visible2
                            bestvalue = value
                            next_stack = candidate_stack
                            we_play = True
            if we_play and next_visible2 is not None and next_stack is not None:
                something_moves = True
                visible2 = sorted(next_visible2 + ([cards2[0]] if cards2 else []))  # type: ignore
                stack = sorted(next_stack)
                if cards2:
                    del cards2[0]
                    if not visible2:
                        return 2
            if not something_moves and cards1 and cards2:
                stack = [cards1[0], cards2[0]]
                del cards1[0]
                del cards2[0]
                something_moves = True
        return 1 if len(visible1) < len(visible2) else 2 if len(visible2) < len(visible1) else 0

    def flip_value(
        self,
        visible1: List[int],
        visible2: List[int],
        l1: int,
        l2: int,
        stack: List[int],
        policy1: Optional[np.ndarray],
    ) -> float:
        if policy1 is None:
            return float(l1)
        state: List[int] = [len(visible1), len(visible2), l1, l2]
        state.append((stack[1] - stack[0]) % 13)
        for i in range(13):
            state.append(sum(1 for s in visible1 if (s - stack[0]) % 13 == i))
            state.append(sum(1 for s in visible1 if (s - stack[1]) % 13 == i))
            state.append(sum(1 for s in visible2 if (s - stack[0]) % 13 == i))
            state.append(sum(1 for s in visible2 if (s - stack[1]) % 13 == i))
        value: float = 0.01 * l1
        for i in range(57):
            for j in range(57):
                value += policy1[i * 57 + j] * state[i] * state[j]
        return value

    def war_play_game(
        self, policy1: Optional[np.ndarray], policy2: Optional[np.ndarray], batawaf: bool = False
    ) -> int:
        self.batawaf = batawaf
        if policy1 is None and policy2 is None:
            if batawaf:
                return 10 * 18 * 6
            return 10 * 26 * 13
        cards: List[int] = [i // 4 for i in range(52)] if not batawaf else [i // 6 for i in range(36)]
        shuffled_cards: List[int] = list(np.random.choice(cards, size=len(cards), replace=False))
        if batawaf:
            cards1: List[int] = shuffled_cards[:18]
            cards2: List[int] = shuffled_cards[18:]
        else:
            cards1 = shuffled_cards[:26]
            cards2 = shuffled_cards[26:]
        assert len(cards1) == len(cards2)
        stack: List[int] = []
        for _ in range(500):
            if not (cards1 or cards2):
                return 0
            elif not cards1:
                return 2
            elif not cards2:
                return 1

            card1: int = cards1[0]
            card2: int = cards2[0]
            del cards1[0]
            del cards2[0]
            stack += [card1, card2]

            if card1 > card2:
                cards1 += self.war_decide(policy1, len(cards1), sorted(stack, reverse=True))
                stack = []
                continue
            if card2 > card1:
                cards2 += self.war_decide(policy2, len(cards2), sorted(stack, reverse=True))
                stack = []
                continue

            if len(cards1) < 2 or len(cards2) < 2:
                return 1 if len(cards2) < len(cards1) else 2 if len(cards1) < len(cards2) else 0

            stack += [cards1[0], cards2[0]]
            del cards1[0]
            del cards2[0]
        return 0

    def war_decide(
        self, policy: Optional[np.ndarray], num_cards: int, list_of_cards: List[int]
    ) -> List[int]:
        cards: List[int] = sorted(list_of_cards, reverse=True)
        if policy is None:
            return cards
        assert len(cards) % 2 == 0
        a: int = min(num_cards, 9)
        b: int = len(cards) // 2
        c: int = cards[0]
        if self.batawaf:
            seed: float = policy[a * 18 * 6 + b * 6 + c]  # type: ignore
        else:
            seed = policy[a * 26 * 13 + b * 13 + c]  # type: ignore
        if seed == 0.0:
            return cards
        state = np.random.RandomState(hash(seed) % (2**32))
        state.shuffle(cards)
        return list(cards)


class Game(ExperimentFunction):
    def __init__(self, game: str = "war") -> None:
        self.game: str = game
        self.game_object: _Game = _Game()
        dimension: int = self.game_object.play_game(self.game) * 2
        self.dimension: int = dimension
        super().__init__(self._simulate_game, p.Array(shape=(dimension,)))
        self.parametrization.function.deterministic = False
        self.parametrization.function.metrizable = game not in ["war", "batawaf"]

    def _simulate_game(self, x: np.ndarray) -> float:
        p1: np.ndarray = x[: (self.dimension // 2)]
        p2: np.ndarray = np.random.normal(size=self.dimension // 2)
        r: int = self.game_object.play_game(self.game, p1, p2)
        result: float = 0.0 if r == 1 else 0.5 if r == 0 else 1.0
        p1 = np.random.normal(size=self.dimension // 2)
        p2 = x[(self.dimension // 2) :]
        r = self.game_object.play_game(self.game, p1, p2)
        return (result + (0.0 if r == 2 else 0.5 if r == 0 else 1.0)) / 2

    def evaluation_function(self, *recommendations: p.Parameter) -> float:
        assert len(recommendations) == 1, "Should not be a pareto set for a singleobjective function"
        x: np.ndarray = recommendations[0].value
        loss: float = sum([self.function(x) for _ in range(42)]) / 42.0
        assert isinstance(loss, float)
        return loss