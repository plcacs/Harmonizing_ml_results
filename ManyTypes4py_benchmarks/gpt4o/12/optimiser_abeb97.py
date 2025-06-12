from typing import Optional, Union, Set
from hypothesis.internal.compat import int_from_bytes, int_to_bytes
from hypothesis.internal.conjecture.choice import ChoiceT, choice_permitted
from hypothesis.internal.conjecture.data import ConjectureResult, Status, _Overrun
from hypothesis.internal.conjecture.engine import ConjectureRunner
from hypothesis.internal.conjecture.junkdrawer import bits_to_bytes, find_integer
from hypothesis.internal.conjecture.pareto import NO_SCORE

class Optimiser:
    def __init__(self, engine: ConjectureRunner, data: ConjectureResult, target: str, max_improvements: int = 100) -> None:
        self.engine = engine
        self.current_data = data
        self.target = target
        self.max_improvements = max_improvements
        self.improvements = 0

    def run(self) -> None:
        self.hill_climb()

    def score_function(self, data: ConjectureResult) -> float:
        return data.target_observations.get(self.target, NO_SCORE)

    @property
    def current_score(self) -> float:
        return self.score_function(self.current_data)

    def consider_new_data(self, data: ConjectureResult) -> bool:
        if data.status < Status.VALID:
            return False
        assert isinstance(data, ConjectureResult)
        score = self.score_function(data)
        if score < self.current_score:
            return False
        if score > self.current_score:
            self.improvements += 1
            self.current_data = data
            return True
        assert score == self.current_score
        if len(data.nodes) <= len(self.current_data.nodes):
            self.current_data = data
            return True
        return False

    def hill_climb(self) -> None:
        nodes_examined: Set[int] = set()
        prev: Optional[ConjectureResult] = None
        i = len(self.current_data.nodes) - 1
        while i >= 0 and self.improvements <= self.max_improvements:
            if prev is not self.current_data:
                i = len(self.current_data.nodes) - 1
                prev = self.current_data
            if i in nodes_examined:
                i -= 1
                continue
            nodes_examined.add(i)
            node = self.current_data.nodes[i]
            assert node.index is not None
            if node.type not in {'integer', 'float', 'bytes', 'boolean'}:
                continue

            def attempt_replace(k: int) -> bool:
                if abs(k) > 2 ** 20:
                    return False
                node = self.current_data.nodes[i]
                assert node.index is not None
                if node.was_forced:
                    return False
                if node.type in {'integer', 'float'}:
                    assert isinstance(node.value, (int, float))
                    new_choice = node.value + k
                elif node.type == 'boolean':
                    assert isinstance(node.value, bool)
                    if abs(k) > 1:
                        return False
                    if k == -1:
                        new_choice = False
                    if k == 1:
                        new_choice = True
                    if k == 0:
                        new_choice = node.value
                else:
                    assert node.type == 'bytes'
                    assert isinstance(node.value, bytes)
                    v = int_from_bytes(node.value)
                    if v + k < 0:
                        return False
                    v += k
                    size = max(len(node.value), bits_to_bytes(v.bit_length()))
                    new_choice = int_to_bytes(v, size)
                if not choice_permitted(new_choice, node.kwargs):
                    return False
                for _ in range(3):
                    choices = self.current_data.choices
                    attempt_choices = choices[:node.index] + (new_choice,) + choices[node.index + 1:]
                    attempt = self.engine.cached_test_function_ir(attempt_choices, extend='full')
                    if self.consider_new_data(attempt):
                        return True
                    if attempt.status is Status.OVERRUN:
                        return False
                    assert isinstance(attempt, ConjectureResult)
                    if len(attempt.nodes) == len(self.current_data.nodes):
                        return False
                    for j, ex in enumerate(self.current_data.examples):
                        if ex.start >= node.index + 1:
                            break
                        if ex.end <= node.index:
                            continue
                        ex_attempt = attempt.examples[j]
                        if ex.choice_count == ex_attempt.choice_count:
                            continue
                        replacement = attempt.choices[ex_attempt.start:ex_attempt.end]
                        if self.consider_new_data(self.engine.cached_test_function_ir(choices[:node.index] + replacement + self.current_data.choices[ex.end:])):
                            return True
                return False
            find_integer(lambda k: attempt_replace(k))
            find_integer(lambda k: attempt_replace(-k))
