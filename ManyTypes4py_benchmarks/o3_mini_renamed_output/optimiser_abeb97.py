from typing import Any, Callable, Optional, Tuple, Set
from hypothesis.internal.compat import int_from_bytes, int_to_bytes
from hypothesis.internal.conjecture.choice import ChoiceT, choice_permitted
from hypothesis.internal.conjecture.data import ConjectureResult, Status, _Overrun
from hypothesis.internal.conjecture.engine import ConjectureRunner
from hypothesis.internal.conjecture.junkdrawer import bits_to_bytes, find_integer
from hypothesis.internal.conjecture.pareto import NO_SCORE


class Optimiser:
    """A fairly basic optimiser designed to increase the value of scores for
    targeted property-based testing.

    This implements a fairly naive hill climbing algorithm based on randomly
    regenerating parts of the test case to attempt to improve the result. It is
    not expected to produce amazing results, because it is designed to be run
    in a fairly small testing budget, so it prioritises finding easy wins and
    bailing out quickly if that doesn't work.

    For more information about targeted property-based testing, see
    Löscher, Andreas, and Konstantinos Sagonas. "Targeted property-based
    testing." Proceedings of the 26th ACM SIGSOFT International Symposium on
    Software Testing and Analysis. ACM, 2017.
    """

    def __init__(
        self,
        engine: ConjectureRunner,
        data: ConjectureResult,
        target: Any,
        max_improvements: int = 100,
    ) -> None:
        """Optimise ``target`` starting from ``data``. Will stop either when
        we seem to have found a local maximum or when the target score has
        been improved ``max_improvements`` times. This limit is in place to
        deal with the fact that the target score may not be bounded above."""
        self.engine: ConjectureRunner = engine
        self.current_data: ConjectureResult = data
        self.target: Any = target
        self.max_improvements: int = max_improvements
        self.improvements: int = 0

    def func_igw25z44(self) -> None:
        self.hill_climb()

    def func_8citasv8(self, data: ConjectureResult) -> int:
        return data.target_observations.get(self.target, NO_SCORE)

    @property
    def func_p028xvmg(self) -> int:
        return self.score_function(self.current_data)

    def func_yyuxrgug(self, data: ConjectureResult) -> bool:
        """Consider a new data object as a candidate target. If it is better
        than the current one, return True."""
        if data.status < Status.VALID:
            return False
        assert isinstance(data, ConjectureResult)
        score: int = self.score_function(data)
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

    def func_9l4mjzzc(self) -> None:
        """The main hill climbing loop where we actually do the work: Take
        data, and attempt to improve its score for target. select_example takes
        a data object and returns an index to an example where we should focus
        our efforts."""
        nodes_examined: Set[int] = set()
        prev: Optional[ConjectureResult] = None
        i: int = len(self.current_data.nodes) - 1
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
                i -= 1
                continue

            def func_5p3shixr(k: int) -> bool:
                """
                Try replacing the current node in the current best test case
                with a value which is "k times larger", where the exact notion
                of "larger" depends on the choice_type.

                Note that we use the *current* best and not the one we started with.
                This helps ensure that if we luck into a good draw when making
                random choices we get to keep the good bits.
                """
                if abs(k) > 2 ** 20:
                    return False
                local_node = self.current_data.nodes[i]
                assert local_node.index is not None
                if local_node.was_forced:
                    return False
                if local_node.type in {'integer', 'float'}:
                    assert isinstance(local_node.value, (int, float))
                    new_choice: int | float = local_node.value + k
                elif local_node.type == 'boolean':
                    assert isinstance(local_node.value, bool)
                    if abs(k) > 1:
                        return False
                    if k == -1:
                        new_choice = False
                    elif k == 1:
                        new_choice = True
                    else:  # k == 0
                        new_choice = local_node.value
                else:
                    assert local_node.type == 'bytes'
                    assert isinstance(local_node.value, bytes)
                    v: int = int_from_bytes(local_node.value)
                    if v + k < 0:
                        return False
                    v += k
                    size: int = max(len(local_node.value), bits_to_bytes(v.bit_length()))
                    new_choice = int_to_bytes(v, size)
                if not choice_permitted(new_choice, local_node.kwargs):
                    return False
                for _ in range(3):
                    choices = self.current_data.choices  # type: Tuple[ChoiceT, ...]
                    attempt_choices: Tuple[ChoiceT, ...] = (
                        choices[:local_node.index]
                        + (new_choice,)
                        + choices[local_node.index + 1 :]
                    )
                    attempt: ConjectureResult = self.engine.cached_test_function_ir(
                        attempt_choices, extend="full"
                    )
                    if self.consider_new_data(attempt):
                        return True
                    if attempt.status is Status.OVERRUN:
                        return False
                    assert isinstance(attempt, ConjectureResult)
                    if len(attempt.nodes) == len(self.current_data.nodes):
                        return False
                    for j, ex in enumerate(self.current_data.examples):
                        if ex.start >= local_node.index + 1:
                            break
                        if ex.end <= local_node.index:
                            continue
                        ex_attempt = attempt.examples[j]
                        if ex.choice_count == ex_attempt.choice_count:
                            continue
                        replacement: Tuple[ChoiceT, ...] = attempt.choices[
                            ex_attempt.start : ex_attempt.end
                        ]
                        new_choices: Tuple[ChoiceT, ...] = (
                            choices[:local_node.index]
                            + replacement
                            + self.current_data.choices[ex.end :]
                        )
                        if self.consider_new_data(
                            self.engine.cached_test_function_ir(new_choices)
                        ):
                            return True
                return False

            find_integer(lambda k: func_5p3shixr(k))
            find_integer(lambda k: func_5p3shixr(-k))
            i -= 1

    # Placeholder for the score function. Should return an int score for the given data.
    def score_function(self, data: ConjectureResult) -> int:
        raise NotImplementedError

    # Placeholder for the consider_new_data method. Should decide whether to keep the new data.
    def consider_new_data(self, data: ConjectureResult) -> bool:
        raise NotImplementedError

    # Property to get the current score from current_data.
    @property
    def current_score(self) -> int:
        return self.score_function(self.current_data)