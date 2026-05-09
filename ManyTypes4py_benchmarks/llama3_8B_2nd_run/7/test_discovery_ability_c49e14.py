from collections import collections
import math
import re
from hypothesis import HealthCheck, settings as Settings
from hypothesis.control import BuildContext
from hypothesis.errors import UnsatisfiedAssumption
from hypothesis.internal import reflection
from hypothesis.internal.conjecture.engine import ConjectureRunner
from hypothesis.strategies import binary, booleans, floats, integers, just, lists, one_of, sampled_from, sets, text, tuples

RUNS: int = 100
INITIAL_LAMBDA: re.Pattern = re.compile('^lambda[^:]*:\\s*')

def strip_lambda(s: str) -> str:
    return INITIAL_LAMBDA.sub('', s)

class HypothesisFalsified(AssertionError):
    pass

def define_test(specifier: collections.abc.Strategy, predicate: callable, condition: callable = None, p: float = 0.5, suppress_health_check: tuple = ()) -> callable:
    required_runs: int = int(RUNS * p)

    def run_test() -> None:
        if condition is None:
            _condition: callable = lambda x: True
            condition_string: str = ''
        else:
            _condition: callable = condition
            condition_string: str = strip_lambda(reflection.get_pretty_function_description(condition))

        def test_function(data: BuildContext) -> None:
            try:
                value: object = data.draw(specifier)
            except UnsatisfiedAssumption:
                data.mark_invalid()
            if not _condition(value):
                data.mark_invalid()
            if predicate(value):
                data.mark_interesting()

        successes: int = 0
        actual_runs: int = 0
        for actual_runs in range(1, RUNS + 1):
            runner: ConjectureRunner = ConjectureRunner(test_function, settings=Settings(max_examples=150, phases=no_shrink, suppress_health_check=suppress_health_check))
            runner.run()
            if runner.interesting_examples:
                successes += 1
                if successes >= required_runs:
                    return
            if required_runs - successes > RUNS - actual_runs:
                break
        event: str = reflection.get_pretty_function_description(predicate)
        if condition is not None:
            event += '|'
            event += condition_string
        raise HypothesisFalsified(f'P({event}) ~ {successes} / {actual_runs} = {successes / actual_runs:.2f} < {required_runs / RUNS:.2f}; rejected')

    return run_test

# ... (rest of the code remains the same)
