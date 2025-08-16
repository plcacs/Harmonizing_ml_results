from typing import Callable

def define_test(specifier: Callable, predicate: Callable, condition: Callable = None, p: float = 0.5, suppress_health_check: tuple = ()) -> Callable:
    required_runs: int = int(RUNS * p)

    def run_test() -> None:
        if condition is None:

            def _condition(x):
                return True
            condition_string: str = ''
        else:
            _condition = condition
            condition_string: str = strip_lambda(reflection.get_pretty_function_description(condition))

        def test_function(data) -> None:
            with BuildContext(data):
                try:
                    value = data.draw(specifier)
                except UnsatisfiedAssumption:
                    data.mark_invalid()
                if not _condition(value):
                    data.mark_invalid()
                if predicate(value):
                    data.mark_interesting()
        successes: int = 0
        actual_runs: int = 0
        for actual_runs in range(1, RUNS + 1):
            runner = ConjectureRunner(test_function, settings=Settings(max_examples=150, phases=no_shrink, suppress_health_check=suppress_health_check))
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
