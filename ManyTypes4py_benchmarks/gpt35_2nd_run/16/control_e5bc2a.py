def _calling_function_location(what: str, frame: inspect.FrameType) -> str:
    where = frame.f_back
    return f'{what}() in {where.f_code.co_name} (line {where.f_lineno})'

def reject() -> NoReturn:
    ...

def assume(condition: bool) -> bool:
    ...

def currently_in_test_context() -> bool:
    ...

def current_build_context() -> 'BuildContext':
    ...

class RandomSeeder:

    def __init__(self, seed: Any):
        ...

def deprecate_random_in_strategy(fmt: str, *args: Any) -> ContextManager[_Checker]:
    ...

class BuildContext:

    def __init__(self, data: ConjectureData, *, is_final: bool = False, close_on_capture: bool = True):
        ...

    def record_call(self, obj: Any, func: Callable, args: Sequence, kwargs: dict) -> None:
        ...

    def prep_args_kwargs_from_strategies(self, kwarg_strategies: dict[str, Any]) -> Tuple[dict[str, Any], dict[str, Tuple[int, int]]]:
        ...

def cleanup(teardown: Callable) -> None:
    ...

def should_note() -> bool:
    ...

def note(value: Any) -> None:
    ...

def event(value: Union[str, int, float], payload: str = '') -> None:
    ...

def target(observation: Union[int, float], *, label: str = '') -> Union[int, float]:
    ...
