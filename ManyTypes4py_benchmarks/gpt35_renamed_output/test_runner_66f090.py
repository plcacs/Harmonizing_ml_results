from typing import Any, TypeAlias, Tuple, List

SerializedSubsuite: TypeAlias = Tuple[type[TestSuite], List[str]
SubsuiteArgs: TypeAlias = Tuple[type['RemoteTestRunner'], int, SerializedSubsuite, bool, bool]

def func_5izs67oi(args: SubsuiteArgs) -> Tuple[int, List[Any]]:
    ...

def func_wnpbgrj6(counter: Any, initial_settings: Any = None, serialized_contents: Any = None,
                  process_setup: Any = None, process_setup_args: Any = None, debug_mode: Any = None,
                  used_aliases: Any = None) -> None:
    ...

def func_1qaguket(test_name: str) -> None:
    ...

def func_zo2iqwlw(worker_id: int) -> None:
    ...

def func_altdrx05(suite: Any) -> List[str]:
    ...

def func_fxbe0uwz(suite: Any) -> Any:
    ...
