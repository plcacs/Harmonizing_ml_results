from typing import Any, TypeAlias, Tuple, Type

SerializedSubsuite = Tuple[Type[TestSuite], list[str]]
SubsuiteArgs = Tuple[Type['RemoteTestRunner'], int, SerializedSubsuite, bool, bool]

def run_subsuite(args: SubsuiteArgs) -> Tuple[int, Any]:
    ...

def init_worker(counter: Any, initial_settings: Any = None, serialized_contents: Any = None, process_setup: Any = None, process_setup_args: Any = None, debug_mode: Any = None, used_aliases: Any = None) -> None:
    ...

class Runner(DiscoverRunner):
    parallel_test_suite: Type[ParallelTestSuite]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def get_resultclass(self) -> Type[TextTestResult]:
        ...

    def on_template_rendered(self, sender: Any, context: Any, **kwargs: Any) -> None:
        ...

    def get_shallow_tested_templates(self) -> set[str]:
        ...

    def setup_test_environment(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def teardown_test_environment(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def test_imports(self, test_labels: list[str], suite: Any) -> None:
        ...

    def run_tests(self, test_labels: list[str], failed_tests_path: Any = None, full_suite: bool = False, include_webhooks: bool = False, **kwargs: Any) -> bool:
        ...

class RemoteTestRunner(django_runner.RemoteTestRunner):
    resultclass: Type[RemoteTestResult]
