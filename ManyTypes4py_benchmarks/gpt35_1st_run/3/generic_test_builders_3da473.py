def synthesize_generic_test_names(test_type: str, test_name: str, args: Dict[str, Any]) -> Tuple[str, str]:
    ...

class TestBuilder(Generic[Testable]):
    TEST_NAME_PATTERN: re.Pattern = re.compile('((?P<test_namespace>([a-zA-Z_][0-9a-zA-Z_]*))\\.)?(?P<test_name>([a-zA-Z_][0-9a-zA-Z_]*))')
    CONFIG_ARGS: Tuple[str, ...] = ('severity', 'tags', 'enabled', 'where', 'limit', 'warn_if', 'error_if', 'fail_calc', 'store_failures', 'store_failures_as', 'meta', 'database', 'schema', 'alias')

    def __init__(self, data_test: Any, target: Any, package_name: str, render_ctx: Any, column_name: Optional[str] = None, version: Optional[str] = None) -> None:
        ...

    def _process_legacy_args(self) -> Dict[str, Any]:
        ...

    def _render_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def _bad_type(self) -> TypeError:
        ...

    @staticmethod
    def extract_test_args(data_test: Any, name: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        ...

    def tags(self) -> List[str]:
        ...

    def macro_name(self) -> str:
        ...

    def get_synthetic_test_names(self) -> Tuple[str, str]:
        ...

    def construct_config(self) -> str:
        ...

    def build_raw_code(self) -> str:
        ...

    def build_model_str(self) -> str:
        ...
