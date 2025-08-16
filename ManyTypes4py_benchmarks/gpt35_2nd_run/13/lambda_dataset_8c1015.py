    def __init__(self, load: Callable[[], Any], save: Callable[[Any], None], exists: Callable[[], bool] = None, release: Callable[[], None] = None, metadata: Any = None) -> None:
    def _load(self) -> Any:
    def _save(self, data: Any) -> None:
    def _exists(self) -> bool:
    def _release(self) -> None:
    def _describe(self) -> dict:
    def _to_str(func: Callable) -> str:
