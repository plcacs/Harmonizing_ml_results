    store: Optional[ResultStore] = None
    key: Optional[str] = None
    children: List[Transaction] = Field(default_factory=list)
    commit_mode: Optional[CommitMode] = None
    isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED
    state: TransactionState = TransactionState.PENDING
    on_commit_hooks: List[Callable[[Transaction], Any]] = Field(default_factory=list)
    on_rollback_hooks: List[Callable[[Transaction], Any]] = Field(default_factory=list)
    overwrite: bool = False
    logger: LoggingAdapter = Field(default_factory=partial(get_logger, 'transactions'))
    write_on_commit: bool = True
    _stored_values: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _staged_value: Any = None
    __var__: ContextVar[Transaction] = ContextVar('transaction')
    def set(self, name: str, value: Any) -> None:
    def get(self, name: str, default: Any = NotSet) -> Any:
    def is_committed(self) -> bool:
    def is_rolled_back(self) -> bool:
    def is_staged(self) -> bool:
    def is_pending(self) -> bool:
    def is_active(self) -> bool:
    def __enter__(self) -> 'Transaction':
    def __exit__(self, *exc_info) -> None:
    def begin(self) -> None:
    def read(self) -> Optional[ResultRecord]:
    def reset(self) -> None:
    def add_child(self, transaction: 'Transaction') -> None:
    def get_parent(self) -> Optional['Transaction']:
    def commit(self) -> bool:
    def run_hook(self, hook: Callable[[Transaction], Any], hook_type: str) -> None:
    def stage(self, value: Any, on_rollback_hooks: Optional[List[Callable[[Transaction], Any]]] = None, on_commit_hooks: Optional[List[Callable[[Transaction], Any]]] = None) -> None:
    def rollback(self) -> bool:
    @classmethod
    def get_active(cls) -> Optional['Transaction']:
    def get_transaction() -> Optional['Transaction']:
    def transaction(key: Optional[str] = None, store: Optional[ResultStore] = None, commit_mode: Optional[CommitMode] = None, isolation_level: Optional[IsolationLevel] = None, overwrite: bool = False, write_on_commit: bool = True, logger: Optional[LoggingAdapter] = None) -> Generator[Transaction, None, None]:
