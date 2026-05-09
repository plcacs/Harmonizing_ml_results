class StateType(AutoEnum):
    """Enumeration of state types."""
    SCHEDULED: StateType = AutoEnum.auto()
    PENDING: StateType = AutoEnum.auto()
    RUNNING: StateType = AutoEnum.auto()
    COMPLETED: StateType = AutoEnum.auto()
    FAILED: StateType = AutoEnum.auto()
    CANCELLED: StateType = AutoEnum.auto()
    CRASHED: StateType = AutoEnum.auto()
    PAUSED: StateType = AutoEnum.auto()
    CANCELLING: StateType = AutoEnum.auto()

class CountByState(PrefectBaseModel):
    COMPLETED: int = Field(default=0)
    PENDING: int = Field(default=0)
    RUNNING: int = Field(default=0)
    FAILED: int = Field(default=0)
    CANCELLED: int = Field(default=0)
    CRASHED: int = Field(default=0)
    PAUSED: int = Field(default=0)
    CANCELLING: int = Field(default=0)
    SCHEDULED: int = Field(default=0)

    @field_validator('*')
    @classmethod
    def check_key(cls, value: Any, info: ValidationInfo) -> Any:
        if info.field_name not in StateType.__members__:
            raise ValueError(f'{info.field_name} is not a valid StateType')
        return value

class StateDetails(PrefectBaseModel):
    flow_run_id: Optional[UUID] = None
    task_run_id: Optional[UUID] = None
    child_flow_run_id: Optional[UUID] = None
    scheduled_time: Optional[datetime] = None
    cache_key: Optional[str] = None
    cache_expiration: Optional[timedelta] = None
    deferred: bool = False
    untrackable_result: bool = False
    pause_timeout: Optional[timedelta] = None
    pause_reschedule: bool = False
    pause_key: Optional[str] = None
    run_input_keyset: Optional[str] = None
    refresh_cache: Optional[bool] = None
    retriable: Optional[bool] = None
    transition_id: Optional[UUID] = None
    task_parameters_id: Optional[UUID] = None
    traceparent: Optional[str] = None

class State(StateBaseModel):
    """Represents the state of a run."""
    model_config: ConfigDict = ConfigDict(from_attributes=True)
    name: Optional[str] = Field(default=None)
    timestamp: datetime = Field(default_factory=lambda: now('UTC'))
    message: Optional[str] = Field(default=None, examples=['Run started'])
    data: Any = Field(default=None, description='Data associated with the state, e.g. a result. Content must be storable as JSON.')
    state_details: StateDetails = Field(default_factory=StateDetails)

    @classmethod
    def from_orm_without_result(cls, orm_state: ORMFlowRunState, with_data: Any = None) -> _State:
        ...

    @model_validator(mode='after')
    def default_name_from_type(self) -> Self:
        ...

    @model_validator(mode='after')
    def default_scheduled_start_time(self) -> Self:
        ...

    def is_scheduled(self) -> bool:
        return self.type == StateType.SCHEDULED

    def is_pending(self) -> bool:
        return self.type == StateType.PENDING

    def is_running(self) -> bool:
        return self.type == StateType.RUNNING

    def is_completed(self) -> bool:
        return self.type == StateType.COMPLETED

    def is_failed(self) -> bool:
        return self.type == StateType.FAILED

    def is_crashed(self) -> bool:
        return self.type == StateType.CRASHED

    def is_cancelled(self) -> bool:
        return self.type == StateType.CANCELLED

    def is_cancelling(self) -> bool:
        return self.type == StateType.CANCELLING

    def is_final(self) -> bool:
        return self.type in TERMINAL_STATES

    def is_paused(self) -> bool:
        return self.type == StateType.PAUSED

    def fresh_copy(self, **kwargs) -> Self:
        ...

    @overload
    def result(self, raise_on_failure: bool = ..., fetch: bool = ...) -> Any:
        ...

    @overload
    def result(self, raise_on_failure: bool = False, fetch: bool = ...) -> Any:
        ...

    @overload
    def result(self, raise_on_failure: bool = ..., fetch: bool = ...) -> Any:
        ...

    def result(self, raise_on_failure: bool = True, fetch: bool = True) -> Any:
        ...

    def to_state_create(self) -> StateCreate:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __hash__(self) -> int:
        return hash((getattr(self.state_details, 'flow_run_id', None), getattr(self.state_details, 'task_run_id', None), self.timestamp, self.type))

def Scheduled(scheduled_time: Optional[datetime] = None, cls: Type[R] = State, **kwargs) -> R:
    ...

def Completed(cls: Type[R] = State, **kwargs) -> R:
    ...

def Running(cls: Type[R] = State, **kwargs) -> R:
    ...

def Failed(cls: Type[R] = State, **kwargs) -> R:
    ...

def Crashed(cls: Type[R] = State, **kwargs) -> R:
    ...

def Cancelling(cls: Type[R] = State, **kwargs) -> R:
    ...

def Cancelled(cls: Type[R] = State, **kwargs) -> R:
    ...

def Pending(cls: Type[R] = State, **kwargs) -> R:
    ...

def Paused(cls: Type[R] = State, timeout_seconds: Optional[int] = None, pause_expiration_time: Optional[datetime] = None, reschedule: bool = False, pause_key: Optional[str] = None, **kwargs) -> R:
    ...

def Suspended(cls: Type[R] = State, timeout_seconds: Optional[int] = None, pause_expiration_time: Optional[datetime] = None, pause_key: Optional[str] = None, **kwargs) -> R:
    ...

def AwaitingRetry(cls: Type[R] = State, scheduled_time: Optional[datetime] = None, **kwargs) -> R:
    ...

def Retrying(cls: Type[R] = State, **kwargs) -> R:
    ...

def Late(cls: Type[R] = State, scheduled_time: Optional[datetime] = None, **kwargs) -> R:
    ...
