class SecretKeySelector(BaseModel):
    pass

class JobV2(BaseModel):
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    deleteTime: Optional[str] = Field(None)
    expireTime: Optional[str] = Field(None)
    creator: Optional[str] = Field(None)
    lastModifier: Optional[str] = Field(None)
    client: Optional[str] = Field(None)
    clientVersion: Optional[str] = Field(None)
    binaryAuthorization: Dict[str, str] = Field(default_factory=dict)
    template: Dict[str, str] = Field(default_factory=dict)
    observedGeneration: Optional[str] = Field(None)
    terminalCondition: Dict[str, str] = Field(default_factory=dict)
    conditions: List[Dict[str, str]] = Field(default_factory=list)
    latestCreatedExecution: Dict[str, str] = Field(default_factory=dict)
    reconciling: bool = Field(False)
    satisfiesPzs: bool = Field(False)
    etag: Optional[str] = Field(None)

    def is_ready(self) -> bool:
        ...

    def get_ready_condition(self) -> Dict[str, str]:
        ...

    @classmethod
    def get(cls, cr_client: Resource, project: str, location: str, job_name: str) -> 'JobV2':
        ...

    @staticmethod
    def create(cr_client: Resource, project: str, location: str, job_id: str, body: Dict[str, str]) -> Dict[str, str]:
        ...

    @staticmethod
    def delete(cr_client: Resource, project: str, location: str, job_name: str) -> Dict[str, str]:
        ...

    @staticmethod
    def run(cr_client: Resource, project: str, location: str, job_name: str) -> Dict[str, str]:
        ...

    @staticmethod
    def _is_missing_container(ready_condition: Dict[str, str]) -> bool:
        ...

class ExecutionV2(BaseModel):
    def is_running(self) -> bool:
        ...

    def succeeded(self) -> bool:
        ...

    def condition_after_completion(self) -> Dict[str, str]:
        ...

    @classmethod
    def get(cls, cr_client: Resource, execution_id: str) -> 'ExecutionV2':
        ...
