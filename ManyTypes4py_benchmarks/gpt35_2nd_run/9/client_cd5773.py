    def send_worker_heartbeat(self, work_pool_name: str, worker_name: str, heartbeat_interval_seconds: int = None, get_worker_id: bool = False, worker_metadata: Any = None) -> Union[UUID, None]:
    def read_workers_for_work_pool(self, work_pool_name: str, worker_filter: WorkerFilter = None, offset: int = None, limit: int = None) -> List[Worker]:
    def read_work_pool(self, work_pool_name: str) -> WorkPool:
    def read_work_pools(self, limit: int = None, offset: int = 0, work_pool_filter: WorkPoolFilter = None) -> List[WorkPool]:
    def create_work_pool(self, work_pool: WorkPoolCreate, overwrite: bool = False) -> WorkPool:
    def update_work_pool(self, work_pool_name: str, work_pool: WorkPoolUpdate):
    def delete_work_pool(self, work_pool_name: str)
    def get_scheduled_flow_runs_for_work_pool(self, work_pool_name: str, work_queue_names: List[str] = None, scheduled_before: datetime = None) -> List[WorkerFlowRunResponse]:

    async def send_worker_heartbeat(self, work_pool_name: str, worker_name: str, heartbeat_interval_seconds: int = None, get_worker_id: bool = False, worker_metadata: Any = None) -> Union[UUID, None]:
    async def read_workers_for_work_pool(self, work_pool_name: str, worker_filter: WorkerFilter = None, offset: int = None, limit: int = None) -> List[Worker]:
    async def read_work_pool(self, work_pool_name: str) -> WorkPool:
    async def read_work_pools(self, limit: int = None, offset: int = 0, work_pool_filter: WorkPoolFilter = None) -> List[WorkPool]:
    async def create_work_pool(self, work_pool: WorkPoolCreate, overwrite: bool = False) -> WorkPool:
    async def update_work_pool(self, work_pool_name: str, work_pool: WorkPoolUpdate):
    async def delete_work_pool(self, work_pool_name: str)
    async def get_scheduled_flow_runs_for_work_pool(self, work_pool_name: str, work_queue_names: List[str] = None, scheduled_before: datetime = None) -> List[WorkerFlowRunResponse]:
