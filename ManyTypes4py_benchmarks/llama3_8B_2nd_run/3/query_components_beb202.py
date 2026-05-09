class AsyncPostgresQueryComponents(BaseQueryComponents):
    def insert(self, obj: sa.Insert) -> sa.Insert:
        return postgresql.insert(obj)

    @property
    def uses_json_strings(self) -> bool:
        return False

    def cast_to_json(self, json_obj: json) -> json:
        return sa.func.json(json_obj)

    # ... rest of the code ...

    @db_injector
    async def get_flow_run_notifications_from_queue(self, db: db_type, session: AsyncSession, limit: int) -> list[FlowRunNotification]:
        # ... rest of the code ...

    @property
    def _get_scheduled_flow_runs_from_work_pool_template_path(self) -> str:
        return 'postgres/get-runs-from-worker-queues.sql.jinja'

    @db_injector
    def _build_flow_run_graph_v2_query(self, db: db_type) -> sa.Select[FlowRunGraphV2Node]:
        # ... rest of the code ...

class AioSqliteQueryComponents(BaseQueryComponents):
    def insert(self, obj: sa.Insert) -> sa.Insert:
        return sqlite.insert(obj)

    @property
    def uses_json_strings(self) -> bool:
        return True

    def cast_to_json(self, json_obj: json) -> json:
        return sa.func.json(json_obj)

    # ... rest of the code ...

    @db_injector
    async def get_flow_run_notifications_from_queue(self, db: db_type, session: AsyncSession, limit: int) -> list[FlowRunNotification]:
        # ... rest of the code ...

    @property
    def _get_scheduled_flow_runs_from_work_pool_template_path(self) -> str:
        return 'sqlite/get-runs-from-worker-queues.sql.jinja'

    @db_injector
    def _build_flow_run_graph_v2_query(self, db: db_type) -> sa.Select[FlowRunGraphV2Node]:
        # ... rest of the code ...
