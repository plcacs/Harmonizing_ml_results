    def __init__(self, origin: Optional[str] = None, tags: Optional[List[str]] = None, create_time: Optional[datetime] = None, timeout: Optional[int] = None, customer: Optional[str] = None, **kwargs: Any) -> None:

    def parse(cls, json: JSON) -> 'Heartbeat':

    @property
    def serialize(self) -> JSON:

    def from_document(cls, doc: JSON) -> 'Heartbeat':

    def from_record(cls, rec: Any) -> 'Heartbeat':

    def from_db(cls, r: Union[Dict[str, Any], Tuple]) -> 'Heartbeat':

    def find_by_id(id: str, customers: Optional[List[str]] = None) -> 'Heartbeat':

    def find_all(query: Optional[Query] = None, page: int = 1, page_size: int = 1000) -> List['Heartbeat']:

    def find_all_by_status(status: Optional[HeartbeatStatus] = None, query: Optional[Query] = None, page: int = 1, page_size: int = 1000) -> List['Heartbeat']:

    def count(query: Optional[Query] = None) -> int:
