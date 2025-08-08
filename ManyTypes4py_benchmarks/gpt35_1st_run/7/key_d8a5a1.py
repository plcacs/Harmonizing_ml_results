    def __init__(self, user: str, scopes: List[str], text: str = '', expire_time: Optional[datetime] = None, customer: Optional[str] = None, **kwargs: Any) -> None:

    def parse(cls, json: JSON) -> 'ApiKey':

    @property
    def serialize(self) -> JSON:

    def from_document(cls, doc: JSON) -> 'ApiKey':

    def from_record(cls, rec: Any) -> 'ApiKey':

    def from_db(cls, r: Union[Dict[str, Any], Tuple]) -> 'ApiKey':

    def create(self) -> 'ApiKey':

    @staticmethod
    def find_by_id(key: str, user: Optional[str] = None) -> 'ApiKey':

    @staticmethod
    def find_all(query: Optional[Query] = None, page: int = 1, page_size: int = 1000) -> List['ApiKey']:

    @staticmethod
    def count(query: Optional[Query] = None) -> int:

    @staticmethod
    def find_by_user(user: str) -> List['ApiKey']:

    def update(self, **kwargs: Any) -> 'ApiKey':

    def verify_key(key: str) -> Optional['ApiKey']:
