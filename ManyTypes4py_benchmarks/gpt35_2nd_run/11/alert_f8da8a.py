    def __init__(self, resource: str, event: str, **kwargs: Any) -> None:
    def parse(cls, json: JSON) -> 'Alert':
    @property
    def serialize(self) -> JSON:
    def get_id(self, short: bool = False) -> str:
    def get_body(self, history: bool = True) -> JSON:
    def from_document(cls, doc: Dict[str, Any]) -> 'Alert':
    def from_record(cls, rec: Any) -> 'Alert':
    def from_db(cls, r: Union[Dict[str, Any], Tuple]) -> 'Alert':
    def is_duplicate(self) -> Optional['Alert']:
    def is_correlated(self) -> Optional['Alert']:
    def is_flapping(self, window: int = 1800, count: int = 2) -> bool:
    def get_status_and_value(self) -> List[Tuple[str, str]]:
    def _get_hist_info(self, action: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
    def deduplicate(self, duplicate_of: 'Alert') -> 'Alert':
    def update(self, correlate_with: 'Alert') -> 'Alert':
    def create(self) -> 'Alert':
    @staticmethod
    def find_by_id(id: str, customers: Optional[List[str]] = None) -> Optional['Alert']:
    def is_blackout(self) -> bool:
    @property
    def is_suppressed(self) -> bool:
    def set_status(self, status: str, text: str = '', timeout: Optional[int] = None) -> 'Alert':
    def tag(self, tags: List[str]) -> None:
    def untag(self, tags: List[str]) -> None:
    def update_tags(self, tags: List[str]) -> None:
    def update_attributes(self, attributes: Dict[str, Any]) -> None:
    def delete(self) -> None:
    @staticmethod
    def tag_find_all(query: Query, tags: List[str]) -> List['Alert']:
    @staticmethod
    def untag_find_all(query: Query, tags: List[str]) -> List['Alert']:
    @staticmethod
    def update_attributes_find_all(query: Query, attributes: Dict[str, Any]) -> List['Alert']:
    @staticmethod
    def delete_find_all(query: Optional[Query] = None) -> None:
    @staticmethod
    def find_all(query: Optional[Query] = None, raw_data: bool = False, history: bool = False, page: int = 1, page_size: int = 1000) -> List['Alert']:
    @staticmethod
    def get_alert_history(alert: 'Alert', page: int = 1, page_size: int = 100) -> List['RichHistory']:
    @staticmethod
    def get_history(query: Optional[Query] = None, page: int = 1, page_size: int = 1000) -> List['RichHistory']:
    @staticmethod
    def get_count(query: Optional[Query] = None) -> int:
    @staticmethod
    def get_counts_by_severity(query: Optional[Query] = None) -> Dict[str, int]:
    @staticmethod
    def get_counts_by_status(query: Optional[Query] = None) -> Dict[str, int]:
    @staticmethod
    def get_top10_count(query: Optional[Query] = None) -> List[Tuple[str, int]]:
    @staticmethod
    def get_topn_count(query: Optional[Query] = None, topn: int = 10) -> List[Tuple[str, int]]:
    @staticmethod
    def get_top10_flapping(query: Optional[Query] = None) -> List[Tuple[str, int]]:
    @staticmethod
    def get_topn_flapping(query: Optional[Query] = None, topn: int = 10) -> List[Tuple[str, int]]:
    @staticmethod
    def get_top10_standing(query: Optional[Query] = None) -> List[Tuple[str, int]]:
    @staticmethod
    def get_topn_standing(query: Optional[Query] = None, topn: int = 10) -> List[Tuple[str, int]]:
    @staticmethod
    def get_environments(query: Optional[Query] = None) -> List[str]:
    @staticmethod
    def get_services(query: Optional[Query] = None) -> List[str]:
    @staticmethod
    def get_groups(query: Optional[Query] = None) -> List[str]:
    @staticmethod
    def get_tags(query: Optional[Query] = None) -> List[str]:
    def add_note(self, text: str) -> 'Note':
    def get_alert_notes(self, page: int = 1, page_size: int = 100) -> List['Note']:
    def delete_note(self, note_id: str) -> None:
    @staticmethod
    def housekeeping(expired_threshold: int, info_threshold: int) -> Tuple[List['Alert'], List['Alert'], List['Alert']]:
    def from_status(self, status: str, text: str = '', timeout: Optional[int] = None) -> 'Alert':
    def from_action(self, action: str, text: str = '', timeout: Optional[int] = None) -> 'Alert':
    def from_expired(self, text: str = '', timeout: Optional[int] = None) -> 'Alert':
    def from_timeout(self, text: str = '', timeout: Optional[int] = None) -> 'Alert':
