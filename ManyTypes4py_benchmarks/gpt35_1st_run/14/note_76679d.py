    def __init__(self, text: str, user: str, note_type: NoteType, **kwargs: Any) -> None:

    @classmethod
    def parse(cls, json: JSON) -> 'Note':

    @property
    def serialize(self) -> JSON:

    def from_document(cls, doc: Dict[str, Any]) -> 'Note':

    def from_record(cls, rec: Any) -> 'Note':

    def from_db(cls, r: Union[Dict[str, Any], Tuple]) -> 'Note':

    def from_alert(alert: Any, text: str) -> 'Note':

    @staticmethod
    def find_by_id(id: str) -> Optional['Note']:

    @staticmethod
    def find_all(query: Optional[Query] = None) -> List['Note']:

    def update(self, **kwargs: Any) -> 'Note':

    @staticmethod
    def delete_by_id(id: str) -> None:
