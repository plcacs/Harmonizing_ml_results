    def __init__(self, text: str, user: str, note_type: NoteType, **kwargs: Any) -> None:

    @classmethod
    def parse(cls, json: JSON) -> 'Note':

    @property
    def serialize(self) -> JSON:

    def __repr__(self) -> str:

    @classmethod
    def from_document(cls, doc: JSON) -> 'Note':

    @classmethod
    def from_record(cls, rec: Any) -> 'Note':

    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple]) -> 'Note':

    def create(self) -> 'Note':

    @staticmethod
    def from_alert(alert: Any, text: str) -> 'Note':

    @staticmethod
    def find_by_id(id: str) -> 'Note':

    @staticmethod
    def find_all(query: Optional[Query] = None) -> List['Note']:

    def update(self, **kwargs: Any) -> 'Note':

    def delete(self) -> None:

    @staticmethod
    def delete_by_id(id: str) -> None:
