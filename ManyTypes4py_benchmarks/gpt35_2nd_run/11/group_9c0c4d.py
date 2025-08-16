    def __init__(self, id: str, login: str, name: str, status: str):
    def serialize(self) -> JSON:
    @classmethod
    def from_document(cls, doc: JSON) -> 'GroupUser':
    @classmethod
    def from_record(cls, rec: Tuple) -> 'GroupUser':
    @classmethod
    def from_db(cls, r: Union[Dict, Tuple]) -> 'GroupUser':

    def __init__(self, id: str, users: List['GroupUser']):
    @staticmethod
    def find_by_id(id: str) -> List['GroupUser']:

    def __init__(self, name: str, text: str, **kwargs):
    @classmethod
    def parse(cls, json: JSON) -> 'Group':
    def serialize(self) -> JSON:
    @classmethod
    def from_document(cls, doc: JSON) -> 'Group':
    @classmethod
    def from_record(cls, rec: Tuple) -> 'Group':
    @classmethod
    def from_db(cls, r: Union[Dict, Tuple]) -> 'Group':
    def create(self) -> 'Group':
    @staticmethod
    def find_by_id(id: str) -> 'Group':
    @staticmethod
    def find_all(query: Optional[Query] = None, page: int = 1, page_size: int = 1000) -> List['Group']:
    @staticmethod
    def count(query: Optional[Query] = None) -> int:
    def update(self, **kwargs) -> 'Group':
    def add_user(self, user_id: str):
    def remove_user(self, user_id: str):
    def delete(self):
