from uuid import uuid4
from typing import Any, Dict, List, Optional, Mapping, TypedDict, Protocol, cast

from boto3.dynamodb.conditions import Key

DEFAULT_USERNAME: str = 'default'


class TodoItem(TypedDict):
    uid: str
    description: str
    state: str
    metadata: Dict[str, Any]
    username: str


class TableResource(Protocol):
    def scan(self, **kwargs: Any) -> Dict[str, Any]: ...
    def query(self, *, KeyConditionExpression: Any, **kwargs: Any) -> Dict[str, Any]: ...
    def put_item(self, *, Item: Mapping[str, Any]) -> Dict[str, Any]: ...
    def get_item(self, *, Key: Mapping[str, Any]) -> Dict[str, Any]: ...
    def delete_item(self, *, Key: Mapping[str, Any]) -> Dict[str, Any]: ...


class TodoDB(object):
    def list_items(self, username: str = DEFAULT_USERNAME) -> List[TodoItem]:
        pass

    def add_item(self, description: str, metadata: Optional[Dict[str, Any]] = None, username: str = DEFAULT_USERNAME) -> str:
        pass

    def get_item(self, uid: str, username: str = DEFAULT_USERNAME) -> TodoItem:
        pass

    def delete_item(self, uid: str, username: str = DEFAULT_USERNAME) -> None:
        pass

    def update_item(
        self,
        uid: str,
        description: Optional[str] = None,
        state: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        username: str = DEFAULT_USERNAME
    ) -> None:
        pass


class InMemoryTodoDB(TodoDB):
    def __init__(self, state: Optional[Dict[str, Dict[str, TodoItem]]] = None) -> None:
        if state is None:
            state = {}
        self._state: Dict[str, Dict[str, TodoItem]] = state

    def list_all_items(self) -> List[TodoItem]:
        all_items: List[TodoItem] = []
        for username in self._state:
            all_items.extend(self.list_items(username))
        return all_items

    def list_items(self, username: str = DEFAULT_USERNAME) -> List[TodoItem]:
        return list(self._state.get(username, {}).values())

    def add_item(self, description: str, metadata: Optional[Dict[str, Any]] = None, username: str = DEFAULT_USERNAME) -> str:
        if username not in self._state:
            self._state[username] = {}
        uid: str = str(uuid4())
        self._state[username][uid] = {
            'uid': uid,
            'description': description,
            'state': 'unstarted',
            'metadata': metadata if metadata is not None else {},
            'username': username,
        }
        return uid

    def get_item(self, uid: str, username: str = DEFAULT_USERNAME) -> TodoItem:
        return self._state[username][uid]

    def delete_item(self, uid: str, username: str = DEFAULT_USERNAME) -> None:
        del self._state[username][uid]

    def update_item(
        self,
        uid: str,
        description: Optional[str] = None,
        state: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        username: str = DEFAULT_USERNAME
    ) -> None:
        item = self._state[username][uid]
        if description is not None:
            item['description'] = description
        if state is not None:
            item['state'] = state
        if metadata is not None:
            item['metadata'] = metadata


class DynamoDBTodo(TodoDB):
    def __init__(self, table_resource: TableResource) -> None:
        self._table: TableResource = table_resource

    def list_all_items(self) -> List[TodoItem]:
        response: Dict[str, Any] = self._table.scan()
        return cast(List[TodoItem], response['Items'])

    def list_items(self, username: str = DEFAULT_USERNAME) -> List[TodoItem]:
        response: Dict[str, Any] = self._table.query(KeyConditionExpression=Key('username').eq(username))
        return cast(List[TodoItem], response['Items'])

    def add_item(self, description: str, metadata: Optional[Dict[str, Any]] = None, username: str = DEFAULT_USERNAME) -> str:
        uid: str = str(uuid4())
        item: TodoItem = {
            'username': username,
            'uid': uid,
            'description': description,
            'state': 'unstarted',
            'metadata': metadata if metadata is not None else {},
        }
        self._table.put_item(Item=item)
        return uid

    def get_item(self, uid: str, username: str = DEFAULT_USERNAME) -> TodoItem:
        response: Dict[str, Any] = self._table.get_item(Key={'username': username, 'uid': uid})
        return cast(TodoItem, response['Item'])

    def delete_item(self, uid: str, username: str = DEFAULT_USERNAME) -> None:
        self._table.delete_item(Key={'username': username, 'uid': uid})

    def update_item(
        self,
        uid: str,
        description: Optional[str] = None,
        state: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        username: str = DEFAULT_USERNAME
    ) -> None:
        item = self.get_item(uid, username)
        if description is not None:
            item['description'] = description
        if state is not None:
            item['state'] = state
        if metadata is not None:
            item['metadata'] = metadata
        self._table.put_item(Item=item)