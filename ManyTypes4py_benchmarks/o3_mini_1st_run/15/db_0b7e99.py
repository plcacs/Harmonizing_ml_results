from uuid import uuid4
from typing import Dict, Any, Optional, List
from boto3.dynamodb.conditions import Key

DEFAULT_USERNAME: str = 'default'


class TodoDB(object):
    def list_items(self, username: str = DEFAULT_USERNAME) -> List[Dict[str, Any]]:
        pass

    def add_item(
        self,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        username: str = DEFAULT_USERNAME,
    ) -> str:
        pass

    def get_item(self, uid: str, username: str = DEFAULT_USERNAME) -> Dict[str, Any]:
        pass

    def delete_item(self, uid: str, username: str = DEFAULT_USERNAME) -> None:
        pass

    def update_item(
        self,
        uid: str,
        description: Optional[str] = None,
        state: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        username: str = DEFAULT_USERNAME,
    ) -> None:
        pass


class InMemoryTodoDB(TodoDB):
    def __init__(self, state: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None) -> None:
        if state is None:
            state = {}
        self._state: Dict[str, Dict[str, Dict[str, Any]]] = state

    def list_all_items(self) -> List[Dict[str, Any]]:
        all_items: List[Dict[str, Any]] = []
        for username in self._state:
            all_items.extend(self.list_items(username))
        return all_items

    def list_items(self, username: str = DEFAULT_USERNAME) -> List[Dict[str, Any]]:
        return list(self._state.get(username, {}).values())

    def add_item(
        self,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        username: str = DEFAULT_USERNAME,
    ) -> str:
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

    def get_item(self, uid: str, username: str = DEFAULT_USERNAME) -> Dict[str, Any]:
        return self._state[username][uid]

    def delete_item(self, uid: str, username: str = DEFAULT_USERNAME) -> None:
        del self._state[username][uid]

    def update_item(
        self,
        uid: str,
        description: Optional[str] = None,
        state: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        username: str = DEFAULT_USERNAME,
    ) -> None:
        item: Dict[str, Any] = self._state[username][uid]
        if description is not None:
            item['description'] = description
        if state is not None:
            item['state'] = state
        if metadata is not None:
            item['metadata'] = metadata


class DynamoDBTodo(TodoDB):
    def __init__(self, table_resource: Any) -> None:
        self._table: Any = table_resource

    def list_all_items(self) -> List[Dict[str, Any]]:
        response: Dict[str, Any] = self._table.scan()
        return response['Items']

    def list_items(self, username: str = DEFAULT_USERNAME) -> List[Dict[str, Any]]:
        response: Dict[str, Any] = self._table.query(KeyConditionExpression=Key('username').eq(username))
        return response['Items']

    def add_item(
        self,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        username: str = DEFAULT_USERNAME,
    ) -> str:
        uid: str = str(uuid4())
        self._table.put_item(
            Item={
                'username': username,
                'uid': uid,
                'description': description,
                'state': 'unstarted',
                'metadata': metadata if metadata is not None else {},
            }
        )
        return uid

    def get_item(self, uid: str, username: str = DEFAULT_USERNAME) -> Dict[str, Any]:
        response: Dict[str, Any] = self._table.get_item(Key={'username': username, 'uid': uid})
        return response['Item']

    def delete_item(self, uid: str, username: str = DEFAULT_USERNAME) -> None:
        self._table.delete_item(Key={'username': username, 'uid': uid})

    def update_item(
        self,
        uid: str,
        description: Optional[str] = None,
        state: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        username: str = DEFAULT_USERNAME,
    ) -> None:
        item: Dict[str, Any] = self.get_item(uid, username)
        if description is not None:
            item['description'] = description
        if state is not None:
            item['state'] = state
        if metadata is not None:
            item['metadata'] = metadata
        self._table.put_item(Item=item)