from typing import Any, Dict, List

VALID_CONFIG: Dict[str, Any] = {'sensor': {'platform': DOMAIN, CONF_CLIENT_ID: 'test_client_id', CONF_CLIENT_SECRET: 'test_client_secret', CONF_USERNAME: 'test_username', CONF_PASSWORD: 'test_password', 'subreddits': ['worldnews', 'news']}}
VALID_LIMITED_CONFIG: Dict[str, Any] = {'sensor': {'platform': DOMAIN, CONF_CLIENT_ID: 'test_client_id', CONF_CLIENT_SECRET: 'test_client_secret', CONF_USERNAME: 'test_username', CONF_PASSWORD: 'test_password', 'subreddits': ['worldnews', 'news'], CONF_MAXIMUM: 1}}
INVALID_SORT_BY_CONFIG: Dict[str, Any] = {'sensor': {'platform': DOMAIN, CONF_CLIENT_ID: 'test_client_id', CONF_CLIENT_SECRET: 'test_client_secret', CONF_USERNAME: 'test_username', CONF_PASSWORD: 'test_password', 'subreddits': ['worldnews', 'news'], 'sort_by': 'invalid_sort_by'}}

class ObjectView:
    def __init__(self, d: Dict[str, Any]) -> None:
        self.__dict__ = d

MOCK_RESULTS: Dict[str, List[ObjectView]] = {'results': [ObjectView({'id': 0, 'url': 'http://example.com/1', 'title': 'example1', 'score': '1', 'num_comments': '1', 'created': '', 'selftext': 'example1 selftext'}), ObjectView({'id': 1, 'url': 'http://example.com/2', 'title': 'example2', 'score': '2', 'num_comments': '2', 'created': '', 'selftext': 'example2 selftext'})]}

class MockPraw:
    def __init__(self, client_id: str, client_secret: str, username: str, password: str, user_agent: str) -> None:
        self._data = MOCK_RESULTS

class MockSubreddit:
    def __init__(self, subreddit: str, data: Dict[str, List[ObjectView]]) -> None:
        self._subreddit = subreddit
        self._data = data

    def top(self, limit: int) -> List[ObjectView]:
        return self._return_data(limit)

    def controversial(self, limit: int) -> List[ObjectView]:
        return self._return_data(limit)

    def hot(self, limit: int) -> List[ObjectView]:
        return self._return_data(limit)

    def new(self, limit: int) -> List[ObjectView]:
        return self._return_data(limit)

    def _return_data(self, limit: int) -> List[ObjectView]:
        data = copy.deepcopy(self._data)
        return data['results'][:limit]
