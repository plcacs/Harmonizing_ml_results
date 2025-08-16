from typing import Any, Dict, List, Optional, Union

def create_bucket(self, bucket: str, location: Optional[str] = None, **create_kwargs: Any) -> MagicMock:
def get_bucket(self, bucket: str) -> MagicMock:
def list_blobs(self, bucket: str, prefix: Optional[str] = None) -> List[Blob]:
def query(self, query: str, **kwargs: Any) -> MagicMock:
def dataset(self, dataset: str) -> MagicMock:
def table(self, table: str) -> MagicMock:
def get_dataset(self, dataset: str) -> MagicMock:
def create_dataset(self, dataset: str) -> MagicMock:
def get_table(self, table: str) -> None:
def create_table(self, table: str) -> str:
def insert_rows_json(self, table: str, json_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
def load_table_from_uri(self, uri: str, *args: Any, **kwargs: Any) -> MagicMock:
def load_table_from_file(self, *args: Any, **kwargs: Any) -> MagicMock:
def create_secret(self, request: Any = None, parent: Optional[str] = None, secret_id: Optional[str] = None, **kwds: Any) -> MagicMock:
def add_secret_version(self, request: Any = None, parent: Optional[str] = None, payload: Any = None, **kwds: Any) -> MagicMock:
def access_secret_version(self, request: Any = None, name: Optional[str] = None, **kwds: Any) -> MagicMock:
def delete_secret(self, request: Any = None, name: Optional[str] = None, **kwds: Any) -> str:
def destroy_secret_version(self, name: str, **kwds: Any) -> str:
