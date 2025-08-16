    def get_entry(resource: KeyValueResource, key: Key) -> KeyValueEntry:
    def get_value(cls, resource: KeyValueResource, key: Key, codec: KeyValueCodec) -> Any:
    def delete_entry(resource: KeyValueResource, key: Key) -> bool:
    def delete_expired_entries(resource: KeyValueResource) -> None:
    def create_entry(resource: KeyValueResource, value: Any, codec: KeyValueCodec, key: UUID = None, expires_on: datetime = None) -> KeyValueEntry:
    def upsert_entry(resource: KeyValueResource, value: Any, codec: KeyValueCodec, key: UUID, expires_on: datetime = None) -> KeyValueEntry:
    def update_entry(resource: KeyValueResource, value: Any, codec: KeyValueCodec, key: UUID, expires_on: datetime = None) -> KeyValueEntry:
