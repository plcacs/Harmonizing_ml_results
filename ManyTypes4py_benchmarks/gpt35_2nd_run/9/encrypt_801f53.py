    def create(self, app_config: dict, *args: Any, **kwargs: Any) -> EncryptedType:
    def init_app(self, app: Flask) -> None:
    def create(self, *args: Any, **kwargs: Any) -> EncryptedType:
    def discover_encrypted_fields(self) -> dict:
    def _read_bytes(self, col_name: str, value: Any) -> bytes:
    def _select_columns_from_table(self, conn: Connection, column_names: list, table_name: str) -> Row:
    def _re_encrypt_row(self, conn: Connection, row: Row, table_name: str, columns: dict) -> None:
    def run(self) -> None:
