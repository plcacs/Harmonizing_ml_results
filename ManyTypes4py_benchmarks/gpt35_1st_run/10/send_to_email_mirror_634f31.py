    def _does_fixture_path_exist(self, fixture_path: str) -> bool:
        return os.path.exists(fixture_path)

    def _parse_email_json_fixture(self, fixture_path: str) -> EmailMessage:
        ...

    def _parse_email_fixture(self, fixture_path: str) -> EmailMessage:
        ...

    def _prepare_message(self, message: EmailMessage, realm: Realm, stream_name: str, creator: UserProfile, sender: UserProfile) -> None:
        ...
