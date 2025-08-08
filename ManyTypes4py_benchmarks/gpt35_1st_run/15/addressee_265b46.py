    def __init__(self, msg_type: str, user_profiles: Iterable[UserProfile] = None, stream: Stream = None, stream_name: str = None, stream_id: int = None, topic_name: str = None) -> None:

    def user_profiles(self) -> Iterable[UserProfile]:

    def stream(self) -> Stream:

    def stream_name(self) -> str:

    def stream_id(self) -> int:

    def topic_name(self) -> str:

    @staticmethod
    def legacy_build(sender: UserProfile, recipient_type_name: str, message_to: Iterable[Union[str, int]], topic_name: str, realm: Realm = None) -> Addressee:

    @staticmethod
    def for_stream(stream: Stream, topic_name: str) -> Addressee:

    @staticmethod
    def for_stream_name(stream_name: str, topic_name: str) -> Addressee:

    @staticmethod
    def for_stream_id(stream_id: int, topic_name: str) -> Addressee:

    @staticmethod
    def for_private(emails: Sequence[str], realm: Realm) -> Addressee:

    @staticmethod
    def for_user_ids(user_ids: Sequence[int], realm: Realm) -> Addressee:

    @staticmethod
    def for_user_profile(user_profile: UserProfile) -> Addressee:

    @staticmethod
    def for_user_profiles(user_profiles: Iterable[UserProfile]) -> Addressee:
