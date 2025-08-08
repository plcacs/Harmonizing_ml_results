from typing import List, Dict, Any

class RawReactionRow(TypedDict):
    emoji_name: str
    emoji_code: str
    reaction_type: str
    user_id: int

def sew_messages_and_reactions(messages: List[Dict[str, Any]], reactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ...

def sew_messages_and_submessages(messages: List[Dict[str, Any]], submessages: List[Dict[str, Any]]) -> None:
    ...

def extract_message_dict(message_bytes: bytes) -> Dict[str, Any]:
    ...

def stringify_message_dict(message_dict: Dict[str, Any]) -> bytes:
    ...

def message_to_encoded_cache(message: Message, realm_id: int = None) -> bytes:
    ...

def update_message_cache(changed_messages: List[Message], realm_id: int = None) -> List[int]:
    ...

def save_message_rendered_content(message: Message, content: str) -> str:
    ...

class ReactionDict:
    @staticmethod
    def build_dict_from_raw_db_row(row: RawReactionRow) -> Dict[str, Any]:
        ...

class MessageDict:
    @staticmethod
    def wide_dict(message: Message, realm_id: int = None) -> Dict[str, Any]:
        ...

    @staticmethod
    def post_process_dicts(objs: List[Dict[str, Any]], *, apply_markdown: bool, client_gravatar: bool, allow_empty_topic_name: bool, realm: Realm, user_recipient_id: int) -> None:
        ...

    @staticmethod
    def finalize_payload(obj: Dict[str, Any], *, apply_markdown: bool, client_gravatar: bool, allow_empty_topic_name: bool, keep_rendered_content: bool = False, skip_copy: bool = False, can_access_sender: bool, realm_host: str, is_incoming_1_to_1: bool) -> Dict[str, Any]:
        ...

    @staticmethod
    def sew_submessages_and_reactions_to_msgs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ...

    @staticmethod
    def messages_to_encoded_cache(messages: List[Message], realm_id: int = None) -> Dict[int, bytes]:
        ...

    @staticmethod
    def ids_to_dict(needed_ids: List[int]) -> List[Dict[str, Any]]:
        ...

    @staticmethod
    def build_dict_from_raw_db_row(row: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @staticmethod
    def build_message_dict(message_id: int, last_edit_time: datetime, edit_history_json: str, content: str, topic_name: str, date_sent: datetime, rendered_content: str, rendered_content_version: str, sender_id: int, sender_realm_id: int, sending_client_name: str, rendering_realm_id: int, recipient_id: int, recipient_type: int, recipient_type_id: int, reactions: List[Dict[str, Any]], submessages: List[Dict[str, Any]]) -> Dict[str, Any]:
        ...

    @staticmethod
    def bulk_hydrate_sender_info(objs: List[Dict[str, Any]]) -> None:
        ...

    @staticmethod
    def hydrate_recipient_info(obj: Dict[str, Any], display_recipient: DisplayRecipientT) -> None:
        ...

    @staticmethod
    def bulk_hydrate_recipient_info(objs: List[Dict[str, Any]]) -> None:
        ...

    @staticmethod
    def set_sender_avatar(obj: Dict[str, Any], client_gravatar: bool, can_access_sender: bool = True) -> None:
        ...
