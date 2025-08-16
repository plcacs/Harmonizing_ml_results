from typing import List, Tuple

def check_schedule_message(sender: UserProfile, client: Client, recipient_type_name: str, message_to: str, topic_name: str, message_content: str, deliver_at: datetime, realm: Realm = None, *, forwarder_user_profile: UserProfile = None, read_by_sender: bool = None) -> int:

def do_schedule_messages(send_message_requests: List[SendMessageRequest], sender: UserProfile, *, read_by_sender: bool = False) -> List[int]:

def notify_update_scheduled_message(user_profile: UserProfile, scheduled_message: ScheduledMessage) -> None:

def edit_scheduled_message(sender: UserProfile, client: Client, scheduled_message_id: int, recipient_type_name: str, message_to: str, topic_name: str, message_content: str, deliver_at: datetime, realm: Realm) -> None:

def notify_remove_scheduled_message(user_profile: UserProfile, scheduled_message_id: int) -> None:

def delete_scheduled_message(user_profile: UserProfile, scheduled_message_id: int) -> None:

def send_scheduled_message(scheduled_message: ScheduledMessage) -> None:

def send_failed_scheduled_message_notification(user_profile: UserProfile, scheduled_message_id: int) -> None:

def try_deliver_one_scheduled_message() -> bool:
