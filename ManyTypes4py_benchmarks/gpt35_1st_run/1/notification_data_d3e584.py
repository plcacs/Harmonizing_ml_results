    def __post_init__(self) -> None:
    def from_user_id_sets(cls, *, user_id: int, flags: Collection[str], private_message: bool, disable_external_notifications: bool, online_push_user_ids: Collection[int], dm_mention_push_disabled_user_ids: Collection[int], dm_mention_email_disabled_user_ids: Collection[int], stream_push_user_ids: Collection[int], stream_email_user_ids: Collection[int], topic_wildcard_mention_user_ids: Collection[int], stream_wildcard_mention_user_ids: Collection[int], followed_topic_push_user_ids: Collection[int], followed_topic_email_user_ids: Collection[int], topic_wildcard_mention_in_followed_topic_user_ids: Collection[int], stream_wildcard_mention_in_followed_topic_user_ids: Collection[int], muted_sender_user_ids: Collection[int], all_bot_user_ids: Collection[int]) -> 'UserMessageNotificationsData':
    def trivially_should_not_notify(self, acting_user_id: int) -> bool:
    def is_notifiable(self, acting_user_id: int, idle: bool) -> bool:
    def is_push_notifiable(self, acting_user_id: int, idle: bool) -> bool:
    def get_push_notification_trigger(self, acting_user_id: int, idle: bool) -> NotificationTriggers:
    def is_email_notifiable(self, acting_user_id: int, idle: bool) -> bool:
    def get_email_notification_trigger(self, acting_user_id: int, idle: bool) -> NotificationTriggers:
def user_allows_notifications_in_StreamTopic(stream_is_muted: bool, visibility_policy: UserTopic.VisibilityPolicy, stream_specific_setting: bool, global_setting: bool) -> bool:
def get_user_group_mentions_data(mentioned_user_ids: Collection[int], mentioned_user_group_ids: Collection[int], mention_data: MentionData) -> dict:
def get_mentioned_user_group(messages: Collection[dict], user_profile: UserProfile) -> MentionedUserGroup:
