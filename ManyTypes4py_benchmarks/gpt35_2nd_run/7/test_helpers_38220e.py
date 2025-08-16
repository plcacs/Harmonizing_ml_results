from typing import Any, TypeVar, cast

def use_s3_backend(method: Callable[..., Any]) -> Callable[..., Any]:
    @mock_aws
    @override_settings(LOCAL_UPLOADS_DIR=None)
    @override_settings(LOCAL_AVATARS_DIR=None)
    @override_settings(LOCAL_FILES_DIR=None)
    def new_method(*args: Any, **kwargs: Any) -> Any:
        backend = S3UploadBackend()
        with mock.patch('zerver.worker.thumbnail.upload_backend', backend), mock.patch('zerver.lib.upload.upload_backend', backend), mock.patch('zerver.views.tusd.upload_backend', backend):
            return method(*args, **kwargs)
    return new_method

def use_db_models(method: Callable[..., Any]) -> Callable[..., Any]:
    def method_patched_with_mock(self: TestCaseT, apps: StateApps) -> None:
        ArchivedAttachment = apps.get_model('zerver', 'ArchivedAttachment')
        ArchivedMessage = apps.get_model('zerver', 'ArchivedMessage')
        ArchivedUserMessage = apps.get_model('zerver', 'ArchivedUserMessage')
        Attachment = apps.get_model('zerver', 'Attachment')
        BotConfigData = apps.get_model('zerver', 'BotConfigData')
        BotStorageData = apps.get_model('zerver', 'BotStorageData')
        Client = apps.get_model('zerver', 'Client')
        CustomProfileField = apps.get_model('zerver', 'CustomProfileField')
        CustomProfileFieldValue = apps.get_model('zerver', 'CustomProfileFieldValue')
        DefaultStream = apps.get_model('zerver', 'DefaultStream')
        DefaultStreamGroup = apps.get_model('zerver', 'DefaultStreamGroup')
        EmailChangeStatus = apps.get_model('zerver', 'EmailChangeStatus')
        DirectMessageGroup = apps.get_model('zerver', 'DirectMessageGroup')
        Message = apps.get_model('zerver', 'Message')
        MultiuseInvite = apps.get_model('zerver', 'MultiuseInvite')
        OnboardingStep = apps.get_model('zerver', 'OnboardingStep')
        PreregistrationUser = apps.get_model('zerver', 'PreregistrationUser')
        PushDeviceToken = apps.get_model('zerver', 'PushDeviceToken')
        Reaction = apps.get_model('zerver', 'Reaction')
        Realm = apps.get_model('zerver', 'Realm')
        RealmAuditLog = apps.get_model('zerver', 'RealmAuditLog')
        RealmDomain = apps.get_model('zerver', 'RealmDomain')
        RealmEmoji = apps.get_model('zerver', 'RealmEmoji')
        RealmFilter = apps.get_model('zerver', 'RealmFilter')
        Recipient = apps.get_model('zerver', 'Recipient')
        Recipient.PERSONAL = 1
        Recipient.STREAM = 2
        Recipient.DIRECT_MESSAGE_GROUP = 3
        ScheduledEmail = apps.get_model('zerver', 'ScheduledEmail')
        ScheduledMessage = apps.get_model('zerver', 'ScheduledMessage')
        Service = apps.get_model('zerver', 'Service')
        Stream = apps.get_model('zerver', 'Stream')
        Subscription = apps.get_model('zerver', 'Subscription')
        UserActivity = apps.get_model('zerver', 'UserActivity')
        UserActivityInterval = apps.get_model('zerver', 'UserActivityInterval')
        UserGroup = apps.get_model('zerver', 'UserGroup')
        UserGroupMembership = apps.get_model('zerver', 'UserGroupMembership')
        UserMessage = apps.get_model('zerver', 'UserMessage')
        UserPresence = apps.get_model('zerver', 'UserPresence')
        UserProfile = apps.get_model('zerver', 'UserProfile')
        UserTopic = apps.get_model('zerver', 'UserTopic')
        zerver_models_patch = mock.patch.multiple('zerver.models', ArchivedAttachment=ArchivedAttachment, ArchivedMessage=ArchivedMessage, ArchivedUserMessage=ArchivedUserMessage, Attachment=Attachment, BotConfigData=BotConfigData, BotStorageData=BotStorageData, Client=Client, CustomProfileField=CustomProfileField, CustomProfileFieldValue=CustomProfileFieldValue, DefaultStream=DefaultStream, DefaultStreamGroup=DefaultStreamGroup, EmailChangeStatus=EmailChangeStatus, DirectMessageGroup=DirectMessageGroup, Message=Message, MultiuseInvite=MultiuseInvite, UserTopic=UserTopic, OnboardingStep=OnboardingStep, PreregistrationUser=PreregistrationUser, PushDeviceToken=PushDeviceToken, Reaction=Reaction, Realm=Realm, RealmAuditLog=RealmAuditLog, RealmDomain=RealmDomain, RealmEmoji=RealmEmoji, RealmFilter=RealmFilter, Recipient=Recipient, ScheduledEmail=ScheduledEmail, ScheduledMessage=ScheduledMessage, Service=Service, Stream=Stream, Subscription=Subscription, UserActivity=UserActivity, UserActivityInterval=UserActivityInterval, UserGroup=UserGroup, UserGroupMembership=UserGroupMembership, UserMessage=UserMessage, UserPresence=UserPresence, UserProfile=UserProfile)
        zerver_test_helpers_patch = mock.patch.multiple('zerver.lib.test_helpers', Client=Client, Message=Message, Subscription=Subscription, UserMessage=UserMessage, UserProfile=UserProfile)
        zerver_test_classes_patch = mock.patch.multiple('zerver.lib.test_classes', Client=Client, Message=Message, Realm=Realm, Recipient=Recipient, Stream=Stream, Subscription=Subscription, UserProfile=UserProfile)
        with zerver_models_patch, zerver_test_helpers_patch, zerver_test_classes_patch:
            method(self, apps)
    return method_patched_with_mock
