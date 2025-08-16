from typing import Union, List, Dict, Any

def b64_to_hex(data: str) -> str:
    return base64.b64decode(data).hex()

def hex_to_b64(data: str) -> str:
    return base64.b64encode(bytes.fromhex(data)).decode()

def get_message_stream_name_from_database(message: Any) -> str:
    stream_id = message.recipient.type_id
    return Stream.objects.get(id=stream_id).name

class UserPushIdentityCompat:
    def __init__(self, user_id: Optional[int] = None, user_uuid: Optional[str] = None) -> None:
        assert user_id is not None or user_uuid is not None
        self.user_id = user_id
        self.user_uuid = user_uuid

    def filter_q(self) -> Any:
        if self.user_id is not None and self.user_uuid is None:
            return Q(user_id=self.user_id)
        elif self.user_uuid is not None and self.user_id is None:
            return Q(user_uuid=self.user_uuid)
        else:
            assert self.user_id is not None and self.user_uuid is not None
            return Q(user_uuid=self.user_uuid) | Q(user_id=self.user_id)

    def __str__(self) -> str:
        result = ''
        if self.user_id is not None:
            result += f'<id:{self.user_id}>'
        if self.user_uuid is not None:
            result += f'<uuid:{self.user_uuid}>'
        return result

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, UserPushIdentityCompat):
            return self.user_id == other.user_id and self.user_uuid == other.user_uuid
        return False

def get_apns_context() -> Any:
    import aioapns
    if not has_apns_credentials():
        return None
    loop = asyncio.new_event_loop()

    async def err_func(request, result):
        pass

    async def make_apns() -> Any:
        return aioapns.APNs(client_cert=settings.APNS_CERT_FILE, key=settings.APNS_TOKEN_KEY_FILE, key_id=settings.APNS_TOKEN_KEY_ID, team_id=settings.APNS_TEAM_ID, max_connection_attempts=APNS_MAX_RETRIES, use_sandbox=settings.APNS_SANDBOX, err_func=err_func, topic='invalid.nonsense')
    apns = loop.run_until_complete(make_apns())
    return APNsContext(apns=apns, loop=loop)

def modernize_apns_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    if 'message_ids' in data:
        return {'alert': data['alert'], 'badge': 0, 'custom': {'zulip': {'message_ids': data['message_ids']}}
    else:
        return data

def send_apple_push_notification(user_identity: Any, devices: List[Any], payload_data: Dict[str, Any], remote: Any = None) -> int:
    if not devices:
        return 0
    import aioapns
    import aioapns.exceptions
    apns_context = get_apns_context()
    if apns_context is None:
        logger.debug('APNs: Dropping a notification because nothing configured.  Set ZULIP_SERVICES_URL (or APNS_CERT_FILE).')
        return 0
    if remote:
        assert settings.ZILENCER_ENABLED
        DeviceTokenClass = RemotePushDeviceToken
    else:
        DeviceTokenClass = PushDeviceToken
    if remote:
        logger.info('APNs: Sending notification for remote user %s:%s to %d devices', remote.uuid, user_identity, len(devices))
    else:
        logger.info('APNs: Sending notification for local user %s to %d devices', user_identity, len(devices))
    payload_data = dict(modernize_apns_payload(payload_data))
    message = {**payload_data.pop('custom', {}), 'aps': payload_data}
    have_missing_app_id = False
    for device in devices:
        if device.ios_app_id is None:
            logger.error('APNs: Missing ios_app_id for user %s device %s', user_identity, device.token)
            have_missing_app_id = True
    if have_missing_app_id:
        devices = [device for device in devices if device.ios_app_id is not None]

    async def send_all_notifications() -> Any:
        requests = [aioapns.NotificationRequest(apns_topic=device.ios_app_id, device_token=device.token, message=message, time_to_live=24 * 3600) for device in devices]
        results = await asyncio.gather(*(apns_context.apns.send_notification(request) for request in requests), return_exceptions=True)
        return zip(devices, results, strict=False)
    results = apns_context.loop.run_until_complete(send_all_notifications())
    successfully_sent_count = 0
    for device, result in results:
        if isinstance(result, aioapns.exceptions.ConnectionError):
            logger.error('APNs: ConnectionError sending for user %s to device %s; check certificate expiration', user_identity, device.token)
        elif isinstance(result, BaseException):
            logger.error('APNs: Error sending for user %s to device %s', user_identity, device.token, exc_info=result)
        elif result.is_successful:
            successfully_sent_count += 1
            logger.info('APNs: Success sending for user %s to device %s', user_identity, device.token)
        elif result.description in ['Unregistered', 'BadDeviceToken', 'DeviceTokenNotForTopic']:
            logger.info('APNs: Removing invalid/expired token %s (%s)', device.token, result.description)
            DeviceTokenClass._default_manager.filter(token=device.token, kind=DeviceTokenClass.APNS).delete()
        else:
            logger.warning('APNs: Failed to send for user %s to device %s: %s', user_identity, device.token, result.description)
    return successfully_sent_count
