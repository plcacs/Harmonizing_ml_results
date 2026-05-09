def do_convert_data(mattermost_data_dir: str, output_dir: str, masking_content: bool) -> None:
    username_to_user: dict[str, dict] = {}
    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        raise Exception('Output directory should be empty!')
    mattermost_data_file: str = os.path.join(mattermost_data_dir, 'export.json')
    mattermost_data: dict = mattermost_data_file_to_dict(mattermost_data_file)
    username_to_user = create_username_to_user_mapping(mattermost_data['user'])
    for team in mattermost_data['team']:
        realm_id: str = NEXT_ID('realm_id')
        team_name: str = team['name']
        user_handler: UserHandler = UserHandler()
        subscriber_handler: SubscriberHandler = SubscriberHandler()
        user_id_mapper: IdMapper[str] = IdMapper[str]()
        stream_id_mapper: IdMapper[str] = IdMapper[str]()
        direct_message_group_id_mapper: IdMapper[frozenset[str]] = IdMapper[frozenset[str]]()
        print('Generating data for', team_name)
        realm: dict = make_realm(realm_id, team)
        realm_output_dir: str = os.path.join(output_dir, team_name)
        reset_mirror_dummy_users(username_to_user)
        label_mirror_dummy_users(len(mattermost_data['team']), team_name, mattermost_data, username_to_user)
        convert_user_data(user_handler=user_handler, user_id_mapper=user_id_mapper, user_data_map=username_to_user, realm_id=realm_id, team_name=team_name)
        zerver_stream: list = convert_channel_data(channel_data=mattermost_data['channel'], user_data_map=username_to_user, subscriber_handler=subscriber_handler, stream_id_mapper=stream_id_mapper, user_id_mapper=user_id_mapper, realm_id=realm_id, team_name=team_name)
        realm['zerver_stream'] = zerver_stream
        zerver_direct_message_group: list = []
        if len(mattermost_data['team']) == 1:
            zerver_direct_message_group = convert_direct_message_group_data(direct_message_group_data=mattermost_data['direct_channel'], user_data_map=username_to_user, subscriber_handler=subscriber_handler, direct_message_group_id_mapper=direct_message_group_id_mapper, user_id_mapper=user_id_mapper, realm_id=realm_id, team_name=team_name)
            realm['zerver_huddle'] = zerver_direct_message_group
        all_users: list = user_handler.get_all_users()
        zerver_recipient: list = build_recipients(zerver_userprofile=all_users, zerver_stream=zerver_stream, zerver_direct_message_group=zerver_direct_message_group)
        realm['zerver_recipient'] = zerver_recipient
        stream_subscriptions: list = build_stream_subscriptions(get_users=subscriber_handler.get_users, zerver_recipient=zerver_recipient, zerver_stream=zerver_stream)
        direct_message_group_subscriptions: list = build_direct_message_group_subscriptions(get_users=subscriber_handler.get_users, zerver_recipient=zerver_recipient, zerver_direct_message_group=zerver_direct_message_group)
        personal_subscriptions: list = build_personal_subscriptions(zerver_recipient=zerver_recipient)
        zerver_subscription: list = personal_subscriptions + stream_subscriptions + direct_message_group_subscriptions
        realm['zerver_subscription'] = zerver_subscription
        zerver_realmemoji: list = write_emoticon_data(realm_id=realm_id, custom_emoji_data=mattermost_data['emoji'], data_dir=mattermost_data_dir, output_dir=realm_output_dir)
        realm['zerver_realmemoji'] = zerver_realmemoji
        subscriber_map: dict = make_subscriber_map(zerver_subscription=zerver_subscription)
        total_reactions: list = []
        uploads_list: list = []
        zerver_attachment: list = []
        write_message_data(num_teams=len(mattermost_data['team']), team_name=team_name, realm_id=realm_id, post_data=mattermost_data['post'], zerver_recipient=zerver_recipient, subscriber_map=subscriber_map, output_dir=realm_output_dir, masking_content=masking_content, stream_id_mapper=stream_id_mapper, direct_message_group_id_mapper=direct_message_group_id_mapper, user_id_mapper=user_id_mapper, user_handler=user_handler, zerver_realmemoji=zerver_realmemoji, total_reactions=total_reactions, uploads_list=uploads_list, zerver_attachment=zerver_attachment, mattermost_data_dir=mattermost_data_dir)
        realm['zerver_reaction'] = total_reactions
        realm['zerver_userprofile'] = user_handler.get_all_users()
        realm['sort_by_date'] = True
        create_converted_data_files(realm, realm_output_dir, '/realm.json')
        create_converted_data_files([], realm_output_dir, '/avatars/records.json')
        attachment: dict = {'zerver_attachment': zerver_attachment}
        create_converted_data_files(uploads_list, realm_output_dir, '/uploads/records.json')
        create_converted_data_files(attachment, realm_output_dir, '/attachment.json')
        do_common_export_processes(realm_output_dir)
