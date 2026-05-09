class TestDatabaseApi(SupersetTestCase):
    def insert_database(self, database_name: str, sqlalchemy_uri: str, extra: str = '', encrypted_extra: str = '', server_cert: str = '', expose_in_sqllab: bool = False, allow_file_upload: bool = False) -> Database:
        database = Database(database_name=database_name, sqlalchemy_uri=sqlalchemy_uri, extra=extra, encrypted_extra=encrypted_extra, server_cert=server_cert, expose_in_sqllab=expose_in_sqllab, allow_file_upload=allow_file_upload)
        db.session.add(database)
        db.session.commit()
        return database

    @pytest.fixture
    def create_database_with_report(self) -> Database:
        with self.create_app().app_context():
            example_db = get_example_database()
            database = self.insert_database('database_with_report', example_db.sqlalchemy_uri_decrypted, expose_in_sqllab=True)
            report_schedule = ReportSchedule(type=ReportScheduleType.ALERT, name='report_with_database', crontab='* * * * *', database=database)
            db.session.add(report_schedule)
            db.session.commit()
            yield database
            db.session.delete(report_schedule)
            db.session.delete(database)
            db.session.commit()

    @pytest.fixture
    def create_database_with_dataset(self) -> Database:
        with self.create_app().app_context():
            example_db = get_example_database()
            self._database = self.insert_database('database_with_dataset', example_db.sqlalchemy_uri_decrypted, expose_in_sqllab=True)
            table = SqlaTable(schema='main', table_name='ab_permission', database=self._database)
            db.session.add(table)
            db.session.commit()
            yield self._database
            db.session.delete(table)
            db.session.delete(self._database)
            db.session.commit()
            self._database = None

    def create_database_import(self) -> BytesIO:
        buf = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('database_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_metadata_config).encode())
            with bundle.open('database_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_config).encode())
            with bundle.open('database_export/datasets/imported_dataset.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(dataset_config).encode())
        buf.seek(0)
        return buf

    def test_get_items(self) -> None:
        # ...

    def test_get_items_filter(self) -> None:
        # ...

    def test_get_items_not_allowed(self) -> None:
        # ...

    def test_create_database(self) -> None:
        # ...

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.commands.database.create.is_feature_enabled')
    @mock.patch('superset.commands.database.update.is_feature_enabled')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_create_database_with_ssh_tunnel(self, mock_get_all_schema_names, mock_get_all_catalog_names, mock_update_is_feature_enabled, mock_create_is_feature_enabled, mock_test_connection_database_command_run) -> None:
        # ...

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.commands.database.create.is_feature_enabled')
    @mock.patch('superset.commands.database.update.is_feature_enabled')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_create_database_with_missing_port_raises_error(self, mock_get_all_schema_names, mock_get_all_catalog_names, mock_update_is_feature_enabled, mock_create_is_feature_enabled, mock_test_connection_database_command_run) -> None:
        # ...

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.commands.database.create.is_feature_enabled')
    @mock.patch('superset.commands.database.update.is_feature_enabled')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_update_database_with_ssh_tunnel(self, mock_get_all_schema_names, mock_get_all_catalog_names, mock_update_is_feature_enabled, mock_create_is_feature_enabled, mock_test_connection_database_command_run) -> None:
        # ...

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.commands.database.create.is_feature_enabled')
    @mock.patch('superset.commands.database.update.is_feature_enabled')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_update_database_with_missing_port_raises_error(self, mock_get_all_schema_names, mock_get_all_catalog_names, mock_update_is_feature_enabled, mock_create_is_feature_enabled, mock_test_connection_database_command_run) -> None:
        # ...

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.commands.database.create.is_feature_enabled')
    @mock.patch('superset.commands.database.update.is_feature_enabled')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_delete_ssh_tunnel(self, mock_get_all_schema_names, mock_get_all_catalog_names, mock_delete_is_feature_enabled, mock_update_is_feature_enabled, mock_create_is_feature_enabled, mock_test_connection_database_command_run) -> None:
        # ...

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.commands.database.create.is_feature_enabled')
    @mock.patch('superset.commands.database.update.is_feature_enabled')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_update_ssh_tunnel_via_database_api(self, mock_get_all_schema_names, mock_get_all_catalog_names, mock_update_is_feature_enabled, mock_create_is_feature_enabled, mock_test_connection_database_command_run) -> None:
        # ...

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.commands.database.create.is_feature_enabled')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_cascade_delete_ssh_tunnel(self, mock_get_all_schema_names, mock_get_all_catalog_names, mock_create_is_feature_enabled, mock_test_connection_database_command_run) -> None:
        # ...

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.commands.database.create.is_feature_enabled')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_do_not_create_database_if_ssh_tunnel_creation_fails(self, mock_get_all_schema_names, mock_get_all_catalog_names, mock_create_is_feature_enabled, mock_test_connection_database_command_run, mock_rollback) -> None:
        # ...

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_get_database_returns_related_ssh_tunnel(self, mock_get_all_schema_names, mock_get_all_catalog_names, mock_test_connection_database_command_run) -> None:
        # ...

    def test_if_ssh_tunneling_flag_is_not_active_it_raises_new_exception(self) -> None:
        # ...

    def test_get_table_details_with_slash_in_table_name(self) -> None:
        # ...

    def test_create_database_invalid_configuration_method(self) -> None:
        # ...

    def test_create_database_no_configuration_method(self) -> None:
        # ...

    def test_create_database_server_cert_validate(self) -> None:
        # ...

    def test_create_database_json_validate(self) -> None:
        # ...

    def test_create_database_extra_metadata_validate(self) -> None:
        # ...

    def test_create_database_unique_validate(self) -> None:
        # ...

    def test_create_database_uri_validate(self) -> None:
        # ...

    @mock.patch('superset.views.core.app.config', {**app.config, 'PREVENT_UNSAFE_DB_CONNECTIONS': True})
    def test_create_database_fail_sqlite(self) -> None:
        # ...

    def test_create_database_conn_fail(self) -> None:
        # ...

    def test_update_database(self) -> None:
        # ...

    def test_update_database_conn_fail(self) -> None:
        # ...

    def test_update_database_uniqueness(self) -> None:
        # ...

    def test_update_database_invalid(self) -> None:
        # ...

    def test_update_database_uri_validate(self) -> None:
        # ...

    def test_update_database_with_invalid_configuration_method(self) -> None:
        # ...

    def test_update_database_with_no_configuration_method(self) -> None:
        # ...

    def test_delete_database(self) -> None:
        # ...

    def test_delete_database_not_found(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_database_with_dataset')
    def test_delete_database_with_datasets(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_database_with_report')
    def test_delete_database_with_report(self) -> None:
        # ...

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_get_table_metadata(self) -> None:
        # ...

    def test_info_security_database(self) -> None:
        # ...

    def test_get_invalid_database_table_metadata(self) -> None:
        # ...

    def test_get_invalid_table_table_metadata(self) -> None:
        # ...

    def test_get_table_metadata_no_db_permission(self) -> None:
        # ...

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_get_table_extra_metadata_deprecated(self) -> None:
        # ...

    def test_get_invalid_database_table_extra_metadata_deprecated(self) -> None:
        # ...

    def test_get_invalid_table_table_extra_metadata_deprecated(self) -> None:
        # ...

    def test_get_select_star(self) -> None:
        # ...

    def test_get_select_star_not_allowed(self) -> None:
        # ...

    def test_get_select_star_not_found_database(self) -> None:
        # ...

    def test_get_select_star_not_found_table(self) -> None:
        # ...

    def test_get_allow_file_upload_filter(self) -> None:
        # ...

    def test_get_allow_file_upload_filter_no_schema(self) -> None:
        # ...

    def test_get_allow_file_upload_filter_allow_file_false(self) -> None:
        # ...

    def test_get_allow_file_upload_false(self) -> None:
        # ...

    def test_get_allow_file_upload_false_no_extra(self) -> None:
        # ...

    @mock.patch('superset.utils.log.logger')
    def test_get_allow_file_upload_true_csv(self, logger_mock) -> None:
        # ...

    def test_get_allow_file_upload_filter_no_permission(self) -> None:
        # ...

    def test_get_allow_file_upload_filter_with_permission(self) -> None:
        # ...

    def test_database_schemas(self) -> None:
        # ...

    def test_database_schemas_not_found(self) -> None:
        # ...

    def test_database_schemas_invalid_query(self) -> None:
        # ...

    def test_database_tables(self) -> None:
        # ...

    @patch('superset.utils.log.logger')
    def test_database_tables_not_found(self, logger_mock) -> None:
        # ...

    def test_database_tables_invalid_query(self) -> None:
        # ...

    @patch('superset.utils.log.logger')
    @mock.patch('superset.security.manager.SupersetSecurityManager.can_access_database')
    @mock.patch('superset.models.core.Database.get_all_table_names_in_schema')
    def test_database_tables_unexpected_error(self, mock_get_all_table_names_in_schema, mock_can_access_database, logger_mock) -> None:
        # ...

    def test_test_connection(self) -> None:
        # ...

    def test_test_connection_failed(self) -> None:
        # ...

    def test_test_connection_unsafe_uri(self) -> None:
        # ...

    @mock.patch('superset.commands.database.test_connection.DatabaseDAO.build_db_for_connection_test')
    @mock.patch('superset.commands.database.test_connection.event_logger')
    def test_test_connection_failed_invalid_hostname(self, mock_event_logger, mock_build_db) -> None:
        # ...

    @pytest.mark.usefixtures('load_unicode_dashboard_with_position', 'load_energy_table_with_slice', 'load_world_bank_dashboard_with_slices', 'load_birth_names_dashboard_with_slices')
    def test_get_database_related_objects(self) -> None:
        # ...

    def test_get_database_related_objects_not_found(self) -> None:
        # ...

    def test_export_database(self) -> None:
        # ...

    def test_export_database_not_allowed(self) -> None:
        # ...

    def test_export_database_non_existing(self) -> None:
        # ...

    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database(self, mock_add_permissions) -> None:
        # ...

    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_overwrite(self, mock_add_permissions) -> None:
        # ...

    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_invalid(self, mock_add_permissions) -> None:
        # ...

    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_password(self, mock_add_permissions) -> None:
        # ...

    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_password_provided(self, mock_add_permissions) -> None:
        # ...

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_password(self, mock_add_permissions, mock_schema_is_feature_enabled) -> None:
        # ...

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_password_provided(self, mock_add_permissions, mock_schema_is_feature_enabled) -> None:
        # ...

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_private_key_and_password(self, mock_add_permissions, mock_schema_is_feature_enabled) -> None:
        # ...

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_private_key_and_password_provided(self, mock_add_permissions, mock_schema_is_feature_enabled) -> None:
        # ...

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_feature_flag_disabled(self, mock_add_permissions, mock_schema_is_feature_enabled) -> None:
        # ...

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_feature_no_credentials(self, mock_add_permissions, mock_schema_is_feature_enabled) -> None:
        # ...

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_feature_mix_credentials(self, mock_add_permissions, mock_schema_is_feature_enabled) -> None:
        # ...

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_feature_only_pk_passwd(self, mock_add_permissions, mock_schema_is_feature_enabled) -> None:
        # ...

    @mock.patch('superset.databases.api.get_available_engine_specs')
    @mock.patch('superset.databases.api.app')
    def test_available(self, app, get_available_engine_specs) -> None:
        # ...

    @mock.patch('superset.databases.api.get_available_engine_specs')
    @mock.patch('superset.databases.api.app')
    def test_available_no_default(self, app, get_available_engine_specs) -> None:
        # ...

    def test_validate_parameters_invalid_payload_format(self) -> None:
        # ...

    def test_validate_parameters_invalid_payload_schema(self) -> None:
        # ...

    def test_validate_parameters_missing_fields(self) -> None:
        # ...

    @mock.patch('superset.db_engine_specs.base.is_hostname_valid')
    @mock.patch('superset.db_engine_specs.base.is_port_open')
    @mock.patch('superset.databases.api.ValidateDatabaseParametersCommand')
    def test_validate_parameters_valid_payload(self, ValidateDatabaseParametersCommand, is_port_open, is_hostname_valid) -> None:
        # ...

    def test_validate_parameters_invalid_port(self) -> None:
        # ...

    @mock.patch('superset.db_engine_specs.base.is_hostname_valid')
    def test_validate_parameters_invalid_host(self, is_hostname_valid) -> None:
        # ...

    @mock.patch('superset.db_engine_specs.base.is_hostname_valid')
    def test_validate_parameters_invalid_port_range(self, is_hostname_valid) -> None:
        # ...

    def test_get_related_objects(self) -> None:
        # ...

    @mock.patch.dict('superset.config.SQL_VALIDATORS_BY_ENGINE', SQL_VALIDATORS_BY_ENGINE, clear=True)
    def test_validate_sql(self) -> None:
        # ...

    @mock.patch.dict('superset.config.SQL_VALIDATORS_BY_ENGINE', SQL_VALIDATORS_BY_ENGINE, clear=True)
    def test_validate_sql_errors(self) -> None:
        # ...

    @mock.patch.dict('superset.config.SQL_VALIDATORS_BY_ENGINE', SQL_VALIDATORS_BY_ENGINE, clear=True)
    def test_validate_sql_not_found(self) -> None:
        # ...

    @mock.patch.dict('superset.config.SQL_VALIDATORS_BY_ENGINE', SQL_VALIDATORS_BY_ENGINE, clear=True)
    def test_validate_sql_validation_fails(self) -> None:
        # ...

    @mock.patch.dict('superset.config.SQL_VALIDATORS_BY_ENGINE', {}, clear=True)
    def test_validate_sql_endpoint_noconfig(self) -> None:
        # ...

    @mock.patch('superset.commands.database.validate_sql.get_validator_by_name')
    @mock.patch.dict('superset.config.SQL_VALIDATORS_BY_ENGINE', PRESTO_SQL_VALIDATORS_BY_ENGINE, clear=True)
    def test_validate_sql_endpoint_failure(self, get_validator_by_name) -> None:
        # ...

    def test_get_databases_with_extra_filters(self) -> None:
        # ...
