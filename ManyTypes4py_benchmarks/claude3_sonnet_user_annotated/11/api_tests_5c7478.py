def test_get_databases_with_extra_filters(self) -> None:
    """
    API: Test get database with extra query filter.
    Here we are testing our default where all databases
    must be returned if nothing is being set in the config.
    Then, we're adding the patch for the config to add the filter function
    and testing it's being applied.
    """
    self.login(ADMIN_USERNAME)
    extra: dict[str, Any] = {
        "metadata_params": {},
        "engine_params": {},
        "metadata_cache_timeout": {},
        "schemas_allowed_for_file_upload": [],
    }
    example_db = get_example_database()

    if example_db.backend == "sqlite":
        return
    # Create our two databases
    database_data: dict[str, Any] = {
        "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
        "configuration_method": ConfigurationMethod.SQLALCHEMY_FORM,
        "server_cert": None,
        "extra": json.dumps(extra),
    }

    uri = "api/v1/database/"
    rv = self.client.post(
        uri, json={**database_data, "database_name": "dyntest-create-database-1"}
    )
    first_response = json.loads(rv.data.decode("utf-8"))
    assert rv.status_code == 201

    uri = "api/v1/database/"
    rv = self.client.post(
        uri, json={**database_data, "database_name": "create-database-2"}
    )
    second_response = json.loads(rv.data.decode("utf-8"))
    assert rv.status_code == 201

    # The filter function
    def _base_filter(query: Query) -> Query:
        from superset.models.core import Database

        return query.filter(Database.database_name.startswith("dyntest"))

    # Create the Mock
    base_filter_mock = Mock(side_effect=_base_filter)
    dbs = db.session.query(Database).all()
    expected_names: list[str] = [db.database_name for db in dbs]
    expected_names.sort()

    uri = "api/v1/database/"  # noqa: F541
    # Get the list of databases without filter in the config
    rv = self.client.get(uri)
    data = json.loads(rv.data.decode("utf-8"))
    # All databases must be returned if no filter is present
    assert data["count"] == len(dbs)
    database_names: list[str] = [item["database_name"] for item in data["result"]]
    database_names.sort()
    # All Databases because we are an admin
    assert database_names == expected_names
    assert rv.status_code == 200
    # Our filter function wasn't get called
    base_filter_mock.assert_not_called()

    # Now we patch the config to include our filter function
    with patch.dict(
        "superset.views.filters.current_app.config",
        {"EXTRA_DYNAMIC_QUERY_FILTERS": {"databases": base_filter_mock}},
    ):
        uri = "api/v1/database/"  # noqa: F541
        rv = self.client.get(uri)
        data = json.loads(rv.data.decode("utf-8"))
        # Only one database start with dyntest
        assert data["count"] == 1
        database_names = [item["database_name"] for item in data["result"]]
        # Only the database that starts with tests, even if we are an admin
        assert database_names == ["dyntest-create-database-1"]
        assert rv.status_code == 200
        # The filter function is called now that it's defined in our config
        base_filter_mock.assert_called()

    # Cleanup
    first_model = db.session.query(Database).get(first_response.get("id"))
    second_model = db.session.query(Database).get(second_response.get("id"))
    db.session.delete(first_model)
    db.session.delete(second_model)
    db.session.commit()
