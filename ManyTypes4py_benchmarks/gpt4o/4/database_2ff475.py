        from prefect_snowflake.database import SnowflakeConnector

        snowflake_connector = SnowflakeConnector.load("BLOCK_NAME")
        