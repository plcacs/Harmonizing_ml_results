        from prefect import flow
        from prefect_aws import AwsCredentials
        from prefect_aws.s3 import adownload_from_bucket

        @flow
        async def example_download_from_bucket_flow():
            aws_credentials = AwsCredentials(
                aws_access_key_id="acccess_key_id",
                aws_secret_access_key="secret_access_key"
            )
            data = await adownload_from_bucket(
                bucket="bucket",
                key="key",
                aws_credentials=aws_credentials,
            )

        await example_download_from_bucket_flow()
        