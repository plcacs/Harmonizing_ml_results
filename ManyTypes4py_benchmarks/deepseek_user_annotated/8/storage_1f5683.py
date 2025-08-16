        from prefect.runner.storage import GitRepository

        storage = GitRepository(
            url="https://github.com/org/repo.git",
            credentials={"username": "oauth2", "access_token": "my-access-token"},
        )

        await storage.pull_code()
        