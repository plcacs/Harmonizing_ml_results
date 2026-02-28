    async with get_client() as client:
        await client.hello()
    