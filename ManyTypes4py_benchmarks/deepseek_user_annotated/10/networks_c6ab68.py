        from pydantic_core import MultiHostUrl

        mhu = MultiHostUrl('https://foo.com:123,foo:bar@bar.com/path')
        print(mhu.hosts())
        """
        [
            {'username': None, 'password': None, 'host': 'foo.com', 'port': 123},
            {'username': 'foo', 'password': 'bar', 'host': 'bar.com', 'port': 443}
        ]
        