        def apply_token_indexers(self, instance: Instance) -> None:
            instance["source"].token_indexers = self._token_indexers
        