            with transaction() as txn:
                txn.set("key", "value")
                ...
                assert txn.get("key") == "value"
            