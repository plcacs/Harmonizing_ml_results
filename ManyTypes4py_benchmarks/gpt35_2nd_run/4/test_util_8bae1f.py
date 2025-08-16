from datetime import datetime, timedelta
from typing import Any
from unittest.mock import Mock

def database_job_retry_wrapper(description: str, max_retries: int):
    def decorator(func):
        def wrapper(instance, *args, **kwargs):
            for _ in range(max_retries):
                try:
                    result = func(instance, *args, **kwargs)
                    return result
                except OperationalError as e:
                    if instance.engine.dialect.name == SupportedDialect.MYSQL and e.orig.args[0] in RETRYABLE_MYSQL_ERRORS:
                        time.sleep(instance.db_retry_wait)
                    else:
                        raise e
            raise e
        return wrapper
    return decorator

def retryable_database_job(description: str):
    def decorator(func):
        def wrapper(instance, *args, **kwargs):
            try:
                return func(instance, *args, **kwargs)
            except OperationalError as e:
                if instance.engine.dialect.name == SupportedDialect.MYSQL and e.orig.args[0] in RETRYABLE_MYSQL_ERRORS:
                    return False
                raise e
        return wrapper
    return decorator

def retryable_database_job_method(description: str):
    def decorator(func):
        def wrapper(self, instance, *args, **kwargs):
            try:
                return func(self, instance, *args, **kwargs)
            except OperationalError as e:
                if instance.engine.dialect.name == SupportedDialect.MYSQL and e.orig.args[0] in RETRYABLE_MYSQL_ERRORS:
                    return False
                raise e
        return wrapper
    return decorator
