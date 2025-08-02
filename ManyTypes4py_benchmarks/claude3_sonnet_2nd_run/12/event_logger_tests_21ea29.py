import logging
import time
import unittest
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast
from unittest.mock import patch
from flask import current_app
from freezegun import freeze_time
from superset import security_manager
from superset.utils.log import AbstractEventLogger, DBEventLogger, get_event_logger_from_cfg_value
from tests.integration_tests.test_app import app

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

class TestEventLogger(unittest.TestCase):

    def test_correct_config_object(self) -> None:
        obj: DBEventLogger = DBEventLogger()
        res: AbstractEventLogger = get_event_logger_from_cfg_value(obj)
        assert obj is res

    def test_config_class_deprecation(self) -> None:
        res: Optional[AbstractEventLogger] = None
        with self.assertLogs(level='WARNING'):
            res = get_event_logger_from_cfg_value(DBEventLogger)
        assert isinstance(res, DBEventLogger)

    def test_raises_typeerror_if_not_abc(self) -> None:
        with self.assertRaises(TypeError):
            get_event_logger_from_cfg_value(logging.getLogger())

    @patch.object(DBEventLogger, 'log')
    def test_log_this(self, mock_log: Any) -> None:
        logger: DBEventLogger = DBEventLogger()

        @logger.log_this
        def test_func() -> int:
            time.sleep(0.05)
            return 1
        with app.test_request_context('/superset/dashboard/1/?myparam=foo'):
            result: int = test_func()
            payload: Dict[str, Any] = mock_log.call_args[1]
            assert result == 1
            assert payload['records'] == [{'myparam': 'foo', 'path': '/superset/dashboard/1/', 'url_rule': '/superset/dashboard/<dashboard_id_or_slug>/', 'object_ref': test_func.__qualname__}]
            assert payload['duration_ms'] >= 50

    @patch.object(DBEventLogger, 'log')
    def test_log_this_with_extra_payload(self, mock_log: Any) -> None:
        logger: DBEventLogger = DBEventLogger()

        @logger.log_this_with_extra_payload
        def test_func(arg1: int, add_extra_log_payload: Callable[..., None], karg1: int = 1) -> int:
            time.sleep(0.1)
            add_extra_log_payload(foo='bar')
            return arg1 * karg1
        with app.test_request_context():
            result: int = test_func(1, karg1=2)
            payload: Dict[str, Any] = mock_log.call_args[1]
            assert result == 2
            assert payload['records'] == [{'foo': 'bar', 'path': '/', 'karg1': 2, 'object_ref': test_func.__qualname__}]
            assert payload['duration_ms'] >= 100

    @patch('superset.utils.core.g', spec={})
    @freeze_time('Jan 14th, 2020', auto_tick_seconds=15)
    def test_context_manager_log(self, mock_g: Any) -> None:

        class DummyEventLogger(AbstractEventLogger):

            def __init__(self) -> None:
                self.records: List[Dict[str, Any]] = []

            def log(self, user_id: Optional[int], action: str, dashboard_id: Optional[int], duration_ms: int, slice_id: Optional[int], referrer: Optional[str], *args: Any, **kwargs: Any) -> None:
                self.records.append({**kwargs, 'user_id': user_id, 'duration': duration_ms})
        logger: DummyEventLogger = DummyEventLogger()
        with app.test_request_context():
            mock_g.user = security_manager.find_user('gamma')
            with logger(action='foo', engine='bar'):
                pass
        assert logger.records == [{'records': [{'path': '/', 'engine': 'bar'}], 'database_id': None, 'user_id': 2, 'duration': 15000, 'curated_payload': {}, 'curated_form_data': {}}]

    @patch('superset.utils.core.g', spec={})
    def test_context_manager_log_with_context(self, mock_g: Any) -> None:

        class DummyEventLogger(AbstractEventLogger):

            def __init__(self) -> None:
                self.records: List[Dict[str, Any]] = []

            def log(self, user_id: Optional[int], action: str, dashboard_id: Optional[int], duration_ms: int, slice_id: Optional[int], referrer: Optional[str], *args: Any, **kwargs: Any) -> None:
                self.records.append({**kwargs, 'user_id': user_id, 'duration': duration_ms})
        logger: DummyEventLogger = DummyEventLogger()
        with app.test_request_context():
            mock_g.user = security_manager.find_user('gamma')
            logger.log_with_context(action='foo', duration=timedelta(days=64, seconds=29156, microseconds=10), object_ref={'baz': 'food'}, log_to_statsd=False, payload_override={'engine': 'sqlite'})
        assert logger.records == [{'records': [{'path': '/', 'object_ref': {'baz': 'food'}, 'payload_override': {'engine': 'sqlite'}}], 'database_id': None, 'user_id': 2, 'duration': 5558756000, 'curated_payload': {}, 'curated_form_data': {}}]

    @patch('superset.utils.core.g', spec={})
    def test_log_with_context_user_null(self, mock_g: Any) -> None:

        class DummyEventLogger(AbstractEventLogger):

            def __init__(self) -> None:
                self.records: List[Dict[str, Any]] = []

            def log(self, user_id: Optional[int], action: str, dashboard_id: Optional[int], duration_ms: int, slice_id: Optional[int], referrer: Optional[str], *args: Any, **kwargs: Any) -> None:
                self.records.append({**kwargs, 'user_id': user_id, 'duration': duration_ms})
        logger: DummyEventLogger = DummyEventLogger()
        with app.test_request_context():
            mock_g.side_effect = Exception('oops')
            logger.log_with_context(action='foo', duration=timedelta(days=64, seconds=29156, microseconds=10), object_ref={'baz': 'food'}, log_to_statsd=False, payload_override={'engine': 'sqlite'})
        assert logger.records[0]['user_id'] == None
