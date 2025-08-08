from contextlib import contextmanager
from collections import deque
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any
import pytest
from mode import label
from mode.utils.mocks import AsyncMock, Mock, patch
from faust.livecheck import Case
from faust.livecheck.exceptions import SuiteFailed
from faust.livecheck.models import State, TestReport

class test_Case:

    def test_constructor(self, arg: str, value: Any, expected: Any, *, livecheck: Any) -> None:
    
    async def test__sampler(self, *, case: Case) -> None:
    
    async def test__sample(self, *, case: Case) -> None:
    
    async def test_maybe_trigger(self, *, case: Case) -> None:
    
    async def test_run(self, *, case: Case) -> None:
    
    async def test_trigger(self, *, case: Case) -> None:
    
    def test_now(self, *, case: Case) -> None:
    
    async def test_resolve_signal(self, *, case: Case) -> None:
    
    async def test_execute(self, *, case: Case, execution: Any, current_execution_stack: Any, frozen_monotonic: Any) -> None:
    
    async def test_on_test_start(self, started: float, last_received: float, frequency: float, *, case: Case, runner: Any) -> None:
    
    @pytest.yield_fixture()
    def frozen_monotonic(self) -> Any:
    
    def _patch_monotonic(self) -> Any:
    
    async def test_on_test_skipped(self, *, case: Case, runner: Any, frozen_monotonic: Any) -> None:
    
    async def test_on_test_failed(self, *, case: Case, runner: Any) -> None:
    
    async def test_on_test_error(self, *, case: Case, runner: Any) -> None:
    
    async def test_on_test_timeout(self, *, case: Case, runner: Any) -> None:
    
    async def test__set_test_error_state(self, state: State, failures: int, fail_suite: bool, *, case: Case) -> None:
    
    async def test_on_suite_pass(self, initial_state: State, ts: float, now: float, failed: float, expected_state: State, *, case: Case, runner: Any, execution: Any) -> None:
    
    async def test_post_report(self, *, case: Case) -> None:
    
    async def test__send_frequency__first_stop(self, *, case: Case, loop: Any) -> None:
    
    async def test__send_frequency__no_frequency(self, *, case: Case, loop: Any) -> None:
    
    async def test__send_frequency__last_stop(self, *, case: Case) -> None:
    
    async def test__send_frequency__no_frequency_None(self, *, case: Case) -> None:
    
    async def test__send_frequency__timer_ends(self, *, case: Case) -> None:
    
    async def test__send_frequency(self, *, case: Case) -> None:
    
    async def test__check_frequency(self, *, case: Case) -> None:
    
    async def test__check_frequency__last(self, *, case: Case, frozen_monotonic: Any) -> None:
    
    async def test__check_frequency__should_stop1(self, *, case: Case) -> None:
    
    async def test__check_frequency__last_stop(self, *, case: Case) -> None:
    
    async def test_on_suite_fail(self, initial_state: State, now: float, failed: float, posts_report: bool, *, case: Case) -> None:
    
    def test__maybe_recover_from_failed_state(self, initial_state: State, now: float, failed: float, expected_state: State, *, case: Case) -> None:
    
    def test_failed_longer_than(self, now: float, failed: float, arg: float, expected: bool, *, case: Case) -> None:
    
    def test_seconds_since_last_fail(self, now: float, failed: float, expected: Any, *, case: Case) -> None:
    
    async def test_get_url(self, *, case: Case) -> None:
    
    async def test_post_url(self, *, case: Case) -> None:
    
    async def assert_url_called(self, case: Case, fut: Any, method: str, url: str, **kwargs: Any) -> None:
    
    async def test_url_request(self, *, case: Case, mock_http_client: Any) -> None:
    
    async def test_url_request_fails(self, *, case: Case, mock_http_client: Any) -> None:
    
    async def test_url_request_fails_recover(self, *, case: Case, mock_http_client: Any) -> None:
    
    def test_current_test(self, *, case: Case) -> None:
    
    def test_current_execution(self, *, case: Case) -> None:
    
    def test_label(self, *, case: Case) -> None:
