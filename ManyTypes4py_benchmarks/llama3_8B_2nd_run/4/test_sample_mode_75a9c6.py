import os
from datetime import datetime
from typing import List, Optional, Tuple
import freezegun
import pytest
import pytz
from pytest_mock import MockerFixture
from dbt.artifacts.resources.types import BatchSize
from dbt.event_time.sample_window import SampleWindow
from dbt.events.types import JinjaLogInfo
from dbt.materializations.incremental.microbatch import MicrobatchBuilder
from dbt.tests.util import read_file, relation_from_name, run_dbt, write_file
from tests.utils import EventCatcher

class BaseSampleMode:
    def assert_row_count(self, project: pytest.Project, relation_name: str, expected_row_count: int) -> None:
        # ...

class TestBasicSampleMode(BaseSampleMode):
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        # ...

    @pytest.fixture
    def event_catcher(self) -> EventCatcher:
        # ...

    @pytest.mark.parametrize('sample_mode_available', [True, False])
    @freezegun.freeze_time('2025-01-03T02:03:0Z')
    def test_sample_mode(self, project: pytest.Project, mocker: MockerFixture, event_catcher: EventCatcher, sample_mode_available: bool) -> None:
        # ...

class TestMicrobatchSampleMode(BaseSampleMode):
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        # ...

    @pytest.fixture
    def event_time_start_catcher(self) -> EventCatcher:
        # ...

    @pytest.fixture
    def event_time_end_catcher(self) -> EventCatcher:
        # ...

    @pytest.mark.parametrize('sample_mode_available', [True, False])
    @freezegun.freeze_time('2025-01-03T02:03:0Z')
    def test_sample_mode(self, project: pytest.Project, mocker: MockerFixture, event_time_end_catcher: EventCatcher, event_time_start_catcher: EventCatcher, sample_mode_available: bool) -> None:
        # ...

class TestIncrementalModelSampleModeRelative(BaseSampleMode):
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        # ...

    @pytest.fixture
    def event_catcher(self) -> EventCatcher:
        # ...

    @pytest.mark.parametrize('sample_mode_available', [True, False])
    @freezegun.freeze_time('2025-01-06T18:03:0Z')
    def test_incremental_model_sample(self, project: pytest.Project, mocker: MockerFixture, event_catcher: EventCatcher, sample_mode_available: bool) -> None:
        # ...

class TestIncrementalModelSampleModeSpecific(BaseSampleMode):
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        # ...

    @pytest.fixture
    def event_catcher(self) -> EventCatcher:
        # ...

    @pytest.mark.parametrize('sample_mode_available', [True, False])
    def test_incremental_model_sample(self, project: pytest.Project, mocker: MockerFixture, event_catcher: EventCatcher, sample_mode_available: bool) -> None:
        # ...

class TestSampleSeedRefs(BaseSampleMode):
    @pytest.fixture(scope='class')
    def seeds(self) -> dict[str, str]:
        # ...

    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        # ...

    @pytest.mark.parametrize('sample_mode_available', [True, False])
    def test_sample_mode(self, project: pytest.Project, mocker: MockerFixture, sample_mode_available: bool) -> None:
        # ...
