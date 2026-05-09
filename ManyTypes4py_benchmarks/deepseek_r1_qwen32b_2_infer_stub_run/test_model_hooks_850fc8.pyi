from pathlib import Path
import pytest
from dbt.exceptions import ParsingError
from dbt_common.exceptions import CompilationError
from dbt.tests.util import run_dbt, write_file

MODEL_PRE_HOOK: str = ...
MODEL_POST_HOOK: str = ...

class BaseTestPrePost:
    @pytest.fixture(scope='class', autouse=True)
    def setUp(self, project) -> pytest.FixtureFunction:
        ...

    def get_ctx_vars(self, state: str, count: int, project: Any) -> list[dict[str, Any]]:
        ...

    def check_hooks(self, state: str, project: Any, host: str, count: int = 1) -> None:
        ...

class TestPrePostModelHooks(BaseTestPrePost):
    @pytest.fixture(scope='class')
    def project_config_update(self) -> pytest.FixtureFunction:
        ...

    @pytest.fixture(scope='class')
    def models(self) -> pytest.FixtureFunction:
        ...

    def test_pre_and_post_run_hooks(self, project: Any, dbt_profile_target: dict) -> None:
        ...

class TestPrePostModelHooksUnderscores(TestPrePostModelHooks):
    @pytest.fixture(scope='class')
    def project_config_update(self) -> pytest.FixtureFunction:
        ...

class TestHookRefs(BaseTestPrePost):
    @pytest.fixture(scope='class')
    def project_config_update(self) -> pytest.FixtureFunction:
        ...

    @pytest.fixture(scope='class')
    def models(self) -> pytest.FixtureFunction:
        ...

    def test_pre_post_model_hooks_refed(self, project: Any, dbt_profile_target: dict) -> None:
        ...

class TestPrePostModelHooksOnSeeds:
    @pytest.fixture(scope='class')
    def seeds(self) -> pytest.FixtureFunction:
        ...

    @pytest.fixture(scope='class')
    def models(self) -> pytest.FixtureFunction:
        ...

    @pytest.fixture(scope='class')
    def project_config_update(self) -> pytest.FixtureFunction:
        ...

    def test_hooks_on_seeds(self, project: Any) -> None:
        ...

class TestPrePostModelHooksWithMacros(BaseTestPrePost):
    @pytest.fixture(scope='class')
    def macros(self) -> pytest.FixtureFunction:
        ...

    @pytest.fixture(scope='class')
    def models(self) -> pytest.FixtureFunction:
        ...

    def test_pre_and_post_run_hooks(self, project: Any, dbt_profile_target: dict) -> None:
        ...

class TestPrePostModelHooksListWithMacros(TestPrePostModelHooksWithMacros):
    @pytest.fixture(scope='class')
    def models(self) -> pytest.FixtureFunction:
        ...

class TestHooksRefsOnSeeds:
    @pytest.fixture(scope='class')
    def seeds(self) -> pytest.FixtureFunction:
        ...

    @pytest.fixture(scope='class')
    def models(self) -> pytest.FixtureFunction:
        ...

    @pytest.fixture(scope='class')
    def project_config_update(self) -> pytest.FixtureFunction:
        ...

    def test_hook_with_ref_on_seeds(self, project: Any) -> None:
        ...

class TestPrePostModelHooksOnSeedsPlusPrefixed(TestPrePostModelHooksOnSeeds):
    @pytest.fixture(scope='class')
    def project_config_update(self) -> pytest.FixtureFunction:
        ...

class TestPrePostModelHooksOnSeedsPlusPrefixedWhitespace(TestPrePostModelHooksOnSeeds):
    @pytest.fixture(scope='class')
    def project_config_update(self) -> pytest.FixtureFunction:
        ...

class TestPrePostModelHooksOnSnapshots:
    @pytest.fixture(scope='class', autouse=True)
    def setUp(self, project) -> pytest.FixtureFunction:
        ...

    @pytest.fixture(scope='class')
    def models(self) -> pytest.FixtureFunction:
        ...

    @pytest.fixture(scope='class')
    def seeds(self) -> pytest.FixtureFunction:
        ...

    @pytest.fixture(scope='class')
    def project_config_update(self) -> pytest.FixtureFunction:
        ...

    def test_hooks_on_snapshots(self, project: Any) -> None:
        ...

class PrePostModelHooksInConfigSetup(BaseTestPrePost):
    @pytest.fixture(scope='class')
    def project_config_update(self) -> pytest.FixtureFunction:
        ...

    @pytest.fixture(scope='class')
    def models(self) -> pytest.FixtureFunction:
        ...

class TestPrePostModelHooksInConfig(PrePostModelHooksInConfigSetup):
    def test_pre_and_post_model_hooks_model(self, project: Any, dbt_profile_target: dict) -> None:
        ...

class TestPrePostModelHooksInConfigWithCount(PrePostModelHooksInConfigSetup):
    @pytest.fixture(scope='class')
    def project_config_update(self) -> pytest.FixtureFunction:
        ...

    def test_pre_and_post_model_hooks_model_and_project(self, project: Any, dbt_profile_target: dict) -> None:
        ...

class TestPrePostModelHooksInConfigKwargs(TestPrePostModelHooksInConfig):
    @pytest.fixture(scope='class')
    def models(self) -> pytest.FixtureFunction:
        ...

class TestPrePostSnapshotHooksInConfigKwargs(TestPrePostModelHooksOnSnapshots):
    @pytest.fixture(scope='class', autouse=True)
    def setUp(self, project) -> pytest.FixtureFunction:
        ...

    @pytest.fixture(scope='class')
    def project_config_update(self) -> pytest.FixtureFunction:
        ...

class TestDuplicateHooksInConfigs:
    @pytest.fixture(scope='class')
    def models(self) -> pytest.FixtureFunction:
        ...

    def test_run_duplicate_hook_defs(self, project: Any) -> None:
        ...