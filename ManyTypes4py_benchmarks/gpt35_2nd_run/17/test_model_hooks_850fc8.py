from pathlib import Path
from typing import List, Dict, Any

class BaseTestPrePost:
    def get_ctx_vars(self, state: str, count: int, project: Any) -> List[Dict[str, Any]]:
        ...

    def check_hooks(self, state: str, project: Any, host: str, count: int = 1) -> None:
        ...

class TestPrePostModelHooks(BaseTestPrePost):
    def test_pre_and_post_run_hooks(self, project: Any, dbt_profile_target: Dict[str, Any]) -> None:
        ...

class TestPrePostModelHooksUnderscores(TestPrePostModelHooks):
    ...

class TestHookRefs(BaseTestPrePost):
    def test_pre_post_model_hooks_refed(self, project: Any, dbt_profile_target: Dict[str, Any]) -> None:
        ...

class TestPrePostModelHooksOnSeeds:
    def test_hooks_on_seeds(self, project: Any) -> None:
        ...

class TestPrePostModelHooksWithMacros(BaseTestPrePost):
    def test_pre_and_post_run_hooks(self, project: Any, dbt_profile_target: Dict[str, Any]) -> None:
        ...

class TestPrePostModelHooksListWithMacros(TestPrePostModelHooksWithMacros):
    ...

class TestHooksRefsOnSeeds:
    def test_hook_with_ref_on_seeds(self, project: Any) -> None:
        ...

class TestPrePostModelHooksOnSeedsPlusPrefixed(TestPrePostModelHooksOnSeeds):
    ...

class TestPrePostModelHooksOnSeedsPlusPrefixedWhitespace(TestPrePostModelHooksOnSeeds):
    ...

class TestPrePostModelHooksOnSnapshots:
    def test_hooks_on_snapshots(self, project: Any) -> None:
        ...

class PrePostModelHooksInConfigSetup(BaseTestPrePost):
    ...

class TestPrePostModelHooksInConfig(PrePostModelHooksInConfigSetup):
    def test_pre_and_post_model_hooks_model(self, project: Any, dbt_profile_target: Dict[str, Any]) -> None:
        ...

class TestPrePostModelHooksInConfigWithCount(PrePostModelHooksInConfigSetup):
    def test_pre_and_post_model_hooks_model_and_project(self, project: Any, dbt_profile_target: Dict[str, Any]) -> None:
        ...

class TestPrePostModelHooksInConfigKwargs(TestPrePostModelHooksInConfig):
    ...

class TestPrePostSnapshotHooksInConfigKwargs(TestPrePostModelHooksOnSnapshots):
    ...

class TestDuplicateHooksInConfigs:
    def test_run_duplicate_hook_defs(self, project: Any) -> None:
        ...
