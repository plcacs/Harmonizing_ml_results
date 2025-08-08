from pathlib import Path
from typing import List, Dict, Any
import pytest

MODEL_PRE_HOOK: str = '\n   insert into {{this.schema}}.on_model_hook (\n        test_state,\n        target_dbname,\n        target_host,\n        target_name,\n        target_schema,\n        target_type,\n        target_user,\n        target_pass,\n        target_threads,\n        run_started_at,\n        invocation_id,\n        thread_id\n   ) VALUES (\n    \'start\',\n    \'{{ target.dbname }}\',\n    \'{{ target.host }}\',\n    \'{{ target.name }}\',\n    \'{{ target.schema }}\',\n    \'{{ target.type }}\',\n    \'{{ target.user }}\',\n    \'{{ target.get("pass", "") }}\',\n    {{ target.threads }},\n    \'{{ run_started_at }}\',\n    \'{{ invocation_id }}\',\n    \'{{ thread_id }}\'\n   )\n'
MODEL_POST_HOOK: str = '\n   insert into {{this.schema}}.on_model_hook (\n        test_state,\n        target_dbname,\n        target_host,\n        target_name,\n        target_schema,\n        target_type,\n        target_user,\n        target_pass,\n        target_threads,\n        run_started_at,\n        invocation_id,\n        thread_id\n   ) VALUES (\n    \'end\',\n    \'{{ target.dbname }}\',\n    \'{{ target.host }}\',\n    \'{{ target.name }}\',\n    \'{{ target.schema }}\',\n    \'{{ target.type }}\',\n    \'{{ target.user }}\',\n    \'{{ target.get("pass", "") }}\',\n    {{ target.threads }},\n    \'{{ run_started_at }}\',\n    \'{{ invocation_id }}\',\n    \'{{ thread_id }}\'\n   )\n'

class BaseTestPrePost(object):

    def get_ctx_vars(self, state: str, count: int, project: Any) -> List[Dict[str, Any]]:
    
    def check_hooks(self, state: str, project: Any, host: str, count: int = 1) -> None:

class TestPrePostModelHooks(BaseTestPrePost):

    def test_pre_and_post_run_hooks(self, project: Any, dbt_profile_target: Dict[str, Any]) -> None:

class TestPrePostModelHooksUnderscores(TestPrePostModelHooks):

class TestHookRefs(BaseTestPrePost):

    def test_pre_post_model_hooks_refed(self, project: Any, dbt_profile_target: Dict[str, Any]) -> None:

class TestPrePostModelHooksOnSeeds(object):

    def test_hooks_on_seeds(self, project: Any) -> None:

class TestPrePostModelHooksWithMacros(BaseTestPrePost):

    def test_pre_and_post_run_hooks(self, project: Any, dbt_profile_target: Dict[str, Any]) -> None:

class TestPrePostModelHooksListWithMacros(TestPrePostModelHooksWithMacros):

class TestHooksRefsOnSeeds:

    def test_hook_with_ref_on_seeds(self, project: Any) -> None:

class TestPrePostModelHooksOnSeedsPlusPrefixed(TestPrePostModelHooksOnSeeds):

class TestPrePostModelHooksOnSeedsPlusPrefixedWhitespace(TestPrePostModelHooksOnSeeds):

class TestPrePostModelHooksOnSnapshots(object):

    def test_hooks_on_snapshots(self, project: Any) -> None:

class PrePostModelHooksInConfigSetup(BaseTestPrePost):

class TestPrePostModelHooksInConfig(PrePostModelHooksInConfigSetup):

    def test_pre_and_post_model_hooks_model(self, project: Any, dbt_profile_target: Dict[str, Any]) -> None:

class TestPrePostModelHooksInConfigWithCount(PrePostModelHooksInConfigSetup):

    def test_pre_and_post_model_hooks_model_and_project(self, project: Any, dbt_profile_target: Dict[str, Any]) -> None:

class TestPrePostModelHooksInConfigKwargs(TestPrePostModelHooksInConfig):

class TestPrePostSnapshotHooksInConfigKwargs(TestPrePostModelHooksOnSnapshots):

class TestDuplicateHooksInConfigs(object):

    def test_run_duplicate_hook_defs(self, project: Any) -> None:
