from typing import List, Dict, Any, Callable

class ManagerTest(FakeFilesystemBase):
    def setUp(self) -> None:
    def test_commands(self) -> None:
    def test_help(self) -> None:
    def test_enqueue(self, mock_print: Any, mock_worker: Any, mock_task_queue: Any) -> None:
    def test_apply(self, mock_subprocess: Any) -> None:
    def test_build(self, mock_build: Any) -> None:
    def test_cb_copy_all_not_enabled(self) -> None:
    def test_clone_rules(self, mock_clone: Any) -> None:
    def test_compile_rules(self, mock_print: Any, mock_compile: Any) -> None:
    def test_configure(self, mock_print: Any, mock_configure: Any) -> None:
    def test_deploy(self, mock_apply: Any, mock_build: Any, mock_test: Any) -> None:
    def test_destroy(self, mock_check_call: Any, mock_call: Any, mock_print: Any, mock_user_input: Any) -> None:
    def test_live_test(self, mock_live_test: Any) -> None:
    def test_purge_queue(self, mock_resource: Any) -> None:
    def test_most_recent_manifest_found(self) -> None:
    def test_most_recent_manifest_not_found(self) -> None:
    def test_s3_batch_iterator(self) -> None:
    def test_s3_msg_summary(self) -> None:
    def test_retro_fast_no_manifest(self, mock_print: Any, mock_manifest: Any, mock_resource: Any) -> None:
    def test_retro_fast(self, mock_print: Any, mock_manifest: Any, mock_enqueue: Any, mock_resource: Any) -> None:
    def test_retro_slow(self, mock_enqueue: Any, mock_resource: Any) -> None:
