"""Unit tests for cli/manager.py"""
import collections
import inspect
import subprocess
from unittest import mock
from cli import config as config_module
from cli import manager as manager_module
from cli.config import BinaryAlertConfig
from cli.exceptions import InvalidConfigError, TestFailureError
from cli.manager import Manager
from tests.cli._common import mock_input, FakeFilesystemBase
MockSummary = collections.namedtuple('MockSummary', ['key'])

class ManagerTest(FakeFilesystemBase):
    """Tests for the Manager class."""

    @mock.patch('sys.stderr', mock.MagicMock())
    def setUp(self):
        super().setUp()
        self.manager = Manager()

    def test_commands(self):
        """Each command should be a function in the class."""
        for command in self.manager.commands:
            self.assertTrue(hasattr(self.manager, command))

    def test_help(self):
        """Help string should contain as many lines as there are commands."""
        self.assertEqual(len(self.manager.commands), len(self.manager.help.split('\n')))

    @mock.patch.object(manager_module, 'JoinableQueue')
    @mock.patch.object(manager_module, 'Worker')
    @mock.patch.object(manager_module, 'print')
    def test_enqueue(self, mock_print, mock_worker, mock_task_queue):
        """SQS messages are batched and enqueued"""
        messages = ({'index': i} for i in range(25))
        self.manager._enqueue('test-queue', messages, lambda msg: (1, msg['index']))
        mock_task_queue.assert_called()
        mock_worker.assert_called()
        mock_print.assert_called()

    @mock.patch.object(subprocess, 'check_call')
    def test_apply(self, mock_subprocess):
        """Validate order of Terraform operations."""
        self.manager.apply()
        mock_subprocess.assert_has_calls([mock.call(['terraform', 'init']), mock.call(['terraform', 'fmt']), mock.call(['terraform', 'apply', '-auto-approve=false'])])

    @mock.patch.object(manager_module, 'lambda_build')
    def test_build(self, mock_build):
        """Calls lambda_build function (tested elsewhere)."""
        self.manager.build()
        mock_build.assert_called_once()

    def test_cb_copy_all_not_enabled(self):
        """Raises InvalidConfigError if the downloader is not enabled."""
        self._write_config(enable_downloader=False)
        self.manager = Manager()
        with self.assertRaises(InvalidConfigError):
            self.manager.cb_copy_all()

    @mock.patch.object(manager_module.clone_rules, 'clone_remote_rules')
    def test_clone_rules(self, mock_clone):
        """Calls clone_remote_rules (tested elsewhere)."""
        self.manager.clone_rules()
        mock_clone.assert_called_once()

    @mock.patch.object(manager_module.compile_rules, 'compile_rules')
    @mock.patch.object(manager_module, 'print')
    def test_compile_rules(self, mock_print, mock_compile):
        """Calls compile_rules (tested elsewhere)."""
        self.manager.compile_rules()
        mock_compile.assert_called_once()
        mock_print.assert_called_once()

    @mock.patch.object(BinaryAlertConfig, 'configure')
    @mock.patch.object(manager_module, 'print')
    def test_configure(self, mock_print, mock_configure):
        """Calls BinaryAlertConfig:configure() (tested elsewhere)."""
        self.manager.configure()
        mock_configure.assert_called_once()
        mock_print.assert_called_once()

    @mock.patch.object(Manager, 'unit_test')
    @mock.patch.object(Manager, 'build')
    @mock.patch.object(Manager, 'apply')
    def test_deploy(self, mock_apply, mock_build, mock_test):
        """Deploy docstring includes each executed command and runs each."""
        for command in ['unit_test', 'build', 'apply']:
            self.assertIn(command, inspect.getdoc(Manager.deploy))
        self.manager.deploy()
        mock_test.assert_called_once()
        mock_build.assert_called_once()
        mock_apply.assert_called_once()

    @mock.patch.object(config_module, 'input', side_effect=mock_input)
    @mock.patch.object(manager_module, 'print')
    @mock.patch.object(subprocess, 'call')
    @mock.patch.object(subprocess, 'check_call')
    def test_destroy(self, mock_check_call, mock_call, mock_print, mock_user_input):
        """Destroy asks whether S3 objects should also be deleted."""
        self.manager.destroy()
        mock_user_input.assert_called_once()
        mock_print.assert_called_once()
        mock_check_call.assert_called_once()
        mock_call.assert_called_once()

    @mock.patch.object(manager_module.live_test, 'run', return_value=False)
    def test_live_test(self, mock_live_test):
        """Live test wrapper raises TestFailureError if appropriate."""
        with self.assertRaises(TestFailureError):
            self.manager.live_test()
        mock_live_test.assert_called_once()

    @mock.patch.object(manager_module.boto3, 'resource')
    def test_purge_queue(self, mock_resource):
        """Purge operation calls out to SQS"""
        self.manager.purge_queue()
        mock_resource.assert_has_calls([mock.call('sqs'), mock.call().get_queue_by_name(QueueName='test_prefix_binaryalert_analyzer_queue'), mock.call().get_queue_by_name().purge()])

    def test_most_recent_manifest_found(self):
        """Finds the first summary key ending in manifest.json"""
        bucket = mock.MagicMock()
        files = [MockSummary('inventory/test-bucket/EntireBucketDaily/2000-01-01/checksum.txt'), MockSummary('inventory/test-bucket/EntireBucketDaily/2000-01-01/manifest.json'), MockSummary('inventory/end')]
        bucket.objects.filter.return_value = files
        self.assertEqual(files[1].key, self.manager._most_recent_manifest(bucket))

    def test_most_recent_manifest_not_found(self):
        """Returns None if no manifest files were found"""
        bucket = mock.MagicMock()
        bucket.objects.filter.return_value = [MockSummary('inventory/end')]
        self.assertIsNone(self.manager._most_recent_manifest(bucket))

    def test_s3_batch_iterator(self):
        """Multiple S3 objects are grouped together by batch iterator"""
        self.manager._config._config['objects_per_retro_message'] = 2
        sqs_messages = list(self.manager._s3_batch_iterator((str(i) for i in range(3))))
        expected = [{'Records': [{'s3': {'bucket': {'name': mock.ANY}, 'object': {'key': '0'}}}, {'s3': {'bucket': {'name': mock.ANY}, 'object': {'key': '1'}}}]}, {'Records': [{'s3': {'bucket': {'name': mock.ANY}, 'object': {'key': '2'}}}]}]
        self.assertEqual(expected, sqs_messages)

    def test_s3_msg_summary(self):
        """S3 message summaries include a record count and print the last key"""
        message = {'Records': [{'s3': {'bucket': {'name': 'bucket-name'}, 'object': {'key': 'ABC'}}}, {'s3': {'bucket': {'name': 'bucket-name'}, 'object': {'key': 'DEF'}}}]}
        count, summary = self.manager._s3_msg_summary(message)
        self.assertEqual(2, count)
        self.assertEqual('DEF', summary)

    @mock.patch.object(manager_module.boto3, 'resource')
    @mock.patch.object(manager_module.Manager, '_most_recent_manifest', return_value=None)
    @mock.patch.object(manager_module, 'print')
    def test_retro_fast_no_manifest(self, mock_print, mock_manifest, mock_resource):
        """Retro fast - error message printed if no manifest exists when"""
        self.manager.retro_fast()
        mock_resource.assert_called_once()
        mock_manifest.assert_called_once()
        mock_print.assert_has_calls([mock.call(mock.ANY)] * 2)

    @mock.patch.object(manager_module.boto3, 'resource')
    @mock.patch.object(manager_module.Manager, '_enqueue')
    @mock.patch.object(manager_module.Manager, '_most_recent_manifest', return_value='inventory/manifest.json')
    @mock.patch.object(manager_module, 'print')
    def test_retro_fast(self, mock_print, mock_manifest, mock_enqueue, mock_resource):
        """Retro fast - enqueue is called if manifest was found"""
        self.manager.retro_fast()
        mock_resource.assert_called_once()
        mock_manifest.assert_called_once()
        mock_enqueue.assert_called_once()
        mock_print.assert_called_once_with('Reading inventory/manifest.json')

    @mock.patch.object(manager_module.boto3, 'resource')
    @mock.patch.object(manager_module.Manager, '_enqueue')
    def test_retro_slow(self, mock_enqueue, mock_resource):
        """Retro slow - enqueue is called"""
        self.manager.retro_slow()
        mock_resource.assert_called_once()
        mock_enqueue.assert_called_once()