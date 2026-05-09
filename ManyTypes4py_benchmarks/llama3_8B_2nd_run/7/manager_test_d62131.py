class ManagerTest(FakeFilesystemBase):
    """Tests for the Manager class."""

    @mock.patch('sys.stderr', mock.MagicMock())
    def setUp(self) -> None:
        super().setUp()
        self.manager: Manager = Manager()

    def test_commands(self) -> None:
        """Each command should be a function in the class."""
        for command in self.manager.commands:
            self.assertTrue(hasattr(self.manager, command))

    # ... other methods ...

    @mock.patch.object(manager_module, 'JoinableQueue')
    @mock.patch.object(manager_module, 'Worker')
    @mock.patch.object(manager_module, 'print')
    def test_enqueue(self, mock_print: mock.MagicMock, mock_worker: mock.MagicMock, mock_task_queue: mock.MagicMock) -> None:
        """SQS messages are batched and enqueued"""
        # ...

    @mock.patch.object(subprocess, 'check_call')
    def test_apply(self, mock_subprocess: mock.MagicMock) -> None:
        """Validate order of Terraform operations."""
        self.manager.apply()
        mock_subprocess.assert_has_calls([mock.call(['terraform', 'init']), mock.call(['terraform', 'fmt']), mock.call(['terraform', 'apply', '-auto-approve=false'])])

    # ... other methods ...

    def test_deploy(self) -> None:
        """Deploy docstring includes each executed command and runs each."""
        for command in ['unit_test', 'build', 'apply']:
            self.assertIn(command, inspect.getdoc(Manager.deploy))
        self.manager.deploy()
        # ...

    # ... other methods ...
