from typing import Optional

@mock.patch.object(sys, 'stderr', mock.MagicMock())
class BinaryAlertConfigTestFakeFilesystem(FakeFilesystemBase):
    """Tests of the BinaryAlertConfig class that use a fake filesystem."""

    def test_property_accesses(self) -> None:
        """Access each property in the BinaryAlertConfig."""
        config: BinaryAlertConfig = BinaryAlertConfig()
        self.assertEqual('123412341234', config.aws_account_id)
        self.assertEqual('us-test-1', config.aws_region)
        self.assertEqual('test_prefix', config.name_prefix)
        self.assertEqual(True, config.enable_carbon_black_downloader)
        self.assertEqual('https://cb-example.com', config.carbon_black_url)
        self.assertEqual('A' * 100, config.encrypted_carbon_black_api_token)
        self.assertEqual('test.prefix.binaryalert-binaries.us-test-1', config.binaryalert_s3_bucket_name)
        self.assertEqual('test_prefix_binaryalert_analyzer_queue', config.binaryalert_analyzer_queue_name)
        self.assertEqual('test_prefix_binaryalert_downloader_queue', config.binaryalert_downloader_queue_name)
        self.assertEqual(5, config.retro_batch_size)

    # ... other methods ...

    @mock.patch.object(config_module, 'input', side_effect=mock_input)
    @mock.patch.object(BinaryAlertConfig, '_encrypt_cb_api_token')
    def test_configure_with_defaults(self, mock_encrypt: mock.MagicMock, mock_user_input: mock.MagicMock) -> None:
        """Test configure() when all variables have already had set values."""
        config: BinaryAlertConfig = BinaryAlertConfig()
        config.configure()
        mock_encrypt.assert_called_once()
        mock_user_input.assert_has_calls([mock.call('AWS Region (us-test-1): '), mock.call('Unique name prefix, e.g. "company_team" (test_prefix): '), mock.call('Enable the CarbonBlack downloader? (yes): '), mock.call('CarbonBlack URL (https://cb-example.com): '), mock.call('Change the CarbonBlack API token? (no): ')])
        self.assertEqual('us-west-2', config.aws_region)
        self.assertEqual('new_name_prefix', config.name_prefix)
        self.assertEqual(1, config.enable_carbon_black_downloader)

    # ... other methods ...

    def test_save(self) -> None:
        """New configuration is successfully written and comments are preserved."""
        config: BinaryAlertConfig = BinaryAlertConfig()
        config._config['force_destroy'] = True
        config.aws_region: str = 'us-west-2'
        config.name_prefix: str = 'new_name_prefix'
        config.enable_carbon_black_downloader: Optional[bool] = False
        config.carbon_black_url: str = 'https://example2.com'
        config.encrypted_carbon_black_api_token: str = 'B' * 100
        config.save()
        # ... rest of the method ...

    # ... other methods ...
