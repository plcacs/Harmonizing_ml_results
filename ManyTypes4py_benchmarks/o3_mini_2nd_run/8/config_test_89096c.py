#!/usr/bin/env python3
"""Unit tests for cli/config.py."""
import base64
import getpass
import subprocess
import sys
from unittest import mock
from typing import Any
import boto3
from cli import config as config_module
from cli.config import BinaryAlertConfig, CONFIG_FILE
from cli.exceptions import InvalidConfigError
from tests.cli._common import mock_input, FakeFilesystemBase


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

    def test_variable_not_defined(self) -> None:
        """InvalidConfigError is raised if a variable declaration is missing."""
        with open(CONFIG_FILE, 'w') as config_file:
            config_file.write('aws_region = "us-east-1"\n')
        with self.assertRaises(InvalidConfigError):
            BinaryAlertConfig()

    def test_invalid_aws_account_id(self) -> None:
        """InvalidConfigError raised if AWS account ID is not a 12-digit number"""
        config: BinaryAlertConfig = BinaryAlertConfig()
        with self.assertRaises(InvalidConfigError):
            config.aws_account_id = '1234'

    def test_invalid_aws_region(self) -> None:
        """InvalidConfigError raised if AWS region is set incorrectly."""
        config: BinaryAlertConfig = BinaryAlertConfig()
        with self.assertRaises(InvalidConfigError):
            config.aws_region = 'us-east-1-'

    def test_invalid_name_prefix(self) -> None:
        """InvalidConfigError raised if name prefix is blank."""
        config: BinaryAlertConfig = BinaryAlertConfig()
        with self.assertRaises(InvalidConfigError):
            config.name_prefix = ''

    def test_invalid_enable_carbon_black_downloader(self) -> None:
        """InvalidConfigError raised if enable_downloader is not an int."""
        config: BinaryAlertConfig = BinaryAlertConfig()
        with self.assertRaises(InvalidConfigError):
            config.enable_carbon_black_downloader = '1'

    def test_invalid_carbon_black_url(self) -> None:
        """InvalidConfigError raised if URL doesn't start with http(s)."""
        config: BinaryAlertConfig = BinaryAlertConfig()
        with self.assertRaises(InvalidConfigError):
            config.carbon_black_url = 'example.com'

    def test_invalid_encrypted_carbon_black_api_token(self) -> None:
        """InvalidConfigError raised if encrypted token is too short."""
        config: BinaryAlertConfig = BinaryAlertConfig()
        with self.assertRaises(InvalidConfigError):
            config.encrypted_carbon_black_api_token = 'ABCD'

    @mock.patch.object(boto3, 'client')
    @mock.patch.object(getpass, 'getpass', return_value='abcd' * 10)
    @mock.patch.object(config_module, 'print')
    @mock.patch.object(subprocess, 'check_call')
    def test_encrypt_cb_api_token(self,
                                  mock_subprocess: Any,
                                  mock_print: Any,
                                  mock_getpass: Any,
                                  mock_client: Any) -> None:
        """Verify that token encryption is done correctly."""
        mock_client('kms').encrypt.return_value = {'CiphertextBlob': base64.b64encode(b'a' * 50)}
        config: BinaryAlertConfig = BinaryAlertConfig()
        config._encrypt_cb_api_token()  # noqa: SLF001
        mock_client('kms').decrypt = lambda **kwargs: {'Plaintext': base64.b64decode(kwargs['CiphertextBlob'])}
        self.assertEqual(b'a' * 50, config.plaintext_carbon_black_api_token)
        mock_client.assert_has_calls([mock.call().encrypt(KeyId=mock.ANY, Plaintext=mock_getpass.return_value)])
        mock_getpass.assert_called_once()
        mock_print.assert_has_calls([mock.call('Terraforming KMS key...'),
                                     mock.call('Encrypting API token...')])
        mock_subprocess.assert_has_calls([mock.call(['terraform', 'init']),
                                          mock.call(['terraform', 'apply', '-target=aws_kms_alias.encrypt_credentials_alias'])])

    @mock.patch.object(config_module, 'input', side_effect=mock_input)
    @mock.patch.object(BinaryAlertConfig, '_encrypt_cb_api_token')
    def test_configure_with_defaults(self,
                                     mock_encrypt: Any,
                                     mock_user_input: Any) -> None:
        """Test configure() when all variables have already had set values."""
        config: BinaryAlertConfig = BinaryAlertConfig()
        config.configure()
        mock_encrypt.assert_called_once()
        mock_user_input.assert_has_calls([mock.call('AWS Region (us-test-1): '),
                                          mock.call('Unique name prefix, e.g. "company_team" (test_prefix): '),
                                          mock.call('Enable the CarbonBlack downloader? (yes): '),
                                          mock.call('CarbonBlack URL (https://cb-example.com): '),
                                          mock.call('Change the CarbonBlack API token? (no): ')])
        self.assertEqual('us-west-2', config.aws_region)
        self.assertEqual('new_name_prefix', config.name_prefix)
        self.assertEqual(1, config.enable_carbon_black_downloader)

    @mock.patch.object(config_module, 'input', side_effect=mock_input)
    @mock.patch.object(BinaryAlertConfig, '_encrypt_cb_api_token')
    def test_configure_with_no_defaults(self,
                                        mock_encrypt: Any,
                                        mock_user_input: Any) -> None:
        """Test configure() without any values set - no defaults should print."""
        self._write_config(region='', prefix='', enable_downloader=False, cb_url='', encrypted_api_token='')
        config: BinaryAlertConfig = BinaryAlertConfig()
        config.configure()
        mock_encrypt.assert_called_once()
        mock_user_input.assert_has_calls([mock.call('AWS Region: '),
                                          mock.call('Unique name prefix, e.g. "company_team": '),
                                          mock.call('Enable the CarbonBlack downloader? (no): '),
                                          mock.call('CarbonBlack URL: ')])

    def test_validate_valid_with_downloader(self) -> None:
        """Test validate() with all values set correctly."""
        config: BinaryAlertConfig = BinaryAlertConfig()
        config.validate()
        self.test_property_accesses()

    def test_validate_valid_without_downloader(self) -> None:
        """Test validate() without any CarbonBlack values set - still valid."""
        self._write_config(enable_downloader=False, cb_url='', encrypted_api_token='')
        config: BinaryAlertConfig = BinaryAlertConfig()
        config.validate()

    def test_validate_invalid(self) -> None:
        """Test validate() with an invalid configuration file."""
        self._write_config(region='BAD_REGION')
        config: BinaryAlertConfig = BinaryAlertConfig()
        with self.assertRaises(InvalidConfigError):
            config.validate()

    def test_save(self) -> None:
        """New configuration is successfully written and comments are preserved."""
        config: BinaryAlertConfig = BinaryAlertConfig()
        config._config['force_destroy'] = True
        config.aws_region = 'us-west-2'
        config.name_prefix = 'new_name_prefix'
        config.enable_carbon_black_downloader = False
        config.carbon_black_url = 'https://example2.com'
        config.encrypted_carbon_black_api_token = 'B' * 100
        config.save()
        with open(CONFIG_FILE) as config_file:
            raw_data: str = config_file.read()
            for i in range(1, 6):
                self.assertIn('comment{}'.format(i), raw_data)
        new_config: BinaryAlertConfig = BinaryAlertConfig()
        self.assertEqual(True, new_config._config['force_destroy'])
        self.assertEqual(config.aws_region, new_config.aws_region)
        self.assertEqual(config.name_prefix, new_config.name_prefix)
        self.assertEqual(config.enable_carbon_black_downloader, new_config.enable_carbon_black_downloader)
        self.assertEqual(config.encrypted_carbon_black_api_token, new_config.encrypted_carbon_black_api_token)
