from cli.config import BinaryAlertConfig, CONFIG_FILE
from cli.exceptions import InvalidConfigError
from tests.cli._common import mock_input, FakeFilesystemBase
from typing import Any

class BinaryAlertConfigTestFakeFilesystem(FakeFilesystemBase):
    def test_property_accesses(self) -> None:
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
        with open(CONFIG_FILE, 'w') as config_file:
            config_file.write('aws_region = "us-east-1"\n')
        with self.assertRaises(InvalidConfigError):
            BinaryAlertConfig()

    def test_invalid_aws_account_id(self) -> None:
        config: BinaryAlertConfig = BinaryAlertConfig()
        with self.assertRaises(InvalidConfigError):
            config.aws_account_id = '1234'

    def test_invalid_aws_region(self) -> None:
        config: BinaryAlertConfig = BinaryAlertConfig()
        with self.assertRaises(InvalidConfigError):
            config.aws_region = 'us-east-1-'

    def test_invalid_name_prefix(self) -> None:
        config: BinaryAlertConfig = BinaryAlertConfig()
        with self.assertRaises(InvalidConfigError):
            config.name_prefix = ''

    def test_invalid_enable_carbon_black_downloader(self) -> None:
        config: BinaryAlertConfig = BinaryAlertConfig()
        with self.assertRaises(InvalidConfigError):
            config.enable_carbon_black_downloader = '1'

    def test_invalid_carbon_black_url(self) -> None:
        config: BinaryAlertConfig = BinaryAlertConfig()
        with self.assertRaises(InvalidConfigError):
            config.carbon_black_url = 'example.com'

    def test_invalid_encrypted_carbon_black_api_token(self) -> None:
        config: BinaryAlertConfig = BinaryAlertConfig()
        with self.assertRaises(InvalidConfigError):
            config.encrypted_carbon_black_api_token = 'ABCD'

    def test_encrypt_cb_api_token(self, mock_subprocess: Any, mock_print: Any, mock_getpass: Any, mock_client: Any) -> None:
        ...

    def test_configure_with_defaults(self, mock_encrypt: Any, mock_user_input: Any) -> None:
        ...

    def test_configure_with_no_defaults(self, mock_encrypt: Any, mock_user_input: Any) -> None:
        ...

    def test_validate_valid_with_downloader(self) -> None:
        ...

    def test_validate_valid_without_downloader(self) -> None:
        ...

    def test_validate_invalid(self) -> None:
        ...

    def test_save(self) -> None:
        ...
