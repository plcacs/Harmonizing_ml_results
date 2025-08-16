import json
import subprocess
import time
from typing import Any, List, Tuple

CMD_BASE: List[str] = ['catt', '-d']
VALIDATE_ARGS: List[str] = ['info', '-j']
STOP_ARGS: List[str] = ['stop']
SCAN_CMD: List[str] = ['catt', 'scan', '-j']

def subp_run(cmd: List[str], allow_failure: bool = False) -> subprocess.CompletedProcess:
    output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if not allow_failure and output.returncode != 0:
        raise CattTestError('The command "{}" failed.'.format(' '.join(cmd)))
    return output

class CattTestError(click.ClickException):
    pass

class CattTest:

    def __init__(self, desc: str, cmd_args: List[str], sleep: int = 10, should_fail: bool = False, substring: bool = False, time_test: bool = False, check_data: Tuple[str, str] = None, check_err: str = ''):
        if should_fail and (not check_err) or (not should_fail and (not check_data)):
            raise CattTestError('Expected outcome mismatch.')
        if substring and time_test:
            raise CattTestError('Test type mismatch.')
        self._cmd_args = cmd_args
        self._cmd: List[str] = []
        self._validate_cmd: List[str] = []
        self._sleep = sleep
        self._should_fail = should_fail
        self._substring = substring
        self._time_test = time_test
        self._check_key, self._check_val = check_data if check_data else (None, None)
        self._check_err = check_err
        self._output = None
        self._failed = False
        self.desc = desc + (' (should fail)' if self._should_fail else '')
        self.dump = ''

    def set_cmd_base(self, base: List[str]):
        self._cmd = base + self._cmd_args
        self._validate_cmd = base + VALIDATE_ARGS

    def _get_val(self, key: str) -> Any:
        output = subp_run(self._validate_cmd)
        catt_json = json.loads(output.stdout)
        return catt_json[key]

    def _should_fail_test(self) -> bool:
        if self._should_fail == self._failed:
            return True
        else:
            self.dump += self._output.stderr if self._failed else self._output.stdout
            return False

    def _failure_test(self) -> bool:
        output_errmsg = self._output.stderr.splitlines()[-1]
        if output_errmsg == 'Error: {}.'.format(self._check_err):
            self.dump += '{}\n - The expected error message.'.format(output_errmsg)
            return True
        else:
            self.dump += self._output.stderr
            return False

    def _regular_test(self, time_margin: int = 5) -> bool:
        catt_val = self._get_val(self._check_key)
        if self._time_test:
            passed = abs(int(catt_val) - int(self._check_val)) <= time_margin
            extra_info = '(time margin is {} seconds)'.format(time_margin)
        elif self._substring:
            passed = self._check_val in catt_val
            extra_info = '(substring)'
        else:
            passed = catt_val == self._check_val
            extra_info = ''
        if not passed:
            self.dump += 'Expected data from "{}" key:\n{} {}\nActual data:\n{}'.format(self._check_key, self._check_val, extra_info, catt_val)
        return passed

    def run(self) -> bool:
        self._output = subp_run(self._cmd, allow_failure=True)
        self._failed = self._output.returncode != 0
        time.sleep(self._sleep)
        if self._should_fail_test():
            if self._failed:
                return self._failure_test()
            else:
                return self._regular_test()
        else:
            return False

DEFAULT_CTRL_TESTS: List[CattTest] = [CattTest('cast h264 1920x1080 / aac content from dailymotion', ['cast', 'http://www.dailymotion.com/video/x6fotne'], substring=True, check_data=('content_id', '/389149466_mp4_h264_aac_fhd.mp4')), ...]

YOUTUBE_CTRL_TESTS: List[CattTest] = [CattTest('cast video from youtube', ['cast', 'https://www.youtube.com/watch?v=mwPSIb3kt_4'], check_data=('content_id', 'mwPSIb3kt_4')), ...]

DASHCAST_CTRL_TESTS: List[CattTest] = [CattTest('cast GitHub website frontpage', ['cast_site', 'https://github.com'], substring=True, check_data=('status_text', 'GitHub'))]

AUDIO_ONLY_TESTS: List[CattTest] = [CattTest('cast audio-only DASH aac content from facebook', ['cast', 'https://www.facebook.com/PixarCars/videos/10158549620120183/'], substring=True, check_data=('content_id', '18106055_10158549666610183_8333687643300691968_n.mp4')), ...]

STANDARD_TESTS: List[CattTest] = DEFAULT_CTRL_TESTS + YOUTUBE_CTRL_TESTS + DASHCAST_CTRL_TESTS
AUDIO_TESTS: List[CattTest] = AUDIO_ONLY_TESTS
ULTRA_TESTS: List[CattTest] = []

def run_tests(standard: str = '', audio: str = '', ultra: str = '') -> bool:
    if not standard and (not audio) and (not ultra):
        raise CattTestError('No test devices were specified.')
    test_outcomes = []
    all_suites = zip([standard, audio, ultra], [STANDARD_TESTS, AUDIO_TESTS, ULTRA_TESTS])
    suites_to_run = {}
    scan_result = json.loads(subp_run(SCAN_CMD).stdout)
    for device_name, suite in all_suites:
        if not device_name:
            continue
        if device_name not in scan_result.keys():
            raise CattTestError('Specified device "{}" not found.'.format(device_name))
        suites_to_run.update({device_name: suite})
    for device_name, suite in suites_to_run.items():
        click.secho('Running some tests on "{}".'.format(device_name), fg='yellow')
        click.secho('------------------------------------------', fg='yellow')
        cbase = CMD_BASE + [device_name]
        for test in suite:
            test.set_cmd_base(cbase)
            click.echo(test.desc + '  ->  ', nl=False)
            if test.run():
                click.secho('test success!', fg='green')
                test_outcomes.append(True)
            else:
                click.secho('test failure!', fg='red')
                test_outcomes.append(False)
            if test.dump:
                click.echo('\n' + test.dump + '\n')
        subp_run(cbase + STOP_ARGS)
    return all((t for t in test_outcomes)) if test_outcomes else False

@click.command()
@click.option('-s', '--standard', help='Name of standard chromecast device.')
@click.option('-a', '--audio', help='Name of audio chromecast device.')
@click.option('-u', '--ultra', help='Name of ultra chromecast device.')
def cli(standard: str, audio: str, ultra: str):
    if run_tests(standard=standard, audio=audio, ultra=ultra):
        click.echo('\nAll tests were successfully completed.')
    else:
        raise CattTestError('Some tests were not successful.')
if __name__ == '__main__':
    cli()
