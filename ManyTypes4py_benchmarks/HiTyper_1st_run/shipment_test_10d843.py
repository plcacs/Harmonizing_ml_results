import os
import os.path
import sys
import datetime
import webbrowser
import argparse
import time
import traceback
import selenium
import selenium.webdriver.chrome.options
import pathlib
import subprocess
from selenium.webdriver.common.by import By

class CommandArgs:

    def __init__(self) -> None:
        self.argParser = argparse.ArgumentParser()
        self.argParser.add_argument('-de', '--dextex', help='show extended exception reports', action='store_true')
        self.argParser.add_argument('-f', '--fcall', help='test fast calls', action='store_true')
        self.argParser.add_argument('-i', '--inst', help='installed version rather than new one', action='store_true')
        self.argParser.add_argument('-b', '--blind', help="don't start browser", action='store_true')
        self.argParser.add_argument('-u', '--unattended', help='unattended mode', action='store_true')
        self.argParser.add_argument('-t', '--transcom', help='transpile command')
        self.__dict__.update(self.argParser.parse_args().__dict__)
commandArgs = CommandArgs()

class BrowserController:

    def __init__(self) -> None:
        self.options = selenium.webdriver.chrome.options.Options()
        self.options.add_argument('start-maximized')
        if commandArgs.unattended:
            self.options.add_argument('--headless')
            self.options.add_argument('--no-sandbox')
            self.options.add_argument('--disable-gpu')
            self.options.add_argument('disable-infobars')
            self.options.add_argument('--disable-extensions')
            self.options.add_argument('--disable-web-security')
        self.webDriver = selenium.webdriver.Chrome(options=self.options)
        self.nrOfTabs = 0

    def waitForNewTab(self) -> None:
        while len(self.webDriver.window_handles) <= self.nrOfTabs:
            time.sleep(0.5)
        self.nrOfTabs = len(self.webDriver.window_handles)

    def open(self, url: Union[str, set[str], None], run: Union[bool, str]) -> bool:
        print(f'Browser controller is opening URL: {url}')
        try:
            if self.nrOfTabs > 0:
                if commandArgs.unattended:
                    self.webDriver.execute_script(f'window.location.href = "{url}";')
                else:
                    self.webDriver.execute_script(f'window.open ("{url}","_blank");')
                    self.waitForNewTab()
                    self.webDriver.switch_to.window(self.webDriver.window_handles[-1])
            else:
                self.webDriver.get(url)
                self.waitForNewTab()
        except:
            self.webDriver.switch_to.alert.accept()
        if run:
            while True:
                self.message = self.webDriver.find_element(By.ID, 'message')
                if 'failed' in self.message.text or 'succeeded' in self.message.text:
                    break
                time.sleep(0.5)
            print()
            print('=========================================================================')
            print(f'Back to back autotest, result: {self.message.text.upper()}')
            print('=========================================================================')
            print()
            if 'succeeded' in self.message.text:
                return True
            else:
                return False
        else:
            print()
            print('=========================================================================')
            print('No back to back autotest')
            print('=========================================================================')
            print()
            return True
browserController = BrowserController()
relSourcePrepathsOfErrors = []
host = 'http://localhost:'
pythonServerPort = '8000'
parcelServerPort = '8001'
nodeServerPort = '8002'
pythonServerUrl = host + pythonServerPort
parcelServerUrl = host + parcelServerPort
nodeServerUrl = host + nodeServerPort
shipDir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
appRootDir = '/'.join(shipDir.split('/')[:-2])
transpileCommand = commandArgs.transcom if commandArgs.transcom else 'ts' if commandArgs.inst else f'/{appRootDir}/ts'
print(f'\nApplication root directory: {appRootDir}\n')

def getAbsPath(relPath: str) -> str:
    return '{}/{}'.format(appRootDir, relPath)
os.system('cls' if os.name == 'nt' else 'clear')
os.system(f'killall node')
print(appRootDir)
if not commandArgs.blind:
    command = f'python3 -m http.server --directory {appRootDir}'
    if commandArgs.unattended:
        os.system(f'{command} &')
    else:
        os.system(f"""konsole --new-tab --hold -e "bash -ic '{command}'"  &""")
    "\n    If 'Address already in use' do:\n        ps -fA | grep python\n        kill <process number>\n    "
os.system(f'{transpileCommand} -h')

def test(relSourcePrepath: Union[pathlib.Path, str, int], run: Union[list[str], str, list], extraSwitches: Union[dict[str, set[str]], dict, tuple[dict]], outputPrename: typing.Text='', nodeJs: bool=False, parcelJs: bool=False, build: bool=True, pause: int=0, needsAttention: bool=False) -> None:
    if commandArgs.unattended and needsAttention:
        return
    print(f'\n\n******** BEGIN TEST {relSourcePrepath} ********\n')
    time.sleep(pause)
    sourcePrepath = getAbsPath(relSourcePrepath)
    sourcePrepathSplit = sourcePrepath.split('/')
    sourceDir = '/'.join(sourcePrepathSplit[:-1])
    moduleName = sourcePrepathSplit[-1]
    targetDir = f'{sourceDir}/__target__'
    targetPrepath = f'{targetDir}/{moduleName}'
    outputPath = f'{targetDir}/{outputPrename}.out'
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    defaultSwitches = '-da -sf -de -m -n '
    if commandArgs.dextex:
        defaultSwitches += '-de '
    if build:
        defaultSwitches += '-b '
    if run:
        os.system(f'{transpileCommand} -r {defaultSwitches}{extraSwitches}{sourcePrepath}')
    if parcelJs:
        origDir = os.getcwd()
        os.chdir(sourceDir)
        os.system(f'node test {parcelServerPort} &')
        os.chdir(origDir)
    else:
        command = f'{transpileCommand} {defaultSwitches}{extraSwitches}{sourcePrepath}'
        if outputPrename:
            "\n            Calling subprocess.check_output while outputFile opened in __target__ folder\n            fails without raising an exception, probably a bug in Python.\n            So we use variable 'output' as intermediate storage.\n            "
            output = subprocess.check_output(command, universal_newlines=True, shell=True)
            with open(outputPath, 'w') as outputFile:
                print(output, file=outputFile)
        else:
            subprocess.run(command, shell=True)
    if nodeJs:
        os.system(f'rollup {targetPrepath}.js --o {targetPrepath}.bundle.js --f cjs')
    if not commandArgs.blind:
        if parcelJs:
            time.sleep(20)
            url = parcelServerUrl
        elif nodeJs:
            os.system(f'chmod 777 {targetPrepath}.bundle.js')
            os.system(f'node {targetPrepath}.bundle.js {nodeServerPort} &')
            time.sleep(5)
            url = nodeServerUrl
        else:
            time.sleep(5)
            url = f'{pythonServerUrl}/{relSourcePrepath}.html'
        success = browserController.open(url, run)
    if commandArgs.unattended and (not success):
        relSourcePrepathsOfErrors.append(relSourcePrepath)
    print(f'\n******** END TEST {relSourcePrepath} ********\n\n')
for switches in ('', '-f ') if commandArgs.fcall else ('',):
    test('development/automated_tests/hello/autotest', True, switches)
    test('development/automated_tests/transcrypt/autotest', True, switches + '-c -xr -xg ')
    test('development/automated_tests/time/autotest', True, switches, needsAttention=True)
    test('development/automated_tests/re/autotest', True, switches)
    test('development/manual_tests/async_await/test', False, switches)
    test('development/manual_tests/import_export_aliases/test', False, switches + '-am ')
    test('development/manual_tests/module_random/module_random', False, switches)
    test('development/manual_tests/static_types/static_types', False, switches + '-ds -dc ', outputPrename='static_types')
    test('development/manual_tests/transcrypt_and_python_results_differ/results', False, switches)
    test('development/manual_tests/transcrypt_only/transcrypt_only', False, switches)
    test('demos/nodejs_demo/nodejs_demo', False, switches, nodeJs=True)
    test('demos/terminal_demo/terminal_demo', False, switches, needsAttention=True)
    test('demos/hello/hello', False, switches, needsAttention=False)
    test('demos/jquery_demo/jquery_demo', False, switches)
    test('demos/d3js_demo/d3js_demo', False, switches)
    test('demos/ios_app/ios_app', False, switches)
    test('demos/react_demo/react_demo', False, switches)
    test('demos/riot_demo/riot_demo', False, switches)
    test('demos/plotly_demo/plotly_demo', False, switches)
    test('demos/three_demo/three_demo', False, switches)
    test('demos/pong/pong', False, switches)
    test('demos/pysteroids_demo/pysteroids', False, switches)
    test('demos/turtle_demos/star', False, switches, pause=2)
    test('demos/turtle_demos/snowflake', False, switches, pause=2)
    test('demos/turtle_demos/mondrian', False, switches, pause=2)
    test('demos/turtle_demos/mandala', False, switches, pause=2)
    test('demos/cyclejs_demo/cyclejs_http_demo', False, switches)
    test('demos/cyclejs_demo/component_demos/isolated_bmi_slider/bmi', False, switches)
    test('demos/cyclejs_demo/component_demos/labeled_slider/labeled_slider', False, switches)
    test('tutorials/baseline/bl_010_hello_world/hello_world', False, switches)
    test('tutorials/baseline/bl_020_assign/assign', False, switches)
    test('tutorials/baseline/bl_030_if_else_prompt/if_else_prompt', False, switches, needsAttention=True)
    test('tutorials/baseline/bl_035_if_else_event/if_else_event', False, switches, needsAttention=True)
    test('tutorials/baseline/bl_040_for_simple/for_simple', False, switches)
    test('tutorials/baseline/bl_042_for_nested/for_nested', False, switches)
    test('tutorials/baseline/bl_045_while_simple/while_simple', False, switches, needsAttention=True)
    test('tutorials/static_typing/static_typing', False, switches + '-c -ds ', outputPrename='static_typing')
    if relSourcePrepathsOfErrors:
        print('\n\n!!!!!!!!!!!!!!!!!!!!\n')
        for relSourcePrepathOfError in relSourcePrepathsOfErrors:
            print(f'SHIPMENT TEST ERROR: {relSourcePrepathOfError}')
        print('\n!!!!!!!!!!!!!!!!!!!!\n\n')
        print('\nSHIPMENT TEST FAILED\n')
        sys.exit(1)
    else:
        if not commandArgs.unattended:
            origDir = os.getcwd()
            sphinxDir = '/'.join([appRootDir, 'docs/sphinx'])
            os.chdir(sphinxDir)
            os.system('touch *.rst')
            os.system('make html')
            os.chdir(origDir)
        print('\nSHIPMENT TEST SUCCEEDED\n')
        sys.exit(0)