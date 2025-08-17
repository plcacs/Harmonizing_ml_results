#!/usr/bin/env python3
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
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from argparse import ArgumentParser
from typing import List, Optional


# ======== Command args singleton

class CommandArgs:
    def __init__(self) -> None:
        self.argParser: ArgumentParser = argparse.ArgumentParser()
    
        self.argParser.add_argument('-de', '--dextex', help="show extended exception reports", action='store_true')
        self.argParser.add_argument('-f', '--fcall', help='test fast calls', action='store_true')
        
        self.argParser.add_argument('-i', '--inst', help='installed version rather than new one', action='store_true')
        self.argParser.add_argument('-b', '--blind', help="don't start browser", action='store_true')
        
        self.argParser.add_argument('-u', '--unattended', help='unattended mode', action='store_true')
        self.argParser.add_argument('-t', '--transcom', help='transpile command')
        
        self.__dict__.update(self.argParser.parse_args().__dict__)
        # The following attributes are expected:
        # dextex: bool, fcall: bool, inst: bool, blind: bool,
        # unattended: bool, transcom: Optional[str]


commandArgs: CommandArgs = CommandArgs()


# ========  Browser controller singleton

class BrowserController:
    def __init__(self) -> None:
        self.options: selenium.webdriver.chrome.options.Options = selenium.webdriver.chrome.options.Options()
        self.options.add_argument('start-maximized')
        
        if commandArgs.unattended:
            self.options.add_argument('--headless')          # Runs Chrome in headless mode.
            self.options.add_argument('--no-sandbox')        # Bypass OS security model
            self.options.add_argument('--disable-gpu')       # Applicable to windows OS only
            self.options.add_argument('disable-infobars')
            self.options.add_argument('--disable-extensions')
            self.options.add_argument('--disable-web-security')
                
        self.webDriver: WebDriver = selenium.webdriver.Chrome(options=self.options)
        self.nrOfTabs: int = 0
        self.message: Optional[WebElement] = None
        
    def waitForNewTab(self) -> None:
        while len(self.webDriver.window_handles) <= self.nrOfTabs:
            time.sleep(0.5)
        self.nrOfTabs = len(self.webDriver.window_handles)
                
    def open(self, url: str, run: bool) -> bool:
        print(f'Browser controller is opening URL: {url}')
    
        try:
            if self.nrOfTabs > 0:
                if commandArgs.unattended:
                    # ---- Show in existing tab
                    self.webDriver.execute_script(f'window.location.href = "{url}";')
                else:
                    # ---- Open new tab
                    self.webDriver.execute_script(f'window.open("{url}","_blank");')  # !!! Avoid redundant open command
                    self.waitForNewTab()
                    self.webDriver.switch_to.window(self.webDriver.window_handles[-1])
            else:
                # ---- Open browser and default tab
                self.webDriver.get(url)
                self.waitForNewTab()
        except Exception:
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


browserController: BrowserController = BrowserController()


# ======== Preparations

relSourcePrepathsOfErrors: List[str] = []

host: str = 'http://localhost:'

pythonServerPort: str = '8000'
parcelServerPort: str = '8001'
nodeServerPort: str = '8002'

pythonServerUrl: str = host + pythonServerPort
parcelServerUrl: str = host + parcelServerPort
nodeServerUrl: str = host + nodeServerPort

shipDir: str = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
appRootDir: str = '/'.join(shipDir.split('/')[:-2])

transpileCommand: str = commandArgs.transcom if commandArgs.transcom else 'ts' if commandArgs.inst else f'/{appRootDir}/ts'

print(f'\nApplication root directory: {appRootDir}\n')

def getAbsPath(relPath: str) -> str:
    return '{}/{}'.format(appRootDir, relPath)

os.system('cls' if os.name == 'nt' else 'clear')
os.system('killall node')
        
# ---- Start an http server in the Transcryp/transcrypt directory

print(appRootDir)

if not commandArgs.blind:
    # command = f'python -m http.server --directory {appRootDir}'
    command: str = f'python3 -m http.server --directory {appRootDir}'

    if commandArgs.unattended:
        os.system(f'{command} &')
    else:
        os.system(f'konsole --new-tab --hold -e "bash -ic \'{command}\'"  &')

    '''
    If 'Address already in use' do:
        ps -fA | grep python
        kill <process number>
    '''

# ---- Allow visual check of all command line options

os.system(f'{transpileCommand} -h')
   
# ======== Individual test function

def test(relSourcePrepath: str, run: bool, extraSwitches: str, outputPrename: str = '', nodeJs: bool = False, parcelJs: bool = False, build: bool = True, pause: int = 0, needsAttention: bool = False) -> None:
    if commandArgs.unattended and needsAttention:
        return  # This test shouldn't be done, since it can't run unattended
            
    print(f'\n\n******** BEGIN TEST {relSourcePrepath} ********\n')
            
    time.sleep(pause)

    # ---- Compute some slugs
    sourcePrepath: str = getAbsPath(relSourcePrepath)
    sourcePrepathSplit: List[str] = sourcePrepath.split("/")
    sourceDir: str = '/'.join(sourcePrepathSplit[:-1])
    moduleName: str = sourcePrepathSplit[-1]
    targetDir: str = f'{sourceDir}/__target__'
    targetPrepath: str = f'{targetDir}/{moduleName}'
    outputPath: str = f'{targetDir}/{outputPrename}.out'
    
    # ---- If there are relevant console messages of the compilation process,
    #       like with the static typechecking tests, write them into a file that can be served for a visual check
    
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)  # Transcrypt will make targetDir too late, so it has to happen heree

    # ---- Default switches
    defaultSwitches: str = '-da -sf -de -m -n '
    if commandArgs.dextex:
        defaultSwitches += '-de '
    if build:
        defaultSwitches += '-b '
    
    # ---- Run with CPython to generate HTML file with back to back reference info
    if run:
        os.system(f'{transpileCommand} -r {defaultSwitches}{extraSwitches}{sourcePrepath}')
        
    # ---- Compile with Transcrypt
    if parcelJs:
        origDir: str = os.getcwd()
        os.chdir(sourceDir)
        os.system(f'node test {parcelServerPort} &')
        os.chdir(origDir)
    else:
        command_str: str = f'{transpileCommand} {defaultSwitches}{extraSwitches}{sourcePrepath}'
        if outputPrename:
            # Calling subprocess.check_output while outputFile opened in __target__ folder
            # fails without raising an exception, probably a bug in Python.
            # So we use variable 'output' as intermediate storage.
            output: str = subprocess.check_output(command_str, universal_newlines=True, shell=True)
            with open(outputPath, 'w') as outputFile:
                print(output, file=outputFile)
        else:
            subprocess.run(command_str, shell=True)
        
    # ---- If it has to run on node, apply rollup to obtain monolith, since node doesn't support named imports and exports
    if nodeJs:
        os.system(f'rollup {targetPrepath}.js --o {targetPrepath}.bundle.js --f cjs')
    
    # --- Compute appropriate URL and wait a while if needed
    if not commandArgs.blind:
        if parcelJs:
            time.sleep(20)
            url: str = parcelServerUrl
        elif nodeJs:
            os.system(f'chmod 777 {targetPrepath}.bundle.js')
            os.system(f'node {targetPrepath}.bundle.js {nodeServerPort} &')
            time.sleep(5)
            url = nodeServerUrl
        else:
            time.sleep(5)
            url = f'{pythonServerUrl}/{relSourcePrepath}.html'
            
        success: bool = browserController.open(url, run)
        
    if commandArgs.unattended and not success:
        relSourcePrepathsOfErrors.append(relSourcePrepath)
        
    print(f'\n******** END TEST {relSourcePrepath} ********\n\n')


# ======== Perform individual tests

for switches in (('', '-f ') if commandArgs.fcall else ('',)):
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

    # test('demos/parcel_demo/test_shipment', False, switches, parcelJs=True)  # ??? Must be before nodejs_demo, why?  # BROKEN - needs update to Parcel V2
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
    
    # test('demos/cyclejs_demo/cyclejs_demo', False, switches)     # Broken
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
        # ---- Make docs, the resulting files are untracked
        if not commandArgs.unattended:
            origDir: str = os.getcwd()
            sphinxDir: str = '/'.join([appRootDir, 'docs/sphinx'])
            os.chdir(sphinxDir)
            os.system('touch *.rst')
            os.system('make html')
            os.chdir(origDir)
            
        # ---- Terminate
        print('\nSHIPMENT TEST SUCCEEDED\n')
        sys.exit(0)