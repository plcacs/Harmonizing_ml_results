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
        self.argParser: argparse.ArgumentParser = argparse.ArgumentParser()
        self.argParser.add_argument('-de', '--dextex', help='show extended exception reports', action='store_true')
        self.argParser.add_argument('-f', '--fcall', help='test fast calls', action='store_true')
        self.argParser.add_argument('-i', '--inst', help='installed version rather than new one', action='store_true')
        self.argParser.add_argument('-b', '--blind', help="don't start browser", action='store_true')
        self.argParser.add_argument('-u', '--unattended', help='unattended mode', action='store_true')
        self.argParser.add_argument('-t', '--transcom', help='transpile command')
        self.__dict__.update(self.argParser.parse_args().__dict__)

class BrowserController:
    def __init__(self) -> None:
        self.options: selenium.webdriver.chrome.options.Options = selenium.webdriver.chrome.options.Options()
        self.options.add_argument('start-maximized')
        if commandArgs.unattended:
            self.options.add_argument('--headless')
            self.options.add_argument('--no-sandbox')
            self.options.add_argument('--disable-gpu')
            self.options.add_argument('disable-infobars')
            self.options.add_argument('--disable-extensions')
            self.options.add_argument('--disable-web-security')
        self.webDriver: selenium.webdriver.Chrome = selenium.webdriver.Chrome(options=self.options)
        self.nrOfTabs: int = 0

    def waitForNewTab(self) -> None:
        while len(self.webDriver.window_handles) <= self.nrOfTabs:
            time.sleep(0.5)
        self.nrOfTabs = len(self.webDriver.window_handles)

    def open(self, url: str, run: bool) -> bool:
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

def getAbsPath(relPath: str) -> str:
    return '{}/{}'.format(appRootDir, relPath)

def test(relSourcePrepath: str, run: bool, extraSwitches: str, outputPrename: str = '', nodeJs: bool = False, parcelJs: bool = False, build: bool = True, pause: int = 0, needsAttention: bool = False) -> None:
    if commandArgs.unattended and needsAttention:
        return
    print(f'\n\n******** BEGIN TEST {relSourcePrepath} ********\n')
    time.sleep(pause)
    sourcePrepath: str = getAbsPath(relSourcePrepath)
    sourcePrepathSplit: list[str] = sourcePrepath.split('/')
    sourceDir: str = '/'.join(sourcePrepathSplit[:-1])
    moduleName: str = sourcePrepathSplit[-1]
    targetDir: str = f'{sourceDir}/__target__'
    targetPrepath: str = f'{targetDir}/{moduleName}'
    outputPath: str = f'{targetDir}/{outputPrename}.out'
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    defaultSwitches: str = '-da -sf -de -m -n '
    if commandArgs.dextex:
        defaultSwitches += '-de '
    if build:
        defaultSwitches += '-b '
    if run:
        os.system(f'{transpileCommand} -r {defaultSwitches}{extraSwitches}{sourcePrepath}')
    if parcelJs:
        origDir: str = os.getcwd()
        os.chdir(sourceDir)
        os.system(f'node test {parcelServerPort} &')
        os.chdir(origDir)
    else:
        command: str = f'{transpileCommand} {defaultSwitches}{extraSwitches}{sourcePrepath}'
        if outputPrename:
            output: str = subprocess.check_output(command, universal_newlines=True, shell=True)
            with open(outputPath, 'w') as outputFile:
                print(output, file=outputFile)
        else:
            subprocess.run(command, shell=True)
    if nodeJs:
        os.system(f'rollup {targetPrepath}.js --o {targetPrepath}.bundle.js --f cjs')
    if not commandArgs.blind:
        if parcelJs:
            time.sleep(20)
            url: str = parcelServerUrl
        elif nodeJs:
            os.system(f'chmod 777 {targetPrepath}.bundle.js')
            os.system(f'node {targetPrepath}.bundle.js {nodeServerPort} &')
            time.sleep(5)
            url: str = nodeServerUrl
        else:
            time.sleep(5)
            url: str = f'{pythonServerUrl}/{relSourcePrepath}.html'
        success: bool = browserController.open(url, run)
    if commandArgs.unattended and (not success):
        relSourcePrepathsOfErrors.append(relSourcePrepath)
    print(f'\n******** END TEST {relSourcePrepath} ********\n\n')

commandArgs: CommandArgs = CommandArgs()
browserController: BrowserController = BrowserController()
relSourcePrepathsOfErrors: list[str] = []
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
