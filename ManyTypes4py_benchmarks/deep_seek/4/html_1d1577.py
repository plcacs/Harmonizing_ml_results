from typing import Dict, List, Tuple, Optional, Any, Union, Iterable, Callable
from org.transcrypt.stubs.browser import __main__, __envir__, __pragma__

__pragma__('nokwargs')

okColor: str = 'green'
errorColor: str = 'red'
highlightColor: str = 'yellow'
testletNameColor: str = 'blue'
messageDivId: str = 'message'
referenceDivId: str = 'python'
refResultDivId: str = 'pyresults'
refPosDivId: str = 'pypos'
testDivId: str = 'transcrypt'
tableId: str = 'resulttable'
resultsDivId: str = 'results'
faultRowClass: str = 'faultrow'
testletHeaderClass: str = 'testletheader'
transValClass: str = 'trans-val'
transPosClass: str = 'trans-pos'
pyValClass: str = 'py-val'
pyPosClass: str = 'py-pos'
excAreaId: str = 'exc-area'
excHeaderClass: str = 'exc-header'
forceCollapseId: str = 'force-collapse'
forceExpandId: str = 'force-expand'

def itemsAreEqual(item0: str, item1: str) -> bool:
    return ' '.join(item0.split()) == ' '.join(item1.split())

class HTMLGenerator(object):
    def __init__(self, filenameBase: Optional[str] = None) -> None:
        self._fnameBase: Optional[str] = filenameBase

    def generate_html(self, refDict: Dict[str, List[Tuple[str, str]]]) -> None:
        minified: bool = False
        if self._fnameBase is None:
            raise ValueError('Filename Base must be defined to generate')
        fname: str = self._fnameBase + '.html'
        print(f'Generating {fname}')
        jsPath: str = f'__target__/{self._fnameBase.split("/")[-1]}.js'
        with open(fname, 'w', encoding='UTF-8') as f:
            f.write("<html><head><meta charset = 'UTF-8'>")
            self._writeCSS(f)
            f.write('</head><body>')
            self._writeStatusHeaderTemplate(f)
            dc: DataConverter = DataConverter()
            dc.writeHiddenResults(f, refDict)
            self._writeTableArea(f)
            f.write('<script type="module" src="{}"></script>\n\n'.format(jsPath))
            f.write('</body></html>')

    def _writeCSS(self, f: Any) -> None:
        cssOut: str = '\n        <style>\n          body {\n            max-width: 100%;\n          }\n          .faultrow > td {\n             background-color: LightCoral;\n          }\n          #resulttable {\n            border-collapse: collapse;\n            width: 100%;\n            table-layout: fixed;\n          }\n          #resulttable th, #resulttable td {\n            border: 1px solid grey;\n          }\n          .testletheader > td {\n            background-color: LightSkyBlue;\n          }\n          .header-pos {\n            width: 20%;\n          }\n          .header-val {\n            width: 30%;\n          }\n          .py-pos,.trans-pos {\n            width: 20%;\n            overflow: hidden;\n          }\n          .py-val, .trans-val {\n            width: 30%;\n            overflow-x: auto;\n          }\n          .exc-header {\n          color: red;\n          }\n          .collapsed {\n            display: None;\n          }\n        </style>\n        '
        f.write(cssOut)

    def _writeStatusHeaderTemplate(self, f: Any) -> None:
        f.write('<b>Status:</b>\n')
        f.write('<div id="{}"></div><br><br>\n\n'.format(messageDivId))

    def _writeTableArea(self, f: Any) -> None:
        f.write('<div id="{}"></div>'.format(excAreaId))
        f.write('<div id="{}">'.format(resultsDivId))
        f.write('<div> <a id="{}" href="#"> Collapse All</a> <a id="{}" href="#">Expand All</a></div>'.format(forceCollapseId, forceExpandId))
        f.write('<table id="{}"><thead><tr> <th colspan="2"> CPython </th> <th colspan="2"> Transcrypt </th> </tr>'.format(tableId))
        f.write('<tr> <th class="header-pos"> Location </th> <th class="header-val"> Value </th> <th class="header-val"> Value </th> <th class="header-pos"> Location </th> </tr></thead><tbody></tbody>')
        f.write('</table>')
        f.write('</div>')

class DataConverter(object):
    def writeHiddenResults(self, f: Any, refDict: Dict[str, List[Tuple[str, str]]]) -> None:
        f.write('<div id="{}" style="display: None">'.format(referenceDivId))
        for key in refDict.keys():
            itemData: str = ' | '.join([x[1] for x in refDict[key]])
            posContent: str = ' | '.join([x[0] for x in refDict[key]])
            f.write('<div id="{}">\n'.format(key))
            f.write('<div id="{}">{}</div>\n\n'.format(refResultDivId, itemData))
            f.write('<div id="{}">{}</div>\n'.format(refPosDivId, posContent))
            f.write('</div>\n')
        f.write('</div></div>\n')

    def getPythonResults(self) -> Dict[str, Iterable[Tuple[str, str]]]:
        refData: Any = document.getElementById(referenceDivId)
        refDict: Dict[str, Iterable[Tuple[str, str]]] = {}
        for child in refData.children:
            keyName: str = child.getAttribute('id')
            posData: List[str]
            resultData: List[str]
            posData, resultData = self._extractPosResult(child)
            refDict[keyName] = zip(posData, resultData)
        return refDict

    def _extractPosResult(self, elem: Any) -> Tuple[List[str], List[str]]:
        resultData: Optional[List[str]] = None
        posData: Optional[List[str]] = None
        for e in elem.children:
            idStr: str = e.getAttribute('id')
            if idStr == refResultDivId:
                resultData = e.innerHTML.split(' | ')
            elif idStr == refPosDivId:
                posData = e.innerHTML.split(' | ')
            else:
                pass
        return (posData or [], resultData or [])

def getRowClsName(name: str) -> str:
    return 'mod-' + name

class JSTesterUI(object):
    def __init__(self) -> None:
        self.expander: TestModuleExpander = TestModuleExpander()

    def setOutputStatus(self, success: bool) -> None:
        if success:
            document.getElementById(messageDivId).innerHTML = '<div style="color: {}">Test succeeded</div>'.format(okColor)
        else:
            document.getElementById(messageDivId).innerHTML = '<div style="color: {}"><b>Test failed</b></div>'.format(errorColor)

    def appendSeqRowName(self, name: str, errCount: int) -> None:
        table: Any = document.getElementById(tableId)
        row: Any = table.insertRow(-1)
        row.id = name
        row.classList.add(testletHeaderClass)
        self.expander.setupCollapseableHeader(row, errCount == 0)
        headerCell: Any = row.insertCell(0)
        headerCell.innerHTML = name + ' | Errors = ' + str(errCount)
        headerCell.colSpan = 4
        headerCell.style.textAlign = 'center'

    def appendTableResult(self, name: str, testPos: Optional[str], testItem: Optional[str], refPos: str, refItem: str, collapse: bool = False) -> None:
        clsName: str = getRowClsName(name)
        table: Any = document.getElementById(tableId)
        row: Any = table.insertRow(-1)
        row.classList.add(clsName)
        if not itemsAreEqual(testItem or '', refItem):
            row.classList.add(faultRowClass)
            refPos = '!!!' + refPos
        else:
            self.expander.setCollapsed(row, collapse)
        cpy_pos: Any = row.insertCell(0)
        cpy_pos.innerHTML = refPos
        cpy_pos.classList.add(pyPosClass)
        cpy_val: Any = row.insertCell(1)
        cpy_val.innerHTML = refItem
        cpy_val.classList.add(pyValClass)
        trans_val: Any = row.insertCell(2)
        if testItem is not None:
            trans_val.innerHTML = testItem
        trans_val.classList.add(transValClass)
        trans_pos: Any = row.insertCell(3)
        if testPos is not None:
            trans_pos.innerHTML = testPos
        trans_pos.classList.add(transPosClass)

    def showException(self, testname: str, exc: Exception) -> None:
        excElem: Any = document.getElementById(excAreaId)
        header: Any = document.createElement('H2')
        header.classList.add(excHeaderClass)
        header.innerHTML = 'Exception Thrown in JS Runtime'
        excElem.appendChild(header)
        content: Any = document.createElement('p')
        content.innerHTML = 'Exception in {}: {}'.format(testname, str(exc))
        excElem.appendChild(content)
        stacktrace: Any = document.createElement('p')
        if hasattr(exc, 'stack') and exc.stack is not None:
            stacktrace.innerHTML = str(exc.stack)
        else:
            stacktrace.innerHTML = 'No Stack Trace Available!'

class TestModuleExpander(object):
    def __init__(self) -> None:
        self.collapsedClass: str = 'collapsed'
        self.modCollapseClass: str = 'mod-collapsed'
        self._expandCollapseAllFuncs()

    def setCollapsed(self, row: Any, collapse: bool) -> None:
        if collapse:
            row.classList.add(self.collapsedClass)
        else:
            row.classList.remove(self.collapsedClass)

    def setupCollapseableHeader(self, row: Any, startCollapsed: bool = False) -> None:
        if startCollapsed:
            row.classList.add(self.modCollapseClass)

        def toggleCollapse(evt: Any) -> None:
            headerRow: Any = evt.target.parentElement
            doCollapse: bool = not headerRow.classList.contains(self.modCollapseClass)
            self.collapseModule(headerRow, doCollapse)
        row.onclick = toggleCollapse

    def collapseModule(self, headerRow: Any, doCollapse: bool) -> None:
        name: str = headerRow.id
        table: Any = document.getElementById(tableId)
        clsName: str = getRowClsName(name)
        allRows: Any = table.tHead.children
        rows: Iterable[Any] = filter(lambda x: x.classList.contains(clsName), allRows)
        for row in rows:
            self.setCollapsed(row, doCollapse)
        if doCollapse:
            headerRow.classList.add(self.modCollapseClass)
        else:
            headerRow.classList.remove(self.modCollapseClass)

    def _expandCollapseAllFuncs(self) -> None:
        def applyToAll(evt: Any, collapse: bool) -> None:
            table: Any = document.getElementById(tableId)
            filtFunc: Callable[[Any], bool] = lambda x: x.classList.contains(testletHeaderClass)
            headerRows: Iterable[Any] = filter(filtFunc, table.tHead.children)
            for headerRow in headerRows:
                self.collapseModule(headerRow, collapse)

        def collapseAll(evt: Any) -> bool:
            evt.preventDefault()
            applyToAll(evt, True)
            return False

        def expandAll(evt: Any) -> bool:
            evt.preventDefault()
            applyToAll(evt, False)
            return False

        forceCollapse: Any = document.getElementById(forceCollapseId)
        forceCollapse.onclick = collapseAll
        forceExpand: Any = document.getElementById(forceExpandId)
        forceExpand.onclick = expandAll
