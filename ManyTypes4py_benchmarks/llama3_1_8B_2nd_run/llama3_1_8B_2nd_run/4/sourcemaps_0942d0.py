import os
import math
import collections
import json
import shutil
from org.transcrypt import utils

'\nA cascaded mini mapping is made as follows:\n    - First generated a non-cascaded pretty map\n    - After that have the minifier generate a shrink map and load that\n    - After that cascade the two to obtain the mini map and save that\n'

lineNrLength: int = 6
maxNrOfSourceLinesPerModule: int = 1000000

class Base64VlqConverter:
    def __init__(self) -> None:
        self.encoding: str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
        self.decoding: dict[str, int] = dict(((char, i) for i, char in enumerate(self.encoding)))
        self.prefabSize: int = 256
        self.prefab: list[str] = [self.encode([i], True) for i in range(self.prefabSize)]

    def encode(self, numbers: list[int], init: bool) -> str:
        segment: str = ''
        for number in numbers:
            if not init and 0 < number < self.prefabSize:
                field: str = self.prefab[number]
            else:
                signed: str = bin(abs(number))[2:] + ('1' if number < 0 else '0')
                nChunks: int = math.ceil(len(signed) / 5.0)
                padded: str = (5 * '0' + signed)[-nChunks * 5:]
                chunks: list[str] = [('1' if iChunk else '0') + padded[iChunk * 5:(iChunk + 1) * 5] for iChunk in range(nChunks - 1, -1, -1)]
                field: str = ''.join([self.encoding[int(chunk, 2)] for chunk in chunks])
            segment += field
        return segment

    def decode(self, segment: str) -> list[int]:
        numbers: list[int] = []
        accu: int = 0
        weight: int = 1
        for char in segment:
            ordinal: int = self.decoding[char]
            isContinuation: bool = ordinal >= 32
            if isContinuation:
                ordinal -= 32
            if weight == 1:
                sign: int = -1 if ordinal % 2 else 1
                ordinal //= 2
            accu += weight * ordinal
            if isContinuation:
                if weight == 1:
                    weight = 16
                else:
                    weight *= 32
            else:
                numbers.append(sign * accu)
                accu = 0
                weight = 1
        return numbers

base64VlqConverter: Base64VlqConverter = Base64VlqConverter()
mapVersion: int = 3
iTargetLine: int
iTargetColumn: int
iSourceIndex: int
iSourceLine: int
iSourceColumn: int

class SourceMapper:
    def __init__(self, moduleName: str, targetDir: str, minify: bool, dump: bool) -> None:
        self.moduleName: str = moduleName
        self.targetDir: str = targetDir
        self.minify: bool = minify
        self.dump: bool = dump

    def generateAndSavePrettyMap(self, sourceLineNrs: list[int]) -> None:
        self.prettyMappings: list[list[int]] = [[targetLineIndex, 0, 0, sourceLineNr - 1, 0] for targetLineIndex, sourceLineNr in enumerate(sourceLineNrs)]
        self.prettyMappings.sort()
        infix: str = '.pretty' if self.minify else ''
        self.save(self.prettyMappings, infix)
        if self.dump:
            self.dumpMap(self.prettyMappings, infix, '.py')
            self.dumpDeltaMap(self.prettyMappings, infix)

    def cascadeAndSaveMiniMap(self) -> None:
        def getCascadedMapping(self, shrinkMapping: list[int]) -> list[int]:
            prettyMapping: list[int] = self.prettyMappings[min(shrinkMapping[iSourceLine], len(self.prettyMappings) - 1)]
            result: list[int] = shrinkMapping[:iTargetColumn + 1] + prettyMapping[iSourceIndex:]
            if self.dump:
                self.cascadeMapdumpFile.write('{} {} {}\n'.format(result, shrinkMapping, prettyMapping))
            return result
        if self.dump:
            self.cascadeMapdumpFile = utils.create(f'{self.targetDir}/{self.moduleName}.cascade_map_dump')
        self.miniMappings: list[list[int]] = [getCascadedMapping(self, shrinkMapping) for shrinkMapping in self.shrinkMappings]
        self.miniMappings.sort()
        self.save(self.miniMappings, '')
        if self.dump:
            self.cascadeMapdumpFile.close()

    def loadShrinkMap(self) -> None:
        with open(f'{self.targetDir}/{self.moduleName}.shrink.map') as mapFile:
            rawMap: dict[str, any] = json.loads(mapFile.read())
        deltaMappings: list[list[list[int]]] = [[base64VlqConverter.decode(segment) for segment in group.split(',')] for group in rawMap['mappings'].split(';')]
        self.shrinkMappings: list[list[int]] = []
        for groupIndex, deltaGroup in enumerate(deltaMappings):
            for segmentIndex, deltaSegment in enumerate(deltaGroup):
                if deltaSegment:
                    if segmentIndex:
                        self.shrinkMappings.append([groupIndex, deltaSegment[0] + self.shrinkMappings[-1][1]])
                    else:
                        self.shrinkMappings.append([groupIndex, deltaSegment[0]])
                    for i in range(1, 4):
                        if groupIndex or segmentIndex:
                            self.shrinkMappings[-1].append(deltaSegment[i] + self.shrinkMappings[-2][i + 1])
                        else:
                            try:
                                self.shrinkMappings[-1].append(deltaSegment[i])
                            except:
                                self.shrinkMappings[-1].append(0)
        self.shrinkMappings.sort()
        if self.dump:
            self.dumpMap(self.shrinkMappings, '.shrink', '.py')
            self.dumpDeltaMap(deltaMappings, '.shrink')

    def save(self, mappings: list[list[int]], infix: str) -> None:
        deltaMappings: list[list[list[int]]] = []
        oldMapping: list[int] = [-1, 0, 0, 0, 0]
        for mapping in mappings:
            newGroup: bool = mapping[iTargetLine] != oldMapping[iTargetLine]
            if newGroup:
                deltaMappings.append([])
            deltaMappings[-1].append([])
            if newGroup:
                deltaMappings[-1][-1].append(mapping[iTargetColumn])
            else:
                deltaMappings[-1][-1].append(mapping[iTargetColumn] - oldMapping[iTargetColumn])
            for i in [iSourceIndex, iSourceLine, iSourceColumn]:
                deltaMappings[-1][-1].append(mapping[i] - oldMapping[i])
            oldMapping = mapping
        rawMap: dict[str, any] = collections.OrderedDict([('version', mapVersion), ('file', f'{self.moduleName}.js'), ('sources', [f'{self.moduleName}{infix}.py']), ('mappings', ';'.join([','.join([base64VlqConverter.encode(segment) for segment in group]) for group in deltaMappings]))])
        with utils.create(f'{self.targetDir}/{self.moduleName}{infix}.map') as mapFile:
            mapFile.write(json.dumps(rawMap, indent='\t'))
        if self.dump:
            self.dumpMap(mappings, infix, '.py')
            self.dumpDeltaMap(deltaMappings, infix)

    def dumpMap(self, mappings: list[list[int]], infix: str, sourceExtension: str) -> None:
        with utils.create(f'{self.targetDir}/{self.moduleName}{infix}.map_dump') as mapdumpFile:
            mapdumpFile.write(f'mapVersion: {mapVersion}\n\n')
            mapdumpFile.write(f'targetPath: {self.moduleName}.js\n\n')
            mapdumpFile.write(f'sourcePath: {self.moduleName}{infix}{sourceExtension}\n\n')
            mapdumpFile.write('mappings:\n')
            for mapping in mappings:
                mapdumpFile.write('\t{}\n'.format(mapping))

    def dumpDeltaMap(self, deltaMappings: list[list[list[int]]], infix: str) -> None:
        with utils.create(f'{self.targetDir}/{self.moduleName}{infix}.delta_map_dump') as deltaMapdumpFile:
            for group in deltaMappings:
                deltaMapdumpFile.write('(New group) ')
                for segment in group:
                    deltaMapdumpFile.write('Segment: {}\n'.format(segment))

    def generateMultilevelMap(self) -> None:
        utils.log(False, 'Saving multi-level sourcemap in: {}\n')
        self.loadShrinkMap()
        self.cascadeAndSaveMiniMap()
